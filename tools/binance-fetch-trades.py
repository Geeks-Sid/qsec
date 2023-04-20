import argparse
import datetime as dt
import json
import logging
from typing import List

import common
import pandas as pd
import requests

import qsec.app
import qsec.logging
import qsec.time

api = "https://api.binance.us"


def buyer_maker_to_aggr_side(buyer_is_maker: bool):
    if buyer_is_maker:
        return -1
    else:
        return +1


def call_http_agg_trades(
    symbol: str, start_time: int = None, end_time: int = None, from_id: int = None
) -> str:
    path = "/api/v3/aggTrades"
    options = {"symbol": symbol, "limit": 1000}
    if start_time is not None:
        options["startTime"] = start_time
    if end_time is not None:
        options["endTime"] = end_time
    if from_id is not None:
        options["fromId"] = from_id

    url = f"{api}{path}"
    logging.info(f"making URL request: {url}, options: {options}")
    response = requests.get(url, params=options)
    response.raise_for_status()

    return response.text


TRADES_PER_API_CALL = 1000
ONE_HOUR = 60 * 60 * 1000


def get_trades(symbol, dt_from, dt_to):
    ts_from = int(dt_from.timestamp() * 1000)
    ts_to = int(dt_to.timestamp() * 1000)

    logging.info(f"from: {ts_from}")
    logging.info(f"to: {ts_to}")

    all_trades = []
    ts0 = ts_from
    window_size_ms = ONE_HOUR

    while ts0 < ts_to:
        while True:
            ts1 = min(ts0 + window_size_ms, ts_to)
            url = f"https://api.example.com/trades?symbol={symbol}&start={ts0}&end={ts1}&limit={TRADES_PER_API_CALL}"
            response = requests.get(url)
            trades = json.loads(response.text)
            if len(trades) >= TRADES_PER_API_CALL:
                window_size_ms //= 2
                logging.info(f"halving window, to {window_size_ms}")
            else:
                break

        logging.info(
            f"[{ts0} -> {ts1}] trades {len(trades)} ({len(trades) * 1000 / window_size_ms:.1f} per sec)"
        )
        all_trades += [trade for trade in trades]
        ts0 = ts1

        window_size_ms = min(window_size_ms * 5 // 4, ONE_HOUR)

    return all_trades


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    renames = {
        "a": "tradeId",
        "p": "price",
        "q": "qty",
        "f": "firstTradeId",
        "l": "lastTradeId",
        "T": "timestamp",
    }
    df.rename(columns=renames, inplace=True)

    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop("timestamp", axis=1, inplace=True)
    df["side"] = df["m"].map(buyer_maker_to_aggr_side) if len(df) else pd.Series()
    df.drop(["m", "firstTradeId", "lastTradeId"], axis=1, inplace=True)

    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    df["price"] = pd.to_numeric(df["price"])
    df["qty"] = pd.to_numeric(df["qty"])

    return df


def plus_one_hour(t_milliseconds: int) -> int:
    return t_milliseconds + (60 * 60) * 1000


def find_any_trade_in_period(symbol: str, beg_ms: int, end_ms: int) -> int:
    logging.info(
        f"searching for any trade within window of interest [{beg_ms}, {end_ms}]"
    )
    end_time = plus_one_hour(beg_ms)
    logging.info(
        f"requesting range {qsec.time.epoch_ms_to_dt(beg_ms)} to {qsec.time.epoch_ms_to_dt(end_time)}"
    )
    raw_json = call_http_agg_trades(symbol, start_time=beg_ms, end_time=end_time)
    trades = pd.DataFrame(json.loads(raw_json))
    if len(trades):
        return trades["a"].min()
    else:
        return None


def find_earliest_trade(
    symbol: str, beg_ms: int, end_ms: int, seek_trade_id: int
) -> int:
    logging.info("searching for earliest trade within window of interest")
    earliest_trade_id = seek_trade_id
    while True:
        seek_trade_id = earliest_trade_id - 1000
        raw_json = call_http_agg_trades(symbol, fromId=seek_trade_id)
        trades = pd.DataFrame(json.loads(raw_json))
        trades.set_index("a", inplace=True)
        trades.sort_index(inplace=True)
        trades = trades.between(beg_ms, end_ms, inclusive=True)
        if trades.empty:
            break
        min_trade_id = trades.index.min()
        if min_trade_id == earliest_trade_id:
            break
        logging.info(f"found new earliest trade id {min_trade_id}")
        earliest_trade_id = min_trade_id
    return earliest_trade_id


def fetch_all_trades(symbol: str, beg_ms: int, end_ms: int, from_id: int) -> pd.DataFrame:
    logging.info("fetching all trades for window")
    all_dfs = []
    cursor = from_id
    count = 0
    while True:
        # fetch trades for current ID range
        raw_json = call_http_agg_trades(symbol, fromId=cursor)
        trades = pd.DataFrame(json.loads(raw_json))
        trades = trades[trades["T"].between(beg_ms, end_ms, inclusive=True)]
        if trades.empty:
            break
        all_dfs.append(trades)
        count += len(trades)
        cursor = trades["a"].max() + 1
        highest_time = trades["T"].idxmax()
        logging.info(f"trades: {count}, time: {qsec.time.epoch_ms_to_dt(highest_time)}")

    df = pd.concat(all_dfs, ignore_index=True)
    df = normalise(df)
    return df


def list_missing_ids(df: pd.DataFrame) -> List[int]:
    if df.empty:
        return []

    expected = pd.RangeIndex(start=df["tradeId"].min(), stop=df["tradeId"].max() + 1)
    missing = expected[~expected.isin(df["tradeId"])].tolist()

    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", type=str, help="symbol", required=True)
    parser.add_argument("--fromDt", type=str, help="begin date", required=True)
    parser.add_argument("--upto", type=str, help="to date", required=True)
    return parser.parse_args()


def process_args(args: argparse.Namespace) -> tuple:
    try:
        fromDt = qsec.time.to_date(args.fromDt)
        uptoDt = qsec.time.to_date(args.upto)
        if fromDt >= uptoDt:
            raise qsec.app.EasyError("'from' date must be before 'upto' date")
        return fromDt, uptoDt
    except qsec.app.EasyError as e:
        raise qsec.app.EasyError(f"Error processing arguments: {e}")


def fetch_trades_for_date(symbol: str, kline_date: dt.date) -> pd.DataFrame:
    logging.info(f"fetching trades for date {kline_date}")
    t0 = qsec.time.date_to_datetime(kline_date)
    t1 = qsec.time.date_to_datetime(kline_date + dt.timedelta(days=1))
    t0 = int(t0.timestamp() * 1000)
    t1 = int(t1.timestamp() * 1000)

    seek_trade_id = find_any_trade_in_period(symbol, t0.value, t1.value)
    logging.info(f"initial seek tradeId: {seek_trade_id}")
    earliest_trade_id = find_earliest_trade(symbol, t0.value, t1.value, seek_trade_id)
    logging.info(f"window earliest tradeId: {earliest_trade_id}")
    df = fetch_all_trades(symbol, t0.value, t1.value, earliest_trade_id)
    missingIds = list_missing_ids(df)
    if missingIds:
        logging.info(f"{len(missingIds)} missing tradeIds detected")
    else:
        logging.info("no missing tradeIds detected")
    return df


def fetch(symbol: str, fromDt: dt.date, uptoDt: dt.date, sid: str) -> None:
    dates = qsec.time.dates_in_range(fromDt, uptoDt)
    for d in dates:
        df = fetch_trades_for_date(symbol, d)
        common.save_dateframe(symbol, d, df, sid, "binance", "trades")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    fromDt, uptoDt = process_args(args)
    sid = common.build_assetid(args.sym, "BNC", is_cash=True)
    logging.info(f"Fetching trades for {args.sym} from {fromDt} to {uptoDt}")
    fetch(args.sym, fromDt, uptoDt, sid)


if __name__ == "__main__":
    main()
