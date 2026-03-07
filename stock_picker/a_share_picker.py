#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma_gap_5",
    "ma_gap_10",
    "ma_gap_20",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "amount_ratio_5",
    "amount_ratio_20",
    "turnover_ratio_5",
    "turnover_ratio_20",
    "price_pos_20",
    "price_pos_60",
    "gap_open",
    "intraday_body",
    "upper_shadow",
    "lower_shadow",
    "amplitude_5",
    "amplitude_20",
]


@dataclass
class PickerConfig:
    top_n: int
    history_start: str
    cache_dir: Path
    output_dir: Path
    universe_file: Path
    max_symbols: int | None
    min_history_rows: int
    min_price: float
    max_price: float
    min_avg_amount: float
    request_pause_seconds: float
    refresh_cache: bool
    disable_live: bool


def parse_args() -> PickerConfig:
    parser = argparse.ArgumentParser(
        description="Pick top A-share candidates for the next trading day."
    )
    parser.add_argument("--top-n", type=int, default=2, help="Number of picks to output.")
    parser.add_argument(
        "--history-start",
        default="20230101",
        help="History fetch start date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--cache-dir",
        default="stock_picker/data/cache",
        help="Directory used to cache per-symbol daily history.",
    )
    parser.add_argument(
        "--output-dir",
        default="stock_picker/output",
        help="Directory used to write the daily pick files.",
    )
    parser.add_argument(
        "--universe-file",
        default="stock_picker/data/universe.default.csv",
        help="CSV file with at least a 'symbol' column. Optional 'name' column is supported.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=12,
        help="Only use the first N symbols from the universe file. Set 0 to use all symbols.",
    )
    parser.add_argument(
        "--min-history-rows",
        type=int,
        default=120,
        help="Minimum historical rows required per symbol.",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=3.0,
        help="Minimum current price for a candidate.",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=300.0,
        help="Maximum current price for a candidate.",
    )
    parser.add_argument(
        "--min-avg-amount",
        type=float,
        default=300_000_000,
        help="Minimum 20-day average turnover amount in CNY.",
    )
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=0.8,
        help="Pause between fresh history requests to reduce upstream throttling.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force refresh of cached historical data.",
    )
    parser.add_argument(
        "--disable-live",
        action="store_true",
        help="Skip the intraday snapshot enhancement and only use daily data.",
    )
    args = parser.parse_args()
    return PickerConfig(
        top_n=args.top_n,
        history_start=args.history_start,
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        universe_file=Path(args.universe_file),
        max_symbols=args.max_symbols or None,
        min_history_rows=args.min_history_rows,
        min_price=args.min_price,
        max_price=args.max_price,
        min_avg_amount=args.min_avg_amount,
        request_pause_seconds=args.request_pause_seconds,
        refresh_cache=args.refresh_cache,
        disable_live=args.disable_live,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def retry_call(callable_obj, *args, retries: int = 3, delay_seconds: float = 2.0, **kwargs):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return callable_obj(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - network errors vary by environment.
            last_error = exc
            logging.warning("Attempt %s/%s failed: %s", attempt, retries, exc)
            time.sleep(delay_seconds * attempt)
    raise last_error


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().zfill(6)


def load_universe(universe_file: Path, max_symbols: int | None) -> pd.DataFrame:
    universe = pd.read_csv(universe_file, dtype={"symbol": str})
    if "symbol" not in universe.columns:
        raise ValueError(f"{universe_file} must contain a 'symbol' column")
    universe["symbol"] = universe["symbol"].map(normalize_symbol)
    if "name" not in universe.columns:
        universe["name"] = universe["symbol"]
    universe = universe[["symbol", "name"]].drop_duplicates().reset_index(drop=True)
    if max_symbols is not None:
        universe = universe.head(max_symbols).reset_index(drop=True)
    return universe


def fetch_history(
    symbol: str,
    history_start: str,
    cache_dir: Path,
    refresh_cache: bool,
    request_pause_seconds: float,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}.csv"
    if cache_file.exists() and not refresh_cache:
        cached = pd.read_csv(cache_file, dtype={"股票代码": str})
        if not cached.empty:
            cached["日期"] = pd.to_datetime(cached["日期"])
            return cached
    if request_pause_seconds > 0:
        time.sleep(request_pause_seconds)
    hist = retry_call(
        ak.stock_zh_a_hist,
        symbol=symbol,
        period="daily",
        start_date=history_start,
        end_date=datetime.now().strftime("%Y%m%d"),
        adjust="qfq",
        timeout=20,
    )
    if hist.empty:
        raise ValueError(f"No history returned for symbol {symbol}")
    hist = hist.copy()
    hist["股票代码"] = hist["股票代码"].astype(str).str.zfill(6)
    hist["日期"] = pd.to_datetime(hist["日期"])
    hist.to_csv(cache_file, index=False)
    return hist


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return numerator.div(denominator)


def build_feature_frame(history: pd.DataFrame, symbol: str, name: str) -> pd.DataFrame:
    df = history.copy().sort_values("日期").reset_index(drop=True)
    for column in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    close = df["收盘"]
    open_ = df["开盘"]
    high = df["最高"]
    low = df["最低"]
    amount = df["成交额"]
    turnover = df["换手率"]
    prev_close = close.shift(1)
    daily_return = close.pct_change()

    ma_5 = close.rolling(5).mean()
    ma_10 = close.rolling(10).mean()
    ma_20 = close.rolling(20).mean()

    df["ret_1"] = daily_return
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)
    df["ma_gap_5"] = safe_ratio(close, ma_5) - 1
    df["ma_gap_10"] = safe_ratio(close, ma_10) - 1
    df["ma_gap_20"] = safe_ratio(close, ma_20) - 1
    df["volatility_5"] = daily_return.rolling(5).std()
    df["volatility_10"] = daily_return.rolling(10).std()
    df["volatility_20"] = daily_return.rolling(20).std()
    df["amount_ratio_5"] = safe_ratio(amount, amount.rolling(5).mean())
    df["amount_ratio_20"] = safe_ratio(amount, amount.rolling(20).mean())
    df["turnover_ratio_5"] = safe_ratio(turnover, turnover.rolling(5).mean())
    df["turnover_ratio_20"] = safe_ratio(turnover, turnover.rolling(20).mean())

    range_20 = high.rolling(20).max() - low.rolling(20).min()
    range_60 = high.rolling(60).max() - low.rolling(60).min()
    df["price_pos_20"] = safe_ratio(close - low.rolling(20).min(), range_20)
    df["price_pos_60"] = safe_ratio(close - low.rolling(60).min(), range_60)

    df["gap_open"] = safe_ratio(open_, prev_close) - 1
    df["intraday_body"] = safe_ratio(close, open_) - 1
    df["upper_shadow"] = safe_ratio(high - pd.concat([open_, close], axis=1).max(axis=1), close)
    df["lower_shadow"] = safe_ratio(pd.concat([open_, close], axis=1).min(axis=1) - low, close)
    df["amplitude_5"] = df["振幅"].rolling(5).mean() / 100
    df["amplitude_20"] = df["振幅"].rolling(20).mean() / 100
    df["avg_amount_20"] = amount.rolling(20).mean()

    df["target_next_return"] = close.shift(-1) / close - 1
    df["symbol"] = symbol
    df["name"] = name
    return df


def fetch_live_snapshot(disable_live: bool) -> pd.DataFrame | None:
    if disable_live:
        logging.info("Live snapshot disabled; using daily features only.")
        return None
    try:
        snapshot = retry_call(ak.stock_zh_a_spot_em, retries=2, delay_seconds=2.5)
    except Exception as exc:
        logging.warning("Live snapshot unavailable, falling back to daily-only mode: %s", exc)
        return None

    snapshot = snapshot.copy()
    snapshot["代码"] = snapshot["代码"].astype(str).str.zfill(6)
    snapshot = snapshot.rename(
        columns={
            "代码": "symbol",
            "名称": "live_name",
            "最新价": "live_price",
            "涨跌幅": "live_pct_chg",
            "成交额": "live_amount",
            "量比": "live_volume_ratio",
            "换手率": "live_turnover_rate",
            "5分钟涨跌": "live_5m_pct_chg",
            "60日涨跌幅": "live_60d_pct_chg",
            "年初至今涨跌幅": "live_ytd_pct_chg",
            "昨收": "live_prev_close",
        }
    )
    return snapshot[
        [
            "symbol",
            "live_name",
            "live_price",
            "live_pct_chg",
            "live_amount",
            "live_volume_ratio",
            "live_turnover_rate",
            "live_5m_pct_chg",
            "live_60d_pct_chg",
            "live_ytd_pct_chg",
            "live_prev_close",
        ]
    ]


def build_training_data(universe: pd.DataFrame, config: PickerConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_frames: list[pd.DataFrame] = []
    latest_frames: list[pd.DataFrame] = []

    for row in universe.itertuples(index=False):
        symbol = row.symbol
        name = row.name
        try:
            history = fetch_history(
                symbol=symbol,
                history_start=config.history_start,
                cache_dir=config.cache_dir,
                refresh_cache=config.refresh_cache,
                request_pause_seconds=config.request_pause_seconds,
            )
        except Exception as exc:
            logging.warning("Skip %s due to history fetch failure: %s", symbol, exc)
            continue

        features = build_feature_frame(history, symbol=symbol, name=name)
        if len(features) < config.min_history_rows:
            logging.info("Skip %s because it only has %s rows", symbol, len(features))
            continue
        training_frames.append(features.iloc[:-1].copy())
        latest_frames.append(features.iloc[[-1]].copy())

    if not training_frames or not latest_frames:
        raise RuntimeError("No valid symbols were available to train the model.")

    training_df = pd.concat(training_frames, ignore_index=True)
    latest_df = pd.concat(latest_frames, ignore_index=True)
    training_df = training_df.dropna(subset=["target_next_return"]).sort_values("日期").reset_index(drop=True)
    latest_df = latest_df.sort_values("日期").reset_index(drop=True)
    return training_df, latest_df


def train_model(training_df: pd.DataFrame) -> tuple[Pipeline, dict]:
    split_index = int(len(training_df) * 0.8)
    split_index = max(split_index, 1)
    train_df = training_df.iloc[:split_index].copy()
    valid_df = training_df.iloc[split_index:].copy()

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df["target_next_return"])

    metrics = {
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(valid_df)),
    }
    if not valid_df.empty:
        predictions = pipeline.predict(valid_df[FEATURE_COLUMNS])
        actual = valid_df["target_next_return"]
        mae = (actual - predictions).abs().mean()
        rmse = ((actual - predictions) ** 2).mean() ** 0.5
        direction_acc = ((predictions >= 0) == (actual >= 0)).mean()
        metrics.update(
            {
                "validation_mae": float(mae),
                "validation_rmse": float(rmse),
                "validation_directional_accuracy": float(direction_acc),
            }
        )
    return pipeline, metrics


def add_reason_text(result_df: pd.DataFrame) -> pd.DataFrame:
    reasons: list[str] = []
    for row in result_df.itertuples(index=False):
        fragments: list[str] = []
        if getattr(row, "ret_5", 0) > 0.03:
            fragments.append("近5日动量较强")
        if getattr(row, "ma_gap_20", 0) > 0:
            fragments.append("股价位于20日均线上方")
        if getattr(row, "amount_ratio_5", 0) > 1.1:
            fragments.append("成交额高于5日均值")
        if getattr(row, "price_pos_20", 0) > 0.7:
            fragments.append("接近20日区间上沿")
        live_volume_ratio = getattr(row, "live_volume_ratio", None)
        if live_volume_ratio is not None and pd.notna(live_volume_ratio) and live_volume_ratio > 1.2:
            fragments.append("盘中量比放大")
        reasons.append("，".join(fragments[:3]) if fragments else "历史特征综合评分较高")
    result_df = result_df.copy()
    result_df["reason"] = reasons
    return result_df


def rank_candidates(
    latest_df: pd.DataFrame,
    model: Pipeline,
    live_snapshot: pd.DataFrame | None,
    config: PickerConfig,
) -> pd.DataFrame:
    scored = latest_df.copy()
    scored["ml_predicted_return"] = model.predict(scored[FEATURE_COLUMNS])
    scored["current_price"] = scored["收盘"]
    scored["current_pct_chg"] = scored["涨跌幅"]

    if live_snapshot is not None:
        scored = scored.merge(live_snapshot, on="symbol", how="left")
        scored["current_price"] = scored["live_price"].fillna(scored["current_price"])
        scored["current_pct_chg"] = scored["live_pct_chg"].fillna(scored["current_pct_chg"])
        scored["current_amount"] = scored["live_amount"].fillna(scored["成交额"])
        live_bonus = (
            0.0008 * scored["live_volume_ratio"].fillna(1.0).clip(lower=0, upper=5)
            + 0.00015 * scored["live_turnover_rate"].fillna(0.0).clip(lower=0, upper=20)
            - 0.0002 * scored["current_pct_chg"].fillna(0.0).abs().clip(lower=0, upper=10)
        )
    else:
        scored["current_amount"] = scored["成交额"]
        live_bonus = 0.0

    scored["expected_next_return"] = (scored["ml_predicted_return"] + live_bonus).clip(-0.1, 0.1)
    filtered = scored[
        (scored["current_price"] >= config.min_price)
        & (scored["current_price"] <= config.max_price)
        & (scored["avg_amount_20"] >= config.min_avg_amount)
    ].copy()

    filtered = filtered[~filtered["name"].astype(str).str.contains("ST|退", na=False)]
    filtered = filtered.sort_values("expected_next_return", ascending=False).reset_index(drop=True)
    filtered = add_reason_text(filtered)
    filtered["expected_next_return_pct"] = (filtered["expected_next_return"] * 100).round(2)
    filtered["ml_predicted_return_pct"] = (filtered["ml_predicted_return"] * 100).round(2)
    filtered["avg_amount_20_yi"] = (filtered["avg_amount_20"] / 100_000_000).round(2)
    return filtered.head(config.top_n).copy()


def print_summary(result_df: pd.DataFrame, metrics: dict, used_live: bool) -> None:
    if result_df.empty:
        print("No candidates passed the filters.")
        return

    summary = result_df[
        [
            "symbol",
            "name",
            "current_price",
            "current_pct_chg",
            "expected_next_return_pct",
            "ml_predicted_return_pct",
            "avg_amount_20_yi",
            "reason",
        ]
    ].copy()
    summary = summary.rename(
        columns={
            "symbol": "代码",
            "name": "名称",
            "current_price": "当前价",
            "current_pct_chg": "当前涨跌幅(%)",
            "expected_next_return_pct": "预期次日涨跌幅(%)",
            "ml_predicted_return_pct": "模型原始预测(%)",
            "avg_amount_20_yi": "20日均成交额(亿)",
            "reason": "入选原因",
        }
    )
    print(summary.to_string(index=False))
    print()
    print(
        json.dumps(
            {
                "used_live_snapshot": used_live,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def save_outputs(result_df: pd.DataFrame, metrics: dict, output_dir: Path, used_live: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    today_tag = datetime.now().strftime("%Y%m%d")

    json_rows = result_df[
        [
            "symbol",
            "name",
            "current_price",
            "current_pct_chg",
            "expected_next_return",
            "expected_next_return_pct",
            "ml_predicted_return",
            "ml_predicted_return_pct",
            "reason",
        ]
    ].to_dict(orient="records")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "used_live_snapshot": used_live,
        "metrics": metrics,
        "picks": json_rows,
    }

    (output_dir / f"daily_picks_{today_tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "latest_picks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result_df.to_csv(output_dir / f"daily_picks_{today_tag}.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(output_dir / "latest_picks.csv", index=False, encoding="utf-8-sig")


def run(config: PickerConfig) -> pd.DataFrame:
    universe = load_universe(config.universe_file, max_symbols=config.max_symbols)
    logging.info("Loaded %s symbols from %s", len(universe), config.universe_file)
    training_df, latest_df = build_training_data(universe, config)
    logging.info("Prepared %s training rows across %s symbols", len(training_df), latest_df["symbol"].nunique())
    model, metrics = train_model(training_df)
    live_snapshot = fetch_live_snapshot(config.disable_live)
    result_df = rank_candidates(
        latest_df=latest_df,
        model=model,
        live_snapshot=live_snapshot,
        config=config,
    )
    save_outputs(
        result_df=result_df,
        metrics=metrics,
        output_dir=config.output_dir,
        used_live=live_snapshot is not None,
    )
    print_summary(result_df, metrics=metrics, used_live=live_snapshot is not None)
    return result_df


def main() -> int:
    configure_logging()
    config = parse_args()
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
