"""
Coinbase intraday paper-trading agent
------------------------------------
Paper trading only. This app uses public Coinbase Exchange market data and never
submits real orders. It scans markets and simulates buys/sells while the app is open.

Install:
    py -m pip install --upgrade streamlit pandas numpy requests
Run:
    py -m streamlit run coinbase_intraday_paper_agent.py
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "paper_state.json"
MEMORY_FILE = APP_DIR / "strategy_memory.json"
BASE_URL = "https://api.exchange.coinbase.com"
STARTING_CASH_GBP = 100.0
FEE_RATE = 0.006  # conservative simulated fee estimate per side
CYCLE_SECONDS = 30
CANDLE_GRANULARITY = 300  # 5-minute candles
CANDLE_CACHE_SECONDS = 30
MAX_OPEN_POSITIONS = 12
MIN_POSITION_GBP = 2.0
MAX_POSITION_FRACTION = 0.40  # of currently available cash
DEFAULT_MARKET_LIMIT = 20
REQUEST_TIMEOUT = 12

STRATEGIES = [
    "EMA Momentum",
    "RSI Reversal",
    "Bollinger Rebound",
    "Volume Breakout",
    "Trend Pullback",
]

st.set_page_config(page_title="Coinbase Paper Trading Agent", layout="wide")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def money(value: float) -> str:
    return f"£{float(value):,.2f}"


def default_state() -> dict[str, Any]:
    return {
        "cash_gbp": STARTING_CASH_GBP,
        "starting_cash_gbp": STARTING_CASH_GBP,
        "positions": {},
        "trades": [],
        "events": [],
        "last_cycle": None,
        "last_cycle_note": "Waiting for first scan.",
        "total_fees_gbp": 0.0,
    }


def load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path: Path, value: Any) -> None:
    try:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
        temp.replace(path)
    except Exception as e:
        st.error(f"Failed to save state: {e}")


def load_state() -> dict[str, Any]:
    state = load_json(STATE_FILE, None)
    if not isinstance(state, dict):
        state = default_state()
        save_json(STATE_FILE, state)
    defaults = default_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    state.setdefault("positions", {})
    state.setdefault("trades", [])
    state.setdefault("events", [])
    return state


def add_event(state: dict[str, Any], message: str, kind: str = "INFO") -> None:
    state.setdefault("events", []).insert(
        0, {"time": utc_now(), "type": kind, "message": str(message)[:500]}
    )
    state["events"] = state["events"][:250]


def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        BASE_URL + path,
        params=params,
        headers={"User-Agent": "CoinbasePaperAgent/1.0"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_gbp_products() -> list[str]:
    try:
        products = api_get("/products")
        result = []
        for item in products:
            pid = item.get("id", "")
            if pid.endswith("-GBP") and item.get("status") == "online":
                result.append(pid)
        preferred = ["BTC-GBP", "ETH-GBP", "SOL-GBP", "XRP-GBP", "ADA-GBP",
                     "DOGE-GBP", "LTC-GBP", "LINK-GBP", "AVAX-GBP", "DOT-GBP"]
        ordered = [p for p in preferred if p in result]
        ordered.extend(sorted(p for p in result if p not in ordered))
        return ordered
    except Exception:
        return ["BTC-GBP", "ETH-GBP", "SOL-GBP", "XRP-GBP"]


@st.cache_data(ttl=5, show_spinner=False)
def get_ticker(product_id: str) -> dict[str, Any]:
    return api_get(f"/products/{product_id}/ticker")


@st.cache_data(ttl=CANDLE_CACHE_SECONDS, show_spinner=False)
def get_candles(product_id: str, granularity: int = CANDLE_GRANULARITY) -> pd.DataFrame:
    try:
        raw = api_get(
            f"/products/{product_id}/candles",
            params={"granularity": granularity},
        )
        if not raw or not isinstance(raw, list):
            return pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])
        df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        for col in ["low", "high", "open", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.dropna(subset=["close", "open", "high", "low"])
        return df.tail(300).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])


def get_price_gbp(product_id: str) -> float:
    ticker = get_ticker(product_id)
    price = float(ticker.get("price", 0))
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"Invalid ticker price for {product_id}")
    return price


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 5:
        return df
    out = df.copy()
    close = out["close"]
    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["rsi"] = out["rsi"].fillna(50)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_upper"] = mid + 2 * std
    out["bb_lower"] = mid - 2 * std
    out["vol_avg20"] = out["volume"].rolling(20).mean()
    out["ret"] = close.pct_change().fillna(0)
    return out


def strategy_signal(df: pd.DataFrame, strategy: str) -> tuple[str, str]:
    if len(df) < 55:
        return "WAIT", f"Need at least 55 candles; received {len(df)}."
    data = add_indicators(df)
    row = data.iloc[-2]
    prev = data.iloc[-3]
    if not np.isfinite(row["close"]) or row["close"] <= 0:
        return "WAIT", "Candle data is incomplete."
    if strategy == "EMA Momentum":
        if row["ema9"] > row["ema21"] and prev["ema9"] <= prev["ema21"]:
            return "BUY", "EMA 9 crossed above EMA 21."
        if row["ema9"] < row["ema21"]:
            return "SELL", "EMA 9 is below EMA 21."
        return "WAIT", "No fresh bullish EMA crossover."
    if strategy == "RSI Reversal":
        if prev["rsi"] < 30 and row["rsi"] >= 30:
            return "BUY", "RSI recovered above 30."
        if row["rsi"] > 70:
            return "SELL", "RSI is above 70."
        return "WAIT", f"RSI is {row['rsi']:.1f}; no reversal trigger."
    if strategy == "Bollinger Rebound":
        if prev["close"] < prev["bb_lower"] and row["close"] >= row["bb_lower"]:
            return "BUY", "Price recovered inside the lower Bollinger Band."
        if row["close"] >= row["bb_upper"]:
            return "SELL", "Price reached the upper Bollinger Band."
        return "WAIT", "No Bollinger rebound trigger."
    if strategy == "Volume Breakout":
        volume_ok = row["volume"] > row["vol_avg20"] * 1.5 if row["vol_avg20"] > 0 else False
        if volume_ok and row["close"] > data.iloc[-22:-2]["high"].max():
            return "BUY", "High-volume close above the recent range."
        if row["close"] < row["ema21"]:
            return "SELL", "Price fell below EMA 21."
        return "WAIT", "No high-volume upside breakout."
    if strategy == "Trend Pullback":
        uptrend = row["ema21"] > row["ema50"]
        near_ema = row["close"] >= row["ema21"] * 0.985
        recovered = row["close"] > row["open"] and prev["close"] <= prev["ema9"]
        if uptrend and near_ema and recovered and 30 <= row["rsi"] <= 70:
            return "BUY", "Pullback recovery in an uptrend."
        if row["close"] < row["ema50"]:
            return "SELL", "Price fell below EMA 50."
        return "WAIT", "No trend-pullback entry."
    return "WAIT", "Unknown strategy."


def portfolio_value(state: dict[str, Any], prices: dict[str, float]) -> tuple[float, float]:
    holdings = 0.0
    for product_id, pos in state.get("positions", {}).items():
        price = prices.get(product_id, pos.get("last_price", pos.get("entry_price", 0.0)))
        holdings += float(pos.get("quantity", 0.0)) * float(price)
    cash = float(state.get("cash_gbp", 0.0))
    return cash + holdings, holdings


def paper_buy(state: dict[str, Any], product_id: str, price: float, strategy: str) -> bool:
    if product_id in state["positions"]:
        return False
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        add_event(state, "Buy skipped: maximum open positions reached.", "SKIP")
        return False
    cash = float(state["cash_gbp"])
    budget = min(cash * MAX_POSITION_FRACTION, cash / max(1, MAX_OPEN_POSITIONS - len(state["positions"])))
    budget = min(cash, max(0.0, budget))
    if budget < MIN_POSITION_GBP:
        add_event(state, f"{product_id}: buy skipped; available cash below minimum.", "SKIP")
        return False
    fee = budget * FEE_RATE
    spend = budget
    quantity = (spend - fee) / price
    if quantity <= 0:
        return False
    state["cash_gbp"] = cash - spend
    state["total_fees_gbp"] = float(state.get("total_fees_gbp", 0.0)) + fee
    state["positions"][product_id] = {
        "product_id": product_id,
        "strategy": strategy,
        "entry_price": price,
        "last_price": price,
        "quantity": quantity,
        "cost_gbp": spend,
        "entry_time": utc_now(),
        "entry_fee_gbp": fee,
    }
    state["trades"].insert(0, {
        "time": utc_now(), "side": "PAPER BUY", "product": product_id,
        "price_gbp": price, "quantity": quantity, "value_gbp": spend,
        "fee_gbp": fee, "strategy": strategy, "pnl_gbp": None,
    })
    add_event(state, f"PAPER BUY {product_id} at {money(price)} using {strategy}.", "BUY")
    return True


def paper_sell(state: dict[str, Any], product_id: str, price: float, reason: str) -> bool:
    pos = state["positions"].get(product_id)
    if not pos:
        return False
    qty = float(pos["quantity"])
    gross = qty * price
    fee = gross * FEE_RATE
    proceeds = gross - fee
    pnl = proceeds - float(pos["cost_gbp"])
    state["cash_gbp"] = float(state["cash_gbp"]) + proceeds
    state["total_fees_gbp"] = float(state.get("total_fees_gbp", 0.0)) + fee
    state["trades"].insert(0, {
        "time": utc_now(), "side": "PAPER SELL", "product": product_id,
        "price_gbp": price, "quantity": qty, "value_gbp": proceeds,
        "fee_gbp": fee, "strategy": pos.get("strategy", ""), "pnl_gbp": pnl,
        "reason": reason,
    })
    add_event(state, f"PAPER SELL {product_id} at {money(price)}; realised P/L {money(pnl)}. {reason}", "SELL")
    del state["positions"][product_id]
    return True


def auto_paper_cycle(market_limit: int) -> dict[str, Any]:
    state = load_state()
    stats = {"markets": 0, "buys": 0, "sells": 0, "errors": 0}
    try:
        products = get_gbp_products()[:market_limit]
    except Exception as exc:
        state["last_cycle"] = utc_now()
        state["last_cycle_note"] = f"Could not load Coinbase markets: {exc}"
        add_event(state, state["last_cycle_note"], "ERROR")
        save_json(STATE_FILE, state)
        return stats

    price_map: dict[str, float] = {}
    market_data: dict[str, pd.DataFrame] = {}
    signal_map: dict[str, dict[str, tuple[str, str]]] = {}

    for product_id in products:
        try:
            price_map[product_id] = get_price_gbp(product_id)
            candles = get_candles(product_id)
            market_data[product_id] = candles
            signal_map[product_id] = {
                strategy: strategy_signal(candles, strategy) for strategy in STRATEGIES
            }
            stats["markets"] +=
