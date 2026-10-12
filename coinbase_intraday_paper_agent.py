"""
Coinbase intraday paper-trading agent (Universal USD Spot Token Mode)
---------------------------------------------------------------------
Paper trading only. Scans online USD-quoted spot tokens on Coinbase
Exchange with volume-weighted entries, trailing stops, and 4-tier risk
allocation (25% each).
"""
from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "paper_state_usd.json"
MEMORY_FILE = APP_DIR / "strategy_memory_usd.json"
BASE_URL = "https://api.exchange.coinbase.com"
STARTING_CASH_USD = 100.0
FEE_RATE = 0.001
CYCLE_SECONDS = 30
CANDLE_GRANULARITY = 300
MAX_OPEN_POSITIONS = 20
MIN_POSITION_USD = 2.0
REQUEST_TIMEOUT = 5

# 4-Tier Risk Classification (25% capital allocation per tier)
RISK_TIERS = {
    "Low Risk": {
        "allocation_fraction": 0.25,
        "assets": ["BTC-USD", "ETH-USD"],
        "stop_loss": -0.05,
        "take_profit": 0.12,
        "trailing_distance": 0.03,
    },
    "Medium Risk": {
        "allocation_fraction": 0.25,
        "assets": ["SOL-USD", "XRP-USD", "ADA-USD", "LTC-USD"],
        "stop_loss": -0.08,
        "take_profit": 0.20,
        "trailing_distance": 0.04,
    },
    "High Risk": {
        "allocation_fraction": 0.25,
        "assets": ["LINK-USD", "AVAX-USD", "DOT-USD", "DOGE-USD"],
        "stop_loss": -0.12,
        "take_profit": 0.30,
        "trailing_distance": 0.05,
    },
    "Very High Risk": {
        "allocation_fraction": 0.25,
        "assets": [],  # Populated dynamically
        "stop_loss": -0.18,
        "take_profit": 0.45,
        "trailing_distance": 0.07,
    },
}

BASE_STRATEGIES = [
    {"name": "EMA Momentum", "fast": 9, "slow": 21},
    {"name": "RSI Reversal", "rsi_buy": 35, "rsi_sell": 75},
    {"name": "Bollinger Rebound", "std_dev": 2.0},
    {"name": "Volume Breakout", "vol_mult": 1.5},
    {"name": "Trend Pullback", "ema_filter": 50},
]

st.set_page_config(
    page_title="Coinbase Paper Trading Agent (Universal USD Spot Mode)",
    layout="wide",
)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def money(value: float) -> str:
    return f"${float(value):,.2f}"

def default_state() -> dict[str, Any]:
    return {
        "cash_usd": STARTING_CASH_USD,
        "starting_cash_usd": STARTING_CASH_USD,
        "positions": {},
        "trades": [],
        "events": [],
        "last_cycle": None,
        "last_cycle_note": "Waiting for first scan.",
        "total_fees_usd": 0.0,
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
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
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

def get_usd_spot_tokens() -> list[str]:
    """Fetches ALL online USD-quoted spot tokens available on Coinbase Exchange."""
    try:
        products = api_get("/products")
        result = []
        for item in products:
            pid = item.get("id", "")
            if pid.endswith("-USD") and item.get("status") == "online":
                result.append(pid)
        return result if result else ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
    except Exception:
        return ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]

def classify_token_tier(token_id: str) -> str:
    for tier_name, config in RISK_TIERS.items():
        if token_id in config["assets"]:
            return tier_name
    return "Very High Risk"

def get_ticker(token_id: str) -> dict[str, Any]:
    return api_get(f"/products/{token_id}/ticker")

def get_candles(token_id: str, granularity: int = CANDLE_GRANULARITY) -> pd.DataFrame:
    try:
        raw = api_get(
            f"/products/{token_id}/candles",
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

def get_price_usd(token_id: str) -> float:
    ticker = get_ticker(token_id)
    price = float(ticker.get("price", 0))
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"Invalid ticker price for {token_id}")
    return price

def add_indicators(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    if df.empty or len(df) < 5:
        return df
    out = df.copy()
    close = out["close"]
    out["ema_fast"] = close.ewm(span=fast, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=slow, adjust=False).mean()
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
    return out

def adaptive_strategy_signal(df: pd.DataFrame, strategy_config: dict[str, Any]) -> tuple[str, str]:
    if len(df) < 30:
        return "WAIT", f"Need at least 30 candles; received {len(df)}."
    
    name = strategy_config.get("name", "EMA Momentum")
    fast = strategy_config.get("fast", 9)
    slow = strategy_config.get("slow", 21)
    rsi_buy = strategy_config.get("rsi_buy", 35)
    rsi_sell = strategy_config.get("rsi_sell", 75)
    vol_mult = strategy_config.get("vol_mult", 1.5)
    data = add_indicators(df, fast=fast, slow=slow)
    row = data.iloc[-2]
    prev = data.iloc[-3]
    if not np.isfinite(row["close"]) or row["close"] <= 0:
        return "WAIT", "Candle data incomplete."
    vol_confirmed = row["volume"] > (row["vol_avg20"] * vol_mult) if row["vol_avg20"] > 0 else True
    if not vol_confirmed:
        return "WAIT", "Volume below institutional confirmation threshold."
    if name == "EMA Momentum":
        if row["ema_fast"] > row["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]:
            return "BUY", f"Volume-confirmed EMA ({fast}/{slow}) bullish cross."
        if row["ema_fast"] < row["ema_slow"]:
            return "SELL", f"EMA ({fast}/{slow}) bearish cross."
    elif name == "RSI Reversal":
        if prev["rsi"] < rsi_buy and row["rsi"] >= rsi_buy:
            return "BUY", f"Volume-backed RSI recovery above {rsi_buy}."
        if row["rsi"] > rsi_sell:
            return "SELL", f"RSI exceeded {rsi_sell}."
    elif name == "Bollinger Rebound":
        if prev["close"] < prev["bb_lower"] and row["close"] >= prev["bb_lower"]:
            return "BUY", "Volume-backed rebound inside lower Bollinger Band."
        if row["close"] >= row["bb_upper"]:
            return "SELL", "Reached upper Bollinger Band."
    elif name == "Volume Breakout":
        if row["close"] > data.iloc[-22:-2]["high"].max():
            return "BUY", "High-volume breakout confirmed."
        if row["close"] < row["ema_slow"]:
            return "SELL", "Price broke below slow EMA."
    elif name == "Trend Pullback":
        uptrend = row["ema_slow"] > row["ema50"]
        near_ema = row["close"] >= row["ema_slow"] * 0.995
        recovered = row["close"] > row["open"] and prev["close"] <= prev["ema_fast"]
        if uptrend and near_ema and recovered:
            return "BUY", "Volume-confirmed trend pullback recovery."
        if row["close"] < row["ema50"]:
            return "SELL", "Trend broken below EMA 50."
    return "WAIT", "No volume-confirmed trigger."

def get_active_strategies() -> list[dict[str, Any]]:
    memory = load_json(MEMORY_FILE, {})
    strategies = memory.get("active_strategies", [])
    if not strategies:
        strategies = BASE_STRATEGIES
        memory["active_strategies"] = strategies
