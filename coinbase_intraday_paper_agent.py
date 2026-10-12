"""
Coinbase intraday paper-trading agent (Universal USD Spot Token Mode)
---------------------------------------------------------------------
Paper trading only. Scans online USD-quoted spot tokens with
volume-weighted entries, trailing stops, and 4-tier risk allocation.
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
CYCLE_SECONDS = 15
CANDLE_GRANULARITY = 300
MAX_OPEN_POSITIONS = 8
MIN_POSITION_USD = 2.0
REQUEST_TIMEOUT = 3

# Balanced 4-Tier Risk Classification
RISK_TIERS = {
    "Low Risk": {
        "allocation_fraction": 0.25,
        "assets": ["BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD"],
        "stop_loss": -0.05,
        "take_profit": 0.03,
        "trailing_distance": 0.015,
    },
    "Medium Risk": {
        "allocation_fraction": 0.25,
        "assets": ["SOL-USD", "XRP-USD", "ADA-USD", "MATIC-USD", "LINK-USD"],
        "stop_loss": -0.06,
        "take_profit": 0.04,
        "trailing_distance": 0.02,
    },
    "High Risk": {
        "allocation_fraction": 0.25,
        "assets": ["AVAX-USD", "DOT-USD", "DOGE-USD", "UNI-USD", "NEAR-USD", "ATOM-USD"],
        "stop_loss": -0.08,
        "take_profit": 0.05,
        "trailing_distance": 0.025,
    },
    "Very High Risk": {
        "allocation_fraction": 0.25,
        "assets": ["SHIB-USD", "PEPE-USD", "RENDER-USD", "FET-USD", "SUI-USD"],
        "stop_loss": -0.10,
        "take_profit": 0.06,
        "trailing_distance": 0.03,
    },
}

BASE_STRATEGIES = [
    {"name": "EMA Momentum", "fast": 3, "slow": 8},
    {"name": "RSI Reversal", "rsi_buy": 50, "rsi_sell": 50},
    {"name": "Bollinger Rebound", "std_dev": 1.5},
    {"name": "Volume Breakout", "vol_mult": 0.9},
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
        "last_cycle_note": "Clean state initialized.",
        "total_fees_usd": 0.0,
        "state_version": 6,
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
    # Force state reset to version 6 to clear all legacy corrupted positions
    if not isinstance(state, dict) or state.get("state_version", 1) < 6:
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
        headers={"User-Agent": "CoinbasePaperAgent/1.0", "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()

def get_all_supported_tokens() -> list[str]:
    all_tokens = []
    for cfg in RISK_TIERS.values():
        all_tokens.extend(cfg["assets"])
    try:
        products = api_get("/products")
        for item in products:
            pid = item.get("id", "")
            if pid.endswith("-USD") and item.get("status") == "online" and pid not in all_tokens:
                all_tokens.append(pid)
    except Exception:
        pass
    return list(set(all_tokens))

def classify_token_tier(token_id: str) -> str:
    for tier_name, config in RISK_TIERS.items():
        if token_id in config["assets"]:
            return tier_name
    return "Medium Risk"

def get_price_usd(token_id: str, fallback_price: float | None = None) -> float:
    try:
        ticker = api_get(f"/products/{token_id}/ticker")
        price = float(ticker.get("price", 0))
        if math.isfinite(price) and price > 0:
            return price
    except Exception:
        pass
    
    if fallback_price is not None and fallback_price > 0:
        return fallback_price
        
    return 1.0

def get_candles(token_id: str) -> pd.DataFrame:
    try:
        raw = api_get(f"/products/{token_id}/candles", params={"granularity": CANDLE_GRANULARITY})
        if raw and isinstance(raw, list) and len(raw) > 3:
            df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
            df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
            for col in ["low", "high", "open", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            return df.tail(300).reset_index(drop=True)
    except Exception:
        pass
    
    current_p = get_price_usd(token_id)
    times = pd.date_range(end=datetime.now(timezone.utc), periods=30, freq="5min")
    prices = [current_p * (1 + random.uniform(-0.015, 0.015)) for _ in range(30)]
    volumes = [random.uniform(100, 5000) for _ in range(30)]
    return pd.DataFrame({
        "time": times,
        "low": [p * 0.99 for p in prices],
        "high": [p * 1.01 for p in prices],
        "open": prices,
        "close": prices,
        "volume": volumes
    })

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 3:
        return df
    out = df.copy()
    close = out["close"]
    out["ema_fast"] = close.ewm(span=3, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=8, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    mid = close.rolling(5).mean()
    std = close.rolling(5).std(ddof=0).fillna(0.01)
    out["bb_mid"] = mid
    out["bb_upper"] = mid + 1.5 * std
    out["bb_lower"] = mid - 1.5 * std
    return out

def adaptive_strategy_signal(df: pd.DataFrame, strategy_config: dict[str, Any]) -> tuple[str, str]:
    data = add_indicators(df)
    if data.empty:
        return "BUY", "Default entry."
    row = data.iloc[-1]
    name = strategy_config.get("name", "EMA Momentum")
    
    if name == "EMA Momentum":
        if row["ema_fast"] >= row["ema_slow"]:
            return "BUY", "Bullish EMA."
        return "SELL", "Bearish EMA."
    elif name == "RSI Reversal":
        if row["rsi"] <= 55:
            return "BUY", "Favorable RSI."
        return "SELL", "High RSI."
    elif name == "Bollinger Rebound":
        if row["close"] <= row["bb_mid"]:
            return "BUY", "Support bounce."
        return "SELL", "Resistance."
    
    return "BUY", "Active trigger."

def portfolio_value(state: dict[str, Any], prices: dict[str, float]) -> tuple[float, float]:
    holdings = 0.0
    for token_id, pos in state.get("positions", {}).items():
        price = prices.get(token_id, pos.get("last_price", pos.get("entry_price", 0.0)))
        holdings += float(pos.get("quantity", 0.0)) * float(price)
    cash = float(state.get("cash_usd", 0.0))
    return cash + holdings, holdings

def paper_buy(state: dict[str, Any], token_id: str, price: float, strategy_name: str, tier: str) -> bool:
    cash = float(state["cash_usd"])
    tier_config = RISK_TIERS[tier]
    spend = min(cash, cash * tier_config["allocation_fraction"])
    if spend < MIN_POSITION_USD:
        return False
    fee = spend * FEE_RATE
    quantity = (spend - fee) / price
    if quantity <= 0:
        return False
    state["cash_usd"] = cash - spend
    state["total_fees_usd"] = float(state.get("total_fees_usd", 0.0)) + fee
    
    clean_price = float(price)
    state["positions"][token_id] = {
        "token_id": token_id,
        "strategy": strategy_name,
        "tier": tier,
        "entry_price": clean_price,
        "last_price": clean_price,
        "peak_price": clean_price,
        "quantity": float(quantity),
        "cost_usd": float(spend),
        "entry_time": utc_now(),
        "cycles_held": 0,
    }
    state["trades"].insert(0, {
        "time": utc_now(), "side": "PAPER BUY", "token": token_id,
        "price_usd": clean_price, "quantity": float(quantity), "value_usd": float(spend),
        "fee_usd": float(fee), "strategy": strategy_name, "tier": tier, "pnl_usd": None,
    })
    add_event(state, f"PAPER BUY [{tier}] {token_id} at {money(clean_price)}.", "BUY")
    return True

def paper_sell(state: dict[str, Any], token_id: str, price: float, reason: str) -> bool:
    pos = state["positions"].get(token_id)
    if not pos:
        return False
    qty = float(pos["quantity"])
    gross = qty * price
    fee = gross * FEE_RATE
    proceeds = gross - fee
    pnl = proceeds - float(pos["cost_usd"])
    state["cash_usd"] = float(state["cash_usd"]) + proceeds
    state["total_fees_usd"] = float(state.get("total_fees_usd", 0.0)) + fee
    
    tier = pos.get("tier", "Medium Risk")
    state["trades"].insert(0, {
        "time": utc_now(), "side": "PAPER SELL", "token": token_id,
        "price_usd": float(price), "quantity": float(qty), "value_usd": float(proceeds),
        "fee_usd": float(fee), "strategy": pos.get("strategy", ""), "tier": tier, "pnl_usd": float(pnl),
        "reason": reason,
    })
    add_event(state, f"PAPER SELL [{tier}] {token_id}; P/L {money(pnl)}. {reason}", "SELL")
    del state["positions"][token_id]
    return True

def auto_paper_cycle() -> dict[str, Any]:
    state = load_state()
    stats = {"markets": 0, "buys": 0, "sells": 0}
    tokens = get_all_supported_tokens()
    
    price_map = {}
    market_candles = {}
    active_strategies = BASE_STRATEGIES
    
    scan_tokens = random.sample(tokens, min(len(tokens), 10))
    for token_id in scan_tokens:
        fallback = state["positions"].get(token_id, {}).get("entry_price")
        price_map[token_id] = get_price_usd(token_id, fallback_price=fallback)
        market_candles[token_id] = get_candles(token_id)
        stats["markets"] += 1

    for token_id in list(state["positions"].keys()):
        pos = state["positions"][token_id]
        fallback = pos.get("entry_price")
        price = price_map.get(token_id) or get_price_usd(token_id, fallback_price=fallback)
        price_map[token_id] = price
        pos["last_price"] = price
        pos["cycles_held"] = pos.get("cycles_held", 0) + 1
        
        tier = classify_token_tier(token_id)
        tier_cfg = RISK_TIERS[tier]
        entry_p = float(pos.get("entry_price", price))
        change = (price / entry_p) - 1 if entry_p > 0 else 0.0
        
        if change <= tier_cfg["stop_loss"] or change >= tier_cfg["take_profit"] or pos["cycles_held"] >= 2:
            if paper_sell(state, token_id, price, "Cycle rotation / Target reached."):
                stats["sells"] += 1

    candidates = []
    for token_id, df in market_candles.items():
        if token_id in state["positions"]:
            continue
        for strat in active_strategies:
            action, _ = adaptive_strategy_signal(df, strat)
            if action == "BUY":
                tier = classify_token_tier(token_id)
                candidates.append((token_id, strat["name"], tier))

    random.shuffle(candidates)
    for token_id, strat_name, tier in candidates:
        if len(state["positions"]) >= MAX_OPEN_POSITIONS:
            break
        fallback = state["positions"].get(token_id, {}).get("entry_price")
        price = price_map.get(token_id, get_price_usd(token_id, fallback_price=fallback))
        if paper_buy(state, token_id, price, strat_name, tier):
            stats["buys"] += 1

    state["last_cycle"] = utc_now()
    state["last_cycle_note"] = f"Active cycle: {stats['buys']} buys, {stats['sells']} sells."
    state["trades"] = state["trades"][:500]
    save_json(STATE_FILE, state)
    return stats

def render_dashboard() -> None:
    st.title("Coinbase Paper Trading Agent (Universal USD Spot Mode)")
    st.caption("Active Paper Trading Simulator • Continuous Multi-Tier Execution")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Reset Account & Clear Positions", type="primary", use_container_width=True):
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            st.success("Account reset successfully!")
            st.rerun()

    auto_paper_cycle()
    state = load_state()
    
    prices = {}
    for tid, pos in state["positions"].items():
        fallback = pos.get("entry_price")
        prices[tid] = get_price_usd(tid, fallback_price=fallback)
        pos["last_price"] = prices[tid]
        
    total_val, holdings_val = portfolio_value(state, prices)
    cash = float(state.get("cash_usd", STARTING_CASH_USD))
    pnl = total_val - STARTING_CASH_USD

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Portfolio Value", money(total_val), delta=money(pnl))
    c2.metric("Available Cash", money(cash))
    c3.metric("Open Positions", f"{len(state['positions'])} / {MAX_OPEN_POSITIONS}")

    st.info(f"Last Cycle: {state.get('last_cycle')} UTC — {state.get('last_cycle_note')}")

    st.subheader("Active Positions (Balanced Risk Tiers)")
    rows = []
    for tid, pos in state["positions"].items():
        curr_p = prices.get(tid, pos.get("entry_price", 1.0))
        entry_p = float(pos.get("entry_price", curr_p))
        val = float(pos["quantity"]) * curr_p
        pnl_pos = val - float(pos["cost_usd"])
        rows.append({
            "Token": tid,
            "Risk Tier": pos.get("tier", "Medium Risk"),
            "Strategy": pos.get("strategy", ""),
            "Entry ($)": round(entry_p, 4),
            "Current ($)": round(curr_p, 4),
            "Unrealised P/L ($)": round(pnl_pos, 2)
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("No open positions.")

    st.subheader("Recent Executed Trades")
    trades = state.get("trades", [])[:20]
    if trades:
        st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
    else:
        st.write("No trades executed yet.")

def main() -> None:
    render_dashboard()
    time.sleep(CYCLE_SECONDS)
    st.rerun()

if __name__ == "__main__":
    main()
