"""
Coinbase intraday paper-trading agent (Universal USD Spot Token Mode)
---------------------------------------------------------------------
Paper trading only. Scans ALL online USD-quoted spot tokens on Coinbase
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
REQUEST_TIMEOUT = 12

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
        "assets": [],  # Populated dynamically from all remaining exchange USD spot tokens
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
        return result
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
        save_json(MEMORY_FILE, memory)
    return strategies

def mutate_strategies() -> None:
    memory = load_json(MEMORY_FILE, {})
    strategies = memory.get("active_strategies", BASE_STRATEGIES)
    scores = memory.get("strategy_scores", {})
    mutated = []
    for strat in strategies:
        name = strat["name"]
        score = scores.get(name, 0.0)
        if score <= -0.1:
            if name == "EMA Momentum":
                strat["fast"] = random.randint(4, 12)
                strat["slow"] = random.randint(14, 35)
            elif name == "RSI Reversal":
                strat["rsi_buy"] = random.randint(25, 42)
            elif name == "Volume Breakout":
                strat["vol_mult"] = round(random.uniform(1.2, 1.8), 2)
        mutated.append(strat)
    memory["active_strategies"] = mutated
    save_json(MEMORY_FILE, memory)

def portfolio_value(state: dict[str, Any], prices: dict[str, float]) -> tuple[float, float]:
    holdings = 0.0
    for token_id, pos in state.get("positions", {}).items():
        price = prices.get(token_id, pos.get("last_price", pos.get("entry_price", 0.0)))
        holdings += float(pos.get("quantity", 0.0)) * float(price)
    cash = float(state.get("cash_usd", 0.0))
    return cash + holdings, holdings

def paper_buy(state: dict[str, Any], token_id: str, price: float, strategy_name: str, tier: str) -> bool:
    if token_id in state["positions"]:
        return False
    
    cash = float(state["cash_usd"])
    tier_config = RISK_TIERS[tier]
    tier_budget_cap = cash * tier_config["allocation_fraction"]
    
    spend = min(cash, tier_budget_cap)
    if spend < MIN_POSITION_USD:
        return False
    fee = spend * FEE_RATE
    quantity = (spend - fee) / price
    if quantity <= 0:
        return False
    state["cash_usd"] = cash - spend
    state["total_fees_usd"] = float(state.get("total_fees_usd", 0.0)) + fee
    state["positions"][token_id] = {
        "token_id": token_id,
        "strategy": strategy_name,
        "tier": tier,
        "entry_price": price,
        "last_price": price,
        "peak_price": price,
        "quantity": quantity,
        "cost_usd": spend,
        "entry_time": utc_now(),
        "entry_fee_usd": fee,
    }
    state["trades"].insert(0, {
        "time": utc_now(), "side": "PAPER BUY", "token": token_id,
        "price_usd": price, "quantity": quantity, "value_usd": spend,
        "fee_usd": fee, "strategy": strategy_name, "tier": tier, "pnl_usd": None,
    })
    add_event(state, f"PAPER BUY [{tier}] {token_id} via {strategy_name} at {money(price)}.", "BUY")
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
    
    strategy_name = pos.get("strategy", "")
    tier = pos.get("tier", "Medium Risk")
    
    memory = load_json(MEMORY_FILE, {})
    scores = memory.setdefault("strategy_scores", {})
    scores[strategy_name] = scores.get(strategy_name, 0.0) + pnl
    save_json(MEMORY_FILE, memory)
    state["trades"].insert(0, {
        "time": utc_now(), "side": "PAPER SELL", "token": token_id,
        "price_usd": price, "quantity": qty, "value_usd": proceeds,
        "fee_usd": fee, "strategy": strategy_name, "tier": tier, "pnl_usd": pnl,
        "reason": reason,
    })
    add_event(state, f"PAPER SELL [{tier}] {token_id}; P/L {money(pnl)}. {reason}", "SELL")
    del state["positions"][token_id]
    return True

def auto_paper_cycle() -> dict[str, Any]:
    state = load_state()
    stats = {"markets": 0, "buys": 0, "sells": 0, "errors": 0}
    try:
        tokens = get_usd_spot_tokens()
    except Exception as exc:
        state["last_cycle"] = utc_now()
        state["last_cycle_note"] = f"Could not load Coinbase markets: {exc}"
        add_event(state, state["last_cycle_note"], "ERROR")
        save_json(STATE_FILE, state)
        return stats
    explicit_listed = set(RISK_TIERS["Low Risk"]["assets"] + RISK_TIERS["Medium Risk"]["assets"] + RISK_TIERS["High Risk"]["assets"])
    RISK_TIERS["Very High Risk"]["assets"] = [p for p in tokens if p not in explicit_listed]
    price_map: dict[str, float] = {}
    market_candles: dict[str, pd.DataFrame] = {}
    active_strategies = get_active_strategies()
    for token_id in tokens:
        try:
            price_map[token_id] = get_price_usd(token_id)
            market_candles[token_id] = get_candles(token_id)
            stats["markets"] += 1
        except Exception:
            pass
    for token_id in list(state["positions"].keys()):
        try:
            price = price_map.get(token_id) or get_price_usd(token_id)
            price_map[token_id] = price
            pos = state["positions"][token_id]
            pos["last_price"] = price
            tier = classify_token_tier(token_id)
            pos["tier"] = tier
            tier_cfg = RISK_TIERS[tier]
            
            peak = max(pos.get("peak_price", pos["entry_price"]), price)
            pos["peak_price"] = peak
            trailing_stop_threshold = peak * (1.0 - tier_cfg["trailing_distance"])
            change = price / float(pos["entry_price"]) - 1
            strat_name = pos.get("strategy", "")
            strat_cfg = next((s for s in active_strategies if s["name"] == strat_name), active_strategies[0])
            action, reason = adaptive_strategy_signal(market_candles.get(token_id, pd.DataFrame()), strat_cfg)
            if price <= trailing_stop_threshold and peak > pos["entry_price"]:
                if paper_sell(state, token_id, price, f"[{tier}] Trailing profit secured from peak {money(peak)}."):
                    stats["sells"] += 1
            elif change <= tier_cfg["stop_loss"]:
                if paper_sell(state, token_id, price, f"[{tier}] Hard stop-loss hit ({tier_cfg['stop_loss']*100}%)."):
                    stats["sells"] += 1
            elif change >= tier_cfg["take_profit"]:
                if paper_sell(state, token_id, price, f"[{tier}] Take-profit target reached (+{tier_cfg['take_profit']*100}%)."):
                    stats["sells"] += 1
            elif action == "SELL":
                if paper_sell(state, token_id, price, f"[{tier}] Evolving strategy flipped to SELL."):
                    stats["sells"] += 1
        except Exception:
            pass
    mutate_strategies()
    memory = load_json(MEMORY_FILE, {})
    strategy_scores = memory.get("strategy_scores", {})
    candidates = []
    for token_id, df in market_candles.items():
        if token_id in state["positions"]:
            continue
        for strat in active_strategies:
            action, reason = adaptive_strategy_signal(df, strat)
            if action == "BUY":
                try:
                    tier = classify_token_tier(token_id)
                    strat_name = strat["name"]
                    score = strategy_scores.get(strat_name, 0.0) + 15
                    candidates.append((score, token_id, strat_name, tier))
                except Exception:
                    pass
    candidates.sort(reverse=True)
    for score, token_id, strategy_name, tier in candidates:
        if len(state["positions"]) >= MAX_OPEN_POSITIONS:
            break
        try:
            price = price_map[token_id]
            if paper_buy(state, token_id, price, strategy_name, tier):
                stats["buys"] += 1
        except Exception:
            pass
    state["last_cycle"] = utc_now()
    state["last_cycle_note"] = (
        f"Scanned {stats['markets']} USD spot tokens; {stats['buys']} buys, "
        f"{stats['sells']} sells. Universal token scan active."
    )
    state["trades"] = state["trades"][:500]
    save_json(STATE_FILE, state)
    return stats

def render_dashboard() -> None:
    st.title("Coinbase Paper Trading Agent (Universal USD Spot Mode)")
    st.caption("Scanning All Coinbase USD Spot Tokens • Volume-Weighted • Trailing Stop • 4-Tier Risk Allocation")
    
    col_top1, col_top2 = st.columns([3, 1])
    with col_top2:
        if st.button("Reset Account & Clear Tiers", type="secondary", use_container_width=True):
            save_json(STATE_FILE, default_state())
            st.success("Account reset successfully!")
            st.rerun()
    auto_paper_cycle()
    state = load_state()
    prices: dict[str, float] = {}
    for token_id in list(state["positions"].keys()):
        try:
            prices[token_id] = get_price_usd(token_id)
        except Exception:
            pass
    total_value, holdings_value = portfolio_value(state, prices)
    cash = float(state.get("cash_usd", 0.0))
    start = float(state.get("starting_cash_usd", STARTING_CASH_USD))
    pnl = total_value - start
    c1, c2, c3 = st.columns(3)
    c1.metric("Total portfolio value", money(total_value), delta=money(pnl))
    c2.metric("Available cash", money(cash))
    c3.metric("Open risk-tiered holdings", money(holdings_value), f"{len(state['positions'])} positions")
    st.info(
        f"Last cycle: {state.get('last_cycle') or 'Not run yet'} UTC  •  "
        f"refresh every {CYCLE_SECONDS} seconds  •  "
        f"{state.get('last_cycle_note', 'Waiting for scan.')}"
    )
    st.subheader("Open Risk-Tiered Positions (Universal USD)")
    positions_rows = []
    for token_id, pos in state["positions"].items():
        price = prices.get(token_id, float(pos.get("last_price", pos["entry_price"])))
        market_value = float(pos["quantity"]) * price
        unrealised = market_value - float(pos["cost_usd"])
        tier_label = classify_token_tier(token_id)
        peak = float(pos.get("peak_price", pos["entry_price"]))
        positions_rows.append({
            "Token": token_id,
            "Risk Tier": tier_label,
            "Strategy": pos.get("strategy", ""),
            "Entry ($)": round(float(pos["entry_price"]), 6),
            "Peak ($)": round(peak, 6),
            "Current ($)": round(price, 6),
            "Unrealised P/L ($)": round(unrealised, 2),
        })
    if positions_rows:
        st.dataframe(pd.DataFrame(positions_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No open positions.")
    st.subheader("Recent paper trades")
    trades = state.get("trades", [])[:30]
    if trades:
        st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
    else:
        st.write("No paper trades recorded yet.")
    st.subheader("Active Strategy Evolution & Mutated Parameters")
    memory = load_json(MEMORY_FILE, {})
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Strategy Profit Scores:**")
        scores_dict = memory.get("strategy_scores", {})
        if scores_dict:
            scores_df = pd.DataFrame([{"Strategy": k, "Realized P/L ($)": round(v, 4)} for k, v in scores_dict.items()])
            st.dataframe(scores_df, use_container_width=True, hide_index=True)
        else:
            st.info("Accumulating P/L scores from closed trades...")
    with col_b:
        st.write("**Mutated Strategy Configurations:**")
        strats_list = memory.get("active_strategies", BASE_STRATEGIES)
        if strats_list:
            strats_df = pd.DataFrame(strats_list)
            st.dataframe(strats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active strategies loaded.")
    st.caption("Educational paper-trading simulator. No real orders are sent.")

def main() -> None:
    render_dashboard()
    time.sleep(CYCLE_SECONDS)
    st.rerun()

if __name__ == "__main__":
    main()
