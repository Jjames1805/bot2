"""
Coinbase intraday paper-trading agent (Universal USD Spot Token Mode)
---------------------------------------------------------------------
Paper trading only. Multi-tier risk allocation with 4-hour timeframes,
1.75x volume filters, strict candidate deduplication, and clean accounting.
"""
from __future__ import annotations

import math
import random
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://api.exchange.coinbase.com"
STARTING_CASH_USD = 100.0
FEE_RATE = 0.001
CYCLE_SECONDS = 30
CANDLE_GRANULARITY = 14400
REQUEST_TIMEOUT = 3

RISK_TIERS = {
    "Low Risk": {
        "assets": ["BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD"],
        "capital_fraction": 0.20,
        "max_positions": 3,
        "stop_loss": -0.04,
        "take_profit": 0.08,
        "trailing_distance": 0.02,
    },
    "Medium Risk": {
        "assets": ["SOL-USD", "XRP-USD", "ADA-USD", "MATIC-USD", "LINK-USD"],
        "capital_fraction": 0.30,
        "max_positions": 4,
        "stop_loss": -0.06,
        "take_profit": 0.12,
        "trailing_distance": 0.03,
    },
    "High Risk": {
        "assets": ["AVAX-USD", "DOT-USD", "DOGE-USD", "UNI-USD", "NEAR-USD", "ATOM-USD"],
        "capital_fraction": 0.30,
        "max_positions": 4,
        "stop_loss": -0.09,
        "take_profit": 0.18,
        "trailing_distance": 0.04,
    },
    "Very High Risk": {
        "assets": ["SHIB-USD", "PEPE-USD", "RENDER-USD", "FET-USD", "SUI-USD"],
        "capital_fraction": 0.20,
        "max_positions": 3,
        "stop_loss": -0.15,
        "take_profit": 0.30,
        "trailing_distance": 0.06,
    },
}

BASE_STRATEGIES = [
    {"name": "4H Macro Momentum", "fast": 8, "slow": 21, "vol_mult": 1.5},
    {"name": "4H Volume Expansion", "fast": 5, "slow": 15, "vol_mult": 1.75},
]

st.set_page_config(
    page_title="Coinbase Paper Trading Agent (4H / 1.75x Volume Mode)",
    layout="wide",
)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def money(value: float) -> str:
    val = float(value)
    if val < 0:
        return f"-${abs(val):,.2f}"
    return f"${val:,.2f}"

def init_session_state() -> None:
    if "realized_pnl" not in st.session_state:
        st.session_state.realized_pnl = 0.0
    if "positions" not in st.session_state:
        st.session_state.positions = {}
    if "trades" not in st.session_state:
        st.session_state.trades = []
    if "events" not in st.session_state:
        st.session_state.events = []
    if "last_cycle" not in st.session_state:
        st.session_state.last_cycle = None
    if "last_cycle_note" not in st.session_state:
        st.session_state.last_cycle_note = "4H High-Conviction engine initialized."
    if "total_fees_usd" not in st.session_state:
        st.session_state.total_fees_usd = 0.0

def add_event(message: str, kind: str = "INFO") -> None:
    st.session_state.events.insert(
        0, {"time": utc_now(), "type": kind, "message": str(message)[:500]}
    )
    st.session_state.events = st.session_state.events[:250]

def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        BASE_URL + path,
        params=params,
        headers={"User-Agent": "CoinbaseRiskAgent/4H", "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()

def get_all_supported_tokens() -> list[str]:
    discovered = []
    for cfg in RISK_TIERS.values():
        discovered.extend(cfg["assets"])
    try:
        products = api_get("/products")
        if isinstance(products, list):
            for item in products:
                pid = item.get("id", "")
                status = item.get("status", "")
                if pid.endswith("-USD") and status == "online" and pid not in discovered:
                    discovered.append(pid)
    except Exception:
        pass
    return list(set(discovered))

def classify_token_tier(token_id: str) -> str:
    for tier_name, config in RISK_TIERS.items():
        if token_id in config["assets"]:
            return tier_name
    tiers = list(RISK_TIERS.keys())
    idx = abs(hash(token_id)) % len(tiers)
    return tiers[idx]

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
        if raw and isinstance(raw, list) and len(raw) > 5:
            df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
            df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
            for col in ["low", "high", "open", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            return df.tail(100).reset_index(drop=True)
    except Exception:
        pass
    
    current_p = get_price_usd(token_id)
    times = pd.date_range(end=datetime.now(timezone.utc), periods=30, freq="4h")
    prices = [current_p * (1 + random.uniform(-0.02, 0.02)) for _ in range(30)]
    volumes = [random.uniform(500, 10000) for _ in range(30)]
    return pd.DataFrame({
        "time": times,
        "low": [p * 0.98 for p in prices],
        "high": [p * 1.02 for p in prices],
        "open": prices,
        "close": prices,
        "volume": volumes
    })

def add_indicators(df: pd.DataFrame, fast: int = 8, slow: int = 21) -> pd.DataFrame:
    if df.empty or len(df) < slow:
        return df
    out = df.copy()
    close = out["close"]
    out["ema_fast"] = close.ewm(span=fast, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=slow, adjust=False).mean()
    out["vol_avg"] = out["volume"].rolling(14).mean().fillna(1.0)
    return out

def adaptive_strategy_signal(df: pd.DataFrame, strategy_config: dict[str, Any]) -> tuple[str, str, float]:
    data = add_indicators(df, fast=strategy_config.get("fast", 8), slow=strategy_config.get("slow", 21))
    if len(data) < 10:
        return "WAIT", "Insufficient 4H data.", 0.0
    row = data.iloc[-1]
    
    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    vol = float(row["volume"])
    vol_avg = float(row["vol_avg"])
    
    required_vol_mult = strategy_config.get("vol_mult", 1.75)
    if ema_fast > ema_slow and vol >= (vol_avg * required_vol_mult):
        strength = (ema_fast - ema_slow) / ema_slow + (vol / vol_avg - 1.0)
        return "BUY", "Confirmed 4H volume breakout.", strength
    
    return "WAIT", "Waiting for 4H volume expansion.", 0.0

def get_total_cost_basis() -> float:
    return sum(float(pos["cost_usd"]) for pos in st.session_state.positions.values())

def get_tier_allocation(tier_name: str) -> float:
    total_bankroll = STARTING_CASH_USD + st.session_state.realized_pnl
    tier_cfg = RISK_TIERS[tier_name]
    return max(0.0, total_bankroll * tier_cfg["capital_fraction"])

def get_tier_current_invested(tier_name: str) -> float:
    return sum(
        float(pos["cost_usd"])
        for pos in st.session_state.positions.values()
        if pos.get("tier") == tier_name
    )

def count_tier_positions(tier_name: str) -> int:
    return sum(1 for pos in st.session_state.positions.values() if pos.get("tier") == tier_name)

def paper_buy(token_id: str, price: float, strategy_name: str, tier: str) -> bool:
    if token_id in st.session_state.positions:
        return False
        
    tier_cfg = RISK_TIERS[tier]
    total_cost_basis = get_total_cost_basis()
    global_cash = STARTING_CASH_USD + st.session_state.realized_pnl - total_cost_basis
    if global_cash < 5.0:
        return False
    
    tier_budget = get_tier_allocation(tier)
    tier_invested = get_tier_current_invested(tier)
    available_tier_cash = tier_budget - tier_invested
    
    target_pos_size = tier_budget / tier_cfg["max_positions"]
    spend = min(available_tier_cash, target_pos_size, global_cash)
    
    if spend < 5.0:
        return False
    
    fee = spend * FEE_RATE
    quantity = (spend - fee) / price
    if quantity <= 0:
        return False
        
    st.session_state.total_fees_usd = float(st.session_state.total_fees_usd) + fee
    
    clean_price = float(price)
    st.session_state.positions[token_id] = {
        "token_id": token_id,
        "strategy": strategy_name,
        "tier": tier,
        "entry_price": clean_price,
        "last_price": clean_price,
        "peak_price": clean_price,
        "quantity": float(quantity),
        "cost_usd": float(spend),
        "entry_time": utc_now(),
    }
    st.session_state.trades.insert(0, {
        "time": utc_now(), "side": "PAPER BUY", "token": token_id,
        "price_usd": clean_price, "quantity": float(quantity), "value_usd": float(spend),
        "fee_usd": float(fee), "strategy": strategy_name, "tier": tier, "pnl_usd": None,
    })
    add_event(f"PAPER BUY [{tier}] {token_id} at {money(clean_price)} (4H Breakout).", "BUY")
    return True

def paper_sell(token_id: str, price: float, reason: str) -> bool:
    pos = st.session_state.positions.get(token_id)
    if not pos:
        return False
    qty = float(pos["quantity"])
    gross = qty * price
    fee = gross * FEE_RATE
    proceeds = gross - fee
    pnl = proceeds - float(pos["cost_usd"])
    
    st.session_state.realized_pnl += pnl
    st.session_state.total_fees_usd = float(st.session_state.total_fees_usd) + fee
    
    tier = pos.get("tier", "Medium Risk")
    st.session_state.trades.insert(0, {
        "time": utc_now(), "side": "PAPER SELL", "token": token_id,
        "price_usd": float(price), "quantity": float(qty), "value_usd": float(proceeds),
        "fee_usd": float(fee), "strategy": pos.get("strategy", ""), "tier": tier, "pnl_usd": float(pnl),
        "reason": reason,
    })
    add_event(f"PAPER SELL [{tier}] {token_id}; P/L {money(pnl)}. {reason}", "SELL")
    del st.session_state.positions[token_id]
    return True

def auto_paper_cycle() -> dict[str, Any]:
    stats = {"markets": 0, "buys": 0, "sells": 0, "rotations": 0}
    tokens = get_all_supported_tokens()
    
    price_map = {}
    market_candles = {}
    active_strategies = BASE_STRATEGIES
    
    scan_tokens = random.sample(tokens, min(len(tokens), 20))
    for token_id in scan_tokens:
        fallback = st.session_state.positions.get(token_id, {}).get("entry_price")
        price_map[token_id] = get_price_usd(token_id, fallback_price=fallback)
        market_candles[token_id] = get_candles(token_id)
        stats["markets"] += 1

    for token_id in list(st.session_state.positions.keys()):
        pos = st.session_state.positions[token_id]
        fallback = pos.get("entry_price")
        price = price_map.get(token_id) or get_price_usd(token_id, fallback_price=fallback)
        price_map[token_id] = price
        pos["last_price"] = price
        
        tier = classify_token_tier(token_id)
        tier_cfg = RISK_TIERS[tier]
        entry_p = float(pos.get("entry_price", price))
        change = (price / entry_p) - 1 if entry_p > 0 else 0.0
        
        peak = max(pos.get("peak_price", entry_p), price)
        pos["peak_price"] = peak
        trailing_stop = peak * (1.0 - tier_cfg["trailing_distance"])
        
        if change <= tier_cfg["stop_loss"]:
            if paper_sell(token_id, price, "4H Stop-loss triggered."):
                stats["sells"] += 1
        elif change >= tier_cfg["take_profit"]:
            if paper_sell(token_id, price, "4H Take-profit target reached."):
                stats["sells"] += 1
        elif price <= trailing_stop and peak > entry_p:
            if paper_sell(token_id, price, "4H Trailing profit secured."):
                stats["sells"] += 1

    # Strict candidate deduplication map per token
    candidates_map = {}
    held_tokens = set(st.session_state.positions.keys())
    
    for token_id, df in market_candles.items():
        if token_id in held_tokens:
            continue
        for strat in active_strategies:
            action, _, strength = adaptive_strategy_signal(df, strat)
            if action == "BUY":
                tier = classify_token_tier(token_id)
                if token_id not in candidates_map or strength > candidates_map[token_id][2]:
                    candidates_map[token_id] = (strat["name"], tier, strength)

    candidates = [(tid, data[0], data[1], data[2]) for tid, data in candidates_map.items()]
    candidates.sort(key=lambda x: x[3], reverse=True)

    for token_id, strat_name, tier, strength in candidates:
        if token_id in st.session_state.positions:
            continue
            
        tier_cfg = RISK_TIERS[tier]
        current_tier_count = count_tier_positions(tier)
        
        if current_tier_count >= tier_cfg["max_positions"]:
            tier_positions = [
                (tid, p) for tid, p in st.session_state.positions.items() if p.get("tier") == tier
            ]
            if tier_positions:
                worst_tid, worst_pos = min(
                    tier_positions,
                    key=lambda x: (price_map.get(x[0], x[1]["last_price"]) / x[1]["entry_price"]) - 1
                )
                worst_curr = price_map.get(worst_tid, worst_pos["last_price"])
                worst_change = (worst_curr / worst_pos["entry_price"]) - 1
                
                if worst_change < 0.02:
                    if paper_sell(worst_tid, worst_curr, f"Rotating out for stronger 4H setup ({token_id})."):
                        stats["sells"] += 1
                        stats["rotations"] += 1

        if paper_buy(token_id, price_map.get(token_id, get_price_usd(token_id)), strat_name, tier):
            stats["buys"] += 1

    st.session_state.last_cycle = utc_now()
    st.session_state.last_cycle_note = f"4H Scan: {stats['markets']} markets, {stats['buys']} buys, {stats['sells']} sells ({stats['rotations']} rotations)."
    st.session_state.trades = st.session_state.trades[:500]
    return stats

@st.fragment(run_every=CYCLE_SECONDS)
def render_live_dashboard() -> None:
    init_session_state()
    
    st.title("Coinbase Paper Trading Agent (4H / 1.75x Volume Mode)")
    st.caption("Active Paper Trading Simulator • High-Conviction 4-Hour Timeframe Engine")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Reset Account & Clear Positions", type="primary", use_container_width=True):
            st.session_state.clear()
            init_session_state()
            st.success("Account reset successfully!")
            st.rerun()

    auto_paper_cycle()
    
    prices = {}
    for tid, pos in st.session_state.positions.items():
        fallback = pos.get("entry_price")
        prices[tid] = get_price_usd(tid, fallback_price=fallback)
        pos["last_price"] = prices[tid]
        
    holdings_val = 0.0
    unrealized_pnl = 0.0
    total_cost_basis = 0.0
    for tid, pos in st.session_state.positions.items():
        curr_p = prices.get(tid, pos.get("entry_price", 1.0))
        val = float(pos.get("quantity", 0.0)) * curr_p
        holdings_val += val
        total_cost_basis += float(pos["cost_usd"])
        unrealized_pnl += (val - float(pos["cost_usd"]))

    total_val = STARTING_CASH_USD + st.session_state.realized_pnl + unrealized_pnl
    cash = max(0.0, STARTING_CASH_USD + st.session_state.realized_pnl - total_cost_basis)
    total_net_pnl = st.session_state.realized_pnl + unrealized_pnl

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Portfolio Value", money(total_val), delta=money(total_net_pnl))
    c2.metric("Available Cash", money(cash))
    c3.metric("Open Positions", f"{len(st.session_state.positions)} active")

    st.info(f"Last Cycle: {st.session_state.last_cycle} UTC — {st.session_state.last_cycle_note}")

    st.subheader("Portfolio Ledger Breakdown")
    ledger_df = pd.DataFrame([
        {"Metric": "Starting Capital", "Amount": money(STARTING_CASH_USD)},
        {"Metric": "Realized P/L", "Amount": money(st.session_state.realized_pnl)},
        {"Metric": "Available Cash", "Amount": money(cash)},
        {"Metric": "Holdings Market Value", "Amount": money(holdings_val)},
        {"Metric": "Total Portfolio Value", "Amount": money(total_val)},
        {"Metric": "Total Net P/L", "Amount": money(total_net_pnl)},
    ])
    st.dataframe(ledger_df, use_container_width=True, hide_index=True)

    st.subheader("Risk Tier Capital Allocation")
    tier_rows = []
    for tier_name, cfg in RISK_TIERS.items():
        budget = get_tier_allocation(tier_name)
        invested = get_tier_current_invested(tier_name)
        pos_count = count_tier_positions(tier_name)
        tier_rows.append({
            "Risk Tier": tier_name,
            "Allocation Target (%)": f"{int(cfg['capital_fraction'] * 100)}%",
            "Budget ($)": round(budget, 2),
            "Invested ($)": round(invested, 2),
            "Open Positions": f"{pos_count} / {cfg['max_positions']}",
            "Stop Loss / Take Profit": f"{cfg['stop_loss']*100:.1f}% / +{cfg['take_profit']*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

    st.subheader("Active Positions")
    rows = []
    for tid, pos in st.session_state.positions.items():
        curr_p =
