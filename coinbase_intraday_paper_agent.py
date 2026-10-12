"""
Coinbase intraday paper-trading agent (Universal USD Spot Token Mode)
---------------------------------------------------------------------
Paper trading only. Multi-tier risk-weighted capital allocation
with color-styled P/L columns and pristine bankroll accounting.
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
CYCLE_SECONDS = 15
CANDLE_GRANULARITY = 300
REQUEST_TIMEOUT = 3

RISK_TIERS = {
    "Low Risk": {
        "assets": ["BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD"],
        "capital_fraction": 0.20,
        "max_positions": 3,
        "stop_loss": -0.03,
        "take_profit": 0.05,
        "trailing_distance": 0.015,
    },
    "Medium Risk": {
        "assets": ["SOL-USD", "XRP-USD", "ADA-USD", "MATIC-USD", "LINK-USD"],
        "capital_fraction": 0.30,
        "max_positions": 4,
        "stop_loss": -0.05,
        "take_profit": 0.08,
        "trailing_distance": 0.02,
    },
    "High Risk": {
        "assets": ["AVAX-USD", "DOT-USD", "DOGE-USD", "UNI-USD", "NEAR-USD", "ATOM-USD"],
        "capital_fraction": 0.30,
        "max_positions": 4,
        "stop_loss": -0.08,
        "take_profit": 0.12,
        "trailing_distance": 0.03,
    },
    "Very High Risk": {
        "assets": ["SHIB-USD", "PEPE-USD", "RENDER-USD", "FET-USD", "SUI-USD"],
        "capital_fraction": 0.20,
        "max_positions": 3,
        "stop_loss": -0.12,
        "take_profit": 0.20,
        "trailing_distance": 0.04,
    },
}

BASE_STRATEGIES = [
    {"name": "Strong Momentum", "fast": 5, "slow": 15},
    {"name": "Trend Breakout", "vol_mult": 1.2},
]

st.set_page_config(
    page_title="Coinbase Paper Trading Agent (Multi-Tier Risk Allocation)",
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
        st.session_state.last_cycle_note = "Multi-tier risk engine initialized."
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
        headers={"User-Agent": "CoinbaseRiskAgent/1.0", "Accept": "application/json"},
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

def add_indicators(df: pd.DataFrame, fast: int = 5, slow: int = 15) -> pd.DataFrame:
    if df.empty or len(df) < slow:
        return df
    out = df.copy()
    close = out["close"]
    out["ema_fast"] = close.ewm(span=fast, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=slow, adjust=False).mean()
    out["vol_avg"] = out["volume"].rolling(10).mean().fillna(1.0)
    return out

def adaptive_strategy_signal(df: pd.DataFrame, strategy_config: dict[str, Any]) -> tuple[str, str]:
    data = add_indicators(df)
    if len(data) < 5:
        return "WAIT", "Insufficient data."
    row = data.iloc[-1]
    
    if row["ema_fast"] > row["ema_slow"] and row["volume"] >= row["vol_avg"] * 1.02:
        return "BUY", "Confirmed momentum breakout."
    
    return "WAIT", "Waiting for high-conviction setup."

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
    tier_cfg = RISK_TIERS[tier]
    if count_tier_positions(tier) >= tier_cfg["max_positions"]:
        return False
    
    tier_budget = get_tier_allocation(tier)
    tier_invested = get_tier_current_invested(tier)
    available_tier_cash = tier_budget - tier_invested
    
    target_pos_size = tier_budget / tier_cfg["max_positions"]
    spend = min(available_tier_cash, target_pos_size)
    
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
    add_event(f"PAPER BUY [{tier}] {token_id} at {money(clean_price)} (Allocation: {money(spend)}).", "BUY")
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
    stats = {"markets": 0, "buys": 0, "sells": 0}
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
            if paper_sell(token_id, price, "Stop-loss triggered."):
                stats["sells"] += 1
        elif change >= tier_cfg["take_profit"]:
            if paper_sell(token_id, price, "Take-profit target reached."):
                stats["sells"] += 1
        elif price <= trailing_stop and peak > entry_p:
            if paper_sell(token_id, price, "Trailing profit secured."):
                stats["sells"] += 1

    candidates = []
    for token_id, df in market_candles.items():
        if token_id in st.session_state.positions:
            continue
        for strat in active_strategies:
            action, _ = adaptive_strategy_signal(df, strat)
            if action == "BUY":
                tier = classify_token_tier(token_id)
                candidates.append((token_id, strat["name"], tier))

    random.shuffle(candidates)
    for token_id, strat_name, tier in candidates:
        if paper_buy(token_id, price_map.get(token_id, get_price_usd(token_id)), strat_name, tier):
            stats["buys"] += 1

    st.session_state.last_cycle = utc_now()
    st.session_state.last_cycle_note = f"Active cycle: {stats['buys']} buys, {stats['sells']} sells."
    st.session_state.trades = st.session_state.trades[:500]
    return stats

@st.fragment(run_every=CYCLE_SECONDS)
def render_live_dashboard() -> None:
    init_session_state()
    
    st.title("Coinbase Paper Trading Agent (Multi-Tier Risk Allocation)")
    st.caption("Active Paper Trading Simulator • Color-Coded P/L Columns")
    
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
    cash = STARTING_CASH_USD + st.session_state.realized_pnl - total_cost_basis
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
            "Cost Basis ($)": round(float(pos["cost_usd"]), 2),
            "Market Val ($)": round(val, 2),
            "Unrealised P/L ($)": round(pnl_pos, 2)
        })
    
    if rows:
        df_pos = pd.DataFrame(rows)
        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #2e7d32; font-weight: bold;"
                elif val < 0:
                    return "color: #c62828; font-weight: bold;"
            return ""

        styled_df = df_pos.style.map(color_pnl, subset=["Unrealised P/L ($)"]) if hasattr(df_pos.style, "map") else df_pos.style.applymap(color_pnl, subset=["Unrealised P/L ($)"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.write("Scanning multi-tier markets for high-conviction breakout opportunities...")

    st.subheader("Recent Executed Trades")
    trades = st.session_state.trades[:20]
    if trades:
        st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
    else:
        st.write("No trades executed yet.")

def main() -> None:
    init_session_state()
    render_live_dashboard()

if __name__ == "__main__":
    main()
