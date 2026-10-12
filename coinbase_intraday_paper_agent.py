"""
Coinbase intraday paper-trading agent (Tiered Risk-Weighted Mode)
-----------------------------------------------------------------
Paper trading only. Automatically segments capital into Low, Medium,
High, and Very High risk buckets (25% each) with active rotation.
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
FEE_RATE = 0.006
CYCLE_SECONDS = 30
CANDLE_GRANULARITY = 300
MAX_OPEN_POSITIONS = 20
MIN_POSITION_GBP = 2.0
REQUEST_TIMEOUT = 12

# Tiered Risk Classification (25% capital allocation per tier)
RISK_TIERS = {
    "Low Risk": {
        "allocation_fraction": 0.25,
        "assets": ["BTC-GBP", "ETH-GBP"],
        "stop_loss": -0.05,
        "take_profit": 0.10,
    },
    "Medium Risk": {
        "allocation_fraction": 0.25,
        "assets": ["SOL-GBP", "XRP-GBP", "ADA-GBP", "LTC-GBP"],
        "stop_loss": -0.08,
        "take_profit": 0.16,
    },
    "High Risk": {
        "allocation_fraction": 0.25,
        "assets": ["LINK-GBP", "AVAX-GBP", "DOT-GBP", "DOGE-GBP"],
        "stop_loss": -0.12,
        "take_profit": 0.24,
    },
    "Very High Risk": {
        "allocation_fraction": 0.25,
        "assets": [],  # Filled dynamically from remaining exchange altcoins
        "stop_loss": -0.18,
        "take_profit": 0.35,
    },
}

STRATEGIES = [
    "EMA Momentum",
    "RSI Reversal",
    "Bollinger Rebound",
    "Volume Breakout",
    "Trend Pullback",
]

st.set_page_config(page_title="Coinbase Paper Trading Agent (Risk-Tiered)", layout="wide")


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


def get_gbp_products() -> list[str]:
    try:
        products = api_get("/products")
        result = []
        for item in products:
            pid = item.get("id", "")
            if pid.endswith("-GBP") and item.get("status") == "online":
                result.append(pid)
        return result
    except Exception:
        return ["BTC-GBP", "ETH-GBP", "SOL-GBP", "XRP-GBP"]


def classify_product_tier(product_id: str) -> str:
    for tier_name, config in RISK_TIERS.items():
        if product_id in config["assets"]:
            return tier_name
    return "Very High Risk"


def get_ticker(product_id: str) -> dict[str, Any]:
    return api_get(f"/products/{product_id}/ticker")


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
    if len(df) < 30:
        return "WAIT", f"Need at least 30 candles; received {len(df)}."
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
        if prev["rsi"] < 35 and row["rsi"] >= 35:
            return "BUY", "RSI recovered above 35."
        if row["rsi"] > 75:
            return "SELL", "RSI is above 75."
        return "WAIT", f"RSI is {row['rsi']:.1f}; no reversal trigger."
    if strategy == "Bollinger Rebound":
        if prev["close"] < prev["bb_lower"] and row["close"] >= row["bb_lower"]:
            return "BUY", "Price recovered inside the lower Bollinger Band."
        if row["close"] >= row["bb_upper"]:
            return "SELL", "Price reached the upper Bollinger Band."
        return "WAIT", "No Bollinger rebound trigger."
    if strategy == "Volume Breakout":
        volume_ok = row["volume"] > row["vol_avg20"] * 1.3 if row["vol_avg20"] > 0 else False
        if volume_ok and row["close"] > data.iloc[-22:-2]["high"].max():
            return "BUY", "High-volume close above recent range."
        if row["close"] < row["ema21"]:
            return "SELL", "Price fell below EMA 21."
        return "WAIT", "No high-volume upside breakout."
    if strategy == "Trend Pullback":
        uptrend = row["ema21"] > row["ema50"]
        near_ema = row["close"] >= row["ema21"] * 0.99
        recovered = row["close"] > row["open"] and prev["close"] <= prev["ema9"]
        if uptrend and near_ema and recovered and 25 <= row["rsi"] <= 75:
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


def paper_buy(state: dict[str, Any], product_id: str, price: float, strategy: str, tier: str) -> bool:
    if product_id in state["positions"]:
        return False
    
    cash = float(state["cash_gbp"])
    tier_config = RISK_TIERS[tier]
    tier_budget_cap = cash * tier_config["allocation_fraction"]
    
    spend = min(cash, tier_budget_cap)
    if spend < MIN_POSITION_GBP:
        return False

    fee = spend * FEE_RATE
    quantity = (spend - fee) / price
    if quantity <= 0:
        return False

    state["cash_gbp"] = cash - spend
    state["total_fees_gbp"] = float(state.get("total_fees_gbp", 0.0)) + fee
    state["positions"][product_id] = {
        "product_id": product_id,
        "strategy": strategy,
        "tier": tier,
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
        "fee_gbp": fee, "strategy": strategy, "tier": tier, "pnl_gbp": None,
    })
    add_event(state, f"PAPER BUY [{tier}] {product_id} at {money(price)} using {strategy}.", "BUY")
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
    
    strategy = pos.get("strategy", "")
    tier = pos.get("tier", "Medium Risk")
    memory = load_json(MEMORY_FILE, {"strategy_scores": {}})
    scores = memory.setdefault("strategy_scores", {})
    scores[strategy] = scores.get(strategy, 0.0) + pnl
    save_json(MEMORY_FILE, memory)

    state["trades"].insert(0, {
        "time": utc_now(), "side": "PAPER SELL", "product": product_id,
        "price_gbp": price, "quantity": qty, "value_gbp": proceeds,
        "fee_gbp": fee, "strategy": strategy, "tier": tier, "pnl_gbp": pnl,
        "reason": reason,
    })
    add_event(state, f"PAPER SELL [{tier}] {product_id} at {money(price)}; P/L {money(pnl)}. {reason}", "SELL")
    del state["positions"][product_id]
    return True


def auto_paper_cycle() -> dict[str, Any]:
    state = load_state()
    stats = {"markets": 0, "buys": 0, "sells": 0, "errors": 0}
    try:
        products = get_gbp_products()
    except Exception as exc:
        state["last_cycle"] = utc_now()
        state["last_cycle_note"] = f"Could not load Coinbase markets: {exc}"
        add_event(state, state["last_cycle_note"], "ERROR")
        save_json(STATE_FILE, state)
        return stats

    # Automatically populate 'Very High Risk' bucket with remaining products not explicitly listed
    explicit_listed = set(RISK_TIERS["Low Risk"]["assets"] + RISK_TIERS["Medium Risk"]["assets"] + RISK_TIERS["High Risk"]["assets"])
    RISK_TIERS["Very High Risk"]["assets"] = [p for p in products if p not in explicit_listed]

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
            stats["markets"] += 1
        except Exception as exc:
            stats["errors"] += 1

    # Active Risk-Tiered Position Monitoring & Rotation
    for product_id in list(state["positions"].keys()):
        try:
            price = price_map.get(product_id) or get_price_gbp(product_id)
            price_map[product_id] = price
            pos = state["positions"][product_id]
            pos["last_price"] = price
            tier = pos.get("tier", "Medium Risk")
            tier_cfg = RISK_TIERS[tier]
            
            action, reason = signal_map.get(product_id, {}).get(pos.get("strategy"), ("WAIT", "No signal"))
            change = price / float(pos["entry_price"]) - 1
            
            # Tier-specific stop-loss and take-profit thresholds
            if action == "SELL" or change <= tier_cfg["stop_loss"] or change >= tier_cfg["take_profit"]:
                if change <= tier_cfg["stop_loss"]:
                    reason = f"[{tier}] Stop-loss reached ({tier_cfg['stop_loss']*100}%)."
                elif change >= tier_cfg["take_profit"]:
                    reason = f"[{tier}] Take-profit reached (+{tier_cfg['take_profit']*100}%)."
                elif action == "SELL":
                    reason = f"[{tier}] Strategy momentum flipped to SELL."
                
                if paper_sell(state, product_id, price, reason):
                    stats["sells"] += 1
        except Exception as exc:
            stats["errors"] += 1

    memory = load_json(MEMORY_FILE, {"strategy_scores": {}})
    strategy_scores = memory.get("strategy_scores", {})

    candidates = []
    for product_id, signals in signal_map.items():
        if product_id in state["positions"]:
            continue
        buy_strategies = [s for s, (action, _) in signals.items() if action == "BUY"]
        if buy_strategies:
            try:
                df = add_indicators(market_data[product_id])
                if not df.empty and len(df) >= 2:
                    row = df.iloc[-2]
                    trend = float(row["ema21"] / row["ema50"] - 1) if ("ema50" in row and row["ema50"]) else 0.0
                    tier = classify_product_tier(product_id)
                    
                    best_strat = max(buy_strategies, key=lambda s: strategy_scores.get(s, 0.0))
                    strat_bonus = max(0.0, strategy_scores.get(best_strat, 0.0) * 0.2)

                    score = (len(buy_strategies) * 20) + strat_bonus + max(-15, min(15, trend * 100))
                    candidates.append((score, product_id, best_strat, tier))
            except Exception:
                pass

    candidates.sort(reverse=True)
    for score, product_id, strategy, tier in candidates:
        if len(state["positions"]) >= MAX_OPEN_POSITIONS:
            break
        try:
            price = price_map[product_id]
            if paper_buy(state, product_id, price, strategy, tier):
                stats["buys"] += 1
        except Exception as exc:
            stats["errors"] += 1

    state["last_cycle"] = utc_now()
    state["last_cycle_note"] = (
        f"Scanned {stats['markets']} markets; {stats['buys']} buys, "
        f"{stats['sells']} sells, {stats['errors']} errors."
    )
    state["trades"] = state["trades"][:500]
    save_json(STATE_FILE, state)
    memory["last_updated"] = utc_now()
    memory["last_cycle_note"] = state["last_cycle_note"]
    save_json(MEMORY_FILE, memory)
    return stats


def render_dashboard() -> None:
    st.title("Coinbase Paper Trading Agent (4-Tier Risk-Weighted)")
    st.caption("25% Low Risk | 25% Medium Risk | 25% High Risk | 25% Very High Risk")
    
    auto_paper_cycle()

    state = load_state()

    prices: dict[str, float] = {}
    for product_id in list(state["positions"].keys()):
        try:
            prices[product_id] = get_price_gbp(product_id)
        except Exception:
            pass

    total_value, holdings_value = portfolio_value(state, prices)
    cash = float(state.get("cash_gbp", 0.0))
    start = float(state.get("starting_cash_gbp", STARTING_CASH_GBP))
    pnl = total_value - start

    c1, c2, c3 = st.columns(3)
    c1.metric("Total paper portfolio", money(total_value), delta=money(pnl))
    c2.metric("Available cash", money(cash))
    c3.metric("Open holdings", money(holdings_value), f"{len(state['positions'])} positions")

    st.info(
        f"Last cycle: {state.get('last_cycle') or 'Not run yet'} UTC  •  "
        f"refresh every {CYCLE_SECONDS} seconds  •  "
        f"{state.get('last_cycle_note', 'Waiting for first scan.')}"
    )

    st.subheader("Open Risk-Tiered Positions")
    positions_rows = []
    for product_id, pos in state["positions"].items():
        price = prices.get(product_id, float(pos.get("last_price", pos["entry_price"])))
        market_value = float(pos["quantity"]) * price
        unrealised = market_value - float(pos["cost_gbp"])
        positions_rows.append({
            "Market": product_id,
            "Risk Tier": pos.get("tier", "Medium Risk"),
            "Strategy": pos.get("strategy", ""),
            "Entry (£)": round(float(pos["entry_price"]), 6),
            "Current (£)": round(price, 6),
            "Value (£)": round(market_value, 2),
            "Unrealised P/L (£)": round(unrealised, 2),
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

    st.subheader("Strategy Profit Scores (Self-Optimization Memory)")
    memory = load_json(MEMORY_FILE, {"strategy_scores": {}})
    st.json(memory.get("strategy_scores", {}))

    st.caption("Educational paper-trading simulator. No real orders are sent.")


def main() -> None:
    render_dashboard()
    time.sleep(CYCLE_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
