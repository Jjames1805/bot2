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

def load_
