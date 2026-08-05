#!/usr/bin/env python3
"""
apex_trader.py — systematic crypto trend bot (paper-first, live-ready)

Strategy (validated in backtests on 2020→2026 Coinbase daily data):
  ENTRY: flat AND close > Donchian upper(30, shifted 1)
         AND Supertrend(10, 3.0) direction == +1
         AND close > SMA200 (regime filter)
  EXIT:  Supertrend flips to -1 (signal exit). Champion+ 2026-08-03: the
         Donchian lower(15) exit leg was removed — it sat below the ST band
         on ~100% of bars and only cut winners early; OOS Sharpe improved on
         both symbols (research/flagship_champion.py).
  Signals act on the most recently CLOSED daily candle only.

Risk overlay (deterministic, runs every cycle — the LLM cannot override it):
  - Position sizing (SIZING_MODE=exposure, default): notional = MAX_POSITION_PCT
    of equity; the stop is catastrophe insurance, not the sizing dial.
    (SIZING_MODE=risk is the legacy 1%-of-equity / stop-distance sizing —
    backtest sweep 2026-08-03 showed it strangles this trend system.)
  - Chandelier trailing stop: stop = max(stop, highest_close_since_entry - ATR_STOP_MULT x ATR_at_entry)
  - Intraday hard-stop check every cycle using the latest price
  - Per-symbol cooldown after a stop exit
  - Daily circuit breaker halts new entries if equity drops
    DAILY_CIRCUIT_BREAKER_PCT from UTC day start
  - Caps: max MAX_POSITIONS positions, MAX_POSITION_PCT of equity per
    position, MAX_EXPOSURE_PCT total exposure

LLM layer: entries only, veto-only, fail-open. The strategy generates the
signal; the LLM may reject it. If the LLM is unavailable or returns junk,
the trade is ALLOWED and the failure is logged/notified.

Data: Coinbase public API is the primary daily-candle source (deep history
for the SMA200, no auth, same source as the validated backtests). Revolut X
candles are the fallback. Revolut X tickers are used for intraday prices
when credentials are configured, otherwise the latest Coinbase daily close.

Modes:
  python apex_trader.py            # loop every CHECK_INTERVAL seconds
  RUN_ONCE=true python apex_trader.py   # single cycle (used by CI cron)
  DRY_RUN=true ...                 # signals + sizing logged, no state mutation
  LIVE_TRADING=true ...            # REAL orders on Revolut X (default: paper)
"""

import argparse
import base64
import json
import os
import requests
import smtplib
import ssl
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from email.header import Header
from email.mime.text import MIMEText

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.environ.get("STATE_FILE", os.path.join(SCRIPT_DIR, "apex_state.json"))
LOG_FILE = os.environ.get("LOG_FILE", os.path.join(SCRIPT_DIR, "apex_decisions.jsonl"))

RUN_ONCE = os.environ.get("RUN_ONCE", "false").lower() == "true"
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
LIVE_TRADING = os.environ.get("LIVE_TRADING", "false").lower() == "true"
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "900"))  # seconds between cycles in loop mode

STARTING_CAPITAL_EUR = float(os.environ.get("STARTING_CAPITAL_EUR", "500"))
SYMBOLS = ["BTC-EUR", "ETH-EUR"]

# Strategy parameters (validated)
DONCHIAN_ENTRY_PERIOD = 30
SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0
SMA_PERIOD = 200
MIN_CANDLES = 210  # need SMA200 + a few warmup bars for supertrend bands

# Risk overlay
# Backtest sweep 2026-08-03: exposure sizing + wide catastrophe stop dominates
# risk sizing at every stop width (2xATR risk sizing strangled the trend system:
# 66/67 BTC exits were stops, ~11d holds vs months for signal exits).
ATR_PERIOD = 14
ATR_STOP_MULT = float(os.environ.get("ATR_STOP_MULT", "6.0"))  # catastrophe insurance; the signal is the real exit
SIZING_MODE = os.environ.get("SIZING_MODE", "exposure")  # "exposure": notional = MAX_POSITION_PCT equity | "risk": legacy 1%-of-equity / stop distance
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "0.01"))  # only used when SIZING_MODE=risk
STOP_COOLDOWN_HOURS = 48
DAILY_CIRCUIT_BREAKER_PCT = 0.05
MAX_POSITIONS = 2
MAX_POSITION_PCT = 0.25   # of equity per position (this IS the position size in exposure mode)
MAX_EXPOSURE_PCT = 0.50   # total across positions

# Execution model (paper): taker fee + slippage per side
FEE_PCT = 0.001
SLIPPAGE_PCT = 0.0005
MIN_TRADE_EUR = 10.0

# Email notifications (Gmail SMTP)
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", GMAIL_ADDRESS)
EMAIL_ON_TRADE = os.environ.get("EMAIL_ON_TRADE", "true").lower() == "true"
EMAIL_ON_FAILURE = os.environ.get("EMAIL_ON_FAILURE", "true").lower() == "true"
EMAIL_DAILY_DIGEST = os.environ.get("EMAIL_DAILY_DIGEST", "true").lower() == "true"
EMAIL_ON_HOLD = os.environ.get("EMAIL_ON_HOLD", "false").lower() == "true"
# Phone push via ntfy.sh — free, no signup; the topic name IS the secret, so
# keep it unguessable. Empty = push silently skipped (fail-open, like email).
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
HOLD_EMAIL_HOURS = 1
FAIL_EMAIL_SECONDS = 3600

# LLM (veto-only). Chain order: override -> Groq 70b -> Groq 8b -> OpenRouter fallbacks.
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))
LLM_TIMEOUT = 90
_LLM_CHAIN = [
    {"model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    {"model": "llama-3.1-8b-instant", "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    {"model": "nvidia/nemotron-3-ultra-550b-a55b:free", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"model": "inclusionai/ling-3.0-flash:free", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
]
if LLM_MODEL:
    # Explicit override: try only this model (on whichever provider has a key).
    _LLM_CHAIN = [dict(e, model=LLM_MODEL) for e in _LLM_CHAIN]
LLM_MODEL_CHAIN = _LLM_CHAIN

# Revolut X
REVX_BASE_URL = "https://revx.revolut.com"
API_KEY = os.environ.get("REVOLUT_X_API_KEY", "")
PRIVATE_KEY_PATH = os.environ.get(
    "REVOLUT_X_PRIVATE_KEY_PATH",
    os.path.join(os.path.expanduser("~"), ".config", "revolut-x", "private.pem"),
)
RAW_PEM_KEY = os.environ.get("REVOLUT_X_PRIVATE_KEY", "")  # for CI (secret-injected key material)

COINBASE_BASE_URL = "https://api.exchange.coinbase.com"
SYMBOL_TO_COINBASE = {"BTC-EUR": "BTC-EUR", "ETH-EUR": "ETH-EUR"}

CANDLE_TIMEOUT = 10
SMTP_TIMEOUT = 20


def _load_dotenv():
    """Load KEY=VALUE lines from .env next to this script. Never overwrites
    variables that are already set in the environment."""
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"⚠️ Could not read .env: {e}")


_load_dotenv()
# Re-read anything the .env may have just provided.
API_KEY = os.environ.get("REVOLUT_X_API_KEY", API_KEY)
RAW_PEM_KEY = os.environ.get("REVOLUT_X_PRIVATE_KEY", RAW_PEM_KEY)
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", GMAIL_ADDRESS)
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", GMAIL_APP_PASSWORD)
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", GMAIL_ADDRESS)
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", NTFY_TOPIC)


# ── Small helpers ─────────────────────────────────────────────────────────────

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def log_decision(entry):
    """Append one JSON line to the decision log. Tolerates a corrupt file."""
    entry = dict(entry)
    entry.setdefault("ts", now_iso())
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        print(f"⚠️ Could not write decision log: {e}")


# ── State management (atomic) ─────────────────────────────────────────────────

def default_state():
    return {
        "cash_eur": STARTING_CAPITAL_EUR,
        "starting_capital_eur": STARTING_CAPITAL_EUR,
        "positions": {},       # symbol -> position dict
        "closed_trades": [],   # list of trade dicts
        "cooldowns": {},       # symbol -> iso timestamp when cooldown ends
        "day": {"date": today_str(), "start_equity": STARTING_CAPITAL_EUR, "halted": False},
        "last_signal_candle": {},  # symbol -> epoch of last daily candle we processed signals for
        "last_digest_date": "",
        "last_hold_email": 0.0,
        "last_fail_email": 0.0,
        "total_fees_eur": 0.0,
        "created_at": now_iso(),
        "live_trading": False,
    }


def load_state():
    """Load state. Missing file starts fresh; a CORRUPT file is preserved as
    .corrupt and raises (never silently reset)."""
    if not os.path.exists(STATE_FILE):
        state = default_state()
        save_state(state)
        return state
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        corrupt_path = STATE_FILE + ".corrupt"
        try:
            os.replace(STATE_FILE, corrupt_path)
        except Exception:
            pass
        send_email(
            "🚨 apex-trader: state file corrupt",
            f"State file {STATE_FILE} failed to parse ({e}).\n"
            f"It was moved to {corrupt_path}. The bot is STOPPED — restore or delete it manually.",
            force=True,
        )
        raise RuntimeError(f"Corrupt state file preserved at {corrupt_path}: {e}")

    # Fill any missing keys from defaults (forward-compatible schema upgrades)
    fresh = default_state()
    for k, v in fresh.items():
        state.setdefault(k, v)
    state["day"] = {**fresh["day"], **state.get("day", {})}
    return state


def save_state(state):
    """Atomic write: .tmp + os.replace."""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, STATE_FILE)


def reset_daily(state):
    """Reset the per-day tracking at UTC midnight."""
    today = today_str()
    if state["day"].get("date") != today:
        state["day"] = {
            "date": today,
            "start_equity": total_equity(state, {}),
            "halted": False,
        }


def total_equity(state, latest_prices):
    """cash + open positions marked to market (latest known prices)."""
    eq = state["cash_eur"]
    for symbol, pos in state["positions"].items():
        price = latest_prices.get(symbol, pos.get("last_price", pos["entry_price"]))
        eq += pos["qty"] * price
    return eq


# ── Revolut X API (Ed25519 signed) ────────────────────────────────────────────

def _load_private_key():
    from cryptography.hazmat.primitives import serialization
    if RAW_PEM_KEY:
        pem = RAW_PEM_KEY
        if "\\n" in pem:  # env vars often escape newlines
            pem = pem.replace("\\n", "\n")
        return serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    with open(PRIVATE_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def revx_headers(method, path, query_string="", body_str=""):
    """Signed headers. Signature message: {ts}{METHOD}{path}{query}{body}
    (query sorted/encoded, NO '?' prefix; body appended raw for POST)."""
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}{method}{path}{query_string}{body_str}"
    private_key = _load_private_key()
    signature = base64.b64encode(private_key.sign(message.encode("utf-8"))).decode("utf-8")
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Revx-API-Key": API_KEY,
        "X-Revx-Timestamp": timestamp,
        "X-Revx-Signature": signature,
    }


def revx_get(path, query_string="", timeout=CANDLE_TIMEOUT):
    url = f"{REVX_BASE_URL}{path}" + (f"?{query_string}" if query_string else "")
    resp = requests.get(url, headers=revx_headers("GET", path, query_string), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data) if isinstance(data, dict) else data


def revx_post(path, body_dict, timeout=15):
    body_str = json.dumps(body_dict, separators=(",", ":"))
    url = f"{REVX_BASE_URL}{path}"
    resp = requests.post(url, headers=revx_headers("POST", path, "", body_str), data=body_str, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data) if isinstance(data, dict) else data


def fetch_revx_candles(symbol, limit=MIN_CANDLES):
    """Daily candles from Revolut X. Returns list of dicts oldest-first, or [] on failure."""
    try:
        rows = revx_get(f"/api/1.0/candles/{symbol}", f"interval=1440&limit={limit}")
    except Exception as e:
        print(f"⚠️ RevX candles failed for {symbol}: {e}")
        return []
    return _normalize_candles(rows)


def fetch_revx_tickers():
    """Latest prices for all pairs. Returns {symbol: price} or {} on failure."""
    if not API_KEY or not (RAW_PEM_KEY or os.path.exists(PRIVATE_KEY_PATH)):
        return {}
    try:
        rows = revx_get("/api/1.0/tickers")
    except Exception as e:
        print(f"⚠️ RevX tickers failed: {e}")
        return {}
    prices = {}
    if isinstance(rows, list):
        for t in rows:
            sym = t.get("symbol")
            p = t.get("price") or t.get("last_price") or t.get("last")
            if sym and p is not None:
                try:
                    prices[sym] = float(p)
                except (TypeError, ValueError):
                    pass
    return prices


def fetch_revx_balances():
    """Live balances (list of {currency, available/total...}) or None on failure."""
    try:
        return revx_get("/api/1.0/balances")
    except Exception as e:
        print(f"⚠️ RevX balances failed: {e}")
        return None


def place_market_order(symbol, side, quote_size_eur=None, base_size=None):
    """Live market order. Exactly one of quote_size_eur / base_size.
    Returns the venue result dict; raises on HTTP failure."""
    market = {}
    if base_size is not None:
        market["base_size"] = f"{base_size:.8f}".rstrip("0").rstrip(".")
    else:
        market["quote_size"] = f"{quote_size_eur:.2f}"
    body = {
        "symbol": symbol,
        "side": side,
        "order_configuration": {"market": market},
        "client_order_id": f"apex-{uuid.uuid4()}",
    }
    print(f"🔴 LIVE ORDER: {side} {symbol} {market}")
    return revx_post("/api/1.0/orders", body)


# ── Coinbase public candles (primary signal source) ──────────────────────────

def fetch_coinbase_daily(symbol, limit=MIN_CANDLES):
    """Daily candles from Coinbase (public, no auth). [time, low, high, open, close, volume]
    newest-first. Returns list of dicts oldest-first, or [] on failure."""
    product = SYMBOL_TO_COINBASE.get(symbol)
    if not product:
        return []
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=limit + 10)  # margin for gaps
    url = f"{COINBASE_BASE_URL}/products/{product}/candles"
    try:
        resp = requests.get(
            url,
            params={"start": start.isoformat(), "end": end.isoformat(), "granularity": 86400},
            timeout=CANDLE_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        print(f"⚠️ Coinbase candles failed for {symbol}: {e}")
        return []
    candles = []
    for r in sorted(rows, key=lambda x: x[0]):
        candles.append({
            "time": int(r[0]),
            "low": float(r[1]),
            "high": float(r[2]),
            "open": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        })
    return candles


def _normalize_candles(rows):
    """Best-effort normalization of RevX candle rows to our dict shape."""
    candles = []
    if not isinstance(rows, list):
        return candles
    for r in rows:
        try:
            if isinstance(r, dict):
                candles.append({
                    "time": int(r.get("time") or r.get("start") or r.get("timestamp")),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r.get("volume", 0) or 0),
                })
            elif isinstance(r, (list, tuple)) and len(r) >= 5:
                # assume [time, open, high, low, close, (volume)]
                candles.append({
                    "time": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]) if len(r) > 5 else 0.0,
                })
        except Exception:
            continue
    candles.sort(key=lambda c: c["time"])
    return candles


def fetch_daily_candles(symbol, limit=MIN_CANDLES):
    """Primary: Coinbase. Fallback: Revolut X. Drops the still-forming current
    daily candle so callers only see CLOSED candles."""
    candles = fetch_coinbase_daily(symbol, limit)
    if len(candles) < MIN_CANDLES:
        alt = fetch_revx_candles(symbol, limit)
        if len(alt) > len(candles):
            print(f"ℹ️ {symbol}: using RevX candles ({len(alt)}) — Coinbase gave {len(candles)}")
            candles = alt
    # Drop incomplete current-day candle: a daily candle opened at t is complete
    # once t + 86400 <= now.
    now_ts = int(time.time())
    closed = [c for c in candles if c["time"] + 86400 <= now_ts]
    return closed


# ── Indicators (exact ports of the validated backtest implementations) ────────

def sma(values, period):
    """SMA aligned to the LAST element: sma[i] uses values[i-period+1..i]. None when not enough data."""
    out = [None] * len(values)
    if len(values) < period:
        return out
    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def rolling_max(values, period):
    """Rolling max over values[i-period+1..i], then shifted 1 (result[i] uses window ending at i-1)."""
    out = [None] * len(values)
    for i in range(period, len(values)):
        out[i] = max(values[i - period:i])  # window ending at i-1 (shifted)
    return out


def rolling_min(values, period):
    out = [None] * len(values)
    for i in range(period, len(values)):
        out[i] = min(values[i - period:i])
    return out


def atr_series(candles, period=ATR_PERIOD):
    """SMA-based True Range average (matches the validated supertrend backtest,
    NOT Wilder's smoothing). First value at index `period`."""
    n = len(candles)
    tr = [None] * n
    for i in range(1, n):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    out = [None] * n
    if n <= period:
        return out
    # first ATR = mean of tr[1..period]
    vals = [t for t in tr[1:period + 1] if t is not None]
    if len(vals) == period:
        out[period] = sum(vals) / period
        for i in range(period + 1, n):
            # simple moving average of TR over the last `period` bars
            window = tr[i - period + 1:i + 1]
            out[i] = sum(window) / period
    return out


def supertrend(candles, period=SUPERTREND_PERIOD, mult=SUPERTREND_MULT):
    """Exact port of the validated backtest supertrend.
    Returns (direction, upper_band, lower_band) lists aligned to candles.
    direction[i] in {+1, -1}; +1 = uptrend (price above lower band)."""
    n = len(candles)
    atr = atr_series(candles, period)
    hl2 = [(c["high"] + c["low"]) / 2 for c in candles]

    upper_band = [None] * n
    lower_band = [None] * n
    direction = [1] * n

    # seed at first index with a valid ATR
    start = None
    for i in range(n):
        if atr[i] is not None:
            start = i
            break
    if start is None:
        return direction, upper_band, lower_band

    upper_band[start] = hl2[start] + mult * atr[start]
    lower_band[start] = hl2[start] - mult * atr[start]

    for i in range(start + 1, n):
        if atr[i] is None:
            atr[i] = atr[i - 1]
        ub_basic = hl2[i] + mult * atr[i]
        lb_basic = hl2[i] - mult * atr[i]
        prev_close = candles[i - 1]["close"]

        if prev_close > upper_band[i - 1]:
            upper_band[i] = ub_basic
        else:
            upper_band[i] = min(ub_basic, upper_band[i - 1])

        if prev_close < lower_band[i - 1]:
            lower_band[i] = lb_basic
        else:
            lower_band[i] = max(lb_basic, lower_band[i - 1])

        if direction[i - 1] == -1 and candles[i]["close"] > upper_band[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and candles[i]["close"] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return direction, upper_band, lower_band


def compute_indicators(candles):
    """All indicator series for the candle list. Returns dict of lists (aligned)."""
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    direction, upper_band, lower_band = supertrend(candles)
    return {
        "close": closes,
        "entry_high": rolling_max(highs, DONCHIAN_ENTRY_PERIOD),  # shifted by 1
        "sma200": sma(closes, SMA_PERIOD),
        "atr": atr_series(candles, ATR_PERIOD),
        "st_dir": direction,
        "st_upper": upper_band,
        "st_lower": lower_band,
    }


def signal_for(candles, ind):
    """Evaluate entry/exit on the LAST (closed) candle. Returns dict:
    {entry: bool, exit: bool, reason...}. Entry requires donchian breakout
    + supertrend up + close above SMA200."""
    i = len(candles) - 1
    close = candles[i]["close"]
    entry_high = ind["entry_high"][i]
    sma200 = ind["sma200"][i]
    st_dir = ind["st_dir"][i]

    if entry_high is None or sma200 is None:
        return {"entry": False, "exit": False, "warmup": True}

    breakout = close > entry_high
    trend_up = st_dir == 1
    regime_ok = close > sma200
    entry = breakout and trend_up and regime_ok
    exit_sig = (st_dir == -1)  # champion+: supertrend flip is the only signal exit

    return {
        "entry": entry,
        "exit": exit_sig,
        "breakout": breakout,
        "trend_up": trend_up,
        "regime_ok": regime_ok,
        "close": close,
        "entry_high": entry_high,
        "sma200": sma200,
        "st_dir": st_dir,
        "atr": ind["atr"][i],
        "candle_time": candles[i]["time"],
    }


# ── Email ─────────────────────────────────────────────────────────────────────

def send_email(subject, body, force=False):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD or not NOTIFY_EMAIL:
        if force:
            print(f"⚠️ Email not configured; would have sent: {subject}")
        return False
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = NOTIFY_EMAIL
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=SMTP_TIMEOUT, context=context) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, [NOTIFY_EMAIL], msg.as_string())
        print(f"📧 Sent: {subject}")
        return True
    except Exception as e:
        print(f"⚠️ Email failed: {e}")
        return False


def send_push(title, body, priority="default"):
    """Phone push via ntfy.sh. Fail-open: unset topic or network error just
    prints a warning — a dead push channel must never block a trade."""
    if not NTFY_TOPIC:
        return False
    try:
        resp = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=body.encode("utf-8"),
            headers={"Title": Header(title, "utf-8").encode(), "Priority": priority},
            timeout=10,
        )
        ok = resp.status_code == 200
        print(f"📲 Push {'sent' if ok else f'HTTP {resp.status_code}'}: {title}")
        return ok
    except Exception as e:
        print(f"⚠️ Push failed: {e}")
        return False


def maybe_send_hold_email(state, equity, actions):
    """Throttled HOLD/status email (off by default in the new bot)."""
    if not EMAIL_ON_HOLD:
        return
    now = time.time()
    if now - state.get("last_hold_email", 0) < HOLD_EMAIL_HOURS * 3600:
        return
    state["last_hold_email"] = now
    send_email(
        f"⏸ apex-trader HOLD — equity €{equity:.2f}",
        "No action this cycle.\n\n" + _portfolio_text(state, equity) + f"\nActions: {actions}",
    )


def maybe_send_failure_email(state, subject, body):
    if not EMAIL_ON_FAILURE:
        return
    now = time.time()
    if now - state.get("last_fail_email", 0) < FAIL_EMAIL_SECONDS:
        return
    state["last_fail_email"] = now
    send_email(subject, body, force=True)


def maybe_send_daily_digest(state, equity):
    if not EMAIL_DAILY_DIGEST:
        return
    today = today_str()
    if state.get("last_digest_date") == today:
        return
    state["last_digest_date"] = today
    pnl = equity - state["starting_capital_eur"]
    pnl_pct = (pnl / state["starting_capital_eur"]) * 100 if state["starting_capital_eur"] else 0
    trades = state["closed_trades"]
    wins = sum(1 for t in trades if t.get("pnl_eur", 0) > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0
    send_email(
        f"📊 apex-trader daily digest — €{equity:.2f} ({pnl_pct:+.2f}%)",
        _portfolio_text(state, equity)
        + f"\nClosed trades: {len(trades)} (win rate {win_rate:.0f}%)\n"
        + f"Total P&L: €{pnl:+.2f} ({pnl_pct:+.2f}%)\n"
        + f"Fees paid: €{state.get('total_fees_eur', 0):.2f}\n"
        + f"Mode: {'LIVE' if LIVE_TRADING else 'paper'}{'(DRY RUN)' if DRY_RUN else ''}\n",
        force=True,
    )


def _portfolio_text(state, equity):
    lines = [
        f"Equity: €{equity:.2f}",
        f"Cash: €{state['cash_eur']:.2f}",
        f"Open positions: {len(state['positions'])}",
    ]
    for symbol, pos in state["positions"].items():
        pnl_pct = ((pos.get("last_price", pos["entry_price"]) / pos["entry_price"]) - 1) * 100
        lines.append(
            f"  - {symbol}: {pos['qty']:.8f} @ €{pos['entry_price']:.2f} "
            f"({pnl_pct:+.1f}%), stop €{pos['hard_stop']:.2f}"
        )
    return "\n".join(lines)


# ── LLM veto layer (entries only, fail-open) ──────────────────────────────────

def _api_key_for(entry):
    return os.environ.get(entry["key_env"], "")


def llm_veto(symbol, sig, state, equity):
    """Ask the LLM whether to VETO this entry. Returns (allow: bool, note: str).
    Fail-open: any provider/parse failure -> allow=True."""
    if not any(_api_key_for(e) for e in LLM_MODEL_CHAIN):
        return True, "no LLM key configured — fail-open allow"

    pos_text = _portfolio_text(state, equity)
    prompt = f"""You are the risk officer for a systematic crypto trend-following bot.

The strategy has produced an ENTRY signal:
  Symbol: {symbol}
  Closed-candle close: €{sig['close']:.2f}
  Donchian upper(30): €{sig['entry_high']:.2f} (breakout: {sig['breakout']})
  SMA200: €{sig['sma200']:.2f} (above: {sig['regime_ok']})
  Supertrend(10,3): {'UP' if sig['trend_up'] else 'DOWN'}
  ATR(14): €{(sig.get('atr') or 0):.2f}

Current portfolio:
{pos_text}
Circuit breaker halted today: {state['day'].get('halted', False)}

You may ONLY veto (reject) this entry — you cannot create trades. Veto only for
clear, severe risk conditions (e.g. extreme vertical overextension far above the
breakout level, obvious blow-off top structure, market-wide crash in progress).
When in doubt, ALLOW — the deterministic risk overlay (2×ATR stops, 1% risk
sizing, 50% exposure cap) already bounds the downside.

Respond with ONLY a JSON object (no markdown, no fences):
{{"veto": true or false, "reason": "one sentence"}}"""

    for entry in LLM_MODEL_CHAIN:
        api_key = _api_key_for(entry)
        if not api_key:
            continue
        result = _call_llm(entry, prompt, api_key)
        if result is None:
            continue
        veto = bool(result.get("veto"))
        reason = str(result.get("reason", ""))[:200]
        return (not veto), f"{entry['model']}: veto={veto} — {reason}"

    return True, "all LLM providers failed — fail-open allow"


def _call_llm(entry, prompt, api_key):
    """One provider call. Returns parsed dict or None."""
    model_name = entry["model"]
    base_url = entry["base_url"].rstrip("/")
    text = ""
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_name, "max_tokens": LLM_MAX_TOKENS,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=LLM_TIMEOUT,
        )
        if response.status_code != 200:
            print(f"⚠️ LLM error ({model_name}): {response.status_code} {response.text[:200]}")
            return None
        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ LLM call failed ({model_name}): {e}")
        if text:
            print(f"   raw: {text[:200]}")
        return None


# ── Trade execution (paper + live) ────────────────────────────────────────────

def paper_fill_price(side, price):
    return price * (1 + FEE_PCT + SLIPPAGE_PCT) if side == "buy" else price * (1 - FEE_PCT - SLIPPAGE_PCT)


def execute_buy(state, symbol, sig, equity, dry_run=False):
    """Size and open a position. Returns trade dict or None."""
    atr = sig.get("atr")
    if not atr or atr <= 0:
        print(f"⚠️ {symbol}: no ATR — cannot size position")
        return None

    stop_dist = ATR_STOP_MULT * atr
    price = sig["close"]

    if SIZING_MODE == "exposure":
        # Position target IS the cap; the stop is catastrophe insurance, not the sizing dial.
        notional = equity * MAX_POSITION_PCT
    else:
        # Legacy risk sizing: qty = 1% equity / stop distance (shrinks as stop widens).
        risk_eur = equity * RISK_PER_TRADE_PCT
        notional = min(risk_eur * price / stop_dist, equity * MAX_POSITION_PCT)

    # Total exposure cap
    current_exposure = sum(p["qty"] * p.get("last_price", p["entry_price"]) for p in state["positions"].values())
    remaining_exposure = equity * MAX_EXPOSURE_PCT - current_exposure
    notional = max(0.0, min(notional, remaining_exposure))

    # Cash constraint
    notional = min(notional, state["cash_eur"])
    qty = notional / price if price > 0 else 0

    if notional < MIN_TRADE_EUR:
        print(f"⛔ {symbol}: sized notional €{notional:.2f} below minimum €{MIN_TRADE_EUR:.0f} — skip")
        log_decision({"type": "skip", "symbol": symbol, "reason": "below_min_notional",
                      "notional": notional, "signal": sig})
        return None

    if dry_run:
        print(f"🧪 DRY RUN buy {symbol}: €{notional:.2f} ({qty:.8f}) @ €{price:.2f}, stop €{price - stop_dist:.2f}")
        log_decision({"type": "dry_buy", "symbol": symbol, "notional": notional, "qty": qty,
                      "price": price, "stop": price - stop_dist, "signal": sig})
        return {"symbol": symbol, "notional": notional, "qty": qty, "dry": True}

    if LIVE_TRADING:
        result = place_market_order(symbol, "buy", quote_size_eur=notional)
        fill_price = price  # market order — assume ~last price for state; venue truth reconciled on restart
        venue_id = result.get("venue_order_id") if isinstance(result, dict) else None
        print(f"🔴 LIVE BUY placed: {venue_id}")
    else:
        fill_price = paper_fill_price("buy", price)
        venue_id = None

    fee = notional * (FEE_PCT + SLIPPAGE_PCT) if not LIVE_TRADING else 0.0
    pos = {
        "symbol": symbol,
        "qty": qty,
        "entry_price": fill_price,
        "entry_date": now_iso(),
        "notional_eur": notional,
        "fees_eur": fee,
        "hard_stop": fill_price - stop_dist,
        "highest_close": sig["close"],
        "atr_at_entry": atr,
        "last_price": price,
        "venue_order_id": venue_id,
    }
    state["cash_eur"] -= notional
    state["positions"][symbol] = pos
    state["total_fees_eur"] = state.get("total_fees_eur", 0) + fee

    trade = {"type": "buy", "symbol": symbol, "qty": qty, "price": fill_price,
             "notional": notional, "stop": pos["hard_stop"], "live": LIVE_TRADING,
             "venue_order_id": venue_id, "signal": sig}
    log_decision(trade)
    send_push(
        f"🟢 apex BUY {symbol}{' (LIVE)' if LIVE_TRADING else ''}",
        f"@ €{fill_price:.2f} · €{notional:.2f} · stop €{pos['hard_stop']:.2f}",
        priority="high",
    )
    if EMAIL_ON_TRADE:
        send_email(
            f"🟢 apex-trader BUY {symbol} €{notional:.2f}{' (LIVE)' if LIVE_TRADING else ''}",
            f"Entry signal confirmed.\n\n{json.dumps({k: v for k, v in trade.items() if k != 'signal'}, indent=2, default=str)}\n\n"
            f"Signal: close €{sig['close']:.2f} > Donchian €{sig['entry_high']:.2f}, "
            f"supertrend UP, above SMA200 €{sig['sma200']:.2f}\n"
            f"Stop: €{pos['hard_stop']:.2f} (2×ATR = €{stop_dist:.2f})\n",
            force=True,
        )
    return trade


def execute_sell(state, symbol, reason, price=None, dry_run=False):
    """Close a position. Returns trade dict or None."""
    pos = state["positions"].get(symbol)
    if not pos:
        return None
    price = price if price is not None else pos.get("last_price", pos["entry_price"])

    if dry_run:
        print(f"🧪 DRY RUN sell {symbol}: {pos['qty']:.8f} @ €{price:.2f} — {reason}")
        log_decision({"type": "dry_sell", "symbol": symbol, "qty": pos["qty"], "price": price, "reason": reason})
        return {"symbol": symbol, "dry": True}

    if LIVE_TRADING:
        result = place_market_order(symbol, "sell", base_size=pos["qty"])
        venue_id = result.get("venue_order_id") if isinstance(result, dict) else None
        fill_price = price
        proceeds = pos["qty"] * fill_price
        fee = 0.0
        print(f"🔴 LIVE SELL placed: {venue_id}")
    else:
        fill_price = paper_fill_price("sell", price)
        proceeds = pos["qty"] * fill_price
        fee = pos["qty"] * price * (FEE_PCT + SLIPPAGE_PCT)
        venue_id = None

    pnl = proceeds - pos["notional_eur"]
    state["cash_eur"] += proceeds
    state["total_fees_eur"] = state.get("total_fees_eur", 0) + fee
    del state["positions"][symbol]

    closed = {
        "symbol": symbol,
        "qty": pos["qty"],
        "entry_price": pos["entry_price"],
        "exit_price": fill_price,
        "entry_date": pos["entry_date"],
        "exit_date": now_iso(),
        "notional_eur": pos["notional_eur"],
        "proceeds_eur": proceeds,
        "pnl_eur": pnl,
        "pnl_pct": (pnl / pos["notional_eur"]) * 100 if pos["notional_eur"] else 0,
        "reason": reason,
        "live": LIVE_TRADING,
    }
    state["closed_trades"].append(closed)
    log_decision({"type": "sell", **closed})

    emoji = "💰" if pnl >= 0 else "🔴"
    send_push(
        f"{emoji} apex SELL {symbol} {closed['pnl_pct']:+.2f}%{' (LIVE)' if LIVE_TRADING else ''}",
        f"@ €{fill_price:.2f} · P&L €{pnl:+.2f} · {reason}",
        priority="high",
    )
    if EMAIL_ON_TRADE:
        send_email(
            f"{emoji} apex-trader SELL {symbol} {closed['pnl_pct']:+.2f}%{' (LIVE)' if LIVE_TRADING else ''}",
            f"Exit reason: {reason}\n\n{json.dumps(closed, indent=2, default=str)}\n",
            force=True,
        )
    return closed


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle():
    print(f"\n=== Cycle {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC ===")
    state = load_state()
    reset_daily(state)

    # 1) fetch daily candles
    daily_data = {}
    for symbol in SYMBOLS:
        candles = fetch_daily_candles(symbol)
        if len(candles) < MIN_CANDLES:
            print(f"⚠️ {symbol}: only {len(candles)} closed daily candles (need {MIN_CANDLES})")
        daily_data[symbol] = candles

    if all(len(c) == 0 for c in daily_data.values()):
        maybe_send_failure_email(state, "🚨 apex-trader: total market-data outage",
                                 "Could not fetch daily candles for any symbol from Coinbase or RevX.")
        log_decision({"type": "error", "reason": "total_market_data_outage"})
        save_state(state)
        return []

    # 2) latest prices for intraday stop checks + mark-to-market
    latest_prices = fetch_revx_tickers()
    for symbol in SYMBOLS:
        if symbol not in latest_prices and daily_data[symbol]:
            latest_prices[symbol] = daily_data[symbol][-1]["close"]

    actions = []
    indicators = {}
    for symbol in SYMBOLS:
        candles = daily_data[symbol]
        if not candles:
            continue
        ind = compute_indicators(candles)
        indicators[symbol] = ind
        sig = signal_for(candles, ind)
        last_candle_time = candles[-1]["time"]
        already_processed = state["last_signal_candle"].get(symbol) == last_candle_time

        pos = state["positions"].get(symbol)
        price = latest_prices.get(symbol, sig.get("close"))
        if pos:
            pos["last_price"] = price

        # ── intraday hard stop (runs every cycle, even on already-processed candles)
        stopped = False
        if pos and price is not None and price <= pos["hard_stop"]:
            trade = execute_sell(state, symbol, f"hard stop hit (price €{price:.2f} <= stop €{pos['hard_stop']:.2f})",
                                 price=price, dry_run=DRY_RUN)
            if trade and not DRY_RUN:
                state["cooldowns"][symbol] = (datetime.now(timezone.utc) + timedelta(hours=STOP_COOLDOWN_HOURS)).isoformat()
            actions.append(f"SELL {symbol} (hard stop)")
            stopped = True
            pos = None

        if already_processed:
            continue

        # ── new closed daily candle: strategy logic
        state["last_signal_candle"][symbol] = last_candle_time

        if pos:
            # update chandelier trail with the new closed candle
            pos["highest_close"] = max(pos.get("highest_close", pos["entry_price"]), sig["close"])
            trail = pos["highest_close"] - ATR_STOP_MULT * pos["atr_at_entry"]
            if trail > pos["hard_stop"]:
                print(f"📈 {symbol}: trail stop €{pos['hard_stop']:.2f} -> €{trail:.2f}")
                pos["hard_stop"] = trail

            if not stopped and sig["exit"]:
                reason = "supertrend flipped DOWN"  # champion+: ST flip is the only signal exit
                trade = execute_sell(state, symbol, reason, price=sig["close"], dry_run=DRY_RUN)
                actions.append(f"SELL {symbol} ({reason})")
            else:
                print(f"📊 {symbol}: holding ({sig['close']:.2f}, stop {pos['hard_stop']:.2f})")

        elif sig.get("entry") and not sig.get("warmup"):
            # Entry path: overlay gates first, then LLM veto
            if state["day"].get("halted"):
                print(f"⛔ {symbol}: entry signal but circuit breaker halted")
                log_decision({"type": "skip", "symbol": symbol, "reason": "circuit_breaker", "signal": sig})
            elif len(state["positions"]) >= MAX_POSITIONS:
                print(f"⛔ {symbol}: entry signal but max positions reached")
                log_decision({"type": "skip", "symbol": symbol, "reason": "max_positions", "signal": sig})
            else:
                cooldown_end = state["cooldowns"].get(symbol)
                if cooldown_end and cooldown_end > now_iso():
                    print(f"⛔ {symbol}: in cooldown until {cooldown_end}")
                    log_decision({"type": "skip", "symbol": symbol, "reason": "cooldown", "signal": sig})
                else:
                    equity = total_equity(state, latest_prices)
                    allow, note = llm_veto(symbol, sig, state, equity)
                    log_decision({"type": "llm_veto_check", "symbol": symbol, "allow": allow, "note": note})
                    print(f"🧠 LLM veto check {symbol}: {'ALLOW' if allow else 'VETO'} ({note})")
                    if allow:
                        trade = execute_buy(state, symbol, sig, equity, dry_run=DRY_RUN)
                        if trade:
                            actions.append(f"BUY {symbol} €{trade.get('notional', 0):.2f}")
                    else:
                        maybe_send_failure_email(
                            state, f"🧠 apex-trader: LLM vetoed {symbol} entry",
                            f"Signal was rejected by the LLM risk officer.\n\n{note}\n\nSignal: {json.dumps(sig, default=str)}")
        else:
            print(f"📊 {symbol}: no signal (close {sig.get('close')}, entry {sig.get('entry')})")

    # ── daily circuit breaker (halts new entries for the rest of the UTC day)
    equity = total_equity(state, latest_prices)
    if not state["day"].get("halted") and equity < state["day"]["start_equity"] * (1 - DAILY_CIRCUIT_BREAKER_PCT):
        state["day"]["halted"] = True
        msg = (f"Equity €{equity:.2f} is down more than {DAILY_CIRCUIT_BREAKER_PCT * 100:.0f}% from "
               f"day start €{state['day']['start_equity']:.2f}. New entries halted until UTC midnight.")
        print(f"🚨 CIRCUIT BREAKER: {msg}")
        send_email("🚨 apex-trader: daily circuit breaker", msg + "\n\n" + _portfolio_text(state, equity), force=True)
        log_decision({"type": "circuit_breaker", "equity": equity, "start_equity": state["day"]["start_equity"]})

    # ── housekeeping
    maybe_send_daily_digest(state, equity)
    if not actions:
        maybe_send_hold_email(state, equity, actions)
    log_decision({"type": "cycle", "equity": equity, "cash": state["cash_eur"],
                  "positions": {s: {"qty": p["qty"], "entry": p["entry_price"], "stop": p["hard_stop"]}
                                for s, p in state["positions"].items()},
                  "actions": actions})
    save_state(state)
    print(f"✅ Cycle done. Equity €{equity:.2f} | cash €{state['cash_eur']:.2f} | positions {len(state['positions'])} | actions: {actions or 'none'}")
    return actions


def main():
    # Windows consoles default to cp1252 and crash on emoji output; replace
    # unencodable characters instead (no-op on UTF-8 platforms like CI).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="apex-trader — systematic crypto trend bot")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args()

    mode = "LIVE" if LIVE_TRADING else "paper"
    print(f"🚀 apex-trader starting ({mode}{' + DRY RUN' if DRY_RUN else ''})")
    print(f"   symbols: {', '.join(SYMBOLS)} | capital €{STARTING_CAPITAL_EUR:.2f} | state: {STATE_FILE}")
    if LIVE_TRADING and not (API_KEY and (RAW_PEM_KEY or os.path.exists(PRIVATE_KEY_PATH))):
        print("❌ LIVE_TRADING=true but Revolut X credentials are missing — refusing to start")
        sys.exit(1)

    if RUN_ONCE or args.once:
        try:
            run_cycle()
        except Exception as e:
            traceback.print_exc()
            send_email("🚨 apex-trader: cycle crashed", f"{e}\n\n{traceback.format_exc()}", force=True)
            raise
        return

    # Local loop mode
    send_email(f"🚀 apex-trader started ({mode})", f"Local loop started. Interval {CHECK_INTERVAL}s.\n"
               f"Symbols: {', '.join(SYMBOLS)}\nState: {STATE_FILE}", force=True)
    while True:
        try:
            run_cycle()
        except KeyboardInterrupt:
            print("\n👋 Stopped by user")
            break
        except Exception as e:
            traceback.print_exc()
            send_email("🚨 apex-trader: cycle crashed", f"{e}\n\n{traceback.format_exc()}", force=True)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
