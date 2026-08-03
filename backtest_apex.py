#!/usr/bin/env python3
"""
backtest_apex.py — pre-deployment validation of the apex-trader combined strategy.

Follows C:\\Users\\Alex\\.claude\\skills\\backtest protocol:
  - 0.1% fees per side always (deployment variant adds 0.05% slippage, stated explicitly)
  - walk-forward: params fixed a priori, headline = last 30% OOS segment
  - parameter-neighbor sensitivity grid
  - drawdown, Sharpe, win rate, trade count reported alongside return
  - baselines on the SAME data (Donchian-only 30/15, Supertrend-only 10/3.0)

Variants per symbol:
  A) combined signal, full-compounding (validated convention, 0.1%/side)
  B) combined signal + ATR overlay, EUR 500 start (deployment-realistic, 0.1% + 0.05% slip/side)
  C) baselines: Donchian 30/15 alone and Supertrend 10/3.0 alone (full compounding, 0.1%/side)

Indicator parity: indicators are imported from apex_trader.py (the exact code
that trades). A parity check against a faithful pandas port of the original
validated implementations runs first and aborts on mismatch.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apex_trader as bot  # exact indicator implementations used by the live bot

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SYMBOLS = ["BTC-EUR", "ETH-EUR"]
FEE = 0.001          # per side, validated convention
SLIP = 0.0005        # deployment variant adds this per side
START_EUR = 500.0

# Strategy params under test (fixed a priori — chosen from prior OOS analysis)
DON_E, DON_X = 30, 15
ST_P, ST_M = 10, 3.0
SMA_P = 200


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_history(symbol):
    """Full daily history from Coinbase, chunked (max 300 candles/request).
    Cached to CSV. Same conventions as the validated sweep tool."""
    cache = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["time"])
        return df
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    end = datetime.now(timezone.utc)
    start = datetime(2014, 1, 1, tzinfo=timezone.utc)
    chunk = pd.Timedelta(seconds=86400 * 290)
    frames = []
    while start < end:
        chunk_end = min(start + chunk, end)
        resp = requests.get(url, params={"start": start.isoformat(), "end": chunk_end.isoformat(),
                                         "granularity": 86400}, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            frames.append(pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]))
        start = chunk_end
        time.sleep(0.3)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["low", "high", "open", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def to_candles(df):
    return [{"time": int(t.timestamp()), "open": o, "high": h, "low": l, "close": c}
            for t, o, h, l, c in zip(df["time"], df["open"], df["high"], df["low"], df["close"])]


# ── Parity check: bot indicators vs faithful port of validated originals ──────

def original_supertrend_dirs(df, period=10, mult=3.0):
    """Faithful port of the validated supertrend_backtest (pandas semantics)."""
    high, low, close = df["high"], df["low"], df["close"]
    hl2 = (high + low) / 2
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    ub_basic = hl2 + mult * atr
    lb_basic = hl2 - mult * atr
    n = len(df)
    ub = np.full(n, np.nan)
    lb = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)
    ub[0], lb[0] = ub_basic.iloc[0], lb_basic.iloc[0]
    for i in range(1, n):
        pc, c = close.iloc[i - 1], close.iloc[i]
        ub[i] = ub_basic.iloc[i] if pc > ub[i - 1] else min(ub_basic.iloc[i], ub[i - 1])
        lb[i] = lb_basic.iloc[i] if pc < lb[i - 1] else max(lb_basic.iloc[i], lb[i - 1])
        if direction[i - 1] == -1 and c > ub[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and c < lb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return direction, ub, lb


def parity_check(df, symbol):
    candles = to_candles(df)
    ind = bot.compute_indicators(candles)
    o_dir, o_ub, o_lb = original_supertrend_dirs(df, ST_P, ST_M)

    # supertrend direction: compare from warmup onward (skip first 30 bars)
    mism = 0
    for i in range(30, len(df)):
        if ind["st_dir"][i] != o_dir[i]:
            mism += 1
    print(f"  parity {symbol} supertrend dir mismatches after warmup: {mism}/{len(df) - 30}")

    # donchian channels vs pandas rolling().shift(1)
    e_high = df["high"].rolling(DON_E).max().shift(1).values
    x_low = df["low"].rolling(DON_X).min().shift(1).values
    eh_mism = sum(1 for i in range(DON_E, len(df))
                  if not np.isclose(ind["entry_high"][i], e_high[i], rtol=1e-9))
    xl_mism = sum(1 for i in range(DON_X, len(df))
                  if not np.isclose(ind["exit_low"][i], x_low[i], rtol=1e-9))
    print(f"  parity {symbol} donchian mismatches: entry_high {eh_mism}, exit_low {xl_mism}")

    if mism > 0 or eh_mism > 0 or xl_mism > 0:
        print("❌ PARITY CHECK FAILED — bot indicators diverge from validated originals")
        sys.exit(1)
    print(f"  ✅ {symbol} indicator parity confirmed")


# ── Signal evaluation (parameterized, mirrors bot.signal_for) ─────────────────

def build_indicators(candles, don_e, don_x, st_p, st_m, sma_p):
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    direction, ub, lb = bot.supertrend(candles, st_p, st_m)
    return {
        "entry_high": bot.rolling_max(highs, don_e),
        "exit_low": bot.rolling_min(lows, don_x),
        "sma": bot.sma(closes, sma_p),
        "atr": bot.atr_series(candles, bot.ATR_PERIOD),
        "st_dir": direction,
    }


def entry_sig(c, i, ind, use_sma=True, use_st=True, use_don=True):
    ok = True
    if use_don:
        eh = ind["entry_high"][i]
        ok = ok and eh is not None and c[i]["close"] > eh
    if use_st:
        ok = ok and ind["st_dir"][i] == 1
    if use_sma:
        s = ind["sma"][i]
        ok = ok and s is not None and c[i]["close"] > s
    return ok


def exit_sig(c, i, ind, use_st=True, use_don=True):
    if use_don:
        xl = ind["exit_low"][i]
        if xl is not None and c[i]["close"] < xl:
            return True
    if use_st and ind["st_dir"][i] == -1:
        return True
    return False


# ── Variant A/C: full-compounding, validated convention ──────────────────────

def backtest_compounding(candles, ind, fee, start_i, end_i,
                         entry_flags=(True, True, True), exit_flags=(True, True)):
    """Full equity compounding per trade (validated convention: entry fill =
    close*(1+fee), exit fill = close*(1-fee), all-in/all-out, long-only).
    entry_flags = (use_don, use_st, use_sma); exit_flags = (use_don, use_st).
    Equity curve is marked to market daily. Returns (curve, trades)."""
    use_don_e, use_st_e, use_sma_e = entry_flags
    use_don_x, use_st_x = exit_flags
    cash = 1.0
    qty = 0.0
    entry_eq = 0.0
    entry_i = None
    curve = []
    trades = []
    for i in range(start_i, end_i):
        close = candles[i]["close"]
        if qty == 0.0 and entry_sig(candles, i, ind, use_sma_e, use_st_e, use_don_e):
            fill = close * (1 + fee)
            qty = cash / fill
            entry_eq = cash
            entry_i = i
            cash = 0.0
        elif qty > 0.0 and exit_sig(candles, i, ind, use_st_x, use_don_x):
            cash = qty * close * (1 - fee)
            trades.append({"entry_i": entry_i, "exit_i": i, "ret": cash / entry_eq - 1})
            qty = 0.0
            entry_i = None
        curve.append(cash + qty * close)
    if qty > 0.0:
        close = candles[end_i - 1]["close"]
        cash = qty * close * (1 - fee)
        trades.append({"entry_i": entry_i, "exit_i": end_i - 1, "ret": cash / entry_eq - 1})
        curve[-1] = cash
    return curve, trades


# ── Variant B: combined signal + ATR overlay (deployment-realistic) ──────────

def backtest_overlay(candles, ind, fee, slip, start_i, end_i, start_equity=START_EUR,
                     sizing="risk"):
    """Mirrors the live bot: ATR hard stop, chandelier trail (using ATR at
    entry, as the bot does), 48h cooldown after stop exit, 25%/50% exposure
    caps, EUR 10 min trade, -5% daily circuit breaker.
    sizing="risk":     qty = 1% equity / stop distance (stop width sets size)
    sizing="exposure": notional = 25% equity (stop is catastrophe insurance only)"""
    cost = fee + slip
    cash = start_equity
    pos = None          # dict qty, entry, stop, highest_close, atr_at_entry, notional
    cooldown_until = -1
    halted_day = None
    day_start_equity = start_equity
    curve = []
    trades = []
    fees_paid = 0.0
    diag = {"exits_stop": 0, "exits_signal": 0, "blocked_cooldown": 0,
            "blocked_halt": 0, "blocked_mintrade": 0, "entries": 0}

    def equity_at(price):
        return cash + (pos["qty"] * price if pos else 0.0)

    for i in range(start_i, end_i):
        bar = candles[i]
        day = bar["time"] // 86400
        if day != halted_day:
            pass
        if day_start_equity is None or bar["time"] // 86400 != candles[i - 1]["time"] // 86400:
            day_start_equity = equity_at(bar["open"])

        # 1) intraday hard stop (daily bar: gap-then-stop approximation)
        if pos is not None:
            stop = pos["hard_stop"]
            if bar["open"] <= stop:
                fill = bar["open"] * (1 - cost)
            elif bar["low"] <= stop:
                fill = stop * (1 - cost)
            else:
                fill = None
            if fill is not None:
                proceeds = pos["qty"] * fill
                pnl = proceeds - pos["notional"]
                cash += proceeds
                fees_paid += pos["qty"] * min(fill, stop) * cost
                trades.append({"entry_i": pos["entry_i"], "exit_i": i, "ret": pnl / pos["notional"],
                               "reason": "stop", "hold_days": i - pos["entry_i"]})
                diag["exits_stop"] += 1
                cooldown_until = day + 2  # 48h
                pos = None

        # 2) trail update on the closed bar (bot parity: uses atr_at_entry)
        if pos is not None:
            pos["highest_close"] = max(pos["highest_close"], bar["close"])
            trail = pos["highest_close"] - bot.ATR_STOP_MULT * pos["atr_at_entry"]
            if trail > pos["hard_stop"]:
                pos["hard_stop"] = trail

        # 3) signal exit at close
        if pos is not None and exit_sig(candles, i, ind):
            fill = bar["close"] * (1 - cost)
            proceeds = pos["qty"] * fill
            pnl = proceeds - pos["notional"]
            cash += proceeds
            fees_paid += proceeds * cost
            trades.append({"entry_i": pos["entry_i"], "exit_i": i, "ret": pnl / pos["notional"],
                           "reason": "signal", "hold_days": i - pos["entry_i"]})
            diag["exits_signal"] += 1
            pos = None

        # 4) circuit breaker check (halts entries for this UTC day)
        eq_close = equity_at(bar["close"])
        halted = eq_close < day_start_equity * (1 - bot.DAILY_CIRCUIT_BREAKER_PCT)
        if halted:
            halted_day = day

        # 5) entry at close
        if pos is None and entry_sig(candles, i, ind) and ind["atr"][i]:
            if halted or halted_day == day:
                diag["blocked_halt"] += 1
            elif day < cooldown_until:
                diag["blocked_cooldown"] += 1
            else:
                atr = ind["atr"][i]
                stop_dist = bot.ATR_STOP_MULT * atr
                if sizing == "exposure":
                    notional = eq_close * bot.MAX_POSITION_PCT
                else:
                    risk_eur = eq_close * bot.RISK_PER_TRADE_PCT
                    qty = risk_eur / stop_dist
                    notional = qty * bar["close"]
                notional = min(notional, eq_close * bot.MAX_POSITION_PCT,
                               eq_close * bot.MAX_EXPOSURE_PCT, cash)
                qty = notional / bar["close"]
                if notional < bot.MIN_TRADE_EUR:
                    diag["blocked_mintrade"] += 1
                else:
                    fill = bar["close"] * (1 + cost)
                    fees_paid += notional * cost
                    cash -= notional
                    pos = {"qty": qty, "entry": fill, "entry_i": i, "notional": notional,
                           "hard_stop": fill - stop_dist, "highest_close": bar["close"],
                           "atr_at_entry": atr}
                    diag["entries"] += 1

        curve.append(equity_at(bar["close"]))

    if pos is not None:
        fill = candles[end_i - 1]["close"] * (1 - cost)
        proceeds = pos["qty"] * fill
        cash += proceeds
        trades.append({"entry_i": pos["entry_i"], "exit_i": end_i - 1,
                       "ret": (proceeds - pos["notional"]) / pos["notional"], "reason": "end",
                       "hold_days": end_i - 1 - pos["entry_i"]})
        curve[-1] = cash

    return curve, trades, fees_paid, diag


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(curve, trades, bars_per_year=365):
    if len(curve) < 2:
        return {}
    eq = np.array(curve, dtype=float)
    total_ret = eq[-1] / eq[0] - 1
    years = len(eq) / bars_per_year
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else 0
    peak = np.maximum.accumulate(eq)
    max_dd = ((eq - peak) / peak).min()
    rets = np.diff(eq) / eq[:-1]
    sharpe = (rets.mean() / rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0
    wins = sum(1 for t in trades if t["ret"] > 0)
    win_rate = wins / len(trades) if trades else 0
    return {"return_pct": total_ret * 100, "cagr_pct": cagr * 100, "max_dd_pct": max_dd * 100,
            "sharpe": sharpe, "trades": len(trades), "win_rate_pct": win_rate * 100}


def fmt(m):
    if not m:
        return "n/a"
    return (f"ret {m['return_pct']:+9.1f}% | CAGR {m['cagr_pct']:+6.1f}% | "
            f"DD {m['max_dd_pct']:6.1f}% | Sharpe {m['sharpe']:5.2f} | "
            f"trades {m['trades']:3d} | win {m['win_rate_pct']:4.0f}%")


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # cp1252 consoles can't encode → etc.
    print("=" * 100)
    print("APEX-TRADER PRE-DEPLOYMENT BACKTEST")
    print(f"Strategy: Donchian {DON_E}/{DON_X} + Supertrend {ST_P}/{ST_M} + SMA{SMA_P} regime filter")
    print(f"Fees: 0.1%/side (variants A, C) | 0.1% + 0.05% slippage/side (variant B, deployment)")
    print("=" * 100)

    results = {}
    for symbol in SYMBOLS:
        df = fetch_history(symbol)
        print(f"\n{symbol}: {len(df)} daily bars, {df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}")
        parity_check(df, symbol)

        candles = to_candles(df)
        ind = build_indicators(candles, DON_E, DON_X, ST_P, ST_M, SMA_P)
        start_i = bot.MIN_CANDLES
        n = len(candles)
        oos_start = start_i + int((n - start_i) * 0.7)  # last 30% = OOS headline

        sym = {}

        # A) combined, full compounding — full period and OOS
        curve, trades = backtest_compounding(candles, ind, FEE, start_i, n)
        sym["A_full"] = metrics(curve, trades)
        curve, trades = backtest_compounding(candles, ind, FEE, oos_start, n)
        sym["A_oos"] = metrics(curve, trades)

        # B) combined + overlay (deployment) — full period and OOS
        curve, trades_b, fees_b, diag_b = backtest_overlay(
            candles, ind, FEE, SLIP, start_i, n,
            sizing=getattr(bot, "SIZING_MODE", "exposure"))  # verify the shipped config
        sym["B_full"] = metrics(curve, trades_b)
        sym["B_full"]["fees_eur"] = fees_b
        sym["B_full"]["diag"] = diag_b
        sym["B_full"]["avg_hold_days"] = (sum(t["hold_days"] for t in trades_b)
                                          / len(trades_b) if trades_b else 0)
        sym["B_full"]["trades_detail"] = [
            {"entry": str(candles[t["entry_i"]]["time"]), "ret_pct": round(t["ret"] * 100, 2),
             "reason": t["reason"], "hold_days": t["hold_days"]} for t in trades_b]
        curve, trades, _, _ = backtest_overlay(
            candles, ind, FEE, SLIP, oos_start, n,
            sizing=getattr(bot, "SIZING_MODE", "exposure"))
        sym["B_oos"] = metrics(curve, trades)

        # C) baselines on same data: donchian-only, supertrend-only (OOS)
        curve, trades = backtest_compounding(candles, ind, FEE, oos_start, n,
                                             entry_flags=(True, False, False),
                                             exit_flags=(True, False))
        sym["C_donchian_oos"] = metrics(curve, trades)
        curve, trades = backtest_compounding(candles, ind, FEE, oos_start, n,
                                             entry_flags=(False, True, False),
                                             exit_flags=(False, True))
        sym["C_supertrend_oos"] = metrics(curve, trades)
        # buy & hold OOS reference
        bh = [candles[i]["close"] / candles[oos_start]["close"] for i in range(oos_start, n)]
        sym["C_buyhold_oos"] = metrics(bh, [])

        results[symbol] = sym

        print(f"\n  ── {symbol} headline (OOS = last 30%, fixed params) ──")
        print(f"  A combined, full-compounding : {fmt(sym['A_oos'])}")
        print(f"  B combined + overlay (deploy): {fmt(sym['B_oos'])}")
        print(f"  C donchian-only 30/15        : {fmt(sym['C_donchian_oos'])}")
        print(f"  C supertrend-only 10/3.0     : {fmt(sym['C_supertrend_oos'])}")
        print(f"  C buy & hold                 : {fmt(sym['C_buyhold_oos'])}")
        print(f"  ── {symbol} full period ──")
        print(f"  A combined, full-compounding : {fmt(sym['A_full'])}")
        print(f"  B combined + overlay (deploy): {fmt(sym['B_full'])}  (fees €{sym['B_full'].get('fees_eur', 0):.0f})")
        d = sym["B_full"]["diag"]
        print(f"    B diag: entries {d['entries']} | exits: stop {d['exits_stop']}, signal {d['exits_signal']} "
              f"| avg hold {sym['B_full']['avg_hold_days']:.0f}d | blocked: cooldown {d['blocked_cooldown']}, "
              f"halt {d['blocked_halt']}, mintrade {d['blocked_mintrade']}")

    # ── Parameter sensitivity (neighbors), OOS, variant A on both symbols ──
    print("\n" + "=" * 100)
    print("PARAMETER SENSITIVITY (OOS, combined signal, full-compounding, 0.1%/side)")
    grid = [(25, 12, 10, 3.0), (30, 15, 10, 2.5), (30, 15, 10, 3.0),
            (30, 15, 10, 3.5), (35, 18, 10, 3.0), (40, 20, 10, 3.0), (30, 15, 14, 3.0)]
    sens = {}
    for (de, dx, sp, sm) in grid:
        row = []
        for symbol in SYMBOLS:
            df = fetch_history(symbol)
            candles = to_candles(df)
            ind = build_indicators(candles, de, dx, sp, sm, SMA_P)
            n = len(candles)
            oos_start = bot.MIN_CANDLES + int((n - bot.MIN_CANDLES) * 0.7)
            curve, trades = backtest_compounding(candles, ind, FEE, oos_start, n)
            row.append(metrics(curve, trades))
        sens[(de, dx, sp, sm)] = row
        print(f"  DC {de}/{dx} ST {sp}/{sm}: BTC {fmt(row[0])}")
        print(f"{'':>22} ETH {fmt(row[1])}")
    results["sensitivity"] = {str(k): v for k, v in sens.items()}

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
