#!/usr/bin/env python3
"""One-off: sweep ATR_STOP_MULT x sizing mode in the deployment overlay (B).
Finding from 2026-08-03: 2xATR + 1%-risk sizing churns (66/67 BTC exits are
stops, ~11d holds) and wider stops shrink positions (qty = 1% equity / stop
distance), so neither extreme captures the trend. Test exposure-based sizing
(notional = 25% equity; stop = catastrophe insurance) vs risk sizing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apex_trader as bot
import backtest_apex as bt

MULTS = [2.0, 3.0, 4.0, 6.0]
MODES = ["risk", "exposure"]

for mode in MODES:
    print("=" * 150)
    print(f"SIZING MODE: {mode}")
    print(f"{'mult':>5} | {'BTC full':>38} | {'BTC OOS':>38} | stop/sig exits, avg hold (full | OOS)")
    print("-" * 150)
    for m in MULTS:
        bot.ATR_STOP_MULT = m
        row = []
        diags = []
        for symbol in bt.SYMBOLS:
            df = bt.fetch_history(symbol)
            candles = bt.to_candles(df)
            ind = bt.build_indicators(candles, bt.DON_E, bt.DON_X, bt.ST_P, bt.ST_M, bt.SMA_P)
            n = len(candles)
            start_i = bot.MIN_CANDLES
            oos_start = bot.MIN_CANDLES + int((n - bot.MIN_CANDLES) * 0.7)
            curve, trades, fees, diag = bt.backtest_overlay(candles, ind, bt.FEE, bt.SLIP, start_i, n, sizing=mode)
            mf = bt.metrics(curve, trades)
            hold_f = sum(t["hold_days"] for t in trades) / len(trades) if trades else 0
            curve, trades, fees, diag_o = bt.backtest_overlay(candles, ind, bt.FEE, bt.SLIP, oos_start, n, sizing=mode)
            mo = bt.metrics(curve, trades)
            hold_o = sum(t["hold_days"] for t in trades) / len(trades) if trades else 0
            row.append((mf, mo))
            diags.append(f"{diag['exits_stop']}/{diag['exits_signal']} {hold_f:.0f}d | {diag_o['exits_stop']}/{diag_o['exits_signal']} {hold_o:.0f}d")
        print(f"{m:>5.1f} | {bt.fmt(row[0][0]):>38} | {bt.fmt(row[0][1]):>38} | BTC {diags[0]}")
        print(f"{'':>5} | {bt.fmt(row[1][0]):>38} | {bt.fmt(row[1][1]):>38} | ETH {diags[1]}")
    print()
