# apex-trader

Systematic BTC/ETH trend-following bot for Revolut X (EUR pairs), with an
LLM acting as a **risk-off veto only** — never as a buy trigger. Paper mode
by default; one variable flip to go live.

## Strategy

**Universe:** BTC-EUR, ETH-EUR (daily candles, Coinbase public API).

**Entry** (all required, evaluated on the last closed daily candle):
- Close > Donchian upper(30, shifted 1)
- Supertrend(10, 3.0) direction = up
- Close > SMA(200)

**Exit** (first trigger wins):
- Close < Donchian lower(15, shifted 1), or Supertrend flips down (signal exit)
- Chandelier hard stop: highest close since entry − 6×ATR(14) at entry (stop exit, 48h per-symbol cooldown)

**Risk overlay (locked deployment config):**
- Exposure sizing: notional = 25% of equity per position (stop is catastrophe insurance, not the sizer)
- Max 2 positions, max 50% total exposure, min trade €10
- −5% daily equity circuit breaker (halts new entries for the day)
- Fees modeled 0.10% + 0.05% slippage per side (Revolut X taker)

**LLM veto:** a 4-model fallback chain (Groq 70B → Groq 8B → 2× OpenRouter
free tier) reviews each *entry* with portfolio context and can block it.
Fail-open by design: if every model is unreachable, the systematic signal
stands. The LLM can never *create* a trade.

## Validated performance

Backtest: Coinbase daily candles → 2026-08-03, full compounding, fees+slippage
as above. OOS = last 30% of history after a 210-bar warmup, parameters frozen.
**Deployed config is variant B.**

| Variant | Symbol | OOS return | OOS max DD | OOS Sharpe | OOS trades | Full-period return | Full max DD |
|---|---|---|---|---|---|---|---|
| **B (deployed)** | BTC | +8.5% | −10.9% | 0.37 | 12 | +136.9% | −19.1% |
| **B (deployed)** | ETH | +17.0% | −6.9% | 0.84 | 5 | +105.0% | −18.4% |
| A (signal only, all-in) | BTC | +69.6% | −28.6% | 0.73 | 10 | +7,723% | −50.0% |
| A (signal only, all-in) | ETH | +70.4% | −24.2% | 0.88 | 5 | +1,733.9% | −50.8% |
| Buy & hold (context) | BTC | +121.1% | −51.7% | 0.77 | — | — | — |
| Buy & hold (context) | ETH | −25.2% | −66.7% | — | — | — | — |

Robustness: all 7 neighboring parameter combos (Donchian ±10, Supertrend
±1.0 mult) are OOS-positive on both symbols — the chosen point is mid-pack,
not a lone peak. Parity check between bot logic and backtest engine: 0
mismatches.

**Read these caveats before trusting any number above:**
- OOS trade counts (5–12) are far below the ~30 needed for statistical
  significance. Treat OOS figures as indicative, not proven.
- In the bull-heavy 2023–2026 OOS window, BTC buy-and-hold returned more
  than the strategy on raw return — the strategy's edge is drawdown
  (−10.9% vs −51.7%), i.e. risk-adjusted survival, not bull-market capture.
- Variant B deliberately trades return for risk: the 6×ATR stop and 25%
  exposure cap cut full-period drawdown roughly in half vs variant A
  (−19% vs −50%) at a large cost to upside.
- A 1.5%/week (~117%/yr) goal is not realistic for this system; variant A's
  OOS CAGR is ~18–23%/yr. This bot is built for durable compounding with
  controlled drawdowns, not weekly income targets.

## Repo layout

| File | Purpose |
|---|---|
| `apex_trader.py` | The bot. Single file, stdlib + requests + cryptography. |
| `backtest_apex.py` | Walk-forward backtest w/ parity check vs the bot. |
| `sweep_stop_mult.py` | Sizing × stop-multiplier sensitivity sweep. |
| `.github/workflows/trader.yml` | Cloud runner (GitHub Actions). |
| `run_local.ps1` | Local backup runner (git-synced). |
| `requirements.txt` / `requirements-dev.txt` | Runtime / backtest deps. |

## Quickstart (local)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements-dev.txt                     # dev incl. backtest
cp .env.example .env                                    # fill in what you have
python backtest_apex.py                                 # verify the engine
python apex_trader.py --once                            # single paper cycle
python apex_trader.py                                   # loop mode (15-min ticks)
```

Everything runs paper (`LIVE_TRADING=false`) with zero secrets configured:
email and LLM simply stay silent. `.env` is loaded but **never overwrites**
already-set environment variables.

## Cloud deployment (primary)

The runner is GitHub Actions; **state is persisted by committing
`apex_state.json` / `apex_decisions.jsonl` back to this repo** (that's why
they're git-tracked). Setup:

1. Private repo. Push this tree.
2. Secrets (`gh secret set <NAME>`): `GROQ_API_KEY`, `OPENROUTER_API_KEY`,
   `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, and for live mode
   `REVOLUT_X_API_KEY` + `REVOLUT_X_PRIVATE_KEY` (raw PEM, multi-line OK).
3. Variable: `gh variable set LIVE_TRADING --body false` (flip later).
4. Metronome: GitHub's `*/15` cron misses ticks, so point cron-job.org at
   `POST https://api.github.com/repos/<you>/apex-trader/dispatches` with
   body `{"event_type":"run-trader"}`, headers
   `Authorization: Bearer <PAT>`, `Accept: application/vnd.github+json`.
   Use a **dedicated fine-grained PAT** scoped to this repo only
   (Contents: read/write) — not your main `gh` OAuth token.

## Going live checklist

1. ≥2 weeks of clean paper cycles in CI (check the decision JSONL history).
2. Revolut X API key with trading permission; Ed25519 private key in secrets.
3. `gh variable set LIVE_TRADING --body true`.
4. Fund the account; watch the first real order's confirmation email closely.
5. To go back to paper: set the variable to `false`. The bot also refuses to
   start live without credentials, so a misconfiguration fails safe.

## Local backup runner

`run_local.ps1` mirrors CI semantics (pull → one cycle → commit/push state).
**Never run it concurrently with GitHub Actions** — both share the same
git-persisted state and overlapping runs can double-trade. Use it only when
CI is down, then hand control back.

## Safety design notes

- LLM is veto-only and fail-open; a dead LLM provider can silence
  notifications but never block or invent trades.
- Atomic state writes (tmp + `os.replace`); corrupt state is quarantined to
  `*.corrupt`, never silently overwritten.
- Crash notifications via email with 1h dedup; daily digest; per-trade emails.
- No hardcoded credentials anywhere; `.env` never overrides the environment.

*Not financial advice. Crypto can go to zero; past backtests say nothing
certain about the future.*
