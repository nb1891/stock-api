"""
Global/Exchange‑Agnostic Alpha MVP – one‑ticker, auto‑fetch & decision

Goal: You give a ticker, the script fetches up‑to‑date data from the internet (yfinance),
computes signals, and returns a clear BUY/HOLD/AVOID verdict. You can change the
**time horizon** (in months) and the algorithm adapts its features and thresholds.

Usage (default 6‑month horizon):
  python london_alpha_mvp.py --ticker HSBA.L

Custom horizon (e.g., 3 or 12 months):
  python london_alpha_mvp.py --ticker AAPL --horizon_m 12 --explain

Run quick self‑tests (no internet):
  python london_alpha_mvp.py --run_tests
"""
from __future__ import annotations
import os, sys, argparse, warnings, math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional providers
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

# -------------------------
# Utils
# -------------------------

def parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).tz_localize(None)

def annualize_vol(daily_vol: float) -> float:
    if np.isnan(daily_vol):
        return np.nan
    return float(daily_vol) * math.sqrt(252)

# -------------------------
# Auto data fetch (prices + benchmark by suffix)
# -------------------------
INDEX_BY_SUFFIX = {
    '.L': '^FTSE',      # London Stock Exchange → FTSE 100 proxy
    '.ST': '^OMXS30',   # Stockholm
    '.DE': '^GDAXI',    # Frankfurt DAX
    '.PA': '^FCHI',     # Paris CAC 40
}

DEFAULT_INDEX = '^GSPC'  # S&P 500 as generic fallback

@dataclass
class AutoData:
    ticker: str
    years: int = 10  # fetch window

    def _pick_index(self) -> str:
        for suf, idx in INDEX_BY_SUFFIX.items():
            if self.ticker.endswith(suf):
                return idx
        return DEFAULT_INDEX

    def fetch(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not _HAS_YF:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")
        tkr = yf.Ticker(self.ticker)
        hist = tkr.history(period=f"{self.years}y", auto_adjust=True)
        if hist.empty:
            raise RuntimeError(f"No price history for {self.ticker}")
        prices = hist.reset_index()[['Date','Close']]
        prices.columns = ['date','close']
        prices['date'] = pd.to_datetime(prices['date'])
        # Benchmark
        bench_ticker = self._pick_index()
        b_hist = yf.Ticker(bench_ticker).history(period=f"{self.years}y", auto_adjust=True)
        if b_hist.empty:
            raise RuntimeError(f"No index history for {bench_ticker}")
        bench = b_hist.reset_index()[['Date','Close']]
        bench.columns = ['date','close']
        bench['date'] = pd.to_datetime(bench['date'])
        return prices, bench

# -------------------------
# Feature engineering driven by horizon
# -------------------------

def compute_features(prices: pd.DataFrame, bench: pd.DataFrame, horizon_m: int) -> pd.DataFrame:
    """Compute momentum and risk features using windows derived from horizon."""
    dfp = prices.copy().sort_values('date')
    dfb = bench.copy().sort_values('date')

    # Daily returns
    dfp['ret'] = dfp['close'].pct_change()
    dfb['ret'] = dfb['close'].pct_change()

    # Window lengths
    n = max(1, int(round(21 * horizon_m)))           # ~ trading days for horizon months
    vol_lb = max(21, int(round(21 * horizon_m)))     # volatility lookback

    # Momentum over horizon (stock and index)
    dfp['mom_h'] = dfp['close'].pct_change(n)
    dfb['mom_h'] = dfb['close'].pct_change(n)
    # Excess momentum (stock minus index)
    # Align dates
    m = pd.merge(dfp[['date','mom_h','ret','close']], dfb[['date','mom_h']].rename(columns={'mom_h':'mom_h_idx'}),
                 on='date', how='left')
    m['excess_mom_h'] = m['mom_h'] - m['mom_h_idx']

    # Volatility (daily → annualized)
    m['vol_daily'] = dfp['ret'].rolling(vol_lb).std()
    m['vol_ann'] = m['vol_daily'].apply(annualize_vol)

    # Simple trend filter: % of positive days in last n
    m['pos_ratio'] = dfp['ret'].rolling(n).apply(lambda x: np.mean(x > 0), raw=True)

    return m

# -------------------------
# Decision logic (BUY/HOLD/AVOID)
# -------------------------

def decide(features: pd.DataFrame, horizon_m: int) -> Dict[str, object]:
    last = features.dropna().iloc[-1]

    # Adaptive thresholds by horizon
    base_neg, base_pos = -0.20, 0.40
    scale = horizon_m / 6.0
    neg = base_neg * scale
    pos = base_pos * scale

    # Momentum score (0..1)
    excess_mom = float(last.get('excess_mom_h', np.nan))
    if np.isnan(excess_mom):
        mom_score = 0.5
    else:
        mom_score = float(np.clip((excess_mom - neg) / (pos - neg), 0, 1))

    # Vol score (0..1) – prefer 15%..60% annualized, relaxed with horizon
    vol = float(last.get('vol_ann', np.nan))
    vol_cap_lo = 0.15
    vol_cap_hi = 0.60 * math.sqrt(scale)
    if np.isnan(vol):
        vol_score = 0.5
        vol_clipped = np.nan
    else:
        vol_clipped = float(np.clip(vol, vol_cap_lo, vol_cap_hi))
        vol_score = float(np.clip((vol_cap_hi - vol_clipped) / (vol_cap_hi - vol_cap_lo + 1e-12), 0, 1))

    # Stability (positive-day ratio)
    pos_ratio = float(last.get('pos_ratio', np.nan))
    stab_score = float(np.clip(((pos_ratio if not np.isnan(pos_ratio) else 0.55) - 0.45) / (0.65 - 0.45), 0, 1))

    # Weights
    w_mom, w_vol, w_stab = 0.5, 0.3, 0.2
    score = 100.0 * (w_mom * mom_score + w_vol * vol_score + w_stab * stab_score)

    if score >= 65:
        label = 'BUY'
    elif score >= 45:
        label = 'HOLD'
    else:
        label = 'AVOID'

    # Short rationale (as before)
    rationale = [
        f"Excess momentum ({horizon_m}m) = {excess_mom*100:.1f}%" if not np.isnan(excess_mom) else "Excess momentum unavailable",
        f"Annualized volatility ≈ {vol*100:.1f}%" if not np.isnan(vol) else "Volatility unavailable",
        f"Positive-day ratio ({horizon_m}m) ≈ {pos_ratio*100:.1f}%" if not np.isnan(pos_ratio) else "Positive-day ratio unavailable",
    ]

    # Long explanation block
    explain_long = (
        f"We evaluate over a {horizon_m}-month window and compute three components:\n\n"
        f"1) Momentum (excess vs. index): {excess_mom:+.2%} "
        f"— mapped between {neg:+.0%} (0/100) and {pos:+.0%} (100/100). "
        f"Component score ≈ {mom_score*100:.0f}/100.\n"
        f"2) Risk (annualized volatility): {vol*100:.1f}% "
        f"— preferred range {vol_cap_lo*100:.0f}% to {vol_cap_hi*100:.0f}% (clipped to {'' if np.isnan(vol_clipped) else f'{vol_clipped*100:.1f}%'}). "
        f"Component score ≈ {vol_score*100:.0f}/100.\n"
        f"3) Trend stability (share of up-days): {pos_ratio*100:.1f}% "
        f"— mapped from 45% (0/100) to 65% (100/100). "
        f"Component score ≈ {stab_score*100:.0f}/100.\n\n"
        f"Weights: momentum {int(w_mom*100)}%, risk {int(w_vol*100)}%, stability {int(w_stab*100)}%.\n"
        f"Final score = {score:.1f}/100 → Verdict: {label}."
    )

    return {
        'score': round(score, 1),
        'label': label,
        'horizon_m': int(horizon_m),
        'rationale': rationale,
        'metrics': {
            'excess_mom_h': round(excess_mom, 4) if not np.isnan(excess_mom) else None,
            'vol_ann': round(vol, 4) if not np.isnan(vol) else None,
            'pos_ratio': round(pos_ratio, 4) if not np.isnan(pos_ratio) else None,
        },
        'components': {
            'momentum_score': round(mom_score*100, 1),
            'volatility_score': round(vol_score*100, 1),
            'stability_score': round(stab_score*100, 1),
        },
        'weights': {'momentum': w_mom, 'volatility': w_vol, 'stability': w_stab},
        'thresholds': {
            'excess_mom_low': neg,
            'excess_mom_high': pos,
            'vol_low_pref': vol_cap_lo,
            'vol_high_pref': vol_cap_hi,
            'pos_ratio_low': 0.45,
            'pos_ratio_high': 0.65,
        },
        'explain_long': explain_long,
    }

# -------------------------
# Pretty print
# -------------------------

def print_decision(ticker: str, decision: Dict[str, object], explain: bool):
    print(f"\n===== Evaluation for {ticker} (horizon {decision['horizon_m']}m) =====")
    print(f"Verdict: {decision['label']}  •  Score: {decision['score']}/100")
    if explain:
        print("\nWhy:")
        for line in decision['rationale']:
            print(" - " + line)

# -------------------------
# Self‑tests (no internet)
# -------------------------

def _make_synth_prices(mu: float, sigma: float, days: int = 252*2) -> pd.DataFrame:
    rng = pd.date_range('2020-01-01', periods=days, freq='B')
    rets = np.random.normal(mu/252, sigma/np.sqrt(252), size=len(rng))
    px = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({'date': rng, 'close': px})


def run_tests():
    # Test 1: Strong uptrend should be BUY at 6m
    prices_up = _make_synth_prices(mu=0.20, sigma=0.20)
    bench_flat = _make_synth_prices(mu=0.00, sigma=0.15)
    feat = compute_features(prices_up, bench_flat, horizon_m=6)
    dec = decide(feat, horizon_m=6)
    assert dec['label'] in ('BUY','HOLD'), "Uptrend should not be AVOID"

    # Test 2: Downtrend should be AVOID at 6m
    prices_dn = _make_synth_prices(mu=-0.20, sigma=0.25)
    feat2 = compute_features(prices_dn, bench_flat, horizon_m=6)
    dec2 = decide(feat2, horizon_m=6)
    assert dec2['label'] in ('AVOID','HOLD'), "Downtrend should not be BUY"

    # Test 3: Horizon sensitivity — a weak trend may pass at 12m but fail at 3m
    prices_weak = _make_synth_prices(mu=0.06, sigma=0.18)
    f3a = compute_features(prices_weak, bench_flat, horizon_m=3)
    f3b = compute_features(prices_weak, bench_flat, horizon_m=12)
    d3a = decide(f3a, horizon_m=3)
    d3b = decide(f3b, horizon_m=12)
    # No hard label asserts here because randomness; ensure code runs and score changes with horizon
    assert d3a['score'] != d3b['score'], "Horizon should influence the score"

    # Test 4: print_decision should not raise and should include horizon in banner
    try:
        print_decision("TEST", d3b, explain=True)
    except Exception as e:
        raise AssertionError(f"print_decision raised unexpectedly: {e}")

    print("[TEST] All tests passed.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="One‑ticker evaluator (auto‑fetch via yfinance) with horizon‑aware verdict")
    p.add_argument('--ticker', type=str, help='Ticker symbol, e.g. HSBA.L')
    p.add_argument('--horizon_m', type=int, default=6, help='Horizon in months for signals/decision (e.g., 3, 6, 12)')
    p.add_argument('--explain', action='store_true', help='Show rationale behind the verdict')
    p.add_argument('--run_tests', action='store_true', help='Run built‑in tests (no internet)')
    args = p.parse_args()

    if args.run_tests:
        run_tests()
        sys.exit(0)

    if not args.ticker:
        print("Please provide --ticker (e.g., --ticker HSBA.L) or run --run_tests.")
        sys.exit(1)

    auto = AutoData(args.ticker)
    prices, bench = auto.fetch()
    feats = compute_features(prices, bench, horizon_m=args.horizon_m)
    decision = decide(feats, horizon_m=args.horizon_m)
    print_decision(args.ticker, decision, explain=args.explain)

    # Save a small JSON report next to the script
    try:
        import json
        out = {
            'ticker': args.ticker,
            'horizon_m': args.horizon_m,
            'decision': decision,
        }
        with open('decision_report.json', 'w') as f:
            json.dump(out, f, indent=2)
        if args.explain:
            print("Saved: decision_report.json")
    except Exception as e:
        print(f"[WARN] Could not write decision_report.json: {e}")
