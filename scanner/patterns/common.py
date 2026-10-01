"""Shared pattern-scan helpers: metrics, parameter pack, RS matrix."""

import numpy as np
import pandas as pd
import streamlit as st

from scanner.constants import MARKETS, SCAN_EXTRA_DEFAULTS

ss = st.session_state

# ============================================================================
# 3. Indicators for every stock (vectorized)
# ============================================================================
def _ago(X, n):
    return X.iloc[-1 - n] if len(X) > n else pd.Series(np.nan, index=X.columns)


def _bars_since_high(H, end_offset, window=20):
    """Days since the highest high in the `window` bars ending `end_offset` bars ago."""
    arr = H.to_numpy(dtype=float)
    stop = len(arr) - end_offset
    w = arr[stop - window:stop][::-1]
    w = np.where(np.isnan(w), -np.inf, w)
    return pd.Series(np.argmax(w, axis=0).astype(float), index=H.columns)


def compute_metrics(panels, pp, vh=None):
    O, H, L, C, V = (panels[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    last = lambda X: X.iloc[-1]
    prev = lambda X: X.iloc[-2]
    with np.errstate(all="ignore"):
        e10 = C.ewm(span=10, adjust=False).mean()
        e21 = C.ewm(span=21, adjust=False).mean()
        s20 = C.rolling(20).mean()
        s50 = C.rolling(50).mean()
        s150 = C.rolling(150).mean()
        s200 = C.rolling(200).mean()
        vol50 = V.rolling(50).mean()
        vol5 = V.rolling(5).mean()
        hi20 = H.rolling(20).max()
        hi252 = H.rolling(252, min_periods=200).max()
        lo252 = L.rolling(252, min_periods=200).min()

        d = C.diff()
        g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        lo = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + g / lo)
        pc = C.shift(1)
        tr = pd.DataFrame(np.fmax.reduce([(H - L).to_numpy(), (H - pc).abs().to_numpy(),
                                          (L - pc).abs().to_numpy()]), index=C.index, columns=C.columns)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

        c = last(C)
        m = pd.DataFrame(index=C.columns)
        m["Price"] = c
        m["Chg %"] = (c / prev(C) - 1) * 100
        m["Gap %"] = (last(O) / prev(C) - 1) * 100
        m["Volume"] = last(V)
        m["Avg vol 50d"] = last(vol50)
        prior50 = V.iloc[-51:-1].mean()
        m["Rel vol"] = last(V) / prior50
        m["$ Vol M"] = c * last(vol50) / 1e6
        m["Avg $ Vol 50d M"] = last((C * V).rolling(50, min_periods=20).mean()) / 1e6
        m["ADR % 14d"] = last(((H / L - 1) * 100).rolling(14, min_periods=10).mean())
        # sessions in a row the stock closed at/below its 50-day SMA just before today — only for stocks that closed
        # ABOVE it today ("first close back above the 50-day in N days")
        _s50 = C.rolling(50).mean()
        _ok = _s50.notna().to_numpy()
        _below = (_ok & ~(C > _s50).to_numpy())[:-1][::-1]
        _run = np.where(_below.all(axis=0), _below.shape[0], _below.argmin(axis=0)) if len(_below) else np.zeros(C.shape[1])
        m["Under 50d days"] = np.where((C > _s50).to_numpy()[-1] & _ok[-1], _run, np.nan)
        m["Chg 5d %"] = (c / _ago(C, 5) - 1) * 100
        m["Chg 20d %"] = (c / _ago(C, 20) - 1) * 100
        for lbl, n in [("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]:
            m[f"Perf {lbl} %"] = (c / _ago(C, n) - 1) * 100
        prev_year = C[C.index.year < C.index[-1].year]
        m["Perf YTD %"] = (c / prev_year.iloc[-1] - 1) * 100 if len(prev_year) else np.nan
        # IBD-style relative strength, ranked 1-99 against the scanned stocks
        r = lambda n: c / _ago(C, n) - 1
        rs_raw = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)
        rs_raw = rs_raw.fillna(0.4 * r(63) + 0.6 * r(126))
        m["RS"] = (rs_raw.rank(pct=True) * 98 + 1).round()
        m["RSI 14"] = last(rsi)
        m["ATR %"] = last(atr) / c * 100
        m["vs EMA21 %"] = (c / last(e21) - 1) * 100
        m["vs SMA20 %"] = (c / last(s20) - 1) * 100
        m["vs SMA50 %"] = (c / last(s50) - 1) * 100
        m["vs SMA200 %"] = (c / last(s200) - 1) * 100
        m["Below 52W high %"] = (1 - c / last(hi252)) * 100
        m["Off 52W high %"] = -m["Below 52W high %"]
        m["Above 52W low %"] = (c / last(lo252) - 1) * 100
        m["New 52W high"] = last(H) >= last(hi252)
        m["Stack 50>200"] = (c > last(s50)) & (last(s50) > last(s200))
        m["Trend template"] = ((c > last(s150)) & (c > last(s200)) & (last(s150) > last(s200)) &
                               (last(s200) > _ago(s200, 21)) & (last(s50) > last(s150)) &
                               (c > last(s50)) & (c >= 1.3 * last(lo252)) & (c >= 0.75 * last(hi252)))
        m["EMA stack"] = (c > last(e21)) & (last(e10) > last(e21)) & (last(e21) > last(s50))
        m["7d range %"] = (H.iloc[-7:].max() - L.iloc[-7:].min()) / c * 100
        m["10d range %"] = (H.iloc[-10:].max() - L.iloc[-10:].min()) / c * 100
        m["Pullback %"] = (last(hi20) - c) / last(hi20) * 100
        m["Days since high"] = _bars_since_high(H, 0)
        m["EMA gap %"] = (last(e10) - last(s20)).abs() / last(s20) * 100      # 10 EMA vs 20 SMA
        m["Vol ratio"] = last(vol5) / last(vol50)

        # ---- tight flag pattern: SETUP today / yesterday, TRIGGER today ----
        def setup_at(k):  # k = 0 today, 1 yesterday
            ck = C.iloc[-1 - k]
            s50k, s200k = s50.iloc[-1 - k], s200.iloc[-1 - k]
            e21k = s20.iloc[-1 - k]                       # the flag's line: the 20-day SMA
            pull = (hi20.iloc[-1 - k] - ck) / hi20.iloc[-1 - k] * 100
            bars = _bars_since_high(H, k)
            end = len(C) - k
            rng7 = (H.iloc[end - 7:end].max() - L.iloc[end - 7:end].min()) / ck * 100
            dist = (ck - e21k) / e21k * 100
            gap = (e10.iloc[-1 - k] - e21k).abs() / e21k * 100
            return ((ck > s50k) & (s50k > s200k) & (s50k > s50.iloc[-11 - k]) &
                    (bars >= pp["min_bars"]) & (bars <= 19) &
                    (pull >= pp["min_pull"]) & (pull <= pp["max_pull"]) &
                    (rng7 < pp["max_range"]) & (ck > e21k) & (dist < pp["max_ema_dist"]) &
                    (gap < pp["max_ema_gap"]) & (vol5.iloc[-1 - k] < vol50.iloc[-1 - k] * pp["dry_up"]))

        setup_today, setup_yday = setup_at(0), setup_at(1)
        trigger = setup_yday & (c > H.iloc[-8:-1].max()) & (last(V) > last(vol50) * pp["breakout_vol"])
        m["Pattern"] = np.where(trigger, "TRIGGER", np.where(setup_today, "SETUP", ""))
    m.loc[C.notna().sum() < 60, "Pattern"] = ""
    m.index.name = "Symbol"
    return m


# ============================================================================
# Backtest: replay a scan over past days and trade every signal it gave
# ============================================================================
def build_pp(universe=None):
    """The calculation settings of every scan, exactly as set in the Stock Scanner tab."""
    pp = dict(max_range=ss["max_range"], min_pull=ss["pull_range"][0], max_pull=ss["pull_range"][1],
              min_bars=int(ss["min_bars"]), max_ema_dist=ss["max_ema_dist"], max_ema_gap=ss["max_ema_gap"],
              dry_up=ss["dry_up"], breakout_vol=ss["breakout_vol"],
              vr_days=int(ss["vr_days"]), vr_min_ratio=float(ss["vr_min_ratio"]), vr_dir=ss["vr_dir"],
              # only settings that change the calculations (a status choice or LiqLead values don't need a recompute)
              **{k: ss[k] for k in SCAN_EXTRA_DEFAULTS if k.startswith(("pb_", "ep_", "xb_", "gl_", "so_", "vcp_", "mac_", "htf_", "sh_", "rc_", "f5_"))
                 and k not in ("pb_status", "ep_status", "xb_status", "gl_status", "gl_ath", "so_status",
                               "vcp_status", "mac_status", "htf_status", "sh_status", "rc_status", "f5_status")})
    pp["xb_bench"] = MARKETS[universe]["bench"] if universe in MARKETS else "SPY"
    return pp


def rs_matrix(C):
    """IBD-style RS rating (1-99) for every stock on every day, ranked against the other stocks that day."""
    r = lambda n: C / C.shift(n) - 1
    raw = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)
    raw = raw.fillna(0.4 * r(63) + 0.6 * r(126))
    return (raw.rank(axis=1, pct=True) * 98 + 1).round()
