"""EMA crossback scan."""

import numpy as np
import pandas as pd
import streamlit as st

from scanner.constants import XB_20
from scanner.rs import rs_prices

def ema_crossback(O, H, L, C, V, rs, pp):
    """EMA crossback: a leading stock pulls back to its rising 10-day EMA / 20-day SMA, shows strength, then breaks out.
    SETUP = at the EMA and holding · TRIGGER = broke above its recent consolidation high today."""
    n = len(C)
    e10, e20 = C.ewm(span=10, adjust=False).mean(), C.rolling(20).mean()     # e20 = the 20-day SMA
    s50 = C.rolling(50).mean()
    tol = pp["xb_tol"] / 100
    T = int(pp["xb_days"])
    with np.errstate(all="ignore"):
        # 1) leading stock: sharp rally off the lows, above the 50-day, strong RS, beating the market
        win_hi = int(pp["xb_rally_days"])
        Ha, La = H.to_numpy(float), L.to_numpy(float)
        cols = np.arange(C.shape[1])
        hi_w = np.where(np.isnan(Ha[-win_hi:]), -np.inf, Ha[-win_hi:])
        hi_i = n - win_hi + np.argmax(hi_w, axis=0)
        lo_w = np.where(np.isnan(La[-(win_hi + 40):]), np.inf, La[-(win_hi + 40):])
        lo_i = n - (win_hi + 40) + np.argmin(lo_w, axis=0)
        rally = (Ha[hi_i, cols] / La[lo_i, cols] - 1) * 100
        rally = np.where(lo_i < hi_i, rally, 0.0)                      # the low must come before the high
        c = C.iloc[-1]
        above50 = c > s50.iloc[-1]
        bt = pp.get("xb_bench", "SPY")                              # SPY, or the local index for Asian markets
        bench = rs_prices((bt,), "5y")
        spy = bench[bt].dropna() if bt in bench else pd.Series(dtype=float)
        spy = spy[spy.index <= C.index[-1]]
        spy_1m = (spy.iloc[-1] / spy.iloc[-22] - 1) * 100 if len(spy) > 22 else 0.0
        vs_spy = (c / C.iloc[-22] - 1) * 100 - spy_1m if n > 22 else pd.Series(np.nan, index=C.columns)
        lead = (rally >= pp["xb_rally"]) & above50 & (rs >= pp["xb_rs"])
        if pp["xb_beat"]:
            lead &= vs_spy > 0
        # 2) pullback to the rising 10 EMA / 20 SMA — the low tags the line, the close holds it
        t10 = (L <= e10 * (1 + tol)) & (C >= e10 * (1 - tol))
        t20 = (L <= e20 * (1 + tol)) & (C >= e20 * (1 - tol))
        which = pp["xb_ema"]
        touch = t10 if which == "10-day EMA" else t20 if which == XB_20 else (t10 | t20)
        touched_now = touch.iloc[-T:].any()                           # within the last T days (incl. today)
        touched_before = touch.iloc[-(T + 1):-1].any()                # before today (for a breakout today)
        rising = (e10.iloc[-1] > e10.iloc[-6]) & (e20.iloc[-1] > e20.iloc[-6])
        hi20 = H.iloc[-20:].max()
        pull = (hi20 - c) / hi20 * 100
        pb_ok = pull <= pp["xb_pull_max"]
        # 3) signs of strength
        pc = C.shift(1)
        tr = np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy())
        atr = pd.DataFrame(tr, index=C.index, columns=C.columns).rolling(20).mean().iloc[-1]
        rng3 = (H - L).iloc[-3:].mean()
        tight_ratio = rng3 / atr
        up = (C > pc).iloc[-10:]
        vv = V.iloc[-10:]
        upv, dnv = vv.where(up).mean(), vv.where(~up).mean()
        vol_ratio = upv / dnv
        o, h, l, cl = O.iloc[-1], H.iloc[-1], L.iloc[-1], C.iloc[-1]
        body = (cl - o).abs()
        hammer = ((np.minimum(o, cl) - l) >= 2 * body) & ((h - np.maximum(o, cl)) <= body.clip(lower=(h - l) * 0.1)) \
            & ((cl - l) >= 0.66 * (h - l)) & ((h - l) > 0)
        po, pcl = O.iloc[-2], C.iloc[-2]
        engulf = (cl > o) & (pcl < po) & (cl >= po) & (o <= pcl)
        candle = np.where(engulf, "Engulfing", np.where(hammer, "Hammer", ""))
        strong = pd.Series(True, index=C.columns)
        if pp["xb_tight"]:
            strong &= tight_ratio <= pp["xb_tight_max"]
        if pp["xb_vol"]:
            strong &= vol_ratio >= 1.0
        if pp["xb_candle"]:
            strong &= pd.Series(candle != "", index=C.columns)
        # 4) stop = recent swing low · 5) entry = break above the recent consolidation high
        stop = L.iloc[-6:].min()
        range_hi = H.iloc[-6:-1].max()
        risk = (c - stop) / c * 100
        breakout = (c > range_hi) & (c > C.iloc[-2])
    base = lead & rising & pb_ok & (C.iloc[-1] > s50.iloc[-1])
    trigger = base & touched_before & breakout
    if pp["xb_vol"]:                      # buyers confirmed: more volume on up days, and a real volume push today
        trigger &= (vol_ratio >= 1.0) & (V.iloc[-1] >= V.iloc[-21:-1].mean())
    setup = base & touched_now & strong & ~breakout & (c >= np.minimum(e10.iloc[-1], e20.iloc[-1]) * (1 - tol))
    out = pd.DataFrame(index=C.columns)
    out["Crossback"] = np.where(trigger, "TRIGGER", np.where(setup, "SETUP", ""))
    has = out["Crossback"] != ""
    last10, last20 = t10.iloc[-T:].any(), t20.iloc[-T:].any()
    out["EMA tagged"] = np.where(has, np.where(last10 & last20, "10 & 20", np.where(last10, "10", np.where(last20, "20", ""))), "")
    out["vs EMA10 %"] = (c / e10.iloc[-1] - 1) * 100
    out["Rally %"] = rally
    out["Pullback from high %"] = pull
    out["vs SPY 1M pts"] = vs_spy
    out["Tight (3d ÷ ATR)"] = tight_ratio
    out["Up/Down vol"] = vol_ratio
    out["Candle"] = candle
    out["Stop"] = stop
    out["Risk %"] = risk
    out["Entry above"] = range_hi
    return out


def _xb_path(rally, n_up, use20, tol, T, tight, tight_max, vol, candle, seed=7):
    """Made-up daily bars: quiet base -> rally -> pullback that tags the EMA -> tight bars -> breakout."""
    rnd = np.random.default_rng(seed)
    B = dict(O=[], H=[], L=[], C=[], V=[], E10=[], E20=[])
    st_ = {"e10": None, "e20": None}

    def add(o, h, l, c, v):
        for k, x in zip("OHLCV", (o, h, l, c, v)):
            B[k].append(x)
        st_["e10"] = c if st_["e10"] is None else st_["e10"] + (c - st_["e10"]) * 2 / 11
        last20 = B["C"][-20:]
        st_["e20"] = sum(last20) / len(last20)                   # the 20-day SMA
        B["E10"].append(st_["e10"]); B["E20"].append(st_["e20"])

    px = 100.0
    for i in range(24):                                   # long enough for the EMAs to settle
        c = px * (1 + rnd.normal(0, 0.006)); add(px, max(px, c) * 1.008, min(px, c) * 0.992, c, 1.0); px = c
    lo_i = len(B["C"]) - 1
    lo_px = B["L"][lo_i] = min(B["L"][lo_i], px * 0.988)
    step = (lo_px * (1 + max(rally, 5) / 100 * 1.03) / px) ** (1 / n_up)
    for i in range(n_up):
        c = px * step * (1 + rnd.normal(0, 0.005))
        add(px, max(px, c) * 1.006, min(px, c) * 0.994, c, (1.5 if vol else 1.1) * (1 + rnd.normal(0, .12)))
        px = c
    atr = float(np.mean(np.array(B["H"][-10:]) - np.array(B["L"][-10:])))
    ema = lambda: B["E20"][-1] if use20 else B["E10"][-1]
    for _ in range(40):                                   # pull back until the low is at the EMA
        if px - atr * 0.5 <= ema() * (1 + tol / 100):
            break
        c = px - max(atr * 0.6, (px - ema()) * 0.3)
        add(px, px + atr * 0.15, c - atr * 0.25, c, 0.65 if vol else 1.0)
        px = c
    tag_low = ema() * (1 - tol / 100 * 0.3)
    c = max(ema() * 1.004, tag_low + atr * (0.8 if candle else 0.5))
    add(c - atr * (0.05 if candle else 0.2), c + atr * 0.1, tag_low, c, 0.75)
    tag_i = len(B["C"]) - 1
    rb = atr * (tight_max if tight else 1.3) * 0.8
    for _ in range(max(0, int(T) - 1)):
        c = max(B["C"][-1], ema() * 1.012) * (1 + rnd.normal(0, 0.0015))
        o = B["C"][-1]
        add(o, max(o, c) + rb * .3, min(o, c) - rb * .45, c, 0.65 if vol else 1.0)
    entry, stop = max(B["H"][-5:]), min(B["L"][-6:])
    c = entry + atr * 0.8
    add(B["C"][-1], c + atr * 0.12, B["C"][-1] - atr * 0.1, c, 2.0 if vol else 1.3)
    hi_i = int(np.argmax(B["H"][:tag_i + 1]))
    return B, lo_i, hi_i, tag_i, len(B["C"]) - 1, entry, stop, atr
