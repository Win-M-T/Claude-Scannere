"""Swing-short setups."""

import numpy as np
import pandas as pd
import streamlit as st

from scanner.constants import SH_SETUPS
from scanner.patterns.common import rs_matrix
from scanner.patterns.htf import _win_view

def _since(flag):
    """Days since `flag` was last True (0 = today), NaN before the first time. flag: (days x stocks) bool array."""
    n, k = flag.shape
    rows = np.where(flag, np.arange(n)[:, None], np.nan)
    last = pd.DataFrame(rows).ffill().to_numpy()
    return np.arange(n)[:, None] - last


def _ffill_at(flag, A):
    """Value of A on the day `flag` was last True, carried forward."""
    return pd.DataFrame(np.where(flag, A, np.nan)).ffill().to_numpy()


def short_frames(P, pp):
    """Both swing-short setups for every day at once (days x stocks), each day using only data up to that day.

    🐻 Bear flag — a stock in a downtrend (50-day SMA under the 200-day and falling, close under the 50) bounces
    ≥ x% off its 15-day low on light volume and tags a falling 20/50-day SMA (high within tol % of it) but closes
    below it. That touch day arms the setup for `bf_days` days: SELL-STOP at the lowest low since the touch, STOP at
    the high of the bounce. TRIGGER = a close below the armed level. Optional: RS rating ≤ x (a laggard).

    🪤 Failed breakout — a close above the prior N-day high that had stood for ≥ 5 days (a real breakout from a base),
    then within `fb_days` days a close back BELOW that old high (the pivot), optionally on ≥ x× volume and under the
    200-day SMA. Armed while still above the pivot: SELL-STOP at the pivot, STOP at the highest high since the
    breakout. Only the first failure of each breakout counts."""
    H, L, C, V = (P[k].to_numpy(float) for k in ["High", "Low", "Close", "Volume"])
    idx, cols = P["Close"].index, P["Close"].columns
    n, k = C.shape
    D = lambda A: pd.DataFrame(A, index=idx, columns=cols)
    roll = lambda A, w, f: getattr(pd.DataFrame(A).rolling(w, min_periods=w), f)().to_numpy()
    shift1 = lambda A, fill=np.nan: np.vstack([np.full((1, k), fill), A[:-1]])
    with np.errstate(all="ignore"):
        s20, s50 = roll(C, 20, "mean"), roll(C, 50, "mean")
        s200 = pd.DataFrame(C).rolling(200, min_periods=150).mean().to_numpy()
        avg50 = shift1(roll(V, 50, "mean"))                          # average volume BEFORE today
        dvol = roll(C * V, 50, "mean") / 1e6
        liq = (C >= float(pp["sh_minpx"])) & (dvol >= float(pp["sh_mindvol"])) & (np.arange(n)[:, None] >= 60)

        # ---------------- 🐻 bear flag ----------------
        tol = float(pp["sh_bf_tol"]) / 100
        t20 = (H >= s20 * (1 - tol)) & (C < s20)
        t50 = (H >= s50 * (1 - tol)) & (C < s50)
        ma = pp["sh_bf_ma"]
        touch = t20 if ma == "20-day SMA" else t50 if ma == "50-day SMA" else (t20 | t50)
        down = (s50 < s200) & (s50 < np.vstack([np.full((10, k), np.nan), s50[:-10]])) & (C < s50)
        lo15 = roll(L, 15, "min")
        bounce = (H / lo15 - 1) * 100
        dry = roll(V, 5, "mean") / avg50
        bf_touch = touch & down & (bounce >= float(pp["sh_bf_bounce"])) & liq
        if float(pp["sh_bf_dry"]) > 0:
            bf_touch &= dry <= float(pp["sh_bf_dry"])
        rs = None
        if float(pp.get("sh_bf_rs", 0)) > 0:
            rs = rs_matrix(P["Close"]).to_numpy()
            bf_touch &= rs <= float(pp["sh_bf_rs"])
        K = int(pp["sh_bf_days"])
        ds = _since(bf_touch)                                       # days since the last touch
        bf_lvl = np.full((n, k), np.nan)                            # sell-stop for TOMORROW (lowest low since touch)
        bf_stp = np.full((n, k), np.nan)                            # stop (high of the bounce: touch day - 2 .. today)
        for j in range(K):
            m = ds == j
            bf_lvl = np.where(m, roll(L, j + 1, "min"), bf_lvl)
            bf_stp = np.where(m, roll(H, j + 3, "max"), bf_stp)
        armed0 = (ds <= K - 1) & (C < bf_stp) & (C < s50)
        lvl_y, stp_y, arm_y = shift1(bf_lvl), shift1(bf_stp), shift1(armed0, False).astype(bool)
        t0 = arm_y & (C < lvl_y)                                    # closed below yesterday's sell level
        t0p = shift1(t0.astype(float), 0.0)                         # … on the day before
        ds_y = shift1(ds)
        # only the first trigger after each touch: none on the days between that touch and today
        prior = np.zeros((n, k), bool)
        spent = np.zeros((n, k), bool)
        for j in range(1, K + 1):
            prior |= (ds_y == j) & (roll(t0p, j, "max") > 0)
            spent |= (ds == j) & (roll(t0.astype(float), j, "max") > 0)   # triggered since today's touch
        bf_trig = t0 & ~prior
        bf_setup = armed0 & ~spent

        # ---------------- 🪤 failed breakout ----------------
        N, FD = int(pp["sh_fb_len"]), int(pp["sh_fb_days"])
        Hn = np.where(np.isnan(H), -np.inf, H)
        age = np.full((n, k), -1)
        for c0 in range(0, k, 250):
            sl = slice(c0, c0 + 250)
            wv = _win_view(shift1(Hn, -np.inf)[:, sl], N)            # the N days BEFORE today, newest first
            age[:, sl] = np.argmax(np.where(np.isnan(wv), -np.inf, wv), axis=2) + 1
        piv = roll(shift1(H), N, "max")                             # prior N-day high (excl. today)
        qbo = (C > piv) & (age >= int(pp["sh_fb_base"])) & (shift1(C) <= shift1(piv)) & liq
        dsb = _since(qbo)
        pivot = _ffill_at(qbo, piv)
        peak = np.full((n, k), np.nan)
        for j in range(FD + 1):
            peak = np.where(dsb == j, roll(H, j + 1, "max"), peak)
        f0 = (dsb >= 1) & (dsb <= FD) & (C < pivot)
        fprior = np.zeros((n, k), bool)
        f0f = f0.astype(float)
        for j in range(2, FD + 1):
            fprior |= (dsb == j) & (roll(shift1(f0f, 0.0), j - 1, "max") > 0)
        volx = V / avg50
        fb_trig = f0 & ~fprior & liq
        if float(pp["sh_fb_vol"]) > 0:
            fb_trig &= volx >= float(pp["sh_fb_vol"])
        if pp["sh_fb_200"]:
            fb_trig &= C < s200
        fb_trig_stop = peak                                          # highest high from the breakout to today
        fb_setup = (dsb >= 0) & (dsb <= FD - 1) & (C >= pivot) & ~(f0 | fprior) & liq
        if pp["sh_fb_200"]:
            fb_setup &= C < s200
    return dict(bf_trig=D(bf_trig), bf_setup=D(bf_setup), bf_level=D(bf_lvl), bf_stop=D(bf_stp),
                bf_trig_level=D(lvl_y), bf_trig_stop=D(stp_y), bf_bounce=D(bounce), bf_dry=D(dry),
                bf_days=D(ds), fb_trig=D(fb_trig), fb_setup=D(fb_setup), fb_pivot=D(pivot), fb_stop=D(peak),
                fb_trig_stop=D(fb_trig_stop), fb_days=D(dsb), volx=D(volx),
                s20=D(s20), s50=D(s50), s200=D(s200), rs=D(rs) if rs is not None else None)


def short_scan(P, pp):
    """The Swing-short scan on the latest day (one row per stock)."""
    C = P["Close"]
    out = pd.DataFrame(index=C.columns)
    out["Short"] = ""
    if len(C) < 80:
        return out
    F_ = short_frames(P, pp)
    last = lambda k: F_[k].iloc[-1].to_numpy()
    which = pp["sh_which"]
    use_bf, use_fb = which != SH_SETUPS[2], which != SH_SETUPS[1]
    bt, bs = last("bf_trig").astype(bool) & use_bf, last("bf_setup").astype(bool) & use_bf
    ft, fs = last("fb_trig").astype(bool) & use_fb, last("fb_setup").astype(bool) & use_fb
    # one row per stock: a trigger beats a setup; bear flag first when both
    is_bf = bt | (bs & ~ft)
    is_fb = ~is_bf & (ft | fs)
    trig = (is_bf & bt) | (is_fb & ft)
    out["Short"] = np.where(trig, "TRIGGER", np.where(is_bf | is_fb, "SETUP", ""))
    out["Short setup"] = np.where(is_bf, SH_SETUPS[1], np.where(is_fb, SH_SETUPS[2], ""))
    sell = np.where(is_bf, np.where(bt, last("bf_trig_level"), last("bf_level")), last("fb_pivot"))
    stop = np.where(is_bf, np.where(bt, last("bf_trig_stop"), last("bf_stop")),
                    np.where(ft, last("fb_trig_stop"), last("fb_stop")))
    px = C.iloc[-1].to_numpy()
    out["Sell below"] = np.where(is_bf | is_fb, sell, np.nan)
    out["Short stop"] = np.where(is_bf | is_fb, stop, np.nan)
    ref = np.where(trig, px, sell)
    out["Short risk %"] = (out["Short stop"] - ref) / ref * 100
    out["To sell level %"] = np.where(trig, np.nan, (px - sell) / px * 100)
    out["Bounce %"] = np.where(is_bf, last("bf_bounce"), np.nan)
    out["Bounce vol ×"] = np.where(is_bf, last("bf_dry"), np.nan)
    out["Breakout days ago"] = np.where(is_fb, last("fb_days"), np.nan)
    out["Short vol ×"] = last("volx")
    return out
