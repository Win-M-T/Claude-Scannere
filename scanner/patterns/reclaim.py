"""50-day reclaim setups."""

import numpy as np
import pandas as pd
import streamlit as st

from scanner.constants import RC_STOPS, RC_TRENDS
from scanner.patterns.common import rs_matrix

def rc_frames(P, pp, rs_all=None):
    """The 50-day reclaim for every day at once (days x stocks), each day using only data up to that day.
    POP = the first close above the 50-day SMA after >= `rc_below` closes under it; it may stay above for up to
    `rc_pop_max` days. DIP = then closes back under the SMA, staying shallow (every low within `rc_dip_max` % under
    the SMA). HIGHER LOW = inside the dip, after its lowest low, price bounces and pulls back to a swing low that
    stays ABOVE that lowest low (a low lower than both neighbouring days' lows — confirmed the day after).
    TRIGGER = after >= `rc_min_dip` closes under, the first close back above the SMA within `rc_max_days` of the pop
    (with a higher low in place if required). Stop = under the higher low (or the dip's lowest low).
    SETUP = in a valid dip (buy-stop at the SMA). Optional: the dip's low above the low before the pop, the SMA's
    direction at the pop, reclaim-day volume, RS rating, price / $ volume."""
    H, L, C, V = (P[k].to_numpy(float) for k in ["High", "Low", "Close", "Volume"])
    idx, cols = P["Close"].index, P["Close"].columns
    n, k = C.shape
    D = lambda A: pd.DataFrame(A, index=idx, columns=cols)
    with np.errstate(all="ignore"):
        s50 = pd.DataFrame(C).rolling(50, min_periods=50).mean().to_numpy()
        slope = s50 - np.vstack([np.full((20, k), np.nan), s50[:-20]])       # 20-day change of the SMA
        avg50 = np.vstack([np.full((1, k), np.nan), pd.DataFrame(V).rolling(50, min_periods=30).mean().to_numpy()[:-1]])
        dvol = pd.DataFrame(C * V).rolling(50, min_periods=30).mean().to_numpy() / 1e6
        lowN = pd.DataFrame(L).rolling(int(pp["rc_base"]), min_periods=5).min().to_numpy()
        # closes under the SMA among the last B + slack closes (a brief poke above doesn't reset the count)
        W_ = int(pp["rc_below"]) + int(pp.get("rc_slack", 2))
        under_n = pd.DataFrame(np.isfinite(s50) & (C <= s50)).astype(float).rolling(W_, min_periods=1).sum().to_numpy()
    B, POPMAX, DIPMAX = int(pp["rc_below"]), int(pp["rc_pop_max"]), float(pp["rc_dip_max"]) / 100
    MINDIP, MAXD, HL = int(pp["rc_min_dip"]), int(pp["rc_max_days"]), bool(pp["rc_hl"])
    PRIOR = bool(pp.get("rc_prior", False))
    STOP_HL = pp.get("rc_stop", RC_STOPS[0]) == RC_STOPS[0]
    trend = pp["rc_trend"]
    pop_above = np.zeros(k, int)              # closes above the SMA since the pop (a 1-day whipsaw keeps counting)
    phase = np.zeros(k, int)                  # 0 none · 1 in the pop (above) · 2 in the dip (below)
    pop_i = np.full(k, -1)
    prior_low = np.full(k, np.nan)
    dip_low = np.full(k, np.nan)
    dip_low_i = np.full(k, -1)
    hl_low = np.full(k, np.nan)               # the latest higher low inside the dip (NaN = none yet)
    dip_days = np.zeros(k, int)
    depth_max = np.zeros(k)
    pop_trend_ok = np.zeros(k, bool)
    trig = np.zeros((n, k), bool); setup = np.zeros((n, k), bool)
    stop = np.full((n, k), np.nan); level = np.full((n, k), np.nan)
    o_prior = np.full((n, k), np.nan); o_dip = np.full((n, k), np.nan); o_hl = np.full((n, k), np.nan)
    o_pop = np.full((n, k), np.nan); o_depth = np.full((n, k), np.nan)
    dipw = np.zeros((n, k), bool)              # in a valid dip, but not armed yet (no higher low yet / dip too short)
    liq = lambda d: (C[d] >= float(pp["rc_minpx"])) & (dvol[d] >= float(pp["rc_mindvol"]))
    rs_ok = lambda d: (rs_all[d] >= float(pp["rc_rs"])) if (rs_all is not None and float(pp.get("rc_rs", 0)) > 0) \
        else np.ones(k, bool)
    for d in range(2, n):
        c, sm, lo = C[d], s50[d], L[d]
        valid = np.isfinite(c) & np.isfinite(sm)
        ab = valid & (c > sm)
        bl = valid & (c <= sm)
        # ---- a higher low confirmed today: yesterday's low was a swing low above the dip's lowest low ----
        sw = (phase == 2) & (d - 1 > dip_low_i) & (L[d - 1] <= L[d - 2]) & (L[d - 1] <= lo) & (L[d - 1] > dip_low)
        hl_low = np.where(sw, L[d - 1], hl_low)
        need = np.isfinite(hl_low) if HL else np.ones(k, bool)
        base_ok = (dip_low > prior_low) if PRIOR else np.ones(k, bool)
        stp = np.where(STOP_HL & np.isfinite(hl_low), hl_low, dip_low)
        # ---- a close back above ----
        rec = (phase == 2) & ab
        t_ = rec & (dip_days >= MINDIP) & (d - pop_i <= MAXD) & pop_trend_ok & need & base_ok & liq(d) & rs_ok(d)
        if float(pp["rc_vol"]) > 0:
            t_ &= V[d] >= float(pp["rc_vol"]) * avg50[d]
        trig[d] = t_
        for A, val in ((stop, stp), (level, sm), (o_prior, prior_low), (o_dip, dip_low), (o_hl, hl_low),
                       (o_pop, d - pop_i), (o_depth, depth_max * 100)):
            A[d] = np.where(t_, val, np.nan)
        short = rec & (dip_days < MINDIP) & (d - pop_i <= MAXD)             # a too-short dip: still the pop
        phase = np.where(rec & ~short, 0, np.where(short, 1, phase))
        # ---- a new pop: a close above after >= B of the last B + slack closes were under ----
        was_bl = np.isfinite(s50[d - 1]) & (C[d - 1] <= s50[d - 1])
        newpop = ab & was_bl & (under_n[d - 1] >= B) & (phase == 0)
        pop_above = np.where((phase == 1) & ab & ~newpop, pop_above + 1, pop_above)
        if newpop.any():
            pop_above = np.where(newpop, 1, pop_above)
            pop_i = np.where(newpop, d, pop_i)
            prior_low = np.where(newpop, lowN[d - 1], prior_low)
            up = slope[d] > 0
            tr_ok = (np.ones(k, bool) if trend == RC_TRENDS[0] else ~up if trend == RC_TRENDS[1] else up)
            pop_trend_ok = np.where(newpop, tr_ok & np.isfinite(slope[d]), pop_trend_ok)
            phase = np.where(newpop, 1, phase)
        phase = np.where((phase == 1) & ab & ((pop_above > POPMAX) | (d - pop_i > MAXD)), 0, phase)  # held above too long
        # ---- the dip ----
        start = (phase == 1) & bl
        if start.any():
            phase = np.where(start, 2, phase)
            dip_low = np.where(start, lo, dip_low)
            dip_low_i = np.where(start, d, dip_low_i)
            hl_low = np.where(start, np.nan, hl_low)
            dip_days = np.where(start, 0, dip_days)
            depth_max = np.where(start, 0.0, depth_max)
        indip = (phase == 2) & bl
        newlow = indip & (lo < dip_low)                                     # a new lowest low resets the higher low
        dip_low = np.where(newlow, lo, dip_low)
        dip_low_i = np.where(newlow, d, dip_low_i)
        hl_low = np.where(newlow, np.nan, hl_low)
        dip_days = np.where(indip, dip_days + 1, dip_days)
        depth_max = np.where(indip, np.fmax(depth_max, 1 - lo / sm), depth_max)
        bad = indip & ((lo < sm * (1 - DIPMAX)) | (d - pop_i > MAXD) | ((dip_low <= prior_low) if PRIOR else False))
        phase = np.where(bad, 0, phase)
        # ---- armed: a valid dip, a close above tomorrow would trigger ----
        need = np.isfinite(hl_low) if HL else np.ones(k, bool)
        arm = ((phase == 2) & (dip_days >= MINDIP) & (d - pop_i < MAXD) & pop_trend_ok & need & liq(d) & rs_ok(d))
        setup[d] = arm
        wait = ((phase == 2) & (d - pop_i < MAXD) & pop_trend_ok & liq(d) & rs_ok(d) & ~arm
                & (((dip_low > prior_low) | ~np.isfinite(prior_low)) if PRIOR else True))
        dipw[d] = wait
        stp = np.where(STOP_HL & np.isfinite(hl_low), hl_low, dip_low)
        for A, val in ((stop, stp), (level, sm), (o_prior, prior_low), (o_dip, dip_low), (o_hl, hl_low),
                       (o_pop, d - pop_i), (o_depth, depth_max * 100)):
            A[d] = np.where(arm | wait, val, A[d])
    return dict(trig=D(trig), setup=D(setup), dip=D(dipw), level=D(level), stop=D(stop), prior_low=D(o_prior), dip_low=D(o_dip),
                hl_low=D(o_hl), pop_days=D(o_pop), depth=D(o_depth), s50=D(s50), volx=D(V / avg50))


def rc_scan(P, pp):
    """The 50-day reclaim scan on the latest day (one row per stock)."""
    C = P["Close"]
    out = pd.DataFrame(index=C.columns)
    out["Reclaim"] = ""
    if len(C) < 80:
        return out
    rs_all = rs_matrix(C).to_numpy() if float(pp.get("rc_rs", 0)) > 0 else None
    F_ = rc_frames(P, pp, rs_all)
    # a TRIGGER in the last N days still counts while the close holds above the SMA and the stop
    N_ = max(1, int(pp.get("rc_recent", 1)))
    tr = F_["trig"].iloc[-N_:].to_numpy()
    ago = np.where(tr.any(0), np.argmax(tr[::-1], 0), np.nan)            # days since the latest trigger (0 = today)
    px = C.iloc[-1].to_numpy()
    s50_now = F_["s50"].iloc[-1].to_numpy()
    ok_, rows_ = np.isfinite(ago), len(C) - 1 - np.nan_to_num(ago).astype(int)

    def pick(k):                                                  # each stock's value on its trigger day
        v = np.full(len(ago), np.nan)
        v[ok_] = F_[k].to_numpy(float)[rows_[ok_], np.flatnonzero(ok_)]
        return v
    t_stop = pick("stop")
    with np.errstate(invalid="ignore"):
        t = np.isfinite(ago) & ((ago == 0) | ((px > s50_now) & (px > t_stop)))
    TRIG_K = ("stop", "level", "prior_low", "dip_low", "hl_low", "pop_days", "depth")   # as of the trigger day
    last = lambda k: np.where(t, pick(k), F_[k].iloc[-1].to_numpy()) if k in TRIG_K else F_[k].iloc[-1].to_numpy()
    s_ = F_["setup"].iloc[-1].to_numpy().astype(bool) & ~t
    w_ = F_["dip"].iloc[-1].to_numpy().astype(bool) & ~t & ~s_
    out["Reclaim"] = np.where(t, "TRIGGER", np.where(s_, "SETUP", np.where(w_, "DIP", "")))
    out["Trigger days ago"] = np.where(t, ago, np.nan)
    out["50 SMA"] = last("s50")
    out["vs 50 SMA %"] = (px / last("s50") - 1) * 100
    out["Low before pop"] = last("prior_low")
    out["Higher low"] = last("hl_low")
    out["Dip low"] = last("dip_low")
    out["Dip depth %"] = last("depth")
    out["Days since pop"] = last("pop_days")
    out["Reclaim stop"] = last("stop")
    ref = np.where(t, px, last("s50"))
    out["Reclaim risk %"] = np.where(t | s_ | w_, (ref - last("stop")) / ref * 100, np.nan)
    out["Reclaim vol ×"] = last("volx")
    return out
