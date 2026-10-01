"""High-tight-flag frames/scan."""

import numpy as np
import pandas as pd
import streamlit as st


def _win_view(A, w):
    """(days, stocks, w) window of the last w values ending at each day, newest first (NaN-padded at the start)."""
    pad = np.full((w - 1, A.shape[1]), np.nan)
    Ap = np.vstack([pad, A])
    return np.lib.stride_tricks.sliding_window_view(Ap, w, axis=0)[..., ::-1]


def htf_frames(P, pp):
    """High Tight Flag for every day at once (days x stocks), each day using only data up to that day.
    Pole: the flag's top close is >= pole% above the lowest close of the `pole_days` days before it, and the gain
    isn't mostly 1–2 giant days. Flag: the high was set flag_min..flag_max days ago, price is 10–25% (settable)
    under it, recent ranges are tighter than the pole's, volume is drying up, few heavy-volume red days, above the
    20-day SMA (10 over 20 optional). Liquidity: price and 50-day $ volume minimums.
    TRIGGER = yesterday was a valid flag, today closes above its high on volume (x avg, and above yesterday's),
    closing in the top part of the day's range. Stop = the flag's low."""
    H, L, C, V = (P[k].to_numpy(float) for k in ["High", "Low", "Close", "Volume"])
    O = P["Open"].to_numpy(float)
    idx, cols = P["Close"].index, P["Close"].columns
    n, k = C.shape
    fmin, fmax, pd_ = int(pp["htf_flag_min"]), int(pp["htf_flag_max"]), int(pp["htf_pole_days"])
    dmin, dmax = pp["htf_depth"]
    with np.errstate(all="ignore"):
        Cs = pd.DataFrame(C)
        avg50 = pd.DataFrame(V).rolling(50, min_periods=30).mean().shift(1).to_numpy()   # before today
        dvol = (pd.DataFrame(C * V).rolling(50, min_periods=30).mean() / 1e6).to_numpy()
        s10, s20 = Cs.rolling(10).mean().to_numpy(), Cs.rolling(20).mean().to_numpy()
        # the flag's high and how long ago it was set
        Hn = np.where(np.isnan(H), -np.inf, H)
        f = np.full((n, k), -1)
        W = fmax + 1
        for c0 in range(0, k, 250):                           # chunks keep memory small
            sl = slice(c0, c0 + 250)
            wv = _win_view(Hn[:, sl], W)
            f[:, sl] = np.argmax(np.where(np.isnan(wv), -np.inf, wv), axis=2)
        rows = np.arange(n)[:, None]
        h = rows - f                                          # index of the flag high
        ok_h = (h >= pd_ + 1) & np.isfinite(Hn[np.clip(h, 0, n - 1), np.arange(k)])
        hc = np.clip(h, 0, n - 1)
        g = lambda A: np.take_along_axis(A, hc, axis=0)
        pivot = g(H)
        # pole: highest close around the top vs the lowest close of the pole_days before the high
        top_c = pd.DataFrame(C).rolling(W, min_periods=1).max().to_numpy()
        base = g(pd.DataFrame(C).rolling(pd_ + 1, min_periods=pd_ // 2).min().to_numpy())
        pole = (top_c / base - 1) * 100
        # pole quality: share of the (log) gain made by the 2 biggest up days in the pole window
        lr = np.log(C / np.roll(C, 1, axis=0)); lr[0] = np.nan
        top2 = np.full((n, k), np.nan)
        for c0 in range(0, k, 150):
            sl = slice(c0, c0 + 150)
            wv = _win_view(np.nan_to_num(lr[:, sl], nan=-1.0), pd_)
            part = -np.partition(-wv, 1, axis=2)[..., :2]
            top2[:, sl] = np.clip(part, 0, None).sum(axis=2)
        top2 = g(top2)
        share = top2 / np.log(top_c / base) * 100
        # flag low since the high, and depth
        Ln = np.where(np.isnan(L), np.inf, L)
        flag_lo = np.full((n, k), np.nan)
        for c0 in range(0, k, 250):
            sl = slice(c0, c0 + 250)
            wv = _win_view(Ln[:, sl], W)
            mask = np.arange(W)[None, None, :] <= f[:, sl][..., None]
            flag_lo[:, sl] = np.min(np.where(mask & ~np.isnan(wv), wv, np.inf), axis=2)
        depth = (pivot - flag_lo) / pivot * 100
        # tightness: last 5 days' average range vs the pole's average range
        rng = (H - L) / C
        cs = np.nancumsum(np.nan_to_num(rng), axis=0)
        r5 = (cs - np.vstack([np.zeros((5, k)), cs[:-5]])) / 5
        pole_rng = (g(cs) - np.take_along_axis(cs, np.clip(hc - pd_, 0, n - 1), axis=0)) / pd_
        tight = r5 / pole_rng
        # volume: dry-up and heavy red days inside the flag
        dry = pd.DataFrame(V).rolling(10).mean().to_numpy() / avg50
        red = ((C < np.roll(C, 1, axis=0)) & (V > 1.5 * avg50)).astype(float)
        red[0] = 0
        csr = np.cumsum(red, axis=0)
        red_n = csr - g(csr)
        # volume signature: a highest-volume-in-a-year day inside the pole
        hv1 = (V >= pd.DataFrame(V).rolling(252, min_periods=120).max().to_numpy()).astype(float)
        hv_in = g(pd.DataFrame(hv1).rolling(pd_ + 1, min_periods=1).max().to_numpy()) > 0
        valid = (ok_h & (f >= fmin) & (pole >= pp["htf_pole"]) & (depth >= dmin) & (depth <= dmax)
                 & (C <= pivot) & (C >= s20 * (1 - 0.01)) & (tight <= pp["htf_tight"])
                 & (C >= pp["htf_minpx"]) & (dvol >= pp["htf_mindvol"]))
        if pp["htf_share"] < 100:
            valid &= share <= pp["htf_share"]
        if pp["htf_dry"] > 0:
            valid &= dry <= pp["htf_dry"]
        if pp["htf_nored"]:
            valid &= red_n <= 1
        if pp["htf_ma"]:
            valid &= s10 > s20
        if pp["htf_hv"]:
            valid &= hv_in
        valid &= np.arange(n)[:, None] >= 60
        vy = np.vstack([np.zeros((1, k), bool), valid[:-1]])
        py = np.vstack([np.full((1, k), np.nan), pivot[:-1]])
        sy = np.vstack([np.full((1, k), np.nan), flag_lo[:-1]])
        volx = V / avg50
        close_pos = (C - L) / (H - L)
        trig = vy & (C > py)
        if pp["htf_vol"] > 0:
            trig &= (volx >= pp["htf_vol"]) & (V > np.roll(V, 1, axis=0))
        if pp["htf_close_top"] > 0:
            trig &= close_pos >= 1 - pp["htf_close_top"] / 100
        setup = valid & ~trig
    D = lambda A: pd.DataFrame(A, index=idx, columns=cols)
    return dict(valid=D(valid), trig=D(trig), setup=D(setup), pivot=D(pivot), stop=D(flag_lo), trig_pivot=D(py),
                trig_stop=D(sy), pole=D(pole), depth=D(depth), flag_days=D(f.astype(float)), share=D(share),
                tight=D(tight), dry=D(dry), red=D(red_n), hv=D(hv_in), volx=D(volx), close_pos=D(close_pos))


def htf_scan(P, pp):
    """The High Tight Flag scan on the latest day (one row per stock)."""
    C = P["Close"]
    out = pd.DataFrame(index=C.columns)
    out["HTF"] = ""
    if len(C) < 80:
        return out
    F_ = htf_frames(P, pp)
    last = lambda k: F_[k].iloc[-1]
    prev = lambda k: F_[k].iloc[-2]
    t = last("trig").astype(bool)
    s_ = last("setup").astype(bool)
    pick = lambda k: np.where(t, prev(k), last(k))
    out["HTF"] = np.where(t, "TRIGGER", np.where(s_, "SETUP", ""))
    out["Pole %"] = pick("pole")
    out["Flag depth %"] = pick("depth")
    out["Flag days"] = pick("flag_days")
    out["Top-2 days %"] = pick("share")
    out["Flag high"] = np.where(t, last("trig_pivot"), last("pivot"))
    out["Flag low"] = np.where(t, last("trig_stop"), last("stop"))
    out["To flag high %"] = (out["Flag high"] - C.iloc[-1]) / out["Flag high"] * 100
    out["HTF risk %"] = (out["Flag high"] - out["Flag low"]) / out["Flag high"] * 100
    out["Tightness"] = pick("tight")
    out["HTF dry-up"] = pick("dry")
    out["HV1 in pole"] = np.where(pick("hv").astype(bool), "✓", "")
    out["HTF vol ×"] = last("volx")
    return out
