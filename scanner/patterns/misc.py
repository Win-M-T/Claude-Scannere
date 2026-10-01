"""First-close-above-50 scan."""

import numpy as np
import pandas as pd
import streamlit as st


def f5_scan(P, pp):
    """First close above the 50-day SMA after N+ sessions at/below it (one row per stock, latest day)."""
    C, L, V = P["Close"], P["Low"], P["Volume"]
    out = pd.DataFrame(index=C.columns)
    out["First50"] = ""
    if len(C) < 80:
        return out
    N = max(1, int(pp.get("f5_days", 30)))
    K = max(1, int(pp.get("f5_recent", 1)))
    near = float(pp.get("f5_near", 3.0))
    s50 = C.rolling(50).mean()
    ok = s50.notna().to_numpy()
    cl, sm = C.to_numpy(float), s50.to_numpy(float)
    above = ok & (cl > sm)
    below = ok & ~above
    run = np.zeros(below.shape)                       # closes in a row at/below the SMA, ending that day
    for d in range(len(C)):
        run[d] = (run[d - 1] + 1 if d else 1) * below[d]
    trig = np.zeros(below.shape, bool)
    trig[1:] = above[1:] & (run[:-1] >= N)
    stp = L.rolling(10).min().to_numpy(float)
    avg50 = V.rolling(50, min_periods=20).mean().to_numpy(float)
    volx = V.to_numpy(float) / avg50
    px, s_now, stp_now = cl[-1], sm[-1], stp[-1]
    tr = trig[-K:]
    ago = np.where(tr.any(0), np.argmax(tr[::-1], 0), np.nan)          # days since the latest trigger (0 = today)
    okk = np.isfinite(ago)
    rows = len(C) - 1 - np.nan_to_num(ago).astype(int)
    cols = np.arange(C.shape[1])
    t_stop = np.where(okk, stp[rows, cols], np.nan)
    with np.errstate(invalid="ignore"):
        t = okk & ((ago == 0) | ((px > s_now) & (px > t_stop)))
        s_ = ~t & below[-1] & (run[-1] >= N) & ((s_now - px) / s_now * 100 <= near)
    prior = np.where(okk, run[np.maximum(rows - 1, 0), cols], np.nan)
    liq = (px >= float(pp.get("f5_minpx", 0))) & (px * avg50[-1] / 1e6 >= float(pp.get("f5_mindvol", 0)))
    vmin = float(pp.get("f5_vol", 0))
    if vmin > 0:
        vt = np.where(okk, volx[rows, cols], np.nan)
        t &= vt >= vmin
    t, s_ = t & liq, s_ & liq
    stop = np.where(t, t_stop, stp_now)
    ref = np.where(t, px, s_now)
    out["First50"] = np.where(t, "TRIGGER", np.where(s_, "SETUP", ""))
    out["F50 days under"] = np.where(t, prior, np.where(s_, run[-1], np.nan))
    out["F50 trig ago"] = np.where(t, ago, np.nan)
    out["F50 SMA"] = s_now
    out["F50 vs SMA %"] = (px / s_now - 1) * 100
    out["F50 stop"] = np.where(t | s_, stop, np.nan)
    out["F50 risk %"] = np.where(t | s_, (ref - stop) / ref * 100, np.nan)
    out["F50 vol ×"] = np.where(t, np.where(okk, volx[rows, cols], np.nan), volx[-1])
    return out
