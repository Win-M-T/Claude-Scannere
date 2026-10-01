"""Record-volume, parabolic and episodic-pivot scans."""

import numpy as np
import pandas as pd
import streamlit as st

from scanner.constants import VOL_REC_NAME, VOL_WINDOWS

def volume_records(C, V, pp, vh=None):
    """Did a day in the last N sessions trade the highest volume of the past 3M / 6M / 1Y / since IPO?"""
    n, cols = len(V), np.arange(V.shape[1])
    N = int(max(1, min(pp["vr_days"], n - 1)))
    Va = V.to_numpy(dtype=float)
    k = np.argmax(Va[-N:], axis=0)
    pos = n - N + k                                    # row of the record day, per stock
    rec_vol = Va[-N:][k, cols]
    prior = V.shift(1)                                 # only days BEFORE the record day count
    at = lambda X: X.to_numpy(dtype=float)[pos, cols]

    Ca = C.to_numpy(dtype=float)
    chg = (Ca[pos, cols] / Ca[np.maximum(pos - 1, 0), cols] - 1) * 100
    avg50 = at(prior.rolling(50, min_periods=20).mean())
    ratio = rec_vol / avg50
    hist = at(C.notna().cumsum())                      # bars of history up to the record day

    ok = (rec_vol > 0) & (hist >= 21) & (ratio >= pp["vr_min_ratio"])
    if pp["vr_dir"] == "Up":
        ok &= chg > 0
    elif pp["vr_dir"] == "Down":
        ok &= chg < 0

    out = pd.DataFrame(index=V.columns)
    for lbl, w in VOL_WINDOWS:
        out[f"VolHi {lbl}"] = ok & (rec_vol > at(prior.rolling(w - 1, min_periods=1).max()))

    old_max = vh["old_max"].reindex(V.columns).fillna(0).to_numpy(float) if vh is not None else 0.0
    complete = (vh["complete"].reindex(V.columns).fillna(False).to_numpy(bool) if vh is not None
                else np.zeros(len(cols), bool))
    ipo_max = np.fmax(np.nan_to_num(at(prior.cummax()), nan=0.0), old_max)
    out["VolHi IPO"] = ok & complete & (rec_vol > ipo_max)

    best = np.full(len(cols), "", dtype=object)
    for lbl in ["3M", "6M", "1Y", "IPO"]:              # smallest → largest, so the largest wins
        best = np.where(out[f"VolHi {lbl}"], VOL_REC_NAME[lbl], best)
    out["Vol record"] = best
    has = best != ""
    out["Record date"] = pd.to_datetime(np.where(has, V.index[pos].values, np.datetime64("NaT")))
    out["Record vol"] = np.where(has, rec_vol, np.nan)
    out["Record × avg"] = np.where(has, ratio, np.nan)
    out["Record day chg %"] = np.where(has, chg, np.nan)
    return out


def parabolic_scan(O, H, L, C, e10, pp):
    """Parabolic short: a huge, fast run-up far above the 10 EMA — then the first crack.
    CRACK = first red day after the run (the short trigger) · EXTENDED = still going up · FADING = rolling over."""
    n, N = len(C), int(pp["pb_days"])
    Ha, La, Ca = H.to_numpy(float), L.to_numpy(float), C.to_numpy(float)
    cols = np.arange(C.shape[1])
    with np.errstate(all="ignore"):
        recent = np.where(np.isnan(Ha[-3:]), -np.inf, Ha[-3:])
        pk = n - 3 + np.argmax(recent, axis=0)                     # peak must be in the last 3 days
        peak = Ha[pk, cols]
        win = np.where(np.isnan(La[-(N + 3):]), np.inf, La[-(N + 3):])
        lo_i = n - (N + 3) + np.argmin(win, axis=0)
        base = La[lo_i, cols]
        run = (peak / base - 1) * 100
        run_days = pk - lo_i
        up = (C > C.shift(1))
        cs = up.cumsum()
        streak = cs - cs.where(~up).ffill().fillna(0)
        max_up = streak.iloc[-(N + 3):].max().to_numpy(float)
        ext = ((C / e10 - 1) * 100).iloc[-3:].max().to_numpy(float)
        c, c1, c2 = Ca[-1], Ca[-2], Ca[-3]
        off_high = (peak - c) / peak * 100
    ok = (run >= pp["pb_gain"]) & (run_days >= 1) & (max_up >= pp["pb_up"]) & (ext >= pp["pb_ext"])
    crack = (c < c1) & (c1 > c2)                                    # first red day after an up day
    extending = (c > c1) & (off_high < 5)
    status = np.where(~ok, "", np.where(crack, "CRACK", np.where(extending, "EXTENDED", "FADING")))
    out = pd.DataFrame(index=C.columns)
    out["Parabolic"] = status
    has = status != ""
    out["Run %"] = np.where(has, run, np.nan)
    out["Run days"] = np.where(has, run_days, np.nan)
    out["Up days in a row"] = np.where(has, max_up, np.nan)
    out["Above EMA10 %"] = np.where(has, ext, np.nan)
    out["Off high %"] = np.where(has, off_high, np.nan)
    return out


def ep_scan(O, H, L, C, V, pp):
    """Episodic pivot: a big gap up on huge volume (usually earnings/news), ideally in a neglected stock."""
    n, N = len(C), int(max(1, min(pp["ep_days"], len(C) - 65)))
    cols = np.arange(C.shape[1])
    with np.errstate(all="ignore"):
        pc = C.shift(1)
        gap = (O / pc - 1) * 100
        volx = V / V.shift(1).rolling(50, min_periods=20).mean()
        dchg = (C / pc - 1) * 100
        prior3m = (pc / C.shift(64) - 1) * 100
        cond = (gap >= pp["ep_gap"]) & (volx >= pp["ep_volx"])
        if pp["ep_strong"]:                                        # closed strong: up on the day, upper half of range
            cond &= (C > pc) & ((C - L) >= 0.5 * (H - L))
        if pp["ep_neglect_on"]:
            cond &= prior3m <= pp["ep_neglect"]
        sub = cond.iloc[-N:].to_numpy(bool)
        has = sub.any(axis=0)
        last = N - 1 - np.argmax(sub[::-1], axis=0)                # most recent EP in the window
        pos = n - N + last
        at = lambda X: X.to_numpy(float)[pos, cols]
        ep_low = at(L)
        c = C.to_numpy(float)[-1]
        since = (c / at(C) - 1) * 100
    out = pd.DataFrame(index=C.columns)
    out["EP"] = np.where(has, np.where(c >= ep_low, "HOLDING", "FAILED"), "")
    out["EP date"] = pd.to_datetime(np.where(has, C.index[pos].values, np.datetime64("NaT")))
    out["Days since EP"] = np.where(has, n - 1 - pos, np.nan)
    out["EP gap %"] = np.where(has, at(gap), np.nan)
    out["EP day chg %"] = np.where(has, at(dchg), np.nan)
    out["EP vol × avg"] = np.where(has, at(volx), np.nan)
    out["Since EP %"] = np.where(has, since, np.nan)
    out["Prior 3M %"] = np.where(has, at(prior3m), np.nan)
    out["EP low"] = np.where(has, ep_low, np.nan)
    return out
