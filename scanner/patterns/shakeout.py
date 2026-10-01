"""Shakeout +3 scan."""

import numpy as np
import pandas as pd
import streamlit as st

from scanner.constants import SO_RECLAIM

def so_rs_gate(x, rs, pp):
    """Shakeout +3 with an RS minimum: stocks rated under `so_rs` get no status (0 = off)."""
    mn = float(pp.get("so_rs", 0) or 0)
    if mn > 0 and "SO+3" in x:
        weak = ~(pd.Series(rs).reindex(x.index) >= mn)
        x = x.copy()
        x.loc[weak.to_numpy(), "SO+3"] = ""
    return x


def shakeout_plus3(H, L, C, V, pp):
    """Shakeout +3 (W-bottom early entry): in a base after an uptrend, price undercuts the base's first low (the
    shakeout), then rallies back. UNDERCUT = shaken out, still below the first low · RECLAIMED = back above the first
    low · TRIGGER = closed above the '+3' level (10% — or $3/$6 — above the first low) today."""
    Ha, La, Ca, Va = (X.to_numpy(float) for X in (H, L, C, V))
    n, cols = len(Ca), np.arange(Ca.shape[1])
    W, S, gap = int(pp["so_base"]), int(pp["so_recent"]), 5
    if n < W + 30:
        return pd.DataFrame(index=C.columns, data={"SO+3": ""})
    Hn = np.where(np.isnan(Ha), -np.inf, Ha)
    Ln = np.where(np.isnan(La), np.inf, La)
    with np.errstate(all="ignore"):
        # the shakeout: the lowest low of the last S days
        sh_i = n - S + np.argmin(Ln[n - S:], axis=0)
        sh_low = Ln[sh_i, cols]
        # the first low: lowest low of the base before the shakeout (at least a few days earlier)
        rows = np.arange(n)[:, None]
        in_first = (rows >= n - W) & (rows < (sh_i - gap)[None, :])
        fl_masked = np.where(in_first, Ln, np.inf)
        fl_i = np.argmin(fl_masked, axis=0)
        first_low = fl_masked[fl_i, cols]
        # W shape: a rally between the two lows, and the base's high on the left side
        mid = (rows > fl_i[None, :]) & (rows < sh_i[None, :])
        mid_high = np.max(np.where(mid, Hn, -np.inf), axis=0)
        left = (rows >= n - W - 20) & (rows <= fl_i[None, :])
        base_high = np.max(np.where(left, Hn, -np.inf), axis=0)
        pw = Ln[max(0, n - W - 252):n - W]                    # the 12 months before the base
        prior_low = np.nanmin(np.where(np.isinf(pw), np.nan, pw), axis=0)
        under = (1 - sh_low / first_low) * 100
        depth = (1 - sh_low / base_high) * 100
        ok = (np.isfinite(first_low) & (under >= pp["so_under"][0]) & (under <= pp["so_under"][1])
              & (mid_high >= first_low * (1 + pp["so_mid"] / 100)) & (depth <= pp["so_depth"])
              & (base_high >= prior_low * (1 + pp["so_trend"] / 100)))
        if pp["so_above200"]:
            s200 = np.nanmean(Ca[-200:], axis=0)
            ok &= base_high > s200
        if pp["so_rule"] == "Livermore $3 ($6 above $60)":
            level = first_low + np.where(first_low >= 60, 6.0, 3.0)
        else:
            level = first_low * (1 + pp["so_pct"] / 100)
        c, pc = Ca[-1], Ca[-2]
        avg50 = np.nanmean(Va[-51:-1], axis=0)
        vol_x = Va[-1] / avg50
        sh_vol = Va[sh_i, cols] / avg50
        since = n - 1 - sh_i                                   # days since the shakeout low
        if float(pp.get("so_shvol", 0) or 0) > 0:             # a real flush: heavy volume on the shakeout day
            ok &= sh_vol >= float(pp["so_shvol"])
        fast = int(pp.get("so_fast", 0) or 0)
        in_time = (since <= fast) if fast > 0 else np.ones_like(ok)
        if pp.get("so_entry") == SO_RECLAIM:                   # buy the close back above the first low
            level = first_low.copy()
            trig = ok & (sh_i < n - 1) & (c > first_low) & (pc <= first_low) & (vol_x >= pp["so_vol"]) & in_time
            reclaimed = ok & ~trig & (c > first_low)
            undercut = ok & (c <= first_low) & ((since < fast) if fast > 0 else True)
        else:
            trig = ok & (sh_i < n - 1) & (c >= level) & (pc < level) & (vol_x >= pp["so_vol"]) & in_time
            reclaimed = ok & ~trig & (c > first_low) & (c < level) & ((since < fast) if fast > 0 else True)
            undercut = ok & (c <= first_low) & ((since < fast) if fast > 0 else True)
    out = pd.DataFrame(index=C.columns)
    out["SO+3"] = np.where(trig, "TRIGGER", np.where(reclaimed, "RECLAIMED", np.where(undercut, "UNDERCUT", "")))
    out["First low"] = first_low
    out["Shakeout low"] = sh_low
    out["Undercut %"] = under
    out["+3 level"] = level
    out["To +3 %"] = (level - c) / level * 100
    out["Days since shakeout"] = n - 1 - sh_i
    out["Shakeout vol ×"] = sh_vol
    out["SO depth %"] = depth
    out["SO stop"] = sh_low * (1 - pp["so_stop"] / 100)
    out.loc[C.notna().sum() < W + 30, "SO+3"] = ""
    return out
