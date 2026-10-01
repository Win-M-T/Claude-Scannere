"""Green-line breakout scan."""

import numpy as np
import pandas as pd
import streamlit as st


def green_line(H, L, C, V, pp):
    """Green Line Breakout (Dr. Eric Wish): the green line is the all-time high, set months ago, that the stock
    has not touched since (3+ months of sideways action).
    BREAKOUT = first close above it today on above-average volume · RETEST = broke out recently, pulled back to the
    line and is holding it · NEAR = within a few % below the line (watchlist)."""
    Ha, La, Ca, Va = (X.to_numpy(float) for X in (H, L, C, V))
    n, cols = len(Ca), np.arange(Ca.shape[1])
    lo_age, hi_age = int(pp["gl_months"][0] * 21), int(pp["gl_months"][1] * 21)
    R = int(pp["gl_retest_days"])

    def line_before(end):
        """Highest high in the bars before `end` (the green line as of that day) and its age in trading days."""
        w = Ha[:end]
        idx = np.argmax(np.where(np.isnan(w), -np.inf, w), axis=0)
        return w[idx, cols], end - 1 - idx
    with np.errstate(all="ignore"):
        line, age = line_before(n - 1)                          # the line as of yesterday (today may break it)
        ok_age = (age >= lo_age) & (age <= hi_age)
        c, pc = Ca[-1], Ca[-2]
        avg50 = np.nanmean(Va[-51:-1], axis=0)
        vol_x = Va[-1] / avg50
        breakout = ok_age & (c > line) & (pc <= line) & (vol_x >= pp["gl_vol"])
        to_line = (line - c) / line * 100                      # + below the line, − above it
        near = ok_age & (to_line >= 0) & (to_line <= pp["gl_near"])
        # retest: the line as it stood R days ago was broken since, price is holding above it and dipped back to it
        line_r, age_r = line_before(n - R)
        ok_r = (age_r >= lo_age) & (age_r <= hi_age)
        broke = np.nanmax(Ca[n - R:n - 1], axis=0) > line_r
        dipped = np.nanmin(La[-3:], axis=0) <= line_r * (1 + pp["gl_tol"] / 100)
        retest = ok_r & broke & (c > line_r) & dipped & ~breakout
    status = np.where(breakout, "BREAKOUT", np.where(retest, "RETEST", np.where(near, "NEAR", "")))
    gl = np.where(retest & ~breakout, line_r, line)
    age_used = np.where(retest & ~breakout, age_r, age)
    out = pd.DataFrame(index=C.columns)
    out["GLB"] = status
    out["Green line"] = gl
    out["Line age (months)"] = age_used / 21
    out["Line date"] = [C.index[max(0, n - 1 - int(a))] if a == a else pd.NaT for a in age_used]
    out["To green line %"] = (gl - c) / gl * 100
    out["GLB vol ×"] = vol_x
    out["GLB stop"] = np.minimum(gl * (1 - pp["gl_stop"] / 100), np.where(breakout, La[-1], np.inf))
    out.loc[C.notna().sum() < 120, "GLB"] = ""
    return out
