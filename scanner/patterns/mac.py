"""MA-consolidation frames/scan."""

import numpy as np
import pandas as pd
import streamlit as st

from scanner.patterns.common import rs_matrix

def mac_frames(P, pp, rs_all=None):
    """MA Consolidation for every day at once (days x stocks), each day using only data up to that day.
    Valid on a day when the last N days: every close (or low) held above the 20/50-day SMA (small tolerance),
    the line is rising, the N-day range is tight, the close isn't extended above the line; optional prior uptrend
    (50 > 200-day) and RS minimum.
    Returns dict: valid, trig, setup, pivot/stop for a trigger (yesterday's box), pivot/stop of today's box,
    the line that held, range %, distance to the line %, volume ×."""
    H, L, C, V = P["High"], P["Low"], P["Close"], P["Volume"]
    N = int(pp["mac_days"])
    tol = float(pp["mac_tol"]) / 100
    with np.errstate(all="ignore"):
        s20, s50, s200 = C.rolling(20).mean(), C.rolling(50).mean(), C.rolling(200, min_periods=150).mean()
        test = L if pp["mac_hold"] == "Every low" else C
        box_hi, box_lo = H.rolling(N).max(), L.rolling(N).min()
        rng = (box_hi - box_lo) / box_lo * 100
        base = (rng <= float(pp["mac_range"])) & (C.notna().cumsum() >= 60)
        if pp["mac_trend"]:
            base &= (C > s200) & (s50 > s200)
        if float(pp.get("mac_rs", 0)) > 0:
            ra = rs_all if rs_all is not None else rs_matrix(C)
            base &= ra >= float(pp["mac_rs"])

        def held(m):
            ok = (test >= m * (1 - tol)).astype(float).where(m.notna())
            allh = ok.rolling(N).min() == 1
            near = ((C / m - 1) * 100 <= float(pp["mac_near"])) & (C >= m * (1 - tol))
            rising = (m > m.shift(10)) if pp["mac_rising"] else pd.DataFrame(True, index=C.index, columns=C.columns)
            return allh & near & rising
        h20, h50 = held(s20), held(s50)
        which = pp["mac_ma"]
        ok_line = h20 if which == "20-day SMA" else h50 if which == "50-day SMA" else (h20 | h50)
        valid = base & ok_line
        vol_x = V / V.rolling(50).mean().shift(1)
        trig = (valid.shift(1, fill_value=False) & (C > box_hi.shift(1))
                & ((vol_x >= float(pp["mac_vol"])) if float(pp["mac_vol"]) > 0 else True))
        setup = valid & ~trig
        line = np.where(h20 & h50, "20 & 50", np.where(h20, "20", np.where(h50, "50", "")))
        near_line = np.where(h20.to_numpy(), s20.to_numpy(), s50.to_numpy())
        vs_line = (C / pd.DataFrame(near_line, index=C.index, columns=C.columns) - 1) * 100
    return dict(valid=valid, trig=trig, setup=setup, trig_pivot=box_hi.shift(1), trig_stop=box_lo.shift(1),
                pivot=box_hi, stop=box_lo, line=pd.DataFrame(line, index=C.index, columns=C.columns),
                rng=rng, vs_line=vs_line, vol_x=vol_x)


def mac_scan(P, rs, pp):
    """The MA Consolidation scan on the latest day (one row per stock)."""
    C = P["Close"]
    out = pd.DataFrame(index=C.columns)
    out["MAC"] = ""
    if len(C) < 60:
        return out
    rs_all = None
    if float(pp.get("mac_rs", 0)) > 0 and rs is not None:
        rs_all = pd.DataFrame([rs.reindex(C.columns).to_numpy(float)] * len(C), index=C.index, columns=C.columns)
    F_ = mac_frames(P, pp, rs_all)
    t = F_["trig"].iloc[-1].fillna(False).astype(bool)
    s_ = F_["setup"].iloc[-1].fillna(False).astype(bool)
    out["MAC"] = np.where(t, "TRIGGER", np.where(s_, "SETUP", ""))
    piv = np.where(t, F_["trig_pivot"].iloc[-1], F_["pivot"].iloc[-1])
    stp = np.where(t, F_["trig_stop"].iloc[-1], F_["stop"].iloc[-1])
    line_now = F_["line"].iloc[-1]
    line_y = F_["line"].iloc[-2]
    out["MA held"] = np.where(t, line_y, line_now)
    out["Box range %"] = np.where(t, F_["rng"].iloc[-2], F_["rng"].iloc[-1])
    out["vs line %"] = F_["vs_line"].iloc[-1]
    out["Box high"] = piv
    out["Box low"] = stp
    out["To box high %"] = (out["Box high"] - C.iloc[-1]) / out["Box high"] * 100
    out["Box risk %"] = (out["Box high"] - out["Box low"]) / out["Box high"] * 100
    out["Box vol ×"] = F_["vol_x"].iloc[-1]
    return out
