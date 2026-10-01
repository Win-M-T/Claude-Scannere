"""Trend template and VCP (Minervini) scans."""

import numpy as np
import pandas as pd
import streamlit as st


def trend_template(H, L, C, rs, min_rs=70):
    """Mark Minervini's Trend Template on the last day (arrays: days x stocks). Returns a bool array."""
    with np.errstate(all="ignore"):
        cnt = np.sum(~np.isnan(C[-200:]), axis=0)
        c = C[-1]
        s50, s150, s200 = np.nanmean(C[-50:], axis=0), np.nanmean(C[-150:], axis=0), np.nanmean(C[-200:], axis=0)
        s200_ago = np.nanmean(C[-221:-21], axis=0) if len(C) >= 221 else np.full(C.shape[1], np.nan)
        hi52, lo52 = np.nanmax(H[-252:], axis=0), np.nanmin(L[-252:], axis=0)
        r = np.asarray(rs, float) if rs is not None else np.full(C.shape[1], 99.0)
        return ((cnt >= 190) & (c > s150) & (c > s200) & (s150 > s200) & (s200 > s200_ago) & (s50 > s150)
                & (s50 > s200) & (c > s50) & (c >= lo52 * 1.25) & (c >= hi52 * 0.75) & (np.nan_to_num(r) >= min_rs))


def vcp_scan(H, L, C, V, rs, pp):
    """Mark Minervini's Volatility Contraction Pattern.
    The base starts at the highest high of the last N days (at least 15 days ago). A zig-zag (swings of at least X%)
    splits it into pullbacks ('contractions'): each smaller than the one before, the last one tight, volume drying
    up. Pivot = the top of the last contraction. SETUP = a valid VCP just under the pivot · TRIGGER = closed above
    yesterday's pivot today on volume. Stop = the low of the last contraction. Optional: the Trend Template."""
    Ha, La, Ca, Va = (X.to_numpy(float) for X in (H, L, C, V))
    if isinstance(rs, pd.Series):
        rs = rs.reindex(C.columns).to_numpy(float)
    n, k = Ca.shape
    out = pd.DataFrame(index=C.columns)
    out["VCP"] = ""
    B = int(pp["vcp_base"])
    if n < B + 5:
        return out
    r = float(pp["vcp_swing"]) / 100
    MAXC = 8
    cols = np.arange(k)
    with np.errstate(all="ignore"):
        Hn = np.where(np.isnan(Ha), -np.inf, Ha)
        Ln = np.where(np.isnan(La), np.inf, La)
        p = n - B + np.argmax(Hn[n - B:], axis=0)                    # the base's left peak
        peak_ok = (p <= n - 15) & np.isfinite(Hn[p, cols])
        dirn = np.ones(k)                                            # +1 = in an up leg, -1 = in a pullback
        ext = Hn[p, cols].copy()
        ph = np.full(k, np.nan)                                      # last swing high
        depths = np.full((k, MAXC + 1), np.nan)
        lows = np.full(k, np.nan)
        cnt = np.zeros(k, int)
        snap = None
        for t in range(n - B + 1, n):
            act = t > p
            h, lo = Hn[t], Ln[t]
            up = act & (dirn == 1)
            ext = np.where(up & (h > ext), h, ext)
            turn_dn = up & (lo <= ext * (1 - r))
            ph = np.where(turn_dn, ext, ph)
            ext = np.where(turn_dn, lo, ext)
            dirn = np.where(turn_dn, -1, dirn)
            dn = act & (dirn == -1) & ~turn_dn
            ext = np.where(dn & (lo < ext), lo, ext)
            turn_up = dn & (h >= ext * (1 + r))
            if turn_up.any():
                idx = np.where(turn_up)[0]
                slot = np.minimum(cnt[idx], MAXC)
                depths[idx, slot] = (ph[idx] - ext[idx]) / ph[idx] * 100
                lows[idx] = ext[idx]
                cnt[idx] += 1
                ext = np.where(turn_up, h, ext)
                dirn = np.where(turn_up, 1, dirn)
            if t == n - 2:
                snap = (dirn.copy(), ext.copy(), ph.copy(), depths.copy(), lows.copy(), cnt.copy())
        now = (dirn, ext, ph, depths, lows, cnt)

        def evaluate(state, upto):
            d_, e_, ph_, dep_, lw_, cn_ = state
            dep_ = dep_.copy()
            cur = d_ == -1                                            # a pullback still forming counts too
            total = cn_ + cur.astype(int)
            slot = np.minimum(cn_, MAXC)
            dep_[cols[cur], slot[cur]] = (ph_[cur] - e_[cur]) / ph_[cur] * 100
            last_low = np.where(cur, e_, lw_)
            last_idx = np.clip(total - 1, 0, MAXC)
            first_d = dep_[:, 0]
            last_d = dep_[cols, last_idx]
            shrink = np.ones(k, bool)
            if pp["vcp_shrink"]:
                for j in range(1, MAXC + 1):
                    has = j < total
                    shrink &= ~has | (dep_[:, j] <= dep_[:, j - 1] + 0.01)
            pivot = ph_
            Cu = Ca[:upto]
            avg50 = np.nanmean(Va[upto - 51:upto - 1], axis=0)
            dry = np.nanmean(Va[upto - 10:upto], axis=0) / avg50
            valid = (peak_ok & (total >= int(pp["vcp_min_c"])) & (total <= 6) & (first_d <= pp["vcp_first_max"])
                     & (last_d <= pp["vcp_last_max"]) & shrink & np.isfinite(pivot) & np.isfinite(last_low)
                     & (((pivot - last_low) / pivot * 100) <= pp["vcp_risk"]))
            if pp["vcp_dry"] > 0:
                valid &= dry <= pp["vcp_dry"]
            txt = [" → ".join(f"{x:.0f}" for x in dep_[i, :min(total[i], MAXC + 1)]) + "%" if total[i] else ""
                   for i in range(k)]
            return valid, pivot, last_low, total, last_d, dry, txt
        v_y, piv_y, low_y, tot_y, ld_y, dry_y, txt_y = evaluate(snap, n - 1)
        v_t, piv_t, low_t, tot_t, ld_t, dry_t, txt_t = evaluate(now, n)
        tt = trend_template(Ha, La, Ca, rs, pp["vcp_rs"]) if pp["vcp_tt"] else np.ones(k, bool)
        c, pc = Ca[-1], Ca[-2]
        avg50 = np.nanmean(Va[-51:-1], axis=0)
        vol_x = Va[-1] / avg50
        trig = tt & v_y & (c > piv_y) & (pc <= piv_y) & (vol_x >= pp["vcp_vol"])
        setup = tt & v_t & ~trig & (c <= piv_t) & (c >= piv_t * (1 - pp["vcp_near"] / 100))
    use_y = trig
    out["VCP"] = np.where(trig, "TRIGGER", np.where(setup, "SETUP", ""))
    out["Contractions"] = np.where(use_y, txt_y, txt_t)
    out["# contractions"] = np.where(use_y, tot_y, tot_t)
    out["Last contraction %"] = np.where(use_y, ld_y, ld_t)
    out["Pivot"] = np.where(use_y, piv_y, piv_t)
    out["To pivot %"] = (out["Pivot"] - c) / out["Pivot"] * 100
    out["VCP stop"] = np.where(use_y, low_y, low_t)
    out["VCP risk %"] = (out["Pivot"] - out["VCP stop"]) / out["Pivot"] * 100
    out["Vol dry-up"] = np.where(use_y, dry_y, dry_t)
    out["Breakout vol ×"] = vol_x
    out["Base days"] = n - 1 - p
    out["Trend template ✓"] = tt
    return out
