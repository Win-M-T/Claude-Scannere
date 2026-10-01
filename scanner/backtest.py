"""Backtest engine and page."""

import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from scanner.charts import load_templates, tradingview_chart, tv_open_url
from scanner.constants import BT_AGREE, BT_ATR_STOPS, BT_COMBINED, BT_ENTRIES, BT_ENTRY_330, BT_LIST_UNIVERSES, BT_MARKET_FILTERS, BT_PRIORITY, BT_SCALE_OUT, BT_STOPS, BT_STRATEGIES, BT_TARGETS, BT_TRAILS, BT_UNIVERSES, DEFAULT_SECTOR_EXCL, DEFAULT_TPL, ETF_OK, ETF_TICKERS, GROUP_CHOICES, GROUP_LEVELS, GRP_CORR_DAYS, MARKETS, MKT_CUSTOM, MKT_LINES, MKT_PULLBACK, OLD_BT_NAMES, PYR_RULES, PYR_STOPS, RS_STAGES, RS_STAGE_ORDER, SO_RECLAIM, STAGE_BENCHES, STAGE_SIZE_DEFAULT, is_trig_entry, BT_ANY, BT_ALL
from scanner.data.prices import build_panels, download_all
from scanner.data.universe import get_exchange_map, get_market_meta, get_screener_meta, market_of, parse_tickers, tv_symbol
from scanner.etf import etf_status_history, group_check, group_used_etfs, market_trend_history, own_mover
from scanner.patterns.common import build_pp, compute_metrics, rs_matrix
from scanner.patterns.crossback import ema_crossback
from scanner.patterns.greenline import green_line
from scanner.patterns.htf import htf_frames
from scanner.patterns.mac import mac_frames
from scanner.patterns.reclaim import rc_frames
from scanner.patterns.shakeout import shakeout_plus3, so_rs_gate
from scanner.patterns.short import short_frames
from scanner.patterns.trend import vcp_scan
from scanner.patterns.volume import ep_scan, parabolic_scan
from scanner.rs import base_universe, rs_classify_stage
from scanner.smart_money import render_cards

ss = st.session_state

def _bt_tight_all(P, pp, first):
    """Tight-flag TRIGGER for every day at once (same rules as the scanner, just vectorised — much faster)."""
    O, H, L, C, V = (P[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    e10, e21 = C.ewm(span=10, adjust=False).mean(), C.rolling(20).mean()    # e21 = the 20-day SMA here
    s50, s200 = C.rolling(50).mean(), C.rolling(200).mean()
    vol5, vol50 = V.rolling(5).mean(), V.rolling(50).mean()
    hi20 = H.rolling(20).max()
    Ha = np.where(np.isnan(H.to_numpy(float)), -np.inf, H.to_numpy(float))
    lo = max(0, first - 30)
    win = np.lib.stride_tricks.sliding_window_view(Ha[lo:], 20, axis=0)[..., ::-1]   # (days, stocks, 20) newest first
    bars = np.full(H.shape, np.nan)
    bars[lo + 19:] = np.argmax(win, axis=2)
    bars = pd.DataFrame(bars, index=H.index, columns=H.columns)
    with np.errstate(all="ignore"):
        pull = (hi20 - C) / hi20 * 100
        rng7 = (H.rolling(7).max() - L.rolling(7).min()) / C * 100
        dist = (C - e21) / e21 * 100
        gap = (e10 - e21).abs() / e21 * 100
        setup = ((C > s50) & (s50 > s200) & (s50 > s50.shift(10)) & (bars >= pp["min_bars"]) & (bars <= 19)
                 & (pull >= pp["min_pull"]) & (pull <= pp["max_pull"]) & (rng7 < pp["max_range"]) & (C > e21)
                 & (dist < pp["max_ema_dist"]) & (gap < pp["max_ema_gap"]) & (vol5 < vol50 * pp["dry_up"]))
        trig = setup.shift(1, fill_value=False) & (C > H.shift(1).rolling(7).max()) & (V > vol50 * pp["breakout_vol"])
    enough = C.notna().cumsum() >= 60
    trig &= enough
    # armed at today's close = still a SETUP; tomorrow's buy-stop sits at the 7-day high, stop at the 7-day low
    return (trig, L.rolling(7).min(), H.shift(1).rolling(7).max(),
            setup & enough, H.rolling(7).max(), L.rolling(7).min())


# strategy -> (lookback rows it needs, side, how to read a signal + its stop from the scan's output)
# Each returns (signal today, its stop, its level, ARMED at today's close, tomorrow's buy level, tomorrow's stop).
# "Armed" = a setup the scan is watching; a buy-stop at tomorrow's level fills if price gets there — whether or not
# that day then closes as a breakout. That is how the "Trigger day" entry is traded without peeking at the close.
def _bt_tight(P, pp, rs):
    m = compute_metrics(P, pp)
    sig = m["Pattern"] == "TRIGGER"
    return (sig, P["Low"].iloc[-7:].min(), P["High"].iloc[-8:-1].max(),
            m["Pattern"] == "SETUP", P["High"].iloc[-7:].max(), P["Low"].iloc[-7:].min())


def _bt_xback(P, pp, rs):
    x = ema_crossback(P["Open"], P["High"], P["Low"], P["Close"], P["Volume"], rs, pp)
    # tomorrow's trigger = a move above the last 5 days' high (today included)
    return (x["Crossback"] == "TRIGGER", x["Stop"], x["Entry above"],
            x["Crossback"] == "SETUP", P["High"].iloc[-5:].max(), P["Low"].iloc[-6:].min())


def _bt_glb(P, pp, rs):
    g = green_line(P["High"], P["Low"], P["Close"], P["Volume"], pp)
    line = g["Green line"]
    # NEAR and today's high stayed under the line (a higher high would become the new line)
    armed = (g["GLB"] == "NEAR") & (P["High"].iloc[-1] <= line)
    return (g["GLB"] == "BREAKOUT", g["GLB stop"], line,
            armed, line, line * (1 - pp["gl_stop"] / 100))


def _bt_ep(P, pp, rs):
    e = ep_scan(P["Open"], P["High"], P["Low"], P["Close"], P["Volume"], pp)
    # an EP is only known after the gap day's volume, so there's no buy-stop version (armed = none)
    return (e["Days since EP"] == 0) & (e["EP"] == "HOLDING"), e["EP low"], P["Open"].iloc[-1], None, None, None


def _bt_so3(P, pp, rs):
    x = so_rs_gate(shakeout_plus3(P["High"], P["Low"], P["Close"], P["Volume"], pp), rs, pp)
    pending = ["UNDERCUT"] if pp.get("so_entry") == SO_RECLAIM else ["RECLAIMED", "UNDERCUT"]
    return (x["SO+3"] == "TRIGGER", x["SO stop"], x["+3 level"],
            x["SO+3"].isin(pending), x["+3 level"], x["SO stop"])


def _bt_vcp(P, pp, rs):
    v = vcp_scan(P["High"], P["Low"], P["Close"], P["Volume"], rs, pp)
    # armed = a SETUP under its pivot -> tomorrow's buy-stop at the pivot, stop at the last contraction's low
    return (v["VCP"] == "TRIGGER", v["VCP stop"], v["Pivot"], v["VCP"] == "SETUP", v["Pivot"], v["VCP stop"])


def _bt_mac_all(P, pp, first):
    F_ = mac_frames(P, pp, rs_matrix(P["Close"]) if float(pp.get("mac_rs", 0)) > 0 else None)
    # signal = TRIGGER (level = yesterday's box high) · armed = still consolidating -> buy-stop at today's box high
    return (F_["trig"].fillna(False).astype(bool), F_["trig_stop"], F_["trig_pivot"],
            F_["setup"].fillna(False).astype(bool), F_["pivot"], F_["stop"])


def _bt_mac(P, pp, rs):                               # only used through the all-days fast path
    raise NotImplementedError


def _bt_htf_all(P, pp, first):
    F_ = htf_frames(P, pp)
    # signal = TRIGGER (level = yesterday's flag high) · armed = a valid flag -> buy-stop at its high
    return (F_["trig"], F_["trig_stop"], F_["trig_pivot"], F_["setup"], F_["pivot"], F_["stop"])


def _bt_htf(P, pp, rs):                               # only used through the all-days fast path
    raise NotImplementedError


def _bt_rc_all(P, pp, first):
    F_ = rc_frames(P, pp, rs_matrix(P["Close"]).to_numpy() if float(pp.get("rc_rs", 0)) > 0 else None)
    # signal = TRIGGER (close back above the SMA) · armed = in a valid dip -> buy-stop at the SMA, stop = the dip low
    return (F_["trig"], F_["stop"], F_["level"], F_["setup"], F_["level"], F_["stop"])


def _bt_rc(P, pp, rs):                                # only used through the all-days fast path
    raise NotImplementedError


def _bt_f50_all(P, pp, first):
    """First close back above the 50-day SMA after N straight closes at/below it. Signal = that close (stop = the lowest low
    of the last 10 sessions); armed = N straight closes at/below it so far -> buy-stop at the SMA."""
    C, L = P["Close"], P["Low"]
    n_ = int(pp.get("f5_days", 30))
    s50 = C.rolling(50).mean()
    ok = s50.notna()
    above = (C > s50) & ok
    below = ok & ~above
    trig = above & (below.shift(1).astype(float).rolling(n_).sum() == n_)
    armed = below & (below.astype(float).rolling(n_).sum() == n_)
    stop = L.rolling(10).min()
    return (trig, stop, s50, armed, s50, stop)


def _bt_f50(P, pp, rs):                               # only used through the all-days fast path
    raise NotImplementedError


def _bt_bf_all(P, pp, first):
    F_ = short_frames(P, pp)
    # signal = TRIGGER (closed below yesterday's sell level) · armed = SETUP -> sell-stop at the lowest low since the
    # SMA tag, stop at the bounce high
    return (F_["bf_trig"], F_["bf_trig_stop"], F_["bf_trig_level"], F_["bf_setup"], F_["bf_level"], F_["bf_stop"])


def _bt_bf(P, pp, rs):                                # only used through the all-days fast path
    raise NotImplementedError


def _bt_fb_all(P, pp, first):
    F_ = short_frames(P, pp)
    # signal = first close back under the old high · armed = still above it -> sell-stop at the old high (pivot),
    # stop at the highest high since the breakout
    return (F_["fb_trig"], F_["fb_trig_stop"], F_["fb_pivot"], F_["fb_setup"], F_["fb_pivot"], F_["fb_stop"])


def _bt_fb(P, pp, rs):                                # only used through the all-days fast path
    raise NotImplementedError


def _bt_para(P, pp, rs):
    p = parabolic_scan(P["Open"], P["High"], P["Low"], P["Close"], P["Close"].ewm(span=10, adjust=False).mean(), pp)
    # short: stop above the run's peak · CRACK = a close below yesterday's close
    # armed = still EXTENDED; tomorrow's sell-stop = a break of today's low
    return (p["Parabolic"] == "CRACK", P["High"].iloc[-3:].max(), P["Close"].iloc[-2],
            p["Parabolic"] == "EXTENDED", P["Low"].iloc[-1], P["High"].iloc[-3:].max())


def bt_agree_signals(sigs, window, trigger):
    """Signals confirmed by every strategy: a signal counts when each of the OTHER strategies also signalled the
    same stock within the last `window` trading days (the same day too for close / next-open entries; strictly
    earlier for buy-stop fills, whose order inside a day is unknown). The trade uses the confirming (latest)
    signal's own entry level and stop. sigs: {strategy: signals DataFrame}."""
    names = [k for k, v in sigs.items()]
    if len(names) < 2 or any(v is None or v.empty for v in sigs.values()):
        return pd.DataFrame()
    idx_of = {k: v.groupby("Symbol")["idx"].apply(lambda x: np.sort(x.to_numpy())).to_dict() for k, v in sigs.items()}
    rows = []
    for k, v in sigs.items():
        others = [o for o in names if o != k]
        for r in v.itertuples(index=False):
            ok, seen = True, []
            for o in others:
                arr = idx_of[o].get(r.Symbol)
                if arr is None:
                    ok = False
                    break
                hi = r.idx if not trigger else r.idx - 1
                j = np.searchsorted(arr, hi, side="right") - 1
                if j < 0 or arr[j] < r.idx - window:
                    ok = False
                    break
                seen.append(o)
            if ok:
                rows.append({**r._asdict(), "Fired by": k, "Also": " + ".join(seen)})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["idx", "Symbol"]).drop_duplicates(["Symbol", "idx"], keep="first")
    return out.assign(Strategy=BT_AGREE).reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def intraday_330(symbols):
    """Hourly bars (last ~2 years, the most Yahoo gives) -> per stock a table by date: the price at 3:30 pm ET (the
    open of the 15:30 bar), the day's low / high before 3:30, the low / high of the last half hour, and the 15:30
    bar's close (≈ the day's close, used to line the hourly prices up with the adjusted daily ones)."""
    out = {}
    syms = list(symbols)
    for i in range(0, len(syms), 40):
        chunk = syms[i:i + 40]
        try:
            data = yf.download(chunk, period="730d", interval="60m", group_by="ticker", auto_adjust=True,
                               prepost=False, threads=True, progress=False)
        except Exception:
            continue
        if data is None or data.empty:
            continue
        for t in chunk:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(how="all")
                if df.empty:
                    continue
                ix = df.index
                ix = ix.tz_convert("America/New_York") if ix.tz is not None else ix
                df = df.set_axis(ix)
                rows = {}
                for day, g in df.groupby(ix.date):
                    b = g[(g.index.hour == 15) & (g.index.minute == 30)]
                    if b.empty:                                # half day / missing bar
                        continue
                    pre = g[g.index < b.index[0]]
                    rows[pd.Timestamp(day)] = (float(b["Open"].iat[0]),
                                               float(pre["Low"].min()) if len(pre) else float(b["Open"].iat[0]),
                                               float(pre["High"].max()) if len(pre) else float(b["Open"].iat[0]),
                                               float(b["Low"].iat[0]), float(b["High"].iat[0]),
                                               float(b["Close"].iat[0]))
                if rows:
                    out[t] = pd.DataFrame.from_dict(rows, orient="index",
                                                    columns=["P330", "LowB", "HighB", "LowA", "HighA", "CloseA"])
            except Exception:
                continue
    return out


def bt_at_330(sig, P, intr, long=True):
    """Buy at 3:30 pm on the trigger day, only if price then is past the trigger level AND up on the day (down for
    shorts). Adds P330 / LowB330 / HighB330 / LowA330 / HighA330 (hourly prices scaled to the adjusted daily ones).
    Returns (kept signals, note)."""
    if sig.empty:
        return sig, ""
    C = P["Close"]
    rows, no_data, not_past, not_green = [], 0, 0, 0
    for r in sig.to_dict("records"):
        t, d = r["Symbol"], int(r["idx"])
        tab = intr.get(t)
        day = C.index[d].normalize()
        if tab is None or day not in tab.index or d < 1:
            no_data += 1
            continue
        p330, lob, hib, loa, hia, cla = tab.loc[day, ["P330", "LowB", "HighB", "LowA", "HighA", "CloseA"]]
        k = C[t].iat[d] / cla if cla and np.isfinite(C[t].iat[d]) else 1.0      # hourly -> adjusted daily scale
        if not np.isfinite(k) or k <= 0:
            k = 1.0
        p330, lob, hib, loa, hia = (x * k for x in (p330, lob, hib, loa, hia))
        prev = C[t].iat[d - 1]
        lvl = r.get("Level", np.nan)
        past = np.isfinite(lvl) and ((p330 >= lvl) if long else (p330 <= lvl))
        green = np.isfinite(prev) and ((p330 > prev) if long else (p330 < prev))
        if not past:
            not_past += 1
            continue
        if not green:
            not_green += 1
            continue
        rows.append({**r, "P330": p330, "LowB330": lob, "HighB330": hib, "LowA330": loa, "HighA330": hia})
    kept = pd.DataFrame(rows, columns=list(sig.columns) + ["P330", "LowB330", "HighB330", "LowA330", "HighA330"])
    note = (f"🕞 3:30 pm entries: {len(kept):,} of {len(sig):,} triggers bought — {not_past:,} weren't past the "
            f"trigger at 3:30, {not_green:,} weren't {'up' if long else 'down'} on the day"
            + (f", {no_data:,} had no hourly data (Yahoo keeps ~2 years; half days have no 3:30 bar)" if no_data else "")
            + ".")
    return kept, note


def bt_signals(P, strategy, pp, days, bar=None, trigger=False, months=None, elig_out=None):
    """Run the scan as of each of the last `days` sessions (using only data up to that day).
    months: optional set of "YYYY-MM" — only signal days in those months are kept (others aren't even scanned).
    trigger=False: the scan's own signal days (known at that day's close).
    trigger=True : setups ARMED at a day's close whose buy-stop level is reached the next day (the fill day).
    elig_out: a dict — gets ["elig"] = days x stocks, True where the stock is a SETUP or TRIGGER at that day's close
    (it still qualifies; used by 'add each day it still qualifies')."""
    look, side, fn, _ = BT_STRATEGIES[strategy]
    long = side == "long"
    C, H, L = P["Close"], P["High"], P["Low"]
    rs_all = rs_matrix(C)
    n = len(C)
    rows, errors = [], 0
    start = max(look, n - days - 1)
    mon = C.index.strftime("%Y-%m")

    def wanted(d):                                           # d = the scan day; its signal lands on d (or d+1)
        if not months:
            return True
        k = d + 1 if trigger else d
        return k < n and mon[k] in months

    def add_signal(d, k, t, stop, lvl):
        rows.append(dict(Symbol=t, Signal=C.index[d], idx=d, Stop=float(stop), Level=float(lvl),
                         RS=float(rs_all.iat[d, k])))

    def add_fills(d, armed, nlvl, nstop):
        """Setups armed at day d's close, filled on day d+1 (the day price reaches the level)."""
        if armed is None or d + 1 >= n:
            return
        armed = armed.reindex(C.columns).fillna(False).astype(bool)
        for t in armed.index[armed.to_numpy()]:
            k = C.columns.get_loc(t)
            lv, sp = float(nlvl.get(t, np.nan)), float(nstop.get(t, np.nan))
            hi, lo = H.iat[d + 1, k], L.iat[d + 1, k]
            if np.isnan(lv) or np.isnan(hi) or np.isnan(lo):
                continue
            if (long and hi >= lv) or (not long and lo <= lv):
                rows.append(dict(Symbol=t, Signal=C.index[d + 1], idx=d + 1, Stop=sp, Level=lv,
                                 RS=float(rs_all.iat[d, k])))          # RS as known when the order was placed

    all_fn = {_bt_tight: _bt_tight_all, _bt_mac: _bt_mac_all, _bt_htf: _bt_htf_all, _bt_bf: _bt_bf_all, _bt_rc: _bt_rc_all, _bt_f50: _bt_f50_all,
              _bt_fb: _bt_fb_all}.get(fn)
    if all_fn is not None:                                   # fast path: all days in one go
        trig, stop, lvl, armed, nlvl, nstop = all_fn(P, pp, start)
        if elig_out is not None:
            e_ = trig.fillna(False).astype(bool)
            if armed is not None:
                e_ = e_ | armed.reindex_like(e_).fillna(False).astype(bool)
            elig_out["elig"] = e_.reindex(index=C.index, columns=C.columns).fillna(False)
        for d in range(start - (1 if trigger else 0), n - 1):
            if not wanted(d):
                continue
            if trigger:
                add_fills(d, armed.iloc[d], nlvl.iloc[d], nstop.iloc[d])
                continue
            hit = trig.iloc[d]
            for t in hit.index[hit.to_numpy()]:
                k = C.columns.get_loc(t)
                add_signal(d, k, t, stop.iat[d, k], lvl.iat[d, k])
        return pd.DataFrame(rows), errors
    days_run = [d for d in range(start - (1 if trigger else 0), n - 1) if wanted(d)]
    if elig_out is not None:                                 # eligibility needs every day, not only the wanted months
        days_run = list(range(start - (1 if trigger else 0), n))
        elig = np.zeros(C.shape, bool)
    for i, d in enumerate(days_run):                         # the last day has no "next day" to enter on
        sl = {k: v.iloc[max(0, d + 1 - look):d + 1] for k, v in P.items()}
        try:
            sig, stop, lvl, armed, nlvl, nstop = fn(sl, pp, rs_all.iloc[d])
        except Exception:
            errors += 1
            continue
        as_s = lambda x: pd.Series(x, index=C.columns) if not isinstance(x, pd.Series) else x
        if elig_out is not None:
            q_ = as_s(sig).reindex(C.columns).fillna(False).astype(bool)
            if armed is not None:
                q_ = q_ | as_s(armed).reindex(C.columns).fillna(False).astype(bool)
            elig[d] = q_.to_numpy()
            if d >= n - 1 or not wanted(d):
                continue
        if trigger:
            if armed is not None:
                add_fills(d, armed, as_s(nlvl), as_s(nstop))
        else:
            stop, lvl = as_s(stop), as_s(lvl)
            for t in sig.index[sig.fillna(False).astype(bool)]:
                add_signal(d, C.columns.get_loc(t), t, stop.get(t, np.nan), lvl.get(t, np.nan))
        if bar is not None and i % 5 == 0:
            bar.progress(min(1.0, (i + 1) / max(1, len(days_run))),
                         text=f"Replaying the scan … {C.index[d]:%b %d, %Y} · {len(rows)} signals so far")
    if elig_out is not None:
        elig_out["elig"] = pd.DataFrame(elig, index=C.index, columns=C.columns)
    return pd.DataFrame(rows), errors


def bt_trades(P, sig, side, o, one_per_stock=True, elig=None):
    """Trade every signal and follow it to the stop / target / moving-average close / time limit.
    one_per_stock: skip a stock's new signals while it's already in a trade (the 'Every signal' view). The account
    simulation passes False and enforces this itself, since it may not have taken the earlier trade.
    Returns (trades, skipped counts)."""
    skipped = {"opened past the stop": 0, "no room below the entry for a stop": 0, "no stop price": 0,
               "stop too far away": 0}
    if sig.empty:
        return pd.DataFrame(), skipped
    O, H, L, C = (P[k] for k in ["Open", "High", "Low", "Close"])
    pc = C.shift(1)
    tr_ = np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy())
    atr = pd.DataFrame(tr_, index=C.index, columns=C.columns).rolling(14, min_periods=10).mean()
    trail = {"10-day SMA": C.rolling(10).mean(), "20-day SMA": C.rolling(20).mean(),
             "50-day SMA": C.rolling(50).mean()}.get(o["trail"])
    if o["target"] == BT_SCALE_OUT:
        adr = ((H / L).rolling(20, min_periods=10).mean() - 1) * 100          # average daily range %, 20 days
        sma_x = C.rolling(int(o.get("atr_ma", 50))).mean()
    n, out, busy_until = len(C), [], {}
    long = side == "long"
    sgn = 1 if long else -1
    trig_mode = is_trig_entry(o["entry"])
    at330 = o["entry"] == BT_ENTRY_330
    cut_n, cut_r = int(o.get("cut_days", 0) or 0), float(o.get("cut_r", 1.0) or 0.0)
    # 🪜 pilot + add to winners: 1 unit at the entry, +1 unit at the close each time price closes another
    # `pyr_step` ATRs (ATR at the entry) past the last buy, up to `pyr_max` units; the stop is raised after each add
    pyr = bool(o.get("pyr")) and o["target"] != BT_SCALE_OUT
    pyr_step, pyr_max, pyr_stop = float(o.get("pyr_step", 1.5)), int(o.get("pyr_max", 5)), o.get("pyr_stop", PYR_STOPS[0])
    rule = o.get("pyr_rule", PYR_RULES[1])
    by_sig = pyr and rule in (PYR_RULES[0], PYR_RULES[2])  # add on each day it qualifies (scan / stacked trend)
    pyr_profit = bool(o.get("pyr_profit", True))
    if by_sig:
        if rule == PYR_RULES[2]:
            e10, e21, e50 = (C.ewm(span=k_, adjust=False).mean() for k_ in (10, 21, 50))
            stk = ((C >= e10) & (e10 >= e21) & (e21 >= e50)) if long else ((C <= e10) & (e10 <= e21) & (e21 <= e50))
            E_ = stk.fillna(False).to_numpy(bool)
        else:
            E_ = (elig.reindex(index=C.index, columns=C.columns).fillna(False).to_numpy(bool) if elig is not None
                  else np.zeros(C.shape, bool))
        col_i = {c: i for i, c in enumerate(C.columns)}
    for s in sig.sort_values("idx").itertuples():
        t, d = s.Symbol, s.idx
        if one_per_stock and busy_until.get(t, -1) >= d:       # already in a trade in this stock
            continue
        at_open = o["entry"] == "Next day's open"
        e = d + 1 if at_open else d                          # the entry day
        if e >= n:
            continue
        known = d - 1 if trig_mode else d                    # last day whose close was known when buying
        filled_open, intraday = at_open, False
        if at_open:
            entry = O[t].iat[e]
        elif at330:
            entry = getattr(s, "P330", np.nan)                 # bought at 3:30 pm on the trigger day
            if not np.isfinite(entry):
                continue
        elif trig_mode:
            # buy-stop at the level (armed the day before): a gap past it fills at the open, else at the level
            lvl, op_ = s.Level, O[t].iat[d]
            if np.isnan(lvl):
                continue
            filled_open = not np.isnan(op_) and ((long and op_ >= lvl) or (not long and op_ <= lvl))
            entry = op_ if filled_open else lvl
            intraday = not filled_open
        else:
            entry = C[t].iat[d]
        if not entry or not np.isfinite(entry):
            continue
        stop_rule = o["stop"]
        basis = stop_rule
        a = atr[t].iat[known] if known >= 0 else np.nan      # ATR as known when the order was placed
        if stop_rule == "The setup's stop":
            if not np.isfinite(s.Stop):
                skipped["no stop price"] += 1
                continue
            stop = s.Stop
        elif stop_rule == "Low of the entry day" and at330:
            # bought at 3:30: the day's low so far is known
            stop, basis = (s.LowB330 if long else s.HighB330), "Day's low (to 3:30 pm)"
        elif stop_rule == "Low of the entry day" and filled_open:
            # bought right at the open: the day's low isn't known yet -> 0.5 ATR under the entry, live right away
            if not np.isfinite(a):
                continue
            stop, basis = entry - sgn * 0.5 * a, "0.5 ATR (bought at the open)"
        elif stop_rule == "Low of the entry day":
            # bought during the day or at the close: the low of that day (the high for shorts), set at the close
            stop, basis = (L[t].iat[e] if long else H[t].iat[e]), "Day's low"
        elif stop_rule in BT_ATR_STOPS:
            if not np.isfinite(a):
                continue
            stop = entry - sgn * BT_ATR_STOPS[stop_rule] * a
        else:
            stop = entry * (1 - sgn * o["stop_pct"] / 100)
        risk = (entry - stop) if long else (stop - entry)
        if not np.isfinite(risk):
            skipped["no stop price"] += 1
            continue
        min_risk = entry * o.get("min_stop_pct", 0.0) / 100
        # a stop only a few cents away gets pushed out to the minimum distance (e.g. bought right at the day's low);
        # a stock that opened PAST its stop is still skipped
        if min_risk > 0 and risk < min_risk and (risk > 0 or basis == "Day's low"):
            stop = entry - sgn * min_risk
            risk = min_risk
            basis = f"{basis} → widened to {o['min_stop_pct']:g}%"
        if risk <= 0:                                       # opened beyond the stop / bought at the day's low
            skipped["opened past the stop" if filled_open or o["entry"] == "Next day's open"
                    else "no room below the entry for a stop"] += 1
            continue
        if o.get("max_stop_pct", 0) > 0 and risk / entry * 100 > o["max_stop_pct"]:
            skipped["stop too far away"] += 1               # a very wide stop: skip the trade
            continue
        target = None
        if o["target"] == "R-multiple":
            target = entry + sgn * o["target_r"] * risk
        elif o["target"] == "Percent":
            target = entry * (1 + o["target_pct"] / 100) if long else entry * max(0.01, 1 - o["target_pct"] / 100)
        scale = o["target"] == BT_SCALE_OUT
        trim1 = None
        if scale:
            ad = adr[t].iat[known] if known >= 0 else np.nan     # ADR % as known when the order was placed
            if np.isfinite(ad) and ad > 0:
                mv = o.get("adr_x", 2.0) * ad / 100
                trim1 = entry * (1 + mv) if long else entry * max(0.01, 1 - mv)
        rem, parts, trims = 1.0, [], []                      # fraction still held · (fraction, price) sold · trims
        t1_done, t2_done = trim1 is None, False
        exit_px, exit_i, why = None, None, None
        # bought at the open / during the day -> the rest of that day counts; bought at the close -> from the next day
        first = e if (filled_open or intraday) else e + 1
        if at330:
            # the last half hour of the entry day: only the stop / target can end it (the close is after 3:30)
            loa, hia = s.LowA330, s.HighA330
            if (long and loa <= stop) or (not long and hia >= stop):
                exit_px, why, exit_i = stop, "Stop", e
            elif target is not None and ((long and hia >= target) or (not long and loa <= target)):
                exit_px, why, exit_i = target, "Target", e
        stop0 = stop
        legs = [(e, entry)]                                  # every unit bought: (day, price) — the pilot first
        rung = pyr_step * (a if np.isfinite(a) else atr[t].iat[e]) if pyr else np.nan
        ci = col_i[t] if by_sig else None
        for j in range(first if exit_px is None else n, n):
            op, hi, lo, cl = O[t].iat[j], H[t].iat[j], L[t].iat[j], C[t].iat[j]
            if np.isnan(cl):
                continue
            stop_hit = ((cl <= stop) if long else (cl >= stop)) if (j == e and intraday) else \
                ((lo <= stop) if long else (hi >= stop))
            if scale and not stop_hit:
                # trim ⅓ at entry ± ADR% × 2, and ⅓ when price is 10 ATRs past the moving average
                # (level from yesterday's MA and ATR, so nothing from the future is used)
                if not t1_done and ((long and hi >= trim1) or (not long and lo <= trim1)):
                    px = trim1 if (j == e and intraday) else (max(op, trim1) if long else min(op, trim1))
                    parts.append((1 / 3, px)); rem -= 1 / 3; t1_done = True
                    trims.append((j, px, f"{o.get('adr_x', 2.0):g}× ADR"))
                if not t2_done and j >= 1 and np.isfinite(sma_x[t].iat[j - 1]) and np.isfinite(atr[t].iat[j - 1]):
                    lvl2 = sma_x[t].iat[j - 1] + sgn * o.get("atr_x", 10.0) * atr[t].iat[j - 1]
                    if (long and hi >= lvl2 and lvl2 > entry) or (not long and lo <= lvl2 and lvl2 < entry):
                        px = lvl2 if (j == e and intraday) else (max(op, lvl2) if long else min(op, lvl2))
                        parts.append((1 / 3, px)); rem -= 1 / 3; t2_done = True
                        trims.append((j, px, f"{o.get('atr_x', 10.0):g} ATR from {o.get('atr_ma', 50)}-day MA"))
                if trail is not None and j > e and ((long and cl < trail[t].iat[j]) or (not long and cl > trail[t].iat[j])):
                    exit_px, why = cl, f"Close {'below' if long else 'above'} {o['trail']}"
                if exit_px is None and cut_n and j - e + 1 == cut_n and not t1_done and \
                        ((long and cl < entry + cut_r * risk) or (not long and cl > entry - cut_r * risk)):
                    exit_px, why = cl, f"Not +{cut_r:g}R after {cut_n}d"
                if exit_px is None and j - e + 1 >= o["max_days"]:
                    exit_px, why = cl, "Time limit"
                if exit_px is not None:
                    exit_i = j
                    break
                continue
            if j == e and intraday:
                # bought during the day: only what surely happened AFTER the fill counts —
                # a close beyond the stop (price was at the entry, then crossed the stop), or the target
                # (price had to pass the entry to get there)
                if (long and cl <= stop) or (not long and cl >= stop):
                    exit_px, why = stop, "Stop"
                elif target is not None and ((long and hi >= target) or (not long and lo <= target)):
                    exit_px, why = target, "Target"
            elif long and lo <= stop:
                exit_px, why = min(op, stop), "Stop"                  # a gap below the stop fills at the open
            elif not long and hi >= stop:
                exit_px, why = max(op, stop), "Stop"
            elif target is not None and long and hi >= target:
                exit_px, why = max(op, target), "Target"
            elif target is not None and not long and lo <= target:
                exit_px, why = min(op, target), "Target"
            elif trail is not None and j > e and ((long and cl < trail[t].iat[j]) or (not long and cl > trail[t].iat[j])):
                exit_px, why = cl, f"Close {'below' if long else 'above'} {o['trail']}"
            if exit_px is None and cut_n and j - e + 1 == cut_n and \
                    ((long and cl < entry + cut_r * risk) or (not long and cl > entry - cut_r * risk)):
                exit_px, why = cl, f"Not +{cut_r:g}R after {cut_n}d"          # a time stop: it didn't get going
            if exit_px is None and j - e + 1 >= o["max_days"]:
                exit_px, why = cl, "Time limit"
            if exit_px is not None:
                exit_i = j
                break
            if by_sig:
                avg_ = len(legs) / sum(1 / p_ for _, p_ in legs)
                add_now = (j >= first and j > d and len(legs) < pyr_max and E_[j, ci]
                           and (not pyr_profit or (cl > avg_ if long else cl < avg_)))
            else:
                add_now = (pyr and j > e and len(legs) < pyr_max and np.isfinite(rung) and rung > 0 and
                           ((long and cl >= legs[-1][1] + rung) or (not long and cl <= legs[-1][1] - rung)))
            if add_now:
                prev = legs[-1][1]
                legs.append((j, cl))                         # add 1 unit at the close
                if pyr_stop == PYR_STOPS[0]:                 # ladder: the stop goes to the previous buy
                    new = prev
                elif pyr_stop == PYR_STOPS[1]:               # breakeven on the whole position (average cost)
                    new = len(legs) / sum(1 / p_ for _, p_ in legs)
                else:
                    new = stop
                if (long and new < cl) or (not long and new > cl):          # never a stop at/through the price
                    stop = max(stop, new) if long else min(stop, new)
        if exit_px is None:                                  # still open at the end of the data
            exit_i = n - 1
            last = C[t].iloc[:n].dropna()
            exit_px, why = (last.iat[-1] if len(last) else entry), "Still open"
        extra = {}
        if scale:
            last_px = exit_px
            parts.append((max(rem, 0.0), last_px))
            exit_px = sum(f * p for f, p in parts) / sum(f for f, _ in parts)   # average price of all the sales
            if trims:
                why = f"Trimmed {'⅓' if len(trims) == 1 else '⅔'} → {why}"
            for q, tag in ((1, "ADR"), (2, "ATR")):            # Trim 1 = the ADR trim, Trim 2 = the ATR trim
                tj = next((x for x in trims if tag in x[2]), None)
                extra[f"Trim {q}"] = tj[1] if tj else np.nan
                extra[f"Trim {q} date"] = C.index[tj[0]] if tj else pd.NaT
            extra["Last exit"] = last_px
        ret = (exit_px / entry - 1) * 100 * sgn
        r_mult = (exit_px - entry) * sgn / risk
        if pyr:
            # every unit is the same $ amount: P&L per $ of one unit, summed over the units
            mult = sum((exit_px / p_ - 1) * sgn for _, p_ in legs)
            r_mult = mult * entry / risk                     # R = P&L ÷ the pilot's risk
            ret = mult / len(legs) * 100                     # return on all the money put in
            extra.update({"Units": len(legs), "Avg cost": len(legs) / sum(1 / p_ for _, p_ in legs),
                          "Final stop": stop, "Legs": [(C.index[j_], float(p_)) for j_, p_ in legs]})
            stop = stop0
        busy_until[t] = exit_i
        out.append(dict(Symbol=t, Signal=s.Signal, Entry_date=C.index[e], Entry=entry, Stop=stop,
                        Exit_date=C.index[exit_i], Exit=exit_px, **{"Return %": ret, "R": r_mult},
                        Days=exit_i - e + 1, Exit_reason=why, RS=s.RS, **{"Risk %": risk / entry * 100},
                        Trigger=float(getattr(s, "Level", np.nan)), Stop_basis=basis,
                        **({"Group": s.Group} if hasattr(s, "Group") else {}), **extra))
    return pd.DataFrame(out), skipped


def bt_pause_filter(shadow, trades, n):
    """The pause rule: take a trade only if the last `n` CLOSED trades of the strategy (every signal, tracked on
    paper even while paused) add up to a profit. Uses only trades closed before the entry day, so it can be
    followed live. Returns (trades kept, paused count, [(pause start, pause end or None)])."""
    if n <= 0 or trades is None or trades.empty or shadow is None or shadow.empty:
        return trades, 0, []
    closed = shadow[shadow["Exit_reason"] != "Still open"].sort_values(["Exit_date", "Entry_date", "Symbol"],
                                                                      kind="mergesort")
    ex_d = closed["Exit_date"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(closed["R"].to_numpy(float))])

    def on(day):
        k = int(np.searchsorted(ex_d, np.datetime64(day), side="left"))   # trades closed BEFORE this day
        return k < n or cum[k] - cum[k - n] > 0

    ent = pd.DatetimeIndex(sorted(set(trades["Entry_date"]) | set(shadow["Entry_date"])))
    days = pd.bdate_range(ent.min(), max(ent.max(), shadow["Exit_date"].max())).union(ent)
    state = pd.Series([on(d) for d in days], index=days)
    keep = trades["Entry_date"].map(state).fillna(True).astype(bool).to_numpy()
    periods, start = [], None
    for d, v in state.items():                              # pause periods, by trading day
        if not v and start is None:
            start = d
        elif v and start is not None:
            periods.append((start, d)); start = None
    if start is not None:
        periods.append((start, None))
    return trades[keep], int((~keep).sum()), periods


def bt_industry_pause(shadow, trades, k, industry_of):
    """Industry pause: skip a trade if the last `k` CLOSED signals in the same industry (tracked on paper) were
    all losers — until one of them wins. Stocks with an unknown industry are never paused.
    Returns (trades kept, paused count, {industry: paused count})."""
    if k <= 0 or trades is None or trades.empty or shadow is None or shadow.empty:
        return trades, 0, {}
    ind = lambda df: df["Symbol"].map(industry_of).fillna("").astype(str)
    closed = shadow[shadow["Exit_reason"] != "Still open"].assign(_ind=lambda d: ind(d)) \
        .sort_values(["Exit_date", "Entry_date", "Symbol"], kind="mergesort")
    by = {g: (d["Exit_date"].to_numpy(), (d["R"].to_numpy(float) <= 0)) for g, d in closed.groupby("_ind") if g}
    keep, per = [], {}
    for g, day in zip(ind(trades), trades["Entry_date"]):
        ok = True
        if g in by:
            ex, lost = by[g]
            j = int(np.searchsorted(ex, np.datetime64(day), side="left"))     # closed BEFORE the entry day
            ok = not (j >= k and lost[j - k:j].all())
        if not ok:
            per[g] = per.get(g, 0) + 1
        keep.append(ok)
    keep = np.array(keep, bool)
    return trades[keep], int((~keep).sum()), dict(sorted(per.items(), key=lambda x: -x[1]))


def bt_by_strategy(fn, shadow, trades, *args):
    """Run a pause rule separately for each strategy of a Combined run (each tracks only its own signals).
    Returns (trades kept, paused count, extra) — extra: a list of (strategy, …) periods / a merged dict."""
    if (trades is None or trades.empty or "Strategy" not in trades or shadow is None or "Strategy" not in shadow
            or shadow["Strategy"].nunique() <= 1):
        kept, n, extra = fn(shadow, trades, *args)
        if isinstance(extra, list):
            extra = [(None, *x) for x in extra]
        return kept, n, extra
    parts, n_all, extra_all = [], 0, None
    for name, sh in shadow.groupby("Strategy", sort=False):
        kept, n, extra = fn(sh, trades[trades["Strategy"] == name], *args)
        parts.append(kept)
        n_all += n
        if isinstance(extra, list):
            extra_all = (extra_all or []) + [(name, *x) for x in extra]
        else:
            extra_all = extra_all or {}
            for k_, v_ in (extra or {}).items():
                extra_all[k_] = extra_all.get(k_, 0) + v_
    rest = trades[~trades["Strategy"].isin(shadow["Strategy"].unique())]
    kept = pd.concat(parts + [rest]) if parts else rest
    if isinstance(extra_all, dict):
        extra_all = dict(sorted(extra_all.items(), key=lambda x: -x[1]))
    return kept, n_all, extra_all if extra_all is not None else []


def bt_size(tr, acct, risk_pct, max_pos_pct, unit_pct=None):
    """Fixed-fractional sizing: risk `risk_pct` of the (starting) account per trade, capped at `max_pos_pct`.
    Pilot + add (trades with 'Legs'): every unit is `unit_pct` % of the account, at its own buy price."""
    if tr.empty:
        return tr
    tr = tr.copy()
    mx = tr["Size ×"].to_numpy(float) if "Size ×" in tr else np.ones(len(tr))
    if "Legs" in tr and unit_pct:
        side = np.where(tr["Entry"] > tr["Stop"], 1, -1)
        u = acct * unit_pct / 100
        sh = [[np.floor(u * m_ / p_) for _, p_ in lg] for lg, m_ in zip(tr["Legs"], mx)]
        tr["Shares"] = [sum(x) for x in sh]
        tr["Position $"] = [sum(q * p_ for q, (_, p_) in zip(x, lg)) for x, lg in zip(sh, tr["Legs"])]
        tr["Position %"] = tr["Position $"] / acct * 100
        tr["P&L $"] = [sum(q * (ex - p_) * sd for q, (_, p_) in zip(x, lg))
                       for x, lg, ex, sd in zip(sh, tr["Legs"], tr["Exit"], side)]
        tr["Account %"] = tr["P&L $"] / acct * 100
        tr.attrs["_acct"] = acct
        return tr
    per_share = (tr["Entry"] - tr["Stop"]).abs()
    sh = np.floor(acct * risk_pct / 100 * mx / per_share)
    sh = np.minimum(sh, np.floor(acct * max_pos_pct / 100 * mx / tr["Entry"])).clip(lower=0)
    tr["Shares"] = sh
    tr["Position $"] = sh * tr["Entry"]
    tr["Position %"] = tr["Position $"] / acct * 100
    side = np.where(tr["Entry"] > tr["Stop"], 1, -1)          # long: stop below entry · short: above
    tr["P&L $"] = sh * (tr["Exit"] - tr["Entry"]) * side
    tr["Account %"] = tr["P&L $"] / acct * 100
    tr.attrs["_acct"] = acct
    return tr


def stage_history(ohlc):
    """The 1A–4C stage of a price series (e.g. the market ETF) for every day, from that day's close — the same
    rules as the Relative Strength page's stages."""
    C, H, L = ohlc["Close"], ohlc["High"], ohlc["Low"]
    e10, e20 = C.ewm(span=10, adjust=False).mean(), C.ewm(span=20, adjust=False).mean()
    s50, s200 = C.rolling(50).mean(), C.rolling(200).mean()
    pc = C.shift(1)
    atr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()], axis=1).max(axis=1).rolling(14).mean()
    return pd.Series([rs_classify_stage(*v) for v in zip(C, e10, e20, s50, s200, atr)], index=C.index)


def bt_bench_stages(bench, res=None):
    """The market ETF's stage for every day (5 years of daily prices, kept for the day)."""
    key_ = (bench, date.today().isoformat())
    cache = ss.setdefault("stage_hist", {})
    if key_ not in cache:
        m_, b_ = st.empty(), st.empty()
        try:
            got = download_all([bench], b_, m_, period="5y").get(bench)
        except Exception:
            got = None
        m_.empty(); b_.empty()
        cache[key_] = stage_history(got.dropna(subset=["Close"])) if got is not None and len(got) > 60 else None
    return cache[key_]


def bt_stage_pct():
    return {k: int(ss.get(f"bt_stg_{k}", v)) for k, v in STAGE_SIZE_DEFAULT.items()}


def bt_stage_sizing(tr, stages, pct, entry):
    """Size ×: each trade's size is scaled by the market's stage on the last close before the buy (the signal day
    for next-open / close entries, the day before for buy-stop / 3:30 entries). 0 % = the trade is skipped.
    Returns (trades with 'Mkt stage' and 'Size ×', number skipped)."""
    if tr is None or tr.empty or stages is None or stages.empty:
        return tr, 0
    st_ = stages.copy()
    st_.index = pd.DatetimeIndex(st_.index).normalize()
    days = st_.index
    ent = pd.DatetimeIndex(tr["Entry_date"]).normalize()
    if entry == "Signal day's close":
        pos = days.searchsorted(ent, side="right") - 1                 # that day's close
    else:
        pos = days.searchsorted(ent, side="left") - 1                  # the close before the entry day
    stg = np.where(pos >= 0, st_.to_numpy()[np.clip(pos, 0, len(days) - 1)], "?")
    mult = np.array([pct.get(x, 100) / 100 for x in stg])
    out = tr.assign(**{"Mkt stage": stg, "Size ×": mult})
    return out[out["Size ×"] > 0].reset_index(drop=True), int((mult <= 0).sum())


def _profit_factor(w, l):
    """Money won ÷ money lost ($ when sized, else R) — ∞ with no losing trades, NaN with no trades at all."""
    col = "P&L $" if "P&L $" in w else "R"
    gains, losses = w[col].sum(), -l[col].sum()
    if losses > 0:
        return gains / losses
    return np.inf if gains > 0 else np.nan


def bt_stats(tr):
    if tr.empty:
        return {}
    closed = tr[tr["Exit_reason"] != "Still open"]
    w, l = tr[tr["R"] > 0], tr[tr["R"] <= 0]
    eq = tr.sort_values("Exit_date").set_index("Exit_date")["R"].cumsum()
    dd = min(0.0, (eq - eq.cummax().clip(lower=0)).min())      # the start (0R) counts as a peak too
    pnl = tr["P&L $"].sum() if "P&L $" in tr else np.nan
    eq_usd = tr.sort_values("Exit_date").set_index("Exit_date")["P&L $"].cumsum() if "P&L $" in tr else None
    dd_pct = np.nan
    if eq_usd is not None and "_acct" in tr.attrs:
        val = tr.attrs["_acct"] + eq_usd
        dd_pct = ((val / val.cummax().clip(lower=tr.attrs["_acct"])) - 1).min() * 100
    streak = best = 0
    for r in tr.sort_values("Exit_date")["R"]:
        streak = streak + 1 if r <= 0 else 0
        best = max(best, streak)
    return dict(n=len(tr), closed=len(closed), win=len(w) / len(tr) * 100, avg_win=w["Return %"].mean(),
                avg_loss=l["Return %"].mean(), exp_r=tr["R"].mean(), total_r=tr["R"].sum(),
                pf=_profit_factor(w, l),
                days=tr["Days"].mean(), dd=dd, streak=best, eq=eq, pnl=pnl, eq_usd=eq_usd, dd_pct=min(0.0, dd_pct),
                avg_win_r=w["R"].mean(), avg_loss_r=l["R"].mean())


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def bt_universe_meta(name):
    """A stock list with today's snapshot: market cap ($B, US$ for Asian markets), price and volume."""
    t, meta = base_universe(BT_LIST_UNIVERSES[name], "")
    if meta is None or not len(meta):
        return pd.DataFrame(index=pd.Index(t, name="Symbol"), columns=["Mkt cap $B", "_last", "_vol"], dtype=float)
    m = meta.reindex(t)
    for c in ["Mkt cap $B", "_last", "_vol"]:
        if c not in m:
            m[c] = np.nan
    for c in ["Sector", "Industry"]:
        if c not in m:
            m[c] = ""
    return m[["Mkt cap $B", "_last", "_vol", "Sector", "Industry"]]


def sector_lookup(tickers, sources=()):
    """Sector / industry for any tickers: from the given tables first, then the US and Asian screeners."""
    out = pd.DataFrame(index=pd.Index(list(dict.fromkeys(tickers)), name="Symbol"), columns=["Sector", "Industry"],
                       dtype=object)
    sources = list(sources)
    if any(not market_of(t) for t in out.index):
        try:
            sources.append(get_screener_meta())
        except Exception:
            pass
    for mk, spec in MARKETS.items():
        if any(t.endswith(spec["suffix"]) for t in out.index):
            try:
                sources.append(get_market_meta(mk))
            except Exception:
                pass
    for m in sources:
        if m is None or not len(m):
            continue
        for c in ["Sector", "Industry"]:
            if c in m:
                have = out[c].notna() & (out[c].astype(str).str.strip() != "")
                out[c] = out[c].where(have, m[c].reindex(out.index))
    return out.fillna("")


def market_lines(c):
    """The moving averages the market filter can use, for one price series."""
    return {"10 EMA": c.ewm(span=10, adjust=False).mean(), "21 EMA": c.ewm(span=21, adjust=False).mean(),
            "50 EMA": c.ewm(span=50, adjust=False).mean(), "50 SMA": c.rolling(50).mean(),
            "200 SMA": c.rolling(200, min_periods=150).mean()}


def market_ok_history(C, market, above=(), below=(), t="SPY"):
    """True on the days SPY passed the market filter (close vs its moving averages, that day's close)."""
    c = C[t]
    if market in ("SPY in an uptrend", "SPY not in a downtrend"):
        m = market_trend_history(C, t)
        return (m == "Uptrend") if market == "SPY in an uptrend" else m.notna() & (m != "Downtrend")
    ln = market_lines(c)
    if market == MKT_PULLBACK:
        above, below = ("50 EMA",), ("21 EMA",)
    ok = pd.Series(True, index=C.index)
    for n in above:
        ok &= c > ln[n]
    for n in below:
        ok &= c < ln[n]
    known = pd.concat([ln[n] for n in (*above, *below)], axis=1).notna().all(axis=1) if (above or below) else True
    return ok & known


def market_filter_label(market, above=(), below=()):
    if market != MKT_CUSTOM:
        return market
    parts = ([f"above {', '.join(above)}"] if above else []) + ([f"below {', '.join(below)}"] if below else [])
    return "SPY " + " and ".join(parts) if parts else "no lines picked"


def bt_liquidity_filter(sig, P, min_px, min_vol, min_dvol, trigger=False):
    """Keep signals whose stock was liquid enough on the signal day (the day before a buy-stop fill):
    close >= min_px, 50-day average volume >= min_vol, 50-day average $ volume >= min_dvol ($M)."""
    C, V = P["Close"], P["Volume"]
    avg_v = V.rolling(50, min_periods=20).mean()
    avg_d = (C * V).rolling(50, min_periods=20).mean() / 1e6
    day = (sig["idx"] - 1 if trigger else sig["idx"]).clip(lower=0).to_numpy()
    cols = C.columns.get_indexer(sig["Symbol"])
    px, av, ad = (X.to_numpy(float)[day, cols] for X in (C, avg_v, avg_d))
    keep = np.ones(len(sig), bool)
    if min_px > 0:
        keep &= px >= min_px
    if min_vol > 0:
        keep &= av >= min_vol
    if min_dvol > 0:
        keep &= ad >= min_dvol
    parts = ([f"price ≥ ${min_px:g}"] if min_px > 0 else []) + ([f"avg vol ≥ {min_vol:,.0f}"] if min_vol > 0 else []) \
        + ([f"avg $ vol ≥ ${min_dvol:g}M"] if min_dvol > 0 else [])
    note = (f"💧 Liquidity ({', '.join(parts)}): {len(sig):,} signals → **{int(keep.sum()):,} kept**.")
    return sig[keep].reset_index(drop=True), note


def bt_funnel_filter(sig, P, uni, ind_top=0, ind_min=3, stages=(), lead_rs=0, corr_min=0.0, trigger=False, rs=None):
    """Funnel filters on the SIGNAL DAY (no hindsight):
    · industry strength — the industry's median RS (1-99) among the tested stocks that day, ranked across industries with
      ≥ ind_min stocks; keep the top ind_top %. A leader (RS ≥ lead_rs) or an own mover (60-day correlation with its
      industry peers < corr_min) is let through anyway.
    · stock stage — the stock's own 1A-4C stage at that day's close must be in `stages`.
    Adds 'Ind rank' (1 = strongest) and 'Stock stage' columns. Returns (kept signals, a one-line note)."""
    if not len(sig) or (ind_top <= 0 and not stages):
        return sig, ""
    C, H, L = P["Close"], P["High"], P["Low"]
    n0 = len(sig)
    day = (sig["idx"] - 1 if trigger else sig["idx"]).clip(lower=0).to_numpy()
    cols = C.columns.get_indexer(sig["Symbol"])
    keep = np.ones(n0, bool)
    parts = []
    rank_txt = np.full(n0, np.nan)
    if ind_top > 0:
        info = bt_sector_info(list(C.columns), uni)
        ind = info["Industry"].reindex(C.columns).astype(str).replace({"nan": "", "None": ""}).fillna("").to_numpy()
        RS = rs if rs is not None else rs_matrix(C)
        RSv = RS.to_numpy(float)
        ind_s = pd.Series(ind)
        valid = ind != ""
        pos_of = {}                                                         # industry -> column positions
        for k_, name in enumerate(ind):
            if name:
                pos_of.setdefault(name, []).append(k_)
        rank_of_day = {}
        for d in np.unique(day):
            row = RSv[d]
            med = {nm: np.nanmedian(row[p]) for nm, p in pos_of.items() if np.isfinite(row[p]).sum() >= ind_min}
            order = sorted(med, key=lambda x: -med[x])
            rank_of_day[d] = {nm: (i + 1, i / max(len(order), 1) * 100) for i, nm in enumerate(order)}
        R = C.pct_change()
        n_lead = n_own = n_noind = 0
        for i in range(n0):
            nm = ind[cols[i]] if cols[i] >= 0 else ""
            rk = rank_of_day[day[i]].get(nm)
            if rk:
                rank_txt[i] = rk[0]
            if rk and rk[1] < ind_top:
                continue
            rs_i = sig["RS"].iat[i] if "RS" in sig else np.nan
            if lead_rs and pd.notna(rs_i) and rs_i >= lead_rs:
                n_lead += 1
                continue
            if corr_min > 0 and nm and day[i] >= 60:
                peers = [p for p in pos_of.get(nm, []) if p != cols[i]]
                if len(peers) >= 2:
                    w = R.iloc[day[i] - 59:day[i] + 1]
                    c_ = w.iloc[:, cols[i]].corr(w.iloc[:, peers].mean(axis=1))
                    if pd.notna(c_) and c_ < corr_min:
                        n_own += 1
                        continue
            keep[i] = False
            n_noind += not rk
        extra = (f"; {n_lead} leaders + {n_own} own movers let through" if (n_lead or n_own) else "")
        parts.append(f"industry in the top {ind_top:g}%{extra}")
    stg = np.full(n0, "?", dtype=object)
    if stages:
        e10, e20 = C.ewm(span=10, adjust=False).mean(), C.ewm(span=20, adjust=False).mean()
        s50, s200 = C.rolling(50).mean(), C.rolling(200, min_periods=150).mean()
        pc = C.shift(1)
        tr = np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy())
        atr = pd.DataFrame(tr, index=C.index, columns=C.columns).ewm(alpha=1 / 14, adjust=False).mean()
        g_ = lambda X: X.to_numpy(float)[day, cols]
        for i, v in enumerate(zip(g_(C), g_(e10), g_(e20), g_(s50), g_(s200), g_(atr))):
            stg[i] = rs_classify_stage(*v)
        keep &= np.isin(stg, list(stages))
        parts.append("stock stage " + "/".join(stages))
    out = sig[keep].reset_index(drop=True)
    if ind_top > 0:
        out["Ind rank"] = rank_txt[keep]
    if stages:
        out["Stock stage"] = stg[keep]
    return out, f"🔻 Funnel ({'; '.join(parts)}): {n0:,} signals → **{int(keep.sum()):,} kept**."


def bt_group_market_filter(sig, P, E, uni, groups, market, trigger=False, above=(), below=(), levels=("Sector",),
                           rs_min=0, corr_min=0.0):
    """Keep only the signals whose group ETF / SPY passed on the signal day (the day before a buy-stop fill).
    Returns (kept signals, a one-line note)."""
    dates = P["Close"].index
    day = sig["idx"] - 1 if trigger else sig["idx"]                 # the last close known when deciding
    when = dates[day.clip(lower=0).to_numpy()]
    keep = pd.Series(True, index=sig.index)
    parts = []
    if groups != "All groups":
        hs = etf_status_history(E)[0].reindex(dates).ffill(limit=5)
        info = bt_sector_info(sorted(sig["Symbol"].unique()), uni)
        ok_set = set(ETF_OK[groups])
        g_ok, no_grp, g_txt, n_own = [], 0, [], 0
        ret_s = P["Close"].pct_change()
        ret_e = E["Close"].reindex(dates).pct_change()
        corr_cache = {}

        def corr_at(sym, etf, d):                      # 60-day correlation of daily returns, up to day d
            k_ = (sym, etf)
            if k_ not in corr_cache:
                corr_cache[k_] = (ret_s[sym].rolling(GRP_CORR_DAYS, min_periods=40).corr(ret_e[etf])
                                  if sym in ret_s and etf in ret_e else None)
            c_ = corr_cache[k_]
            return c_.at[d] if c_ is not None and d in c_.index else np.nan
        rs_col = sig["RS"] if "RS" in sig else pd.Series(np.nan, index=sig.index)
        for sym, d, rs_ in zip(sig["Symbol"], when, rs_col):
            sec = info["Sector"].get(sym) if sym in info.index else None
            ind = info["Industry"].get(sym) if sym in info.index else None
            ok, has, txt = group_check(sec, ind, levels,
                                       lambda t: hs.at[d, t] if t in hs.columns else None, ok_set, sym)
            own = ""
            if not ok and (rs_min or corr_min):
                cs = [corr_at(sym, t, d) for t in group_used_etfs(sec, ind, levels, sym)]
                cs = [c for c in cs if pd.notna(c)]
                own = own_mover(rs_, max(cs) if cs else np.nan, rs_min, corr_min)
            if own:
                ok = True
                n_own += 1
            no_grp += not has
            g_ok.append(ok)
            g_txt.append((f"{own} · " if own else "") + (f"{txt} ({d:%b %d})" if has else "no group"))
        sig = sig.assign(Group=g_txt)
        g_ok = pd.Series(g_ok, index=sig.index)
        parts.append(f"group filter ({groups} · {' + '.join(levels)}) removed {int((~g_ok).sum()):,}"
                     + (f" ({no_grp:,} with no {'/'.join(levels).lower()} ETF)" if no_grp else "")
                     + (f", let {n_own:,} through as leaders / own movers" if n_own else ""))
        keep &= g_ok
    if market != "Off" and "SPY" in E["Close"]:
        if market != MKT_CUSTOM:
            above = below = ()
        ok_h = market_ok_history(E["Close"], market, above, below).astype(float).reindex(dates).ffill(limit=5)
        m_ok = pd.Series(ok_h.reindex(when).fillna(0).to_numpy() > 0.5, index=sig.index)
        parts.append(f"market filter ({market_filter_label(market, above, below)}) removed "
                     f"{int((keep & ~m_ok).sum()):,}" + (" more" if groups != "All groups" else ""))
        keep &= m_ok
    note = (f"🗂 {len(sig):,} signals → **{int(keep.sum()):,} kept**: " + "; ".join(parts) + ".") if parts else ""
    return sig[keep].reset_index(drop=True), note


def bt_sector_info(tickers, uni):
    """Sector and industry for the backtest's stocks: from the chosen stock list, else the US / Asian screeners."""
    sources = []
    try:
        if uni in BT_LIST_UNIVERSES:
            sources.append(base_universe(BT_LIST_UNIVERSES[uni], "")[1])
    except Exception:
        pass
    sources.append(ss.get("meta"))                          # the Stock Scanner's last list
    return sector_lookup(tickers, sources)


def month_list_text(months):
    """['2026-03', '2026-04', '2026-06'] -> 'Mar–Apr 2026, Jun 2026'."""
    ms = sorted(pd.Period(m, "M") for m in months)
    runs, out = [], []
    for p in ms:
        if runs and p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    for a, b in runs:
        f = lambda p, fmt: p.strftime(fmt)
        out.append(f(a, "%b %Y") if a == b else
                   f"{f(a, '%b')}–{f(b, '%b %Y')}" if a.year == b.year else f"{f(a, '%b %Y')}–{f(b, '%b %Y')}")
    return ", ".join(out)


def month_picker(period_days):
    """📅 Months popover (Backtest): a checkbox per calendar month of the test period. Returns the set of ticked
    "YYYY-MM" months, or None when all are ticked (no filter)."""
    ss_ = st.session_state
    today = pd.Timestamp.today().normalize()
    first = today - pd.offsets.BDay(period_days + 1)
    months = [p.strftime("%Y-%m") for p in pd.period_range(first.to_period("M"), today.to_period("M"), freq="M")]
    ss_.setdefault("bt_months_off", [])
    off = set(ss_["bt_months_off"])

    def _toggle(m):
        cur = set(ss_["bt_months_off"])
        cur.discard(m) if ss_.get(f"bmon_{m}") else cur.add(m)
        ss_["bt_months_off"] = sorted(cur)

    def _set(on, which=None):
        ms = months if which is None else which
        cur = set(ss_["bt_months_off"])
        ss_["bt_months_off"] = sorted(cur - set(ms) if on else cur | set(ms))

    on = [m for m in months if m not in off]
    lbl = "all" if len(on) == len(months) else (month_list_text(on) if 0 < len(on) <= 2 else f"{len(on)} of {len(months)}")
    with st.sidebar.popover(f"📅 Months · {lbl}", use_container_width=True,
                            help="Test only some months of the test period — e.g. untick all, then tick one month to "
                                 "see how the setup did in that month. Trades that start in a ticked month are "
                                 "followed to their exit, even into the next month."):
        b1, b2 = st.columns(2)
        b1.button("Tick all", on_click=_set, args=(True,), use_container_width=True, key="bmon__all")
        b2.button("Untick all", on_click=_set, args=(False,), use_container_width=True, key="bmon__none")
        for yr in sorted({m[:4] for m in months}, reverse=True):
            ym = [m for m in months if m[:4] == yr]
            st.markdown(f"**{yr}**")
            cols = st.columns(3)
            for i, m in enumerate(ym):
                ss_[f"bmon_{m}"] = m not in off
                cols[i % 3].checkbox(pd.Period(m, "M").strftime("%b"), key=f"bmon_{m}", on_change=_toggle, args=(m,))
        st.caption("Months at the edges are partly inside the test period.")
    return None if len(on) == len(months) else set(on)


def sector_picker(sec_names, prefix, help_txt, where=None, inline=False):
    """🏷 Sectors popover: a checkbox per sector; the choice (ss['sec_excl']) is shared by the scanner and backtest."""
    where = where or st
    ss_ = st.session_state
    ss_.setdefault("sec_excl", list(DEFAULT_SECTOR_EXCL))

    def _toggle(name):
        ex = [x for x in ss_["sec_excl"] if x.lower() != name.lower()]
        ss_["sec_excl"] = ex if ss_.get(f"{prefix}{name}") else ex + [name]

    def _all(on):
        names_l = {n.lower() for n in sec_names}
        others = [x for x in ss_["sec_excl"] if x.lower() not in names_l]
        ss_["sec_excl"] = others if on else others + list(sec_names)

    excl_l = {x.lower() for x in ss_["sec_excl"]}
    n_in = sum(n.lower() not in excl_l for n in sec_names)
    lbl = "all" if n_in == len(sec_names) else f"{n_in} of {len(sec_names)}"
    if inline:                                           # right inside a panel (e.g. Edit pre-filter)
        where.markdown(f"**Sectors** · {lbl}", help=help_txt)
        b1, b2 = where.columns(2)
        b1.button("Tick all", on_click=_all, args=(True,), use_container_width=True, key=f"{prefix}_all")
        b2.button("Untick all", on_click=_all, args=(False,), use_container_width=True, key=f"{prefix}_none")
        for name in sec_names:
            ss_[f"{prefix}{name}"] = name.lower() not in excl_l
            where.checkbox(name, key=f"{prefix}{name}", on_change=_toggle, args=(name,))
        return
    with where.popover(f"🏷 Sectors · {lbl}", use_container_width=True, help=help_txt):
        b1, b2 = st.columns(2)
        b1.button("Tick all", on_click=_all, args=(True,), use_container_width=True, key=f"{prefix}_all")
        b2.button("Untick all", on_click=_all, args=(False,), use_container_width=True, key=f"{prefix}_none")
        for name in sec_names:
            ss_[f"{prefix}{name}"] = name.lower() not in excl_l
            st.checkbox(name, key=f"{prefix}{name}", on_change=_toggle, args=(name,))
        st.caption("Unticked sectors are left out before prices are downloaded.")


def is_excluded_sector(sector, industry="", excl=None):
    excl = {x.lower() for x in (st.session_state.get("sec_excl", DEFAULT_SECTOR_EXCL) if excl is None else excl)}
    if not excl:
        return False
    sec, ind = str(sector or "").strip().lower(), str(industry or "").lower()
    if sec in excl:
        return True
    # REITs / utilities that a list files under another sector (e.g. REITs under "Finance")
    if "real estate" in excl and re.search(r"real estate|\breits?\b", ind):
        return True
    return "utilities" in excl and "utilit" in ind


def drop_excluded(tickers, meta=None):
    """Remove Real Estate and Utilities stocks (sector or industry) from a ticker list."""
    if not tickers:
        return tickers
    info = sector_lookup(tickers, [meta] if meta is not None else [])
    bad = {t for t, sec, ind in zip(info.index, info["Sector"], info["Industry"]) if is_excluded_sector(sec, ind)}
    return [t for t in tickers if t not in bad]


def bt_universe_list(name, max_n=1500, cmin=None, cmax=None, vmin=0):
    """The stocks to test: filtered by market cap / volume, then the `max_n` largest companies."""
    m = bt_universe_meta(name)
    m = m[~np.array([is_excluded_sector(a, b) for a, b in zip(m["Sector"].fillna(""), m["Industry"].fillna(""))],
                    dtype=bool)]
    if name not in MARKETS and m["_last"].notna().any():
        m = m[m["_last"].fillna(0) >= 5]                      # skip US penny stocks
    if cmin:
        m = m[m["Mkt cap $B"] >= cmin]
    if cmax:
        m = m[m["Mkt cap $B"] <= cmax]
    if vmin:
        m = m[m["_vol"] >= vmin]
    return m["Mkt cap $B"].sort_values(ascending=False, na_position="last").index[:max_n].tolist()


def bt_portfolio(tr, P, acct, risk_pct, max_pos_pct, max_open, priority, compound, unit_pct=None):
    """Replay the backtest's trades through ONE account with limited cash.
    Each entry day: positions that exited earlier free their cash, then new signals are taken in priority order
    while there is room (max open positions) and cash. Size = risk % of the account ÷ (entry − stop), capped by
    the max position size and by the cash available.
    Pilot + add (trades with 'Legs', and unit_pct): the pilot is 1 unit (unit_pct % of the account); each add is
    bought at its day's close if there is cash for it (an add doesn't use a position slot)."""
    C = P["Close"]
    dates = C.index
    t = tr.copy()
    t["_risk"] = (t["Entry"] - t["Stop"]).abs()
    t["_side"] = np.where(t["Entry"] > t["Stop"], 1, -1)
    pyr = "Legs" in t and bool(unit_pct)
    key = {"Highest RS first": ("RS", False), "Tightest stop first": ("Risk %", True),
           "Alphabetical": ("Symbol", True)}[priority]
    t = t.sort_values(["Entry_date", key[0]], ascending=[True, key[1]], kind="mergesort")
    cash, equity_real = float(acct), float(acct)
    open_pos, taken, skipped, adds = [], [], [], []           # adds: pending (day, price, position)

    def close_until(day):
        nonlocal cash, equity_real
        keep = []
        for p in open_pos:
            if p["Exit_date"] < day:
                pnl = sum(q * (p["Exit"] - px_) * p["_side"] for _, px_, q in p["_legs"])
                cash += sum(q * px_ for _, px_, q in p["_legs"]) + pnl
                equity_real += pnl
            else:
                keep.append(p)
        open_pos[:] = keep

    def adds_until(day):                                     # buy the adds of the days before `day`
        nonlocal cash
        adds.sort(key=lambda x: x[0])
        while adds and adds[0][0] < day:
            d_, px_, p = adds.pop(0)
            close_until(d_)
            if not any(p is x for x in open_pos):
                continue
            base = equity_real if compound else float(acct)
            mx = float(p.get("Size ×", 1.0)) if "Size ×" in p else 1.0
            q = min(np.floor(base * unit_pct / 100 * mx / px_), np.floor(cash / px_))
            if q >= 1:
                cash -= q * px_
                p["_legs"].append((d_, px_, q))
            else:
                p["Adds skipped"] = p.get("Adds skipped", 0) + 1

    for rd in t.to_dict("records"):
        day = rd["Entry_date"]
        adds_until(day)
        close_until(day)
        if any(p["Symbol"] == rd["Symbol"] for p in open_pos):
            skipped.append({**rd, "Reason": "already holding this stock"})
            continue
        if len(open_pos) >= max_open:
            skipped.append({**rd, "Reason": f"account full ({max_open} open)"})
            continue
        base = equity_real if compound else float(acct)
        if not (np.isfinite(rd["_risk"]) and rd["_risk"] > 0 and np.isfinite(rd["Entry"]) and rd["Entry"] > 0):
            continue
        mx = float(rd.get("Size ×", 1.0) or 0.0) if "Size ×" in rd else 1.0
        if pyr:
            sh = np.floor(base * unit_pct / 100 * mx / rd["Entry"])
        else:
            sh = np.floor(base * risk_pct / 100 * mx / rd["_risk"])
            sh = min(sh, np.floor(base * max_pos_pct / 100 * mx / rd["Entry"]))
        cash_sh = np.floor(cash / rd["Entry"])
        if cash_sh < sh:
            if cash_sh < max(1, sh * 0.25):                  # less than a quarter of the planned size left
                skipped.append({**rd, "Reason": "not enough cash"})
                continue
            sh = cash_sh
        if sh < 1:
            skipped.append({**rd, "Reason": "position too small"})
            continue
        cost = sh * rd["Entry"]
        cash -= cost
        pos = {**rd, "_legs": [(day, rd["Entry"], sh)], "Account at entry": equity_real,
               "Size %": cost / equity_real * 100 if equity_real else np.nan}
        open_pos.append(pos)
        taken.append(pos)
        if pyr and isinstance(rd.get("Legs"), list):
            adds.extend((d_, px_, pos) for d_, px_ in rd["Legs"][1:])
    adds_until(pd.Timestamp.max)
    close_until(pd.Timestamp.max)

    tk = pd.DataFrame(taken)
    if tk.empty:
        return tk, pd.DataFrame(skipped), pd.Series(dtype=float), pd.Series(dtype=float)
    tk["Shares"] = [sum(q for _, _, q in lg) for lg in tk["_legs"]]
    tk["Cost"] = [sum(q * px_ for _, px_, q in lg) for lg in tk["_legs"]]
    tk["P&L $"] = [sum(q * (ex - px_) * sd for _, px_, q in lg) for lg, ex, sd in zip(tk["_legs"], tk["Exit"], tk["_side"])]
    tk["Account %"] = tk["P&L $"] / tk["Account at entry"] * 100
    if pyr:
        tk["Units"] = [len(lg) for lg in tk["_legs"]]
        tk["Avg cost"] = tk["Cost"] / tk["Shares"]
        tk["Size %"] = tk["Cost"] / tk["Account at entry"] * 100          # the full position, all units
        tk["Legs"] = [[(d_, px_) for d_, px_, _ in lg] for lg in tk["_legs"]]
    # daily account value, marked to market with the closing prices
    i0 = dates.get_indexer([tk["Entry_date"].min()])[0]
    n = len(dates)
    cash_flow = np.zeros(n)
    value = np.zeros(n)
    for p in tk.to_dict("records"):
        ix = dates.get_loc(p["Exit_date"])
        for d_, px_, q in p["_legs"]:
            ie = dates.get_loc(d_)
            cost_ = q * px_
            cash_flow[ie] -= cost_
            cash_flow[ix] += cost_ + q * (p["Exit"] - px_) * p["_side"]
            if ix > ie:
                pxs = C[p["Symbol"]].iloc[ie:ix].ffill().to_numpy(float)
                value[ie:ix] += cost_ + q * (pxs - px_) * p["_side"]
    eq = acct + np.cumsum(cash_flow) + value
    eq = pd.Series(eq[i0:], index=dates[i0:])
    held = np.zeros(n)
    for p in tk.to_dict("records"):
        held[dates.get_loc(p["Entry_date"]):dates.get_loc(p["Exit_date"])] += 1
    held = pd.Series(held[i0:], index=dates[i0:])
    return tk.drop(columns=["_legs"]), pd.DataFrame(skipped), eq, held


def bt_account_view(res, tr, acct, risk_pct, max_pos):
    """The 'Account simulation' view of the Backtest results."""
    P = res.get("P")
    if P is None:
        st.info("Run the backtest again to see the account simulation.")
        return
    st.caption("The same trades as 'Every signal', but run through **one account with limited cash**: a trade is "
               "only taken if there is room and money for it. Cash comes back when a position is sold.")
    c1, c2, c3 = st.columns([1, 1.4, 1], vertical_alignment="bottom")
    c1.number_input("Max open positions", 1, 100, key="bt_max_open",
                    help="The most stocks held at the same time. New signals are skipped while the account is full.")
    c2.selectbox("When there are more signals than room", BT_PRIORITY, key="bt_priority",
                 help="Which signals of the same day get the money first. RS = relative strength on the signal day.")
    c3.checkbox("Compound", key="bt_compound",
                help="On: each trade risks a % of the CURRENT account (profits make positions bigger). "
                     "Off: always a % of the starting account.")
    unit_pct = float(ss["bt_pyr_unit"]) if "Legs" in tr else None
    tk, sk, eq, held = bt_portfolio(tr, P, acct, risk_pct, max_pos, int(ss["bt_max_open"]), ss["bt_priority"],
                                    bool(ss["bt_compound"]), unit_pct)
    if tk.empty:
        st.warning("No trade could be taken with these settings — try a bigger account or more open positions.")
        return
    final = float(eq.iloc[-1])
    dd = float(((eq / eq.cummax()) - 1).min() * 100)
    wins = (tk["P&L $"] > 0).mean() * 100
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 12)
    cagr = ((final / acct) ** (1 / years) - 1) * 100 if final > 0 else -100.0
    tone = lambda v, good: "g" if v > good else "r" if v < 0 else "y"
    reasons = sk["Reason"].value_counts().to_dict() if len(sk) else {}
    render_cards([
        ("Account now", f"${final:,.0f}", f"started with ${acct:,.0f}", tone(final - acct, 0)),
        ("Return", f"{(final / acct - 1) * 100:+.1f}%", f"{cagr:+.0f}% a year" if years >= 0.9 else "over the test",
         tone(final - acct, 0)),
        ("Trades taken", f"{len(tk):,}", f"of {len(tr):,} signals · {len(sk):,} skipped", "n"),
        ("Win rate", f"{wins:.0f}%", f"avg {tk['Account %'].mean():+.2f}% of the account per trade", tone(wins - 40, 0)),
        ("Max drawdown", f"{dd:.1f}%", "worst drop from a peak (daily)", "r" if dd < -10 else "n"),
        ("Avg positions", f"{held.mean():.1f}", f"max {int(held.max())} open at once", "n"),
    ])
    if reasons:
        st.caption("Skipped: " + " · ".join(f"{v} × {k}" for k, v in reasons.items()) +
                   ". Raise **Max open positions**, lower **Risk %** or **Max position size** to fit more trades.")
    fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, mode="lines", line=dict(color="#26a69a", width=2),
                               name="Account", hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=acct, line_dash="dot", line_color="rgba(128,128,128,.6)")
    fig.add_trace(go.Bar(x=held.index, y=held.values, name="Open positions", yaxis="y2",
                         marker_color="rgba(59,130,246,.35)",
                         hovertemplate="%{x|%b %d, %Y}<br>%{y:.0f} open<extra></extra>"))
    fig.update_layout(title="Account value (daily, marked to market) and open positions", height=380,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False, bargap=0,
                      yaxis=dict(title="$", tickformat="$,.0f"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, rangemode="tozero",
                                  range=[0, max(3, held.max()) * 3], title="open", tickformat="d"))
    st.plotly_chart(fig, use_container_width=True)

    tk_show = tk.assign(**{"Position $": tk["Cost"], "Position %": tk["Size %"]})
    bt_detail_sections(tk_show, res, "bta", "Trades taken", month_col="P&L $", csv_name="account_trades.csv")
    if len(sk):
        with st.expander(f"Skipped signals ({len(sk):,})"):
            sk_show = sk.rename(columns={"Entry_date": "Entry date"})
            sk_show = sk_show[[c for c in ["Symbol", "Entry date", "Entry", "Stop", "RS", "Reason"] if c in sk_show]]
            st.dataframe(sk_show.sort_values("Entry date", ascending=False), hide_index=True, use_container_width=True,
                         column_config={"Entry date": st.column_config.DateColumn(format="MMM D, YYYY"),
                                        "Entry": st.column_config.NumberColumn(format="%.2f"),
                                        "Stop": st.column_config.NumberColumn(format="%.2f"),
                                        "RS": st.column_config.NumberColumn(format="%d")})
    st.caption("Cash from a sale can be used from the next trading day. Commissions and slippage are not included.")


def bt_detail_sections(tr, res, k, title, hist_in=None, month_col="R", csv_name="backtest_trades.csv"):
    """The charts and tables under the Backtest cards — shared by 'Every signal' and 'Account simulation'.
    tr: the trades to show (with Shares, Position $, P&L $ …) · k: prefix that keeps each view's widgets apart ·
    hist_in: a column to draw the R histogram into (None = put it at the start of the next row)."""
    hist = px.histogram(tr, x="R", nbins=30, color=(tr["R"] > 0).map({True: "Win", False: "Loss"}),
                        color_discrete_map={"Win": "#26a69a", "Loss": "#ef5350"})
    hist.update_layout(title="Result of each trade (R)", height=320, margin=dict(l=10, r=10, t=40, b=10),
                       showlegend=False, yaxis_title="Trades", bargap=0.05)
    if hist_in is not None:
        hist_in.plotly_chart(hist, use_container_width=True)

    by_col = ("Strategy" if "Strategy" in tr and tr["Strategy"].nunique() > 1 else
              "Fired by" if "Fired by" in tr and tr["Fired by"].nunique() > 1 else None)
    if by_col:
        st.markdown("**By strategy**" if by_col == "Strategy" else "**By the strategy that fired last** (the others "
                    "had already signalled)")
        rows_ = []
        for nm, g_ in tr.groupby(by_col):
            r_ = g_["R"]
            w_, l_ = g_[r_ > 0], g_[r_ <= 0]
            rows_.append({"Strategy": nm, "Trades": len(g_), "Win %": (r_ > 0).mean() * 100, "Avg R": r_.mean(),
                          "Total R": r_.sum(), "P&L $": g_["P&L $"].sum() if "P&L $" in g_ else np.nan,
                          "Profit factor": _profit_factor(w_, l_), "Avg days": g_["Days"].mean()})
        st.dataframe(pd.DataFrame(rows_).sort_values("Total R", ascending=False), hide_index=True,
                     use_container_width=True,
                     column_config={"Win %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Avg R": st.column_config.NumberColumn(format="%+.2f"),
                                    "Total R": st.column_config.NumberColumn(format="%+.1f"),
                                    "P&L $": st.column_config.NumberColumn(format="$%.0f"),
                                    "Profit factor": st.column_config.NumberColumn(format="%.2f"),
                                    "Avg days": st.column_config.NumberColumn(format="%.0f")})
    by = tr.groupby("Exit_reason").agg(Trades=("R", "size"), **{"Avg R": ("R", "mean")},
                                       **{"Win %": ("R", lambda r: (r > 0).mean() * 100)}).reset_index()
    mon = tr.assign(Month=tr["Exit_date"].dt.to_period("M").astype(str)).groupby("Month")[month_col].sum().reset_index()
    if hist_in is None:                                 # no spot for the histogram yet: it leads this row
        c0, c1, c2 = st.columns([1, 1.15, 1.25])
        c0.plotly_chart(hist, use_container_width=True)
    else:
        c1, c2 = st.columns([1, 1.6])
    c1.markdown("**How trades ended**")
    c1.dataframe(by.rename(columns={"Exit_reason": "Exit"}), hide_index=True, use_container_width=True,
                 column_config={"Avg R": st.column_config.NumberColumn(format="%+.2f"),
                                "Win %": st.column_config.NumberColumn(format="%.0f%%")})
    mf = go.Figure(go.Bar(x=mon["Month"], y=mon[month_col],
                          marker_color=np.where(mon[month_col] >= 0, "#26a69a", "#ef5350")))
    mf.update_layout(title=f"{'P&L' if month_col == 'P&L $' else 'R'} by month", height=260,
                     margin=dict(l=10, r=10, t=40, b=10), yaxis_tickformat="$,.0f" if month_col == "P&L $" else None)
    c2.plotly_chart(mf, use_container_width=True)

    # ---- wins and losses per sector / industry ----
    sec_all = bt_sector_info(tr["Symbol"].tolist(), res["uni"])
    g_by = tr.assign(Sector=tr["Symbol"].map(sec_all["Sector"]).replace("", "Unknown").fillna("Unknown"),
                     Industry=tr["Symbol"].map(sec_all["Industry"]).replace("", "Unknown").fillna("Unknown"),
                     Win=tr["R"] > 0)
    s1, s2 = st.columns([3, 1.2], vertical_alignment="bottom")
    s1.markdown("**Winning trades by sector / industry**")
    ss.setdefault(f"{k}_grp", "Sector")
    grp = (s2.segmented_control("Group by", ["Sector", "Industry"], key=f"{k}_grp", label_visibility="collapsed")
           if hasattr(st, "segmented_control") else
           s2.radio("Group by", ["Sector", "Industry"], key=f"{k}_grp", horizontal=True, label_visibility="collapsed")) \
        or "Sector"
    agg = g_by.groupby(grp).agg(Wins=("Win", "sum"), Trades=("Win", "size"), AvgR=("R", "mean"),
                                PnL=("P&L $", "sum")).reset_index()
    agg["Losses"] = agg["Trades"] - agg["Wins"]
    agg["Win %"] = agg["Wins"] / agg["Trades"] * 100
    agg = agg.sort_values(["Wins", "Win %"], ascending=[False, False])
    more = max(0, len(agg) - 25)
    agg = agg.head(25).iloc[::-1]                        # biggest at the top of a horizontal bar chart
    hover = [f"{w:.0f} wins / {t:.0f} trades ({wp:.0f}%)<br>avg {r:+.2f}R · P&L ${p:,.0f}"
             for w, t, wp, r, p in zip(agg["Wins"], agg["Trades"], agg["Win %"], agg["AvgR"], agg["PnL"])]
    gf = go.Figure([
        go.Bar(y=agg[grp], x=agg["Wins"], orientation="h", name="Wins", marker_color="#26a69a",
               text=[f"{w:.0f}" if w else "" for w in agg["Wins"]], textposition="inside", textangle=0,
               insidetextanchor="middle", hovertext=hover, hoverinfo="text"),
        go.Bar(y=agg[grp], x=agg["Losses"], orientation="h", name="Losses", marker_color="rgba(239,83,80,.55)",
               hovertext=hover, hoverinfo="text"),
    ])
    gf.update_traces(cliponaxis=False)
    for yv, wp, t in zip(agg[grp], agg["Win %"], agg["Trades"]):
        gf.add_annotation(x=t, y=yv, text=f" {wp:.0f}%", showarrow=False, xanchor="left",
                          font=dict(size=11, color="#26a69a" if wp >= 50 else "rgba(160,160,160,1)"))
    gf.update_layout(barmode="stack", height=max(240, 26 * len(agg) + 70), margin=dict(l=10, r=40, t=10, b=10),
                     legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", traceorder="normal"),
                     xaxis_title="Trades",
                     yaxis=dict(automargin=True))
    st.plotly_chart(gf, use_container_width=True)
    st.caption(f"Green = winning trades, red = losing trades; the % is the win rate. Sorted by the number of wins"
               + (f" · top 25 shown ({more} more {grp.lower()} groups)" if more else "")
               + ". Hover for the average R and P&L.")

    st.markdown(f"**{title}** ({len(tr):,}) — click a row to see it on the chart")
    show = tr.sort_values("Entry_date", ascending=False).rename(
        columns={"Entry_date": "Entry date", "Exit_date": "Exit date", "Exit_reason": "Exit reason",
                 "Stop_basis": "Stop basis"}).reset_index(drop=True)
    if "Trigger" not in show:
        show["Trigger"] = np.nan
    if "Stop basis" not in show:
        show["Stop basis"] = ""
    sec = sec_all
    show["Sector"] = show["Symbol"].map(sec["Sector"]).fillna("")
    show["Industry"] = show["Symbol"].map(sec["Industry"]).fillna("")
    show = show[[c for c in ["Symbol", "Strategy", "Fired by", "Also", "Sector", "Industry", "Group", "Signal", "Trigger", "Entry date", "Entry", "Stop",
                             "Units", "Avg cost", "Final stop", "Adds skipped", "Shares", "Position $", "Exit date", "Exit", "Return %", "R", "P&L $", "Account %", "Days",
                             "Exit reason", "Trim 1", "Trim 1 date", "Trim 2", "Trim 2 date", "Last exit",
                             "Risk %", "Position %", "Account at entry", "RS", "Stop basis"] if c in show]]
    cfg = {"Group": st.column_config.TextColumn(help="The ETF the group filter checked for this stock, its status and the "
                                                   "day it was checked (the signal day; for buy-stop entries the day "
                                                   "before). 🔥 in play · 🌱 emerging · ⚠️ cooling · ❄️ weak"),
           "Signal": st.column_config.DateColumn(format="MMM D, YYYY"),
           "Entry date": st.column_config.DateColumn(format="MMM D, YYYY"),
           "Exit date": st.column_config.DateColumn(format="MMM D, YYYY"),
           "Entry": st.column_config.NumberColumn(format="%.2f"), "Stop": st.column_config.NumberColumn(format="%.2f"),
           "Trigger": st.column_config.NumberColumn(format="%.2f", help="The breakout level the scan was watching"),
           "Exit": st.column_config.NumberColumn(format="%.2f", help="Average price of everything sold "
                                                 "(with scaling out: ⅓ at each trim + the rest)"),
           "Trim 1": st.column_config.NumberColumn("⅓ at ADR", format="%.2f", help="⅓ sold here — up ADR % × 2"),
           "Trim 2": st.column_config.NumberColumn("⅓ at ATR", format="%.2f",
                                                   help="⅓ sold here — ATRs above the moving average"),
           "Trim 1 date": st.column_config.DateColumn("ADR trim date", format="MMM D, YYYY"),
           "Trim 2 date": st.column_config.DateColumn("ATR trim date", format="MMM D, YYYY"),
           "Last exit": st.column_config.NumberColumn(format="%.2f", help="Where the rest was sold"),
           "Return %": st.column_config.NumberColumn(format="%+.1f%%"), "R": st.column_config.NumberColumn(format="%+.2f"),
           "Risk %": st.column_config.NumberColumn(format="%.1f%%"), "RS": st.column_config.NumberColumn(format="%d"),
           "Shares": st.column_config.NumberColumn(format="localized"),
           "Position $": st.column_config.NumberColumn(format="dollar"),
           "Position %": st.column_config.NumberColumn("Size %", format="%.0f%%", help="Position as % of the account"),
           "P&L $": st.column_config.NumberColumn(format="dollar"),
           "Account %": st.column_config.NumberColumn(format="%+.2f%%", help="P&L as % of the account at entry"),
           "Account at entry": st.column_config.NumberColumn(format="dollar"),
           "Units": st.column_config.NumberColumn(format="%d", help="Pilot + adds bought"),
           "Avg cost": st.column_config.NumberColumn(format="%.2f", help="Average price of all units"),
           "Final stop": st.column_config.NumberColumn(format="%.2f", help="Where the stop was after the last add"),
           "Adds skipped": st.column_config.NumberColumn(format="%d", help="Adds not bought — not enough cash")}
    sty = show.style.map(lambda v: "color:#26a69a" if isinstance(v, float) and v > 0 else
                         "color:#ef5350" if isinstance(v, float) and v < 0 else "", subset=["Return %", "R", "P&L $"])
    kw = dict(hide_index=True, use_container_width=True, column_config=cfg, on_select="rerun", key=f"{k}_tbl",
              height=min(35 * (len(show) + 1) + 3, 420))
    try:
        ev = st.dataframe(sty, selection_mode="single-cell", **kw)
        cells = ev.selection.get("cells", []) if hasattr(ev.selection, "get") else getattr(ev.selection, "cells", [])
        pick_i = (cells[0][0] if isinstance(cells[0], (list, tuple)) else cells[0].get("row")) if cells else 0
    except Exception:
        ev = st.dataframe(sty, selection_mode="single-row", **kw)
        rows_ = list(getattr(ev.selection, "rows", []) or [])
        pick_i = rows_[0] if rows_ else 0
    st.download_button("⬇ Trades CSV", show.to_csv(index=False).encode(), csv_name, "text/csv", key=f"dl_{k}_csv")

    P = res.get("P") or ss.get("bt_panels")
    if P is None or not len(show):
        return
    t = show.iloc[min(pick_i or 0, len(show) - 1)]
    sym = t["Symbol"]
    if sym not in P["Close"]:
        return
    ss.setdefault(f"{k}_chart_src", "TradingView")
    h1, h2 = st.columns([3, 1.4], vertical_alignment="bottom")
    h1.markdown(f"**{sym} · {t['R']:+.2f}R ({t['Return %']:+.1f}%) · {t['Days']} days** — "
                f"entry {t['Entry']:.2f} on {t['Entry date']:%b %d, %Y} · stop {t['Stop']:.2f} · "
                f"exit {t['Exit']:.2f} on {t['Exit date']:%b %d, %Y} ({t['Exit reason']})")
    src = (h2.segmented_control("Chart type", ["TradingView", "Trade chart"], key=f"{k}_chart_src",
                                label_visibility="collapsed") if hasattr(st, "segmented_control") else
           h2.radio("Chart type", ["TradingView", "Trade chart"], key=f"{k}_chart_src", horizontal=True,
                    label_visibility="collapsed")) or "TradingView"
    if src == "TradingView":
        ex_map = get_exchange_map()
        tpls, active = load_templates()
        tpl = {**DEFAULT_TPL, **tpls.get(ss.get("tpl_name", active), tpls["Default"])}
        syms = list(dict.fromkeys(show["Symbol"]))[:100]
        tv_sym = tv_symbol(sym, ex_map)
        tradingview_chart(tv_sym, tpl, [tv_symbol(x, ex_map) for x in syms])
        st.caption(f"Live chart from TradingView · template **{ss.get('tpl_name', active)}** (change it on the Stock "
                   f"Scanner tab) · switch to **Trade chart** to see the entry, exit and stop marked · "
                   f"[Open {tv_sym} on TradingView ↗]({tv_open_url(tv_sym, tpl.get('layout_url', ''))})")
        st.caption("Past results don't guarantee future ones. The test ignores commissions, slippage and position "
                   "sizing limits, and uses today's stock list — stocks that were delisted earlier aren't in it "
                   "(survivorship bias), so real results are usually somewhat worse.")
        return
    a = max(0, P["Close"].index.get_loc(t["Entry date"]) - 60)
    b = min(len(P["Close"]), P["Close"].index.get_loc(t["Exit date"]) + 25)
    sl = {k: v[sym].iloc[a:b] for k, v in P.items()}
    cf = go.Figure(go.Candlestick(x=sl["Close"].index, open=sl["Open"], high=sl["High"], low=sl["Low"],
                                  close=sl["Close"], increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                                  name=sym, showlegend=False))
    full_c = P["Close"][sym]                                 # averages from the full history, then cut to the window
    for w, colr in [(10, "#8bc34a"), (20, "#e57373"), (50, "#cddc39"), (200, "#9e9e9e")]:
        ma = full_c.rolling(w).mean().iloc[a:b]
        if ma.notna().any():
            cf.add_trace(go.Scatter(x=ma.index, y=ma, mode="lines", name=f"SMA {w}", line=dict(color=colr, width=1.2),
                                    hovertemplate=f"SMA {w}: %{{y:.2f}}<extra></extra>"))
    cf.add_trace(go.Scatter(x=[t["Entry date"]], y=[t["Entry"]], mode="markers+text", text=["Entry"], showlegend=False,
                            textposition="bottom center", marker=dict(symbol="triangle-up", size=14, color="#2563eb")))
    cf.add_trace(go.Scatter(x=[t["Exit date"]], y=[t["Last exit"] if pd.notna(t.get("Last exit")) else t["Exit"]],
                            mode="markers+text", text=[t["Exit reason"]], showlegend=False,
                            textposition="top center", marker=dict(symbol="x", size=13, color="#f59e0b")))
    cf.add_hline(y=t["Stop"], line_dash="dash", line_color="#ef5350", annotation_text=f"Stop {t['Stop']:.2f}")
    lg_ = None
    if "Legs" in tr:                                         # the units bought (not a table column)
        m_ = tr[(tr["Symbol"] == sym) & (tr["Entry_date"] == t["Entry date"])]
        lg_ = m_["Legs"].iat[0] if len(m_) else None
    if isinstance(lg_, list) and len(lg_) > 1:              # pilot + add: mark each add and the last stop
        cf.add_trace(go.Scatter(x=[d_ for d_, _ in lg_[1:]], y=[p_ for _, p_ in lg_[1:]], mode="markers+text",
                                text=[f"+{i + 1}" for i in range(len(lg_) - 1)], textposition="bottom center",
                                showlegend=False, marker=dict(symbol="triangle-up", size=11, color="#22c55e")))
        if pd.notna(t.get("Final stop")) and t["Final stop"] != t["Stop"]:
            cf.add_hline(y=t["Final stop"], line_dash="dot", line_color="#f97316",
                         annotation_text=f"Raised stop {t['Final stop']:.2f}")
    for q in (1, 2):                                          # scaling out: mark each ⅓ sold
        if pd.notna(t.get(f"Trim {q}")) and pd.notna(t.get(f"Trim {q} date")):
            cf.add_trace(go.Scatter(x=[t[f"Trim {q} date"]], y=[t[f"Trim {q}"]], mode="markers+text",
                                    text=["⅓ at ADR" if q == 1 else "⅓ at ATR"], textposition="top center",
                                    showlegend=False,
                                    marker=dict(symbol="diamond", size=12, color="#a855f7")))
    if pd.notna(t.get("Trigger")):
        x0 = sl["Close"].index[max(0, sl["Close"].index.get_loc(t["Signal"]) - 10)]
        cf.add_shape(type="line", x0=x0, x1=t["Exit date"], y0=t["Trigger"], y1=t["Trigger"],
                     line=dict(color="#22c55e", width=1.6, dash="dot"))
        cf.add_annotation(x=x0, y=t["Trigger"], text=f"Trigger {t['Trigger']:.2f}", showarrow=False,
                          xanchor="left", yanchor="bottom", font=dict(color="#22c55e"))
    cf.add_vline(x=t["Signal"], line_dash="dot", line_color="rgba(128,128,128,.6)", line_width=1)
    cf.add_annotation(x=t["Signal"], y=1, yref="paper", text="Signal", showarrow=False, yanchor="bottom",
                      font=dict(size=11, color="rgba(160,160,160,1)"))
    cf.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])          # no weekend gaps
    cf.update_layout(height=480, xaxis_rangeslider_visible=False, showlegend=True,
                     legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0, font=dict(size=11)),
                     margin=dict(l=10, r=10, t=56, b=10))
    st.plotly_chart(cf, use_container_width=True)
    st.caption("Past results don't guarantee future ones. The test ignores commissions, slippage and position "
               "sizing limits, and uses today's stock list — stocks that were delisted earlier aren't in it "
               "(survivorship bias), so real results are usually somewhat worse.")


def bt_levels():
    lv = [n for n, k in zip(GROUP_LEVELS, ("bt_grp_sec", "bt_grp_ind", "bt_grp_thm")) if ss.get(k)]
    return lv or ["Sector"]


def bt_apply_funnel_preset():
    """The 🔻 Funnel as one click: every long setup, liquid leaders in leading industries, market-stage sizing,
    pilot + add each day it still qualifies."""
    long_setups = [k for k in ("🚩 Tight Flag Pattern", "🔁 EMA Crossback", "✳️ Green Line Breakout", "📈 Shakeout +3",
                               "🌀 VCP (Minervini)", "📏 MA Consolidation", "⛳ High Tight Flag", "🔂 50-Day Reclaim")
                   if k in BT_STRATEGIES]
    ss.update(bt_strategy=BT_COMBINED, bt_combo=long_setups, bt_combo_mode=BT_ANY, bt_minpx=10.0, bt_mindvol=20.0,
              bt_minrs=85, bt_fn_ind_top=30, bt_fn_ind_min=3, bt_fn_lead=95, bt_fn_corr=0.3,
              bt_fn_stages=["1A", "1B", "2A", "2B"], bt_stage_size=True, bt_pyr=True, bt_pyr_rule=PYR_RULES[0])


def backtest_page():
    ss["bt_strategy"] = OLD_BT_NAMES.get(ss.get("bt_strategy"), ss.get("bt_strategy"))
    if ss.get("bt_strategy") not in BT_STRATEGIES and ss.get("bt_strategy") != BT_COMBINED:
        ss["bt_strategy"] = next(iter(BT_STRATEGIES))
    st.sidebar.title("🧪 Backtest")
    strategy = st.sidebar.selectbox("Strategy", list(BT_STRATEGIES) + [BT_COMBINED], key="bt_strategy",
                                    help="Uses your current settings for that scan in the Stock Scanner tab.")
    combo = [strategy]
    if strategy == BT_COMBINED:
        ss["bt_combo"] = [x for x in ss.get("bt_combo", []) if x in BT_STRATEGIES]
        st.sidebar.multiselect("Strategies to combine", list(BT_STRATEGIES), key="bt_combo",
                               placeholder="Pick 2 or more",
                               help="Each strategy finds its own signals with its own scan settings (Stock Scanner "
                                    "tab); they share the trade rules below and one account. The pause rules run "
                                    "per strategy.")
        combo = list(ss["bt_combo"])
        st.sidebar.radio("Trade", [BT_ANY, BT_ALL], key="bt_combo_mode",
                         help="**Any** — every strategy's signals are traded (one account). **All agree** — only a "
                              "stock that every picked strategy signalled within the window below; bought on the "
                              "latest of those signals, with that signal's entry and stop.")
        if ss["bt_combo_mode"] == BT_ALL:
            st.sidebar.number_input("…all signals within N trading days", 0, 60, key="bt_combo_win",
                                    help="0 = the same day. The strategies rarely fire on the very same day, so a "
                                         "window of 5–20 days gives far more trades. With the 'Trigger day' entry "
                                         "the other signals must come at least a day earlier (the order inside a "
                                         "day is unknown), so use 1 or more.")
    if "🔃 First Close Above 50-day" in combo:
        st.sidebar.number_input("First close above the 50-day: after N days under it", 5, 120, step=5, key="bt_f50_days",
                                help="The stock must have closed at or below its 50-day SMA for N sessions in a row, then "
                                     "close above it. 30 = the first close back above in 30 days.")
    uni = st.sidebar.selectbox("Stocks", BT_UNIVERSES, key="bt_universe")
    if uni == "My own tickers":
        st.sidebar.text_area("Tickers", key="bt_tickers", placeholder="NVDA, AMD, 0700.HK, PTT.BK")
    if uni in BT_LIST_UNIVERSES:
        cur = " (US$)" if uni in MARKETS else ""
        c1, c2 = st.sidebar.columns(2)
        c1.number_input(f"Mkt cap from $B{cur}", min_value=0.0, value=None, step=0.5, key="bt_cmin",
                        placeholder="No limit")
        c2.number_input("to $B", min_value=0.0, value=None, step=10.0, key="bt_cmax", placeholder="No limit",
                        help="Leave blank for no limit")
        if uni != "S&P 500":
            st.sidebar.select_slider("Max stocks (largest first)", [250, 500, 1000, 1500, 2000, 3000, 4000, 6000],
                                     key="bt_max_stocks",
                                     help="After the filters above, the biggest companies by market cap. More stocks "
                                          "= more signals but a longer download (roughly 1 minute per 400 stocks for a "
                                          "1-year test).")
        st.sidebar.caption("Market cap is today's value (the stocks are picked by how they look now).")
    try:
        if uni in BT_LIST_UNIVERSES:
            _bsrc = bt_universe_meta(uni)
        elif uni == "Stock Scanner results":
            _bsrc = sector_lookup(ss.get("last_results", [])[:600], [ss.get("meta")])
        else:
            _bsrc = sector_lookup(parse_tickers(ss.get("bt_tickers", "")))
        bt_secs = sorted({str(x).strip() for x in _bsrc["Sector"].dropna() if str(x).strip()})
    except Exception:
        bt_secs = []
    if not bt_secs and uni not in MARKETS:
        try:
            bt_secs = sorted({str(x).strip() for x in get_screener_meta()["Sector"].dropna() if str(x).strip()})
        except Exception:
            bt_secs = []
    if bt_secs:
        sector_picker(bt_secs, "bsecchk_", "Tick the sectors to test. Shared with the Stock Scanner tab.",
                      where=st.sidebar)
    n_liq = sum(float(ss[k]) > 0 for k in ("bt_minpx", "bt_minvol", "bt_mindvol"))
    with st.sidebar.popover("💧 Liquidity" + (f" · {n_liq} on" if n_liq else " · off"), use_container_width=True,
                            help="Only take signals from stocks that were liquid enough ON THE SIGNAL DAY "
                                 "(50-day averages up to that day — no hindsight). 0 = off."):
        st.number_input("Price ≥ \\$", 0.0, 1000.0, step=1.0, key="bt_minpx")
        st.number_input("Avg volume (50 days) ≥ shares", 0, 50_000_000, step=100_000, key="bt_minvol",
                        help="The Stock Scanner's default is 500,000.")
        st.number_input("Avg \\$ volume (50 days) ≥ \\$M", 0.0, 5000.0, step=5.0, key="bt_mindvol",
                        help="Average of price × volume, in millions of dollars. 20–50 = easy to trade in size.")
    st.sidebar.number_input("RS rating ≥ (any strategy)", 0, 99, step=5, key="bt_minrs",
                            help="Only take signals whose relative-strength rating (1–99, vs all stocks tested) was at "
                                 "least this on the signal day. Works on top of the scan's own settings. 0 = off.")
    n_fn = int(int(ss.get("bt_fn_ind_top", 0) or 0) > 0) + int(bool(ss.get("bt_fn_stages")))
    with st.sidebar.popover("🔻 Funnel" + (f" · {n_fn} on" if n_fn else " · off"), use_container_width=True,
                            help="Trade only leading stocks in leading industries — each judged ON THE SIGNAL DAY "
                                 "(no hindsight). Same rules as the 🔻 Funnel tab."):
        st.button("⚡ Apply the full funnel preset", on_click=bt_apply_funnel_preset, use_container_width=True,
                  help="Sets: all setup strategies combined · price ≥ $10, avg $ vol ≥ $20M · RS ≥ 85 · industry in "
                       "the top 30% · stage 1A–2B · size by market stage · pilot + add each day it still qualifies.")
        st.number_input("Industry in top % (0 = off)", 0, 100, step=5, key="bt_fn_ind_top",
                        help="Industries are ranked each day by the median RS of their stocks among the stocks "
                             "you're testing (so test a big list). 30 = the strongest 30%.")
        st.number_input("Min stocks per industry", 1, 50, key="bt_fn_ind_min")
        e1, e2 = st.columns(2)
        e1.number_input("…unless RS ≥", 0, 99, key="bt_fn_lead", help="Leaders pass anyway (0 = off)")
        e2.number_input("…or ρ with peers <", 0.0, 1.0, step=0.05, format="%.2f", key="bt_fn_corr",
                        help="Own movers (like TSLA) pass anyway: 60-day correlation with their industry peers is "
                             "below this (0 = off)")
        ss["bt_fn_stages"] = [x for x in ss.get("bt_fn_stages", []) if x in RS_STAGE_ORDER]
        st.multiselect("Stock stage (empty = off)", RS_STAGE_ORDER[:-1], key="bt_fn_stages",
                       help="The stock's own 1A–4C stage at the signal day's close.")
    st.sidebar.select_slider("Test period", ["3 months", "6 months", "1 year", "2 years"], key="bt_period")
    bt_months = month_picker({"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}[ss["bt_period"]])
    st.sidebar.selectbox("Group filter: only trade stocks whose group is", GROUP_CHOICES, key="bt_groups",
                         help="The stock's industry ETF (e.g. SMH for chips, KRE for regional banks), else its sector "
                              "ETF, rated ON THE SIGNAL DAY with the Sectors in play rules — no hindsight. "
                              "US stocks only.")
    if ss["bt_groups"] != "All groups":
        st.sidebar.markdown("**Check these groups**")
        st.sidebar.checkbox("Sector — XLK, XLF, XLE …", key="bt_grp_sec")
        st.sidebar.checkbox("Industry — SMH, KRE, XBI, GDX …", key="bt_grp_ind")
        st.sidebar.checkbox("Theme — TAN, URA, LIT, ICLN, GRID …", key="bt_grp_thm")
        if not bt_levels():
            st.sidebar.warning("Tick at least one — using Sector.")
        g1, g2 = st.sidebar.columns(2)
        g1.number_input("Skip if RS ≥", 0, 99, step=1, key="bt_grp_rs",
                        help="Leader override: a stock this strong (RS on the signal day) is traded even if its "
                             "group isn't in play. 0 = off.")
        g2.number_input("…or if ρ with group <", 0.0, 1.0, step=0.05, key="bt_grp_corr", format="%.2f",
                        help="Own mover: if the stock's 60-day correlation of daily returns with its group ETFs "
                             "(the highest of the ticked ones) is below this, the stock doesn't really move with "
                             "its group (e.g. TSLA vs DRIV) — the group check is skipped. 0 = off.")
        st.sidebar.caption("Every ticked group the stock belongs to must pass. A stock with none of the ticked "
                           "groups is skipped (e.g. Theme only → just stocks that have a theme ETF).")
    st.sidebar.selectbox("Market filter", BT_MARKET_FILTERS, key="bt_market",
                         help="Only take signals when SPY was in this state on the signal day (its close that day).\n\n"
                              "**Uptrend** = above the 21 EMA and 50 SMA, with the 50 SMA above the 200 SMA.\n\n"
                              "**Not in a downtrend** = anything except below both the 21 EMA and the 50 SMA.\n\n"
                              "**Pullback only** = below the 21 EMA but still above the 50 EMA — a dip inside an "
                              "uptrend.\n\n**Custom** = pick which lines SPY must close above and/or below.")
    if ss["bt_market"] == MKT_CUSTOM:
        st.sidebar.multiselect("SPY closes above", MKT_LINES, key="bt_mkt_above", placeholder="any line")
        st.sidebar.multiselect("SPY closes below", MKT_LINES, key="bt_mkt_below", placeholder="any line")
        if set(ss["bt_mkt_above"]) & set(ss["bt_mkt_below"]):
            st.sidebar.warning("A line can't be in both lists — no day would pass.")
    st.sidebar.markdown("**Trade rules**")
    if ss.get("bt_entry") not in BT_ENTRIES:
        ss["bt_entry"] = BT_ENTRIES[0]
    if ss.get("bt_stop") not in BT_STOPS:
        ss["bt_stop"] = BT_STOPS[0]
    st.sidebar.selectbox("Entry", BT_ENTRIES, key="bt_entry",
                         help="**Next day's open** — the signal is known at the close, you buy the next morning.\n\n"
                              "**Signal day's close** — buy at the close of the trigger day.\n\n"
                              "**Trigger day, at the breakout price** — a buy-stop order placed the evening before at "
                              "the setup's breakout level, filled the moment price reaches it (at the open if it gaps "
                              "past). Every fill counts — also breakouts that close back below. Setups & levels: "
                              "Tight flag SETUP → 7-day high · EMA crossback SETUP → 5-day high · Green line NEAR → "
                              "the line · Shakeout +3 RECLAIMED/UNDERCUT → the +3 level · Parabolic short EXTENDED → "
                              "below the day's low · VCP SETUP → the pivot · MA consolidation → the box high · HTF → the flag high · 50-day reclaim SETUP → the 50-day SMA · Bear flag (short) SETUP → a sell-stop at the lowest low since the SMA tag · Failed breakout (short) → a sell-stop at the old high. Episodic pivots use the gap day's close."
                              "\n\n**Trigger day at 3:30 pm** — the same setups, but you check them at 3:30 pm and buy at that "
                              "price only if it is above the trigger level AND above yesterday's close (below both "
                              "for shorts). Uses hourly prices from Yahoo (about the last 2 years). Stop 'Low of the "
                              "entry day' = the low up to 3:30.")
    if ss["bt_entry"] == BT_ENTRY_330:
        st.sidebar.caption("🕞 Needs hourly prices — downloaded for the stocks that trigger (can take a minute). "
                           "Yahoo keeps ~2 years of hourly data, so older triggers are skipped.")
    st.sidebar.selectbox("Stop", BT_STOPS, key="bt_stop",
                         help="**The setup's stop** — Tight flag = 7-day low · EMA crossback = 6-day low · Green line = "
                              "below the line · Shakeout +3 = under the shakeout low · EP = the gap day's low · "
                              "Parabolic short = the run's high · VCP = the last contraction's low · MA consolidation = the box low · HTF = the flag low · 50-day reclaim = the dip's low (the higher low) · Bear flag (short) = the bounce high · Failed breakout (short) = the highest high since the breakout.\n\n"
                              "**Low of the entry day** — bought during the day or at the close: that day's low (the "
                              "high for shorts), set at the close and active from the next day. Bought right at the "
                              "open: 0.5 × ATR under the entry, active immediately.\n\n"
                              "**ATR × 0.5 / 0.8 / 1** — entry minus that many 14-day ATRs (average daily range).")
    if ss["bt_stop"] == "Fixed %":
        st.sidebar.number_input("Stop % from entry", 1.0, 30.0, step=0.5, key="bt_stop_pct")
    st.sidebar.number_input("Stop at least % from entry", 0.0, 10.0, step=0.25, key="bt_min_stop",
                            help="A stop closer than this is moved out to this distance, so a stop a few cents under "
                                 "the entry doesn't give a huge position and a quick stop-out. 0 = off. "
                                 "Trades where it happened say 'widened' in the Stop basis column.")
    st.sidebar.number_input("Skip if the stop is more than % away", 0.0, 50.0, step=0.5, key="bt_max_stop",
                            help="Don't take a trade whose stop is further than this from the entry (e.g. 10 = skip "
                                 "stops over 10% away — the position would be tiny and the stock is too extended). "
                                 "0 = off. Checked after the minimum above. Skipped trades are counted under the "
                                 "results as 'stop too far away'.")
    st.sidebar.number_input("Pause after losses: last N signals", 0, 100, step=5, key="bt_pause",
                            help="Only take a new trade when the last N CLOSED signals (every signal, tracked on "
                                 "paper even while paused) add up to a profit. Stops trading in a bad stretch and "
                                 "starts again once the setup works again. Uses only trades closed before the entry "
                                 "day — you can follow it live. 0 = off. Applies to both views; no re-run needed.")
    st.sidebar.number_input("Pause an industry after N losses in a row", 0, 10, step=1, key="bt_pause_ind",
                            help="An industry (e.g. Semiconductors) sits out after its last N closed signals all "
                                 "lost; it's back as soon as one of its signals (tracked on paper) wins. Works "
                                 "together with the pause above. 0 = off. No re-run needed.")
    st.sidebar.selectbox("Profit target", BT_TARGETS, key="bt_target",
                         help="**Scale out** — sell ⅓ when the stock is up ADR % × 2 from the entry, ⅓ when it is "
                              "10 ATRs above its moving average, and trail the rest with the trailing exit below "
                              "(it also closes whatever is left if price closes below it before the trims). "
                              "The stop stays in place for whatever is still held.")
    if ss["bt_target"] == BT_SCALE_OUT:
        s1, s2 = st.sidebar.columns(2)
        s1.number_input("⅓ at ADR ×", 0.5, 10.0, step=0.5, key="bt_adr_x",
                        help="First trim: up this many ADRs (20-day average daily range %) from the entry.")
        s2.number_input("⅓ at ATR ×", 2.0, 30.0, step=0.5, key="bt_atr_x",
                        help="Second trim: price this many 14-day ATRs above the moving average.")
        st.sidebar.selectbox("ATRs measured from the", [10, 20, 50], key="bt_atr_ma",
                             format_func=lambda v: f"{v}-day moving average",
                     help="The moving average the ATR distance is measured from.")
        if ss.get("_bt_scale_trail") != True:            # first time: trail the rest with the 10-day
            ss["_bt_scale_trail"] = True
            ss["bt_trail"] = "10-day SMA"
    elif ss["bt_target"] == "R-multiple":
        st.sidebar.number_input("Target = × risk", 1.0, 10.0, step=0.5, key="bt_target_r")
    elif ss["bt_target"] == "Percent":
        st.sidebar.number_input("Target % gain", 2.0, 200.0, step=1.0, key="bt_target_pct")
    if ss.get("bt_trail") not in BT_TRAILS:
        ss["bt_trail"] = "20-day SMA"
    st.sidebar.selectbox("Trailing exit: close below", BT_TRAILS,
                         key="bt_trail")
    ss["bt_max_days"] = int(min(max(ss.get("bt_max_days", 60), 3), 120))
    st.sidebar.slider("Max days in a trade", 3, 120, key="bt_max_days")
    c1, c2 = st.sidebar.columns(2)
    c1.number_input("Cut if not up", 0.0, 5.0, step=0.5, key="bt_cut_r", format="%.1f",
                    help="R the trade must be up by (at the close) on day N — else it's sold. Works with the next box.")
    c2.number_input("…R after N days", 0, 60, key="bt_cut_days",
                    help="A time stop: on the close of day N (entry day = day 1), sell if the trade isn't up at least "
                         "the R on the left. Losers that never got going are cut early; real winners are usually "
                         "well past +1R by then. 0 = off.")
    st.sidebar.markdown("**Position size**")
    st.sidebar.checkbox("📊 Size by market stage", key="bt_stage_size",
                        help="Scale every trade's size by the market ETF's stage (1A–4C, the same stages as the "
                             "colored candles / the Relative Strength page) on the last close before the buy. "
                             "0% = don't trade in that stage. Changes apply without re-running.")
    if ss["bt_stage_size"]:
        with st.sidebar.expander(f"Stage sizes · market = {ss['bt_stage_bench']}", expanded=False):
            st.selectbox("Market ETF", STAGE_BENCHES, key="bt_stage_bench")
            for row in (("1A", "1B"), ("2A", "2B", "2C"), ("3A", "3B"), ("4A", "4B", "4C")):
                cols = st.columns(len(row))
                for c_, sg in zip(cols, row):
                    c_.number_input(f"{sg} %", 0, 200, step=25, key=f"bt_stg_{sg}",
                                    help=f"{sg} — {RS_STAGES[sg][0]}. Size in this stage, % of normal.")
            st.caption("100% = normal size (your Risk % or Unit %). The stage is from the ETF's close vs its "
                       "10/20 EMA and 50/200 SMA — a close cousin of your TradingView stage candles.")
    st.sidebar.checkbox("📈 Pilot + add to winners", key="bt_pyr",
                        help="Start every trade with a small **pilot** (1 unit). Each time the stock closes another "
                             "N × ATR above your last buy, buy 1 more unit at the close, up to the max units — and "
                             "raise the stop. Losers cost only the pilot; winners grow into big positions.")
    pyr_on = bool(ss["bt_pyr"])
    if pyr_on:
        if ss["bt_target"] == BT_SCALE_OUT:
            st.sidebar.caption("⚠️ Pilot + add doesn't work with **Scale out** — pick another profit target "
                               "(e.g. None, with a trailing exit).")
        a1, a2 = st.sidebar.columns(2)
        a1.number_input("Account $", 1000, 100_000_000, step=10_000, key="bt_acct")
        a2.number_input("Unit, % of account", 0.25, 50.0, step=0.25, key="bt_pyr_unit",
                        help="Each unit — the pilot and every add — is this % of the account (like 1 unit ≈ 1%). "
                             "The pilot then risks unit % × the stop distance.")
        if ss.get("bt_pyr_rule") not in PYR_RULES:
            ss["bt_pyr_rule"] = PYR_RULES[0]
        st.sidebar.selectbox("Add 1 unit", PYR_RULES, key="bt_pyr_rule",
                             help="**Each day it still qualifies** — at every close where the stock is still a SETUP or "
                                  "TRIGGER in this strategy's scan, buy 1 more unit (days in a row are fine). Losers "
                                  "stop qualifying and get no adds. Best for scans whose SETUP lasts while the trend "
                                  "runs (e.g. EMA crossback) — a breakout scan's SETUP ends once it breaks out.\n\n"
                                  "**Stacked trend** — at every close where close ≥ EMA10 ≥ EMA21 ≥ EMA50 (his "
                                  "'Price ≥ EMA10 ≥ EMA21 ≥ EMA50' switch). Works with any strategy.\n\n"
                                  "**Price ladder** — only when the close is N × ATR above the last buy.")
        by_sig_ui = ss["bt_pyr_rule"] != PYR_RULES[1]
        c1, c2 = st.sidebar.columns(2)
        if by_sig_ui:
            c1.checkbox("Only while in profit", key="bt_pyr_profit",
                        help="Add only if the close is above the average cost of what you hold — adding to "
                             "winners, never to losers.")
        else:
            c1.number_input("Add every × ATR", 0.25, 10.0, step=0.25, key="bt_pyr_step",
                            help="The ladder: add 1 unit when the close is this many ATRs (14-day ATR on the entry "
                                 "day) above the last buy. At most one add per day.")
        c2.number_input("Max units", 1, 20, key="bt_pyr_max", help="Pilot + adds. 1 = no adds.")
        st.sidebar.selectbox("After each add, the stop", PYR_STOPS, key="bt_pyr_stop",
                             help="With **each day it still qualifies**, adds can come on days in a row at similar "
                                  "prices — **Keep** or **Breakeven** fit better (a stop is never moved to or above "
                                  "the price).\n\n**Previous buy (ladder)** — the stop moves up to the buy before the add: after the "
                                  "1st add the pilot is at breakeven, after the 2nd the whole position is about "
                                  "flat, after that it locks in profit.\n\n**Breakeven** — the stop goes to the "
                                  "average cost of all units.\n\n**Keep** — the stop never moves (only the exits "
                                  "above end the trade).")
        st.sidebar.caption(f"Max position = {int(ss['bt_pyr_max'])} × {ss['bt_pyr_unit']:g}% = "
                           f"**{int(ss['bt_pyr_max']) * float(ss['bt_pyr_unit']):g}%** of the account. "
                           "Risk % and Max position size aren't used in this mode.")
    else:
        a1, a2 = st.sidebar.columns(2)
        a1.number_input("Account $", 1000, 100_000_000, step=10_000, key="bt_acct")
        a2.number_input("Risk %", 0.1, 10.0, step=0.25, key="bt_risk",
                        help="Each trade risks this % of the account: shares = (account × risk %) ÷ (entry − stop). "
                             "A tighter stop means more shares.")
        st.sidebar.slider("Max position size, % of account", 5, 200, step=5, key="bt_maxpos",
                          help="Caps very tight stops so one trade can't take the whole account. Over 100% = margin "
                               "(the Account simulation never borrows — it's limited to the cash it has).")
    run = st.sidebar.button("▶  RUN BACKTEST", type="primary", use_container_width=True)

    st.title("Backtest")
    if strategy == BT_COMBINED:
        look = max([BT_STRATEGIES[x][0] for x in combo] or [120])
        what = "the signals of " + (" + ".join(combo) if combo else "(none picked yet)")
    else:
        look, side, _, what = BT_STRATEGIES[strategy]
    st.caption(f"**{strategy}** · signals = {what}. Each signal is traded with the rules on the left; one trade per "
               f"stock at a time. **R** = profit ÷ risk (entry − stop), so +2R means you made twice what you risked.")

    period_days = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}[ss["bt_period"]]
    if uni == "Stock Scanner results":
        tickers = drop_excluded(ss.get("last_results", [])[:600], ss.get("meta"))
    elif uni == "My own tickers":
        tickers = drop_excluded(parse_tickers(ss.get("bt_tickers", "")))
    else:
        try:
            tickers = bt_universe_list(uni, 6000 if uni == "S&P 500" else int(ss.get("bt_max_stocks", 1500)),
                                       ss.get("bt_cmin"), ss.get("bt_cmax"))
        except Exception:
            tickers = []
    if tickers:
        st.sidebar.caption(f"**{len(tickers):,}** stocks will be tested."
                           + (f" Left out: {', '.join(ss['sec_excl'])}." if ss.get("sec_excl") else ""))
    elif uni in BT_LIST_UNIVERSES:
        st.sidebar.caption("⚠️ **0** stocks match these filters — loosen the market cap.")
    opts = dict(entry=ss["bt_entry"], stop=ss["bt_stop"], stop_pct=float(ss["bt_stop_pct"]), target=ss["bt_target"],
                min_stop_pct=float(ss["bt_min_stop"]), max_stop_pct=float(ss["bt_max_stop"]),
                cut_days=int(ss["bt_cut_days"]), cut_r=float(ss["bt_cut_r"]),
                target_r=float(ss["bt_target_r"]), target_pct=float(ss["bt_target_pct"]), trail=ss["bt_trail"],
                adr_x=float(ss["bt_adr_x"]), atr_x=float(ss["bt_atr_x"]), atr_ma=int(ss["bt_atr_ma"]),
                max_days=int(ss["bt_max_days"]))
    if ss["bt_pyr"]:
        opts.update(pyr=True, pyr_step=float(ss["bt_pyr_step"]), pyr_max=int(ss["bt_pyr_max"]), pyr_stop=ss["bt_pyr_stop"],
                    pyr_rule=ss["bt_pyr_rule"], pyr_profit=bool(ss["bt_pyr_profit"]))
    pp = build_pp(uni if uni in MARKETS else None)
    pp["f5_days"] = int(ss.get("bt_f50_days", 30))
    key = (strategy, tuple(combo), ss.get("bt_combo_mode") if strategy == BT_COMBINED else "",
           int(ss.get("bt_combo_win", 0)) if strategy == BT_COMBINED else 0, uni, tuple(tickers), period_days, tuple(sorted(pp.items())), tuple(sorted(opts.items())),
           tuple(sorted(bt_months or ())), ss["bt_groups"], ss["bt_market"], tuple(bt_levels()),
           int(ss.get("bt_grp_rs", 0) or 0), float(ss.get("bt_grp_corr", 0) or 0),
           float(ss["bt_minpx"]), int(ss["bt_minvol"]), float(ss["bt_mindvol"]), int(ss.get("bt_minrs", 0) or 0),
           int(ss.get("bt_fn_ind_top", 0) or 0), int(ss.get("bt_fn_ind_min", 3)), int(ss.get("bt_fn_lead", 0) or 0),
           float(ss.get("bt_fn_corr", 0) or 0), tuple(ss.get("bt_fn_stages", [])),
           tuple(ss["bt_mkt_above"]) if ss["bt_market"] == MKT_CUSTOM else (),
           tuple(ss["bt_mkt_below"]) if ss["bt_market"] == MKT_CUSTOM else ())

    if run:
        if not tickers:
            st.warning("No stocks to test — pick a list, type tickers, or run a scan first "
                       "(for 'Stock Scanner results').")
            return
        need = period_days + look + 30
        dl_period = "2y" if need <= 480 else "5y"
        pkey = (tuple(tickers), dl_period, date.today().isoformat())
        if ss.get("bt_pkey") != pkey:
            with st.status(f"Downloading {dl_period} of prices for {len(tickers):,} stocks …", expanded=True) as stt:
                msg, bar = st.empty(), st.progress(0.0)
                prices = download_all(tickers, bar, msg, period=dl_period)
                ss["bt_panels"] = build_panels(prices, rows=100000) if prices else None
                ss["bt_pkey"] = pkey
                stt.update(label=f"Downloaded {len(prices):,} stocks", state="complete", expanded=False)
        P = ss.get("bt_panels")
        if not P:
            st.error("No prices downloaded.")
            return
        if not combo:
            st.warning("Pick at least one strategy under **Strategies to combine**.")
            return
        if bt_months is not None and not bt_months:
            st.warning("No months ticked — tick at least one month under 📅 Months.")
            return
        bar = st.progress(0.0, text="Replaying the scan …")
        sigs, trs, tr_alls, notes, filt_notes = [], [], [], [], []
        agree = strategy == BT_COMBINED and ss.get("bt_combo_mode") == BT_ALL
        if agree and len(combo) < 2:
            st.warning("Pick at least two strategies for **All agree**.")
            return
        agree_sigs, sides, elig_all = {}, set(), {}
        skipped, errors, run_opts = {}, 0, dict(opts)
        for n_s, strat in enumerate(combo):
            s_look, side, s_fn, _ = BT_STRATEGIES[strat]
            run_opts = dict(opts)
            if is_trig_entry(opts["entry"]) and s_fn is _bt_ep:
                run_opts["entry"] = "Signal day's close"         # an EP is only confirmed by the day's volume
                notes.append(("For 🚀 Episodic Pivot: " if len(combo) > 1 else "") +
                             "An episodic pivot is only confirmed by the gap day's full volume, so there's no "
                             "buy-stop version — this run buys at the **gap day's close** instead.")
            if len(combo) > 1:
                bar.progress(n_s / len(combo), text=f"Replaying {strat} ({n_s + 1} of {len(combo)}) …")
            trig_mode = is_trig_entry(run_opts["entry"])
            eo = {} if run_opts.get("pyr") and run_opts.get("pyr_rule") == PYR_RULES[0] else None
            sig, err = bt_signals(P, strat, pp, period_days, bar, trigger=trig_mode, months=bt_months, elig_out=eo)
            if eo is not None:
                elig_all[strat] = eo.get("elig")
            errors += err
            if len(sig) and (float(ss["bt_minpx"]) > 0 or int(ss["bt_minvol"]) > 0 or float(ss["bt_mindvol"]) > 0):
                sig, fn_ = bt_liquidity_filter(sig, P, float(ss["bt_minpx"]), int(ss["bt_minvol"]),
                                               float(ss["bt_mindvol"]), trigger=trig_mode)
                filt_notes.append(fn_)
            if (ss["bt_groups"] != "All groups" or ss["bt_market"] != "Off") and len(sig):
                bar.progress(1.0, text="Rating the sector / industry ETFs on each signal day …")
                epkey = (dl_period, date.today().isoformat())
                if ss.get("bt_etf_pkey") != epkey:
                    eprices = download_all(ETF_TICKERS + ["SPY"], st.empty(), st.empty(), period=dl_period)
                    ss["bt_etf_panels"] = build_panels(eprices, rows=100000) if eprices else None
                    ss["bt_etf_pkey"] = epkey
                E = ss.get("bt_etf_panels")
                if E is None:
                    st.error("Couldn't download the ETF prices for the group / market filter.")
                    return
                sig, fn2 = bt_group_market_filter(sig, P, E, uni, ss["bt_groups"], ss["bt_market"],
                                                  rs_min=int(ss.get("bt_grp_rs", 0) or 0),
                                                  corr_min=float(ss.get("bt_grp_corr", 0) or 0),
                                                  above=tuple(ss["bt_mkt_above"]), below=tuple(ss["bt_mkt_below"]),
                                                  levels=bt_levels(), trigger=trig_mode)
                filt_notes.append(fn2)
            if int(ss.get("bt_minrs", 0) or 0) > 0 and len(sig) and "RS" in sig:
                n0_ = len(sig)
                sig = sig[sig["RS"] >= int(ss["bt_minrs"])]
                filt_notes.append(f"RS ≥ {int(ss['bt_minrs'])}: {len(sig):,} of {n0_:,} signals kept.")
            if len(sig) and (int(ss.get("bt_fn_ind_top", 0) or 0) > 0 or ss.get("bt_fn_stages")):
                bar.progress(1.0, text="Funnel: ranking industries and stock stages on each signal day …")
                if ss.get("bt_rs_key") != ss.get("bt_pkey"):
                    ss["bt_rs_mat"], ss["bt_rs_key"] = rs_matrix(P["Close"]), ss.get("bt_pkey")
                sig, fn3 = bt_funnel_filter(sig, P, uni, int(ss.get("bt_fn_ind_top", 0) or 0),
                                            int(ss.get("bt_fn_ind_min", 3)), tuple(ss.get("bt_fn_stages", [])),
                                            int(ss.get("bt_fn_lead", 0) or 0), float(ss.get("bt_fn_corr", 0) or 0),
                                            trigger=trig_mode, rs=ss["bt_rs_mat"])
                filt_notes.append(fn3)
            agree_sigs[strat] = sig
            sides.add(side)
            if agree:
                continue
            if run_opts["entry"] == BT_ENTRY_330 and len(sig):
                bar.progress(1.0, text=f"Downloading hourly prices for {sig['Symbol'].nunique():,} stocks (3:30 pm) …")
                sig, n330 = bt_at_330(sig, P, intraday_330(tuple(sorted(sig["Symbol"].unique()))), side == "long")
                filt_notes.append(n330)
            bar.progress(1.0, text="Simulating trades …")
            el_ = elig_all.get(strat)
            tr_s, sk_s = bt_trades(P, sig, side, run_opts, elig=el_)
            tr_all_s, _ = bt_trades(P, sig, side, run_opts, one_per_stock=False, elig=el_)   # for the account simulation
            for k_, v_ in sk_s.items():
                skipped[k_] = skipped.get(k_, 0) + v_
            if strategy == BT_COMBINED:
                sig = sig.assign(Strategy=strat)
                tr_s = tr_s.assign(Strategy=strat) if len(tr_s) else tr_s
                tr_all_s = tr_all_s.assign(Strategy=strat) if len(tr_all_s) else tr_all_s
            sigs.append(sig); trs.append(tr_s); tr_alls.append(tr_all_s)
        cat = lambda xs: (pd.concat([x for x in xs if len(x)], ignore_index=True)
                          if any(len(x) for x in xs) else pd.DataFrame())
        if agree:
            if len(sides) > 1:
                bar.empty()
                st.warning("**All agree** needs strategies on the same side — don't mix long and short ones.")
                return
            side = sides.pop()
            win = int(ss.get("bt_combo_win", 0))
            sig = bt_agree_signals(agree_sigs, win, is_trig_entry(run_opts["entry"]))
            n_each = " · ".join(f"{k}: {len(v):,}" for k, v in agree_sigs.items())
            filt_notes.append(f"✅ All agree (within {win} day{'' if win == 1 else 's'}): {len(sig):,} confirmed "
                              f"signals out of {n_each}.")
            if run_opts["entry"] == BT_ENTRY_330 and len(sig):
                bar.progress(1.0, text=f"Downloading hourly prices for {sig['Symbol'].nunique():,} stocks (3:30 pm) …")
                sig, n330 = bt_at_330(sig, P, intraday_330(tuple(sorted(sig["Symbol"].unique()))), side == "long")
                filt_notes.append(n330)
            if len(sig):
                bar.progress(1.0, text="Simulating trades …")
                el_ = None                                     # still qualifies = any of the strategies qualifies
                for v_ in elig_all.values():
                    if v_ is not None:
                        el_ = v_ if el_ is None else (el_ | v_)
                tr, sk_s = bt_trades(P, sig, side, run_opts, elig=el_)
                tr_all, _ = bt_trades(P, sig, side, run_opts, one_per_stock=False, elig=el_)
                for k_, v_ in sk_s.items():
                    skipped[k_] = skipped.get(k_, 0) + v_
                extra = sig.set_index(["Symbol", "Signal"])[["Fired by", "Also"]]
                add = lambda t: t.join(extra, on=["Symbol", "Signal"]).assign(Strategy=BT_AGREE) if len(t) else t
                tr, tr_all = add(tr), add(tr_all)
            else:
                tr = tr_all = pd.DataFrame()
        else:
            sig, tr, tr_all = cat(sigs), cat(trs), cat(tr_alls)
        note = " ".join(dict.fromkeys(notes))
        filt_note = " ".join(x for x in dict.fromkeys(filt_notes) if x)
        bar.empty()
        ss["bt_result"] = dict(key=key, sig=sig, tr=tr, tr_all=tr_all, skipped=skipped, errors=errors, note=note,
                               P=P, entry=run_opts["entry"], strategy=strategy, combo=list(combo), agree=agree, uni=uni,
                               months=sorted(bt_months) if bt_months else None, filt_note=filt_note,
                               n_stocks=P["Close"].shape[1],
                               first=P["Close"].index[max(0, len(P["Close"]) - period_days - 1)],
                               last=P["Close"].index[-1])

    res = ss.get("bt_result")
    if not res:
        st.info("👈 Pick a strategy, the stocks and your trade rules, then press **RUN BACKTEST**.\n\n"
                "**How it works:** the app goes back day by day over the test period and runs the scan as it would "
                "have run on that day (only data up to that day). Every signal becomes a trade — entered at the next "
                "day's open with the setup's stop — and is followed until the stop, the target, the trailing "
                "moving average or the time limit ends it. You get the win rate, average gain and loss, expectancy "
                "in R, an equity curve and every trade on a chart.")
        return
    if res["key"] != key:
        st.caption("⚠️ Settings changed since this run — press **RUN BACKTEST** to update.")
    acct, risk_pct, max_pos = float(ss["bt_acct"]), float(ss["bt_risk"]), float(ss["bt_maxpos"])
    n_pause = int(ss.get("bt_pause", 0) or 0)
    tr_raw, n_paused, pauses = bt_by_strategy(bt_pause_filter, res["tr"], res["tr"], n_pause)
    tr_all_raw, n_paused_all, _ = bt_by_strategy(bt_pause_filter, res["tr"], res.get("tr_all", res["tr"]), n_pause)
    k_ind = int(ss.get("bt_pause_ind", 0) or 0)
    n_ind, ind_per = 0, {}
    if k_ind > 0 and len(res["tr"]):
        ind_of = bt_sector_info(res["tr"]["Symbol"].unique().tolist(), res["uni"])["Industry"]
        ind_of = ind_of.replace("", np.nan)
        tr_raw, n_ind, ind_per = bt_by_strategy(bt_industry_pause, res["tr"], tr_raw, k_ind, ind_of)
        tr_all_raw, _, _ = bt_by_strategy(bt_industry_pause, res["tr"], tr_all_raw, k_ind, ind_of)
    n_stage0 = 0
    if ss.get("bt_stage_size") and len(tr_raw):
        stg_h = bt_bench_stages(ss["bt_stage_bench"], res)
        if stg_h is None:
            st.warning(f"Couldn't download {ss['bt_stage_bench']} for the market-stage sizing — sizing is off.")
        else:
            pct_ = bt_stage_pct()
            tr_raw, n_stage0 = bt_stage_sizing(tr_raw, stg_h, pct_, res.get("entry"))
            tr_all_raw, _ = bt_stage_sizing(tr_all_raw, stg_h, pct_, res.get("entry"))
            st.caption(f"📊 **Sized by {ss['bt_stage_bench']}'s stage** on the last close before each buy"
                       + (f" · {n_stage0:,} trades skipped (stage sized at 0%)" if n_stage0 else "")
                       + f" · {ss['bt_stage_bench']} now: **{stg_h.iat[-1]}** → {pct_.get(stg_h.iat[-1], 100)}% size.")
    unit_pct = float(ss["bt_pyr_unit"]) if "Legs" in tr_raw else None
    tr = bt_size(tr_raw, acct, risk_pct, max_pos, unit_pct)
    s = bt_stats(tr)
    combo_txt = ((" & " if res.get("agree") else " + ").join(res.get("combo") or [])
                 + (" (all agree)" if res.get("agree") else "") if res["strategy"] == BT_COMBINED else res["strategy"])
    st.caption(f"{combo_txt} · {res['uni']} · {res['n_stocks']:,} stocks · "
               f"{res['first']:%b %d, %Y} → {res['last']:%b %d, %Y}"
               + (f" · only {month_list_text(res['months'])}" if res.get("months") else "")
               + f" · {len(res['sig']):,} signals"
               + (" (buy-stop fills: setups armed the day before whose level was reached)"
                  if is_trig_entry(res.get("entry")) else ""))
    if res.get("note"):
        st.info(res["note"])
    if res.get("filt_note"):
        st.caption(res["filt_note"])
    sk = {k: v for k, v in (res.get("skipped") or {}).items() if v}
    if sk:
        st.caption("Not traded: " + " · ".join(f"{v} {k}" for k, v in sk.items())
                   + " (a stock that opens past its stop wouldn't be bought).")
    if n_pause > 0:
        n_sig = len(res["tr"])
        st.caption(f"⏸ **Pause rule on** (last {n_pause} closed signals must be net positive): **{n_paused:,}** of "
                   f"{n_sig:,} signal trades skipped while paused · paused {len(pauses)} time"
                   f"{'' if len(pauses) == 1 else 's'}"
                   + (" · **paused now**" if any(b is None for _, a, b in pauses) else "")
                   + (" · each strategy has its own pause" if res["strategy"] == BT_COMBINED else "")
                   + ". Every signal is still tracked on paper to know when to start again.")
        if pauses:
            with st.expander(f"⏸ When trading was paused ({len(pauses)})"):
                st.dataframe(pd.DataFrame([{**({"Strategy": nm} if nm else {}), "Paused from": a,
                                            "Back on": b if b is not None else pd.NaT,
                                            "Trading days": (len(pd.bdate_range(a, b)) - 1) if b is not None else
                                            len(pd.bdate_range(a, res["last"]))} for nm, a, b in pauses]),
                             hide_index=True, use_container_width=True,
                             column_config={"Paused from": st.column_config.DateColumn(format="MMM D, YYYY"),
                                            "Back on": st.column_config.DateColumn(format="MMM D, YYYY")})
    if k_ind > 0:
        top = ", ".join(f"{g} {v}" for g, v in list(ind_per.items())[:4])
        st.caption(f"⏸ **Industry pause on** (an industry sits out after {k_ind} losing signal{'' if k_ind == 1 else 's'} in a row, until one "
                   f"wins on paper): **{n_ind:,}** more trades skipped" + (f" — most in {top}" if top else "") + ".")
    if res.get("errors"):
        st.warning(f"The scan failed on {res['errors']} of the replayed days (missing or bad price data) — "
                   "those days gave no signals.")
    if not s:
        st.warning("No trades in this period. Try a longer period, more stocks, or looser scan settings "
                   "in the Stock Scanner tab.")
        return
    ss.setdefault("bt_view", "📋 Every signal")
    view = (st.segmented_control("View", ["📋 Every signal", "💼 Account simulation"], key="bt_view",
                                 label_visibility="collapsed") if hasattr(st, "segmented_control") else
            st.radio("View", ["📋 Every signal", "💼 Account simulation"], key="bt_view", horizontal=True,
                     label_visibility="collapsed")) or "📋 Every signal"
    if view == "💼 Account simulation":
        bt_account_view(res, tr_all_raw, acct, risk_pct, max_pos)
        return
    st.caption("Every signal is traded, however many are open at once (the account never runs out of money). "
               "See **💼 Account simulation** for a real account with limited cash.")
    tone = lambda v, good: "g" if v > good else "r" if v < 0 else "y"
    render_cards([("Trades", f"{s['n']:,}", f"{s['closed']:,} closed · {s['n'] - s['closed']} still open", "n"),
                  ("Win rate", f"{s['win']:.0f}%", " · ".join(x for x in [
                      f"avg win {s['avg_win']:+.1f}%" if pd.notna(s["avg_win"]) else "",
                      f"avg loss {s['avg_loss']:+.1f}%" if pd.notna(s["avg_loss"]) else ""] if x) or "—",
                   tone(s["win"] - 40, 0)),
                  ("Expectancy", f"{s['exp_r']:+.2f}R", "average result per trade", tone(s["exp_r"], 0.2)),
                  ("Profit", f"{'-' if s['pnl'] < 0 else '+'}${abs(s['pnl']):,.0f}",
                   f"{s['pnl'] / acct * 100:+.1f}% of ${acct:,.0f} · {s['total_r']:+.1f}R", tone(s["pnl"], 0)),
                  ("Profit factor", "∞" if s["pf"] == np.inf else "—" if pd.isna(s["pf"]) else f"{s['pf']:.2f}",
                   "$ won ÷ $ lost", tone((s["pf"] if pd.notna(s["pf"]) else 1) - 1, 0.3)),
                  ("Max drawdown", f"{s['dd_pct']:.1f}%", f"{s['dd']:.1f}R · worst losing streak {s['streak']}",
                   "r" if s["dd_pct"] < -5 else "n")])
    capped = int((tr["Position %"] >= max_pos - 0.5).sum())
    st.caption(f"Position size: risk **{risk_pct:g}%** of a **\\${acct:,.0f}** account per trade "
               f"(\\${acct * risk_pct / 100:,.0f}), shares = risk ÷ (entry − stop), max {max_pos:g}% of the account in "
               f"one stock · average position {tr['Position %'].mean():.0f}% of the account"
               + (f" · **{capped}** trades were capped by the max position size (they risk less than "
                  f"{risk_pct:g}%)" if capped else "") + ". Sized on the starting account (no compounding).")
    c1, c2 = st.columns([1.6, 1])
    eq = acct + s["eq_usd"]
    eq = pd.concat([pd.Series([acct], index=[eq.index.min() - pd.Timedelta(days=1)]), eq])
    fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, mode="lines", line=dict(color="#26a69a", width=2),
                               name="Account $", hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=acct, line_dash="dot", line_color="rgba(128,128,128,.6)")
    fig.update_layout(title="Account value (by exit date)", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_title="$", yaxis_tickformat="$,.0f", xaxis_title=None)
    c1.plotly_chart(fig, use_container_width=True)
    bt_detail_sections(tr, res, "bt", "Every trade", hist_in=c2)
