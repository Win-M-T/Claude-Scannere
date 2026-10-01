"""Results table, filter pills and detail charts of the Stock Scanner page."""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from scanner.charts import load_templates, price_chart, save_templates, tradingview_chart, tv_open_url
from scanner.constants import COLUMN_SETS, ETF_MARKET, ETF_OK, ETF_TICKERS, F, GROUP_CHOICES, GROUP_LEVELS, GRP_CORR_DAYS, MARKETS, MODE_SPEC, RS_STAGES, SH_SETUPS, SP_WINDOWS, TPL_FIELDS, TV_EXTRAS, TV_INTERVALS, TV_STYLES, VOL_REC_RANK, sp_pick_index
from scanner.data.fundamentals import between, fetch_fundamentals, fetch_mcap
from scanner.data.universe import get_exchange_map, market_of, tv_symbol, tv_watchlist_text
from scanner.etf import etf_panels, etf_status_now, group_used_etfs, own_mover, stock_group_status, stock_theme
from scanner.rs import V, is_earn_mode, is_ep_mode, is_f5_mode, is_gl_mode, is_htf_mode, is_ll_mode, is_mac_mode, is_pb_mode, is_rc_mode, is_record_mode, is_rs_mode, is_sh_mode, is_so_mode, is_sp_mode, is_vcp_mode, is_xb_mode, pwidget
from scanner.smart_money import detail_card, market_mood, render_cards, setup_levels

ss = st.session_state

MODE_FILTER_KEYS = {v[0] for v in MODE_SPEC.values()}
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


# which other scans a stock also shows up in: (scan, icon, column, statuses that count as a match)
ALSO_SCANS = [("pattern", "🚩", "Pattern", ("TRIGGER", "SETUP")),
              ("xback", "🔁", "Crossback", ("TRIGGER", "SETUP")),
              ("glb", "🟢", "GLB", ("BREAKOUT", "RETEST", "NEAR")),
              ("so3", "🪤", "SO+3", ("TRIGGER", "RECLAIMED", "UNDERCUT")),
              ("vcp", "🌀", "VCP", ("TRIGGER", "SETUP")),
              ("mac", "📏", "MAC", ("TRIGGER", "SETUP")),
              ("htf", "⛳", "HTF", ("TRIGGER", "SETUP")),
              ("swshort", "🐻", "Short", ("TRIGGER", "SETUP")),
              ("reclaim", "🔂", "Reclaim", ("TRIGGER", "SETUP", "DIP")),
              ("f50scan", "🔃", "First50", ("TRIGGER", "SETUP")),
              ("ep", "🚀", "EP", ("HOLDING",)),
              ("parabolic", "📉", "Parabolic", ("CRACK", "EXTENDED", "FADING")),
              ("spchg", "🏛", "S&P change", ("ADDED", "REMOVED")),
              ("volrec", "📊", "VolHi 3M", (True,))]


def also_in(df, own):
    """'🟢 NEAR · 🔁 SETUP' — the other scans each stock currently matches (the current scan left out)."""
    parts = []
    for key, icon, col, ok in ALSO_SCANS:
        if key == own or col not in df:
            continue
        v = df[col]
        if col == "VolHi 3M":
            hit = v.fillna(False).astype(bool)
            parts.append(np.where(hit, f"{icon} vol record", ""))
        else:
            hit = v.isin(ok)
            parts.append(np.where(hit, icon + " " + v.astype(str), ""))
    if not parts:
        return pd.Series("", index=df.index), pd.Series(0, index=df.index)
    arr = np.array(parts, dtype=object).T
    txt = [" · ".join(x for x in row if x) for row in arr]
    return pd.Series(txt, index=df.index), pd.Series([sum(bool(x) for x in row) for row in arr], index=df.index)


@_fragment
def results_area(metrics=None, universe=None, sp_changes=None):
    if metrics is None:
        metrics = ss.get("metrics", pd.DataFrame())
    if universe is None:
        universe = ss.get("universe")
    if sp_changes is None:
        sp_changes = ss.get("sp_changes", {})
    """Filter buttons, results table and chart. Clicks in here redraw only this part of the page."""
    if ss.pop("_full_rerun", False):       # a scan-type filter changed → the sidebar must follow
        st.rerun()
    # ============================================================================
    # 9. Filter pills
    # ============================================================================
    def reset_filter(key):
        ss[f"v_f_{key}"] = "Any"
        ss[f"v_fmin_{key}"] = None
        ss[f"v_fmax_{key}"] = None
        ss[f"v_fcat_{key}"] = []


    def reset_all():
        for key in F:
            reset_filter(key)


    def pill_label(key, spec):
        choice = V(f"f_{key}", "Any")
        if spec["kind"] == "cat":
            n = len(V(f"fcat_{key}", []))
            return f"● {spec['label']} · {n}" if n else spec["label"]
        if choice == "Any":
            return spec["label"]
        if choice == "Manual setup":
            lo, hi = V(f"fmin_{key}"), V(f"fmax_{key}")
            if lo is None and hi is None:
                return spec["label"]
            rng = f"{'' if lo is None else f'{lo:g}'}–{'' if hi is None else f'{hi:g}'}"
            return f"● {spec['label']} · {rng}"
        return f"● {spec['label']} · {choice}"


    def render_pill(key, spec, cat_options):
        with st.popover(pill_label(key, spec)):
            top = st.columns([4, 1])
            top[0].markdown(f"**{spec['label']}**")
            top[1].button("🗑", key=f"del_{key}", on_click=reset_filter, args=(key,), help="Clear this filter")
            if spec.get("fund"):
                st.caption("Needs fundamentals: open the **Fundamentals** tab and press *Load fundamentals*.")
            if spec["kind"] == "cat":
                options = cat_options.get(spec["col"], [])
                ss[f"v_fcat_{key}"] = [x for x in V(f"fcat_{key}", []) if x in options]
                pwidget(st.multiselect, f"fcat_{key}", [], spec["label"], options,
                        label_visibility="collapsed", placeholder="Choose one or more")
                return
            opts = ["Any"] + [p[0] for p in spec["presets"]]
            caps = [""] + [p[1] for p in spec["presets"]]
            if spec["kind"] == "num":
                opts.append("Manual setup")
                caps.append(f"Set your own range ({spec.get('unit', '')})")
            if V(f"f_{key}") not in opts:
                ss[f"v_f_{key}"] = "Any"
            pwidget(st.radio, f"f_{key}", "Any", spec["label"], opts, captions=caps, label_visibility="collapsed",
                    on_change=(lambda: ss.__setitem__("_full_rerun", True)) if key in MODE_FILTER_KEYS else None)
            if V(f"f_{key}") == "Manual setup":
                c1, c2 = st.columns(2)
                with c1:
                    pwidget(st.number_input, f"fmin_{key}", None, "From", value=None, placeholder="min", format="%g")
                with c2:
                    pwidget(st.number_input, f"fmax_{key}", None, "To", value=None, placeholder="max", format="%g")


    def apply_filters(d):
        mask = pd.Series(True, index=d.index)
        active = 0
        for key, spec in F.items():
            if spec["kind"] == "cat":
                sel = V(f"fcat_{key}", [])
                if sel:
                    mask &= d[spec["col"]].isin(sel)
                    active += 1
                continue
            choice = V(f"f_{key}", "Any")
            if choice == "Any":
                continue
            if choice == "Manual setup":
                lo, hi = V(f"fmin_{key}"), V(f"fmax_{key}")
                if lo is None and hi is None:
                    continue
                mask &= between(spec["col"], lo, hi)(d)
            else:
                fn = dict((p[0], p[2]) for p in spec["presets"]).get(choice)
                if fn is None:
                    continue
                mask &= fn(d)
            active += 1
        return d[mask], active


    cat_options = {}
    if metrics is not None:
        metrics["Theme"] = [stock_theme(t, i)[0] for t, i in zip(metrics.index, metrics["Industry"])]
        for col in ["Sector", "Industry", "Theme"]:
            cat_options[col] = sorted(x for x in metrics[col].dropna().astype(str).unique().tolist() if x)

    try:
        pill_row = st.container(horizontal=True, gap="small", key="pillrow")
    except TypeError:
        pill_row = st.container()
    LABEL_TO_KEY = {F[k]["label"]: k for k in F}
    mode_key = MODE_SPEC[ss["scan_mode"]][0]

    def is_active(k):
        spec = F[k]
        if spec["kind"] == "cat":
            return bool(V(f"fcat_{k}", []))
        ch = V(f"f_{k}", "Any")
        if ch == "Manual setup":
            return V(f"fmin_{k}") is not None or V(f"fmax_{k}") is not None
        return ch not in (None, "Any")

    # buttons shown: this scan's own filter, every filter that's switched on, and any you pinned with "＋"
    pinned = {LABEL_TO_KEY[l] for l in V("visible", []) if l in LABEL_TO_KEY}
    shown = [mode_key] + [k for k in F if k != mode_key and (is_active(k) or k in pinned)]
    ss["v_visible"] = [F[k]["label"] for k in shown if k != mode_key]

    def visible_changed():
        keep = {LABEL_TO_KEY[lbl] for lbl in V("visible", [])}
        for k in F:
            if k != mode_key and k not in keep:
                reset_filter(k)

    with pill_row:
        for k in shown:
            render_pill(k, F[k], cat_options)
        g_now = V("groups", "All groups")
        with st.popover("🗂 Groups" + ("" if g_now == "All groups" else " · " + g_now.split(" ")[0]
                                        + ("🌱" if "Emerging" in g_now else "")), help="Only stocks whose industry or "
                        "sector ETF is in play (see the 🗂 Sectors in play tab)"):
            pwidget(st.radio, "groups", "All groups", "Only stocks whose group is", GROUP_CHOICES)
            st.markdown("**Check these groups**")
            q1, q2, q3 = st.columns(3)
            with q1:
                pwidget(st.checkbox, "grp_sec", True, "Sector")
            with q2:
                pwidget(st.checkbox, "grp_ind", False, "Industry")
            with q3:
                pwidget(st.checkbox, "grp_thm", False, "Theme")
            st.caption("Sector = XLK, XLF … · Industry = SMH, KRE, XBI, GDX … · Theme = TAN, URA, LIT, ICLN, "
                       "GRID, PAVE … Every ticked group the stock belongs to must pass; stocks with none of the "
                       "ticked groups are left out. The Group column shows what was checked.")
            st.markdown("**Let these through anyway**")
            q1, q2 = st.columns(2)
            with q1:
                pwidget(st.number_input, "grp_rs", 0, "Leaders: RS ≥", 0, 99, step=1,
                        help="A stock this strong is kept even if its group isn't in play. 0 = off.")
            with q2:
                pwidget(st.number_input, "grp_corr", 0.3, "Own movers: ρ <", 0.0, 1.0, step=0.05, format="%.2f",
                        help="If the stock's 60-day correlation of daily returns with its group ETFs is below this, "
                             "it doesn't really move with its group (e.g. TSLA vs DRIV), so the group isn't "
                             "checked. 0 = off.")
        with st.popover("＋ Filter"):
            st.markdown("**Add or remove filters**")
            pwidget(st.multiselect, "visible", [], "Filters to show",
                    [F[k]["label"] for k in F if k != mode_key], label_visibility="collapsed",
                    on_change=visible_changed, placeholder="Pick filters to add")
        st.button("Reset", on_click=reset_all, type="tertiary", help="Clear every filter")

    # ============================================================================
    # 10. Results
    # ============================================================================
    if is_sp_mode() and universe in MARKETS:
        st.info("🏛 S&P index changes are for US stocks only — pick a US universe to use this scan.")
    if metrics is None:
        st.info("👈 Pick the stocks and a scan under **Scan for** in the sidebar, then press **SCAN**. "
                "Then use the filter buttons above — the table updates instantly.")
        with st.expander("What is the Tight-flag pattern?"):
            st.markdown(
                "- **Uptrend:** price above the 50-day, 50-day above the 200-day, 50-day rising\n"
                "- **Shallow pullback:** 3–15% below a 20-day high set 5+ days ago\n"
                "- **Tight base:** last 7 days' range is small (default under 8%)\n"
                "- **On the 20-day SMA:** price just above it, the 10 EMA and 20 SMA bunched together\n"
                "- **Quiet volume:** 5-day average volume below 80% of the 50-day\n\n"
                "**SETUP** = still basing.  **TRIGGER** = broke above the base today on heavy volume.")
        st.stop()

    res, n_active = apply_filters(metrics)
    grp_status = etf_status_now() if (V("groups", "All groups") != "All groups" or ss.get("etf_cache")) \
        and universe not in MARKETS else {}
    if grp_status:
        lv = [n for n, k in zip(GROUP_LEVELS, ("grp_sec", "grp_ind", "grp_thm")) if V(k, n == "Sector")] or ["Sector"]
        want = ETF_OK.get(V("groups", "All groups"), ())
        g_ok, g_t = stock_group_status(res["Sector"], res["Industry"], grp_status, lv, set(want), res.index)
        rs_min_, corr_min_ = int(V("grp_rs", 0) or 0), float(V("grp_corr", 0.3) or 0)
        if V("groups", "All groups") != "All groups" and (rs_min_ or corr_min_):
            try:
                ec = etf_panels(ETF_TICKERS + [t for t, _ in ETF_MARKET])[0]["Close"]
                sc = ss["panels"]["Close"]
                rs_e, rs_s = ec.pct_change().iloc[-GRP_CORR_DAYS:], sc.pct_change().iloc[-GRP_CORR_DAYS:]
                rs_e = rs_e.reindex(rs_s.index)
            except Exception:
                rs_e = rs_s = None
            g_ok, g_t = list(g_ok), list(g_t)
            for i, (sym, sec, ind) in enumerate(zip(res.index, res["Sector"], res["Industry"])):
                if g_ok[i]:
                    continue
                cs = []
                if rs_s is not None and sym in rs_s:
                    for t_ in group_used_etfs(sec, ind, lv, sym):
                        if t_ in rs_e and rs_s[sym].notna().sum() >= 40:
                            cs.append(rs_s[sym].corr(rs_e[t_]))
                cs = [c for c in cs if pd.notna(c)]
                own = own_mover(res["RS"].iat[i] if "RS" in res else np.nan, max(cs) if cs else np.nan,
                                rs_min_, corr_min_)
                if own:
                    g_ok[i] = True
                    g_t[i] = own + (f" · {g_t[i]}" if g_t[i] else "")
        res = res.assign(Group=g_t)
        if V("groups", "All groups") != "All groups":
            res = res[np.array(g_ok, bool)]
            n_active += 1
    if is_sp_mode() and ss["sp_type"] != "Inclusion candidates":
        back = SP_WINDOWS.get(ss["sp_window"], 31)
        res = res[res["Days to change"].between(-back, 400) & ((back > 0) | (res["Days to change"] >= 0))]
    if is_earn_mode() and ss["earn_time"] != "Any":
        res = res[res["Report time"] == ss["earn_time"]]

    # fill missing market caps for a manageable result list
    missing = [t for t in res.index[res["Mkt cap $B"].isna()] if t not in ss.get("mcap_fill", {})]
    if 0 < len(missing) <= 300:
        with st.spinner(f"Looking up market cap for {len(missing)} stocks …"):
            with ThreadPoolExecutor(8) as ex:
                vals = list(ex.map(fetch_mcap, missing))
            ss.setdefault("mcap_fill", {}).update(dict(zip(missing, vals)))
        st.rerun()

    res = res.reset_index()
    if is_so_mode():                   # shakeout +3: triggers, then reclaimed, then undercut; closest to +3 first
        res = res.assign(_rank=res["SO+3"].map({"TRIGGER": 0, "RECLAIMED": 1, "UNDERCUT": 2}).fillna(3)) \
            .sort_values(["_rank", "To +3 %", "RS"], ascending=[True, True, False], na_position="last")
    elif is_rc_mode():                   # 50-day reclaim: triggers first, then closest to the SMA
        res = res.assign(_rank=res["Reclaim"].map({"TRIGGER": 0, "SETUP": 1, "DIP": 2}).fillna(3),
                         _dist=res["vs 50 SMA %"].abs()) \
            .sort_values(["_rank", "_dist", "RS"], ascending=[True, True, False], na_position="last")
    elif is_f5_mode():                   # first close above 50-day: triggers first (longest under first), then nearest SMA
        res = res.assign(_rank=res["First50"].map({"TRIGGER": 0, "SETUP": 1}).fillna(2),
                         _dist=res["F50 vs SMA %"].abs()) \
            .sort_values(["_rank", "F50 days under", "_dist"], ascending=[True, False, True], na_position="last")
    elif is_sh_mode():                   # swing shorts: triggers first, then closest to the sell level
        res = res.assign(_rank=res["Short"].map({"TRIGGER": 0, "SETUP": 1}).fillna(2)) \
            .sort_values(["_rank", "To sell level %", "RS"], ascending=[True, True, True], na_position="last")
    elif is_htf_mode():                  # HTF: triggers first, then the biggest poles
        res = res.assign(_rank=res["HTF"].map({"TRIGGER": 0, "SETUP": 1}).fillna(2)) \
            .sort_values(["_rank", "Pole %"], ascending=[True, False], na_position="last")
    elif is_mac_mode():                  # MA consolidation: triggers first, then the tightest boxes
        res = res.assign(_rank=res["MAC"].map({"TRIGGER": 0, "SETUP": 1}).fillna(2)) \
            .sort_values(["_rank", "Box range %", "RS"], ascending=[True, True, False], na_position="last")
    elif is_vcp_mode():                  # VCP: triggers first, then closest to the pivot
        res = res.assign(_rank=res["VCP"].map({"TRIGGER": 0, "SETUP": 1}).fillna(2)) \
            .sort_values(["_rank", "To pivot %", "RS"], ascending=[True, True, False], na_position="last")
    elif is_gl_mode():                   # green line: breakouts, then retests, then nearest to the line
        res = res.assign(_rank=res["GLB"].map({"BREAKOUT": 0, "RETEST": 1, "NEAR": 2}).fillna(3)) \
            .sort_values(["_rank", "To green line %", "RS"], ascending=[True, True, False], na_position="last")
    elif is_xb_mode():                   # crossback: triggers first, then strongest RS
        res = res.assign(_rank=res["Crossback"].map({"TRIGGER": 0, "SETUP": 1}).fillna(2)) \
            .sort_values(["_rank", "RS"], ascending=[True, False], na_position="last")
    elif is_rs_mode():                   # RS score: highest Score first
        res = res.sort_values(["Score", "COMP"], ascending=False, na_position="last")
    elif is_ll_mode():                   # LiqLead: strongest 20-day movers first
        res = res.sort_values(["Chg 20d %"], ascending=False, na_position="last")
    elif is_sp_mode():                   # S&P: upcoming first, then most recent; candidates by size
        res = res.sort_values(["Change date", "Mkt cap $B"], ascending=[False, False], na_position="last")
    elif is_pb_mode():                   # parabolic: biggest run first
        res = res.sort_values(["Run %"], ascending=False, na_position="last")
    elif is_ep_mode():                   # EP: newest first, then heaviest volume
        res = res.sort_values(["Days since EP", "EP vol × avg"], ascending=[True, False], na_position="last")
    elif V("f_earnwin", "Any") != "Any":  # earnings scan: closest to today first, bigger companies first
        res = res.assign(_rank=res["Days to earnings"].abs()).sort_values(
            ["_rank", "Mkt cap $B"], ascending=[True, False], na_position="last")
    elif V("f_volrec", "Any") != "Any":   # record-volume scan: biggest records first, then most unusual volume
        res["_rank"] = res["Vol record"].map(VOL_REC_RANK).fillna(0)
        res = res.sort_values(["_rank", "Record × avg"], ascending=[False, False], na_position="last")
    else:
        res["_rank"] = res["Pattern"].map({"TRIGGER": 0, "SETUP": 1}).fillna(2)
        res = res.sort_values(["_rank", "Mkt cap $B"], ascending=[True, False], na_position="last")
    res = res.drop(columns="_rank", errors="ignore")
    ss["last_results"] = res["Symbol"].tolist()          # used by Relative Strength → Stocks from watchlist
    ss["last_results_label"] = ss.get("scan_mode", "")

    # column views: "Setup" = the columns for the chosen scan, plus the general views
    setup_tab = "Tight flag" if mode_key == "pattern" else MODE_SPEC[ss["scan_mode"]][2]
    TAB_CHOICES = ["Setup", "Overview", "Performance", "Technicals", "Fundamentals"]
    if V("tab") not in TAB_CHOICES:
        ss["v_tab"] = "Setup"

    n_trig = int((res["Pattern"] == "TRIGGER").sum())
    n_setup = int((res["Pattern"] == "SETUP").sum())
    n_rec = int((res["Vol record"] != "").sum()) if "Vol record" in res else 0
    n_earn = int(res["Days to earnings"].notna().sum()) if "Days to earnings" in res else 0
    n_pb = int(res["Parabolic"].isin(["CRACK", "EXTENDED", "FADING"]).sum()) if "Parabolic" in res else 0
    n_crack = int((res["Parabolic"] == "CRACK").sum()) if "Parabolic" in res else 0
    n_ep = int((res["EP"] != "").sum()) if "EP" in res else 0
    n_hold = int((res["EP"] == "HOLDING").sum()) if "EP" in res else 0
    n_add = int((res["S&P change"] == "ADDED").sum()) if "S&P change" in res else 0
    n_rem = int((res["S&P change"] == "REMOVED").sum()) if "S&P change" in res else 0
    n_cand = int((res["S&P change"] == "CANDIDATE").sum()) if "S&P change" in res else 0
    n_ll = int(res["LiqLead"].sum()) if "LiqLead" in res else 0
    n_xt = int((res["Crossback"] == "TRIGGER").sum()) if "Crossback" in res else 0
    n_xs = int((res["Crossback"] == "SETUP").sum()) if "Crossback" in res else 0
    n_st, n_sr, n_su = ((int((res["SO+3"] == x).sum()) if "SO+3" in res else 0) for x in ("TRIGGER", "RECLAIMED", "UNDERCUT"))
    n_vt, n_vs = ((int((res["VCP"] == x).sum()) if "VCP" in res else 0) for x in ("TRIGGER", "SETUP"))
    n_mt, n_ms = ((int((res["MAC"] == x).sum()) if "MAC" in res else 0) for x in ("TRIGGER", "SETUP"))
    n_ht, n_hs = ((int((res["HTF"] == x).sum()) if "HTF" in res else 0) for x in ("TRIGGER", "SETUP"))
    n_rct, n_rcs, n_rcd = ((int((res["Reclaim"] == x).sum()) if "Reclaim" in res else 0)
                           for x in ("TRIGGER", "SETUP", "DIP"))
    n_f5t, n_f5s = ((int((res["First50"] == x).sum()) if "First50" in res else 0) for x in ("TRIGGER", "SETUP"))
    n_sht, n_shs = ((int((res["Short"] == x).sum()) if "Short" in res else 0) for x in ("TRIGGER", "SETUP"))
    n_bf = int((res["Short setup"] == SH_SETUPS[1]).sum()) if "Short setup" in res else 0
    n_fb = int((res["Short setup"] == SH_SETUPS[2]).sum()) if "Short setup" in res else 0
    n_gb, n_gr, n_gn = ((int((res["GLB"] == x).sum()) if "GLB" in res else 0) for x in ("BREAKOUT", "RETEST", "NEAR"))
    n_strong = int((res["Quadrant"] == "Strong").sum()) if "Quadrant" in res else 0
    n_impr = int((res["Quadrant"] == "Improving").sum()) if "Quadrant" in res else 0
    counts = (f"📊 {n_rec} record volume" if is_record_mode() else
              f"💧 {n_ll} pass Liquid Leaders" if is_ll_mode() else
              f"🧭 {len(res)} pass the RS rules · {n_strong} Strong" if is_rs_mode() else
              f"🟢 {n_xt} TRIGGER · 🔵 {n_xs} SETUP (EMA crossback)" if is_xb_mode() else
              (f"🏛 {n_cand} candidates" if ss["sp_type"] == "Inclusion candidates"
               else f"🏛 {n_add} added · {n_rem} removed · {ss['sp_window'].lower()}") if is_sp_mode() else
              f"📉 {n_pb} parabolic · 🔻 {n_crack} first crack" if is_pb_mode() else
              f"🔂 {n_rct} TRIGGER · 🔵 {n_rcs} SETUP · ⏳ {n_rcd} DIP (50-day reclaim)" if is_rc_mode() else
              f"🔃 {n_f5t} TRIGGER · 🔵 {n_f5s} SETUP (first close above the 50-day)" if is_f5_mode() else
              f"🔻 {n_sht} short TRIGGER · 🔵 {n_shs} SETUP (🐻 {n_bf} · 🪤 {n_fb})" if is_sh_mode() else
              f"🚀 {n_ep} episodic pivots · {n_hold} holding" if is_ep_mode() else
              f"📅 {n_earn} with earnings · {ss['earn_window'].lower()}" if is_earn_mode() else
              f"🟢 {n_trig} TRIGGER · 🔵 {n_setup} SETUP")
    # ---- summary cards ----
    mode_cards = {
        "pattern": [("Triggers today", n_trig, "broke out of the base", "g"), ("Setups", n_setup, "still basing", "b")],
        "reclaim": [("Reclaims today", n_rct, "closed back above the 50-day SMA", "g"),
                    ("Armed", n_rcs, "higher low in — buy-stop at the SMA", "b"),
                    ("Dipping", n_rcd, "under the SMA, higher low not yet", "n")],
        "f50scan": [("First closes today", n_f5t, "closed above the 50-day SMA after a long spell under it", "g"),
                    ("Under it, close", n_f5s, "still below the SMA, within a few % — buy-stop at the SMA", "b")],
        "swshort": [("Short triggers", n_sht, "closed below the sell level", "r"),
                    ("Armed", n_shs, "sell-stop for tomorrow", "b"),
                    ("🐻 Bear flags", n_bf, "bounce into a falling SMA", "n"),
                    ("🪤 Failed breakouts", n_fb, "back under the old high", "n")],
        "htf": [("Breakouts today", n_ht, "closed above the flag high", "g"),
                ("In the flag", n_hs, "pole done, resting", "b")],
        "mac": [("Breakouts today", n_mt, "closed above the box high", "g"),
                ("Consolidating", n_ms, "tight, holding the line", "b")],
        "vcp": [("Triggers today", n_vt, "broke out above the pivot", "g"),
                ("Setups", n_vs, "tight, just under the pivot", "b")],
        "so3": [("Triggers today", n_st, "closed above the +3 level", "g"),
                ("Reclaimed", n_sr, "back above the first low", "b"), ("Undercut", n_su, "shaken out — watch", "n")],
        "glb": [("Breakouts today", n_gb, "closed above the green line", "g"),
                ("Retests", n_gr, "back at the line, holding", "b"), ("Near the line", n_gn, "watchlist", "n")],
        "xback": [("Triggers today", n_xt, "broke out after the EMA tag", "g"), ("Setups", n_xs, "at the EMA, holding", "b")],
        "volrec": [("Record volume", n_rec, ss["vr_window"].replace("Highest ", "highest "), "g")],
        "earnwin": [("With earnings", n_earn, ss["earn_window"].lower(), "b")],
        "parabolic": [("Parabolic runs", n_pb, "extended / cracking / fading", "n"), ("First crack", n_crack, "short trigger", "r")],
        "ep": [("Episodic pivots", n_ep, "big gap on huge volume", "g"), ("Holding", n_hold, "above the gap-day low", "b")],
        "spchg": ([("Candidates", n_cand, "size check only", "b")] if ss["sp_type"] == "Inclusion candidates" else
                  [("Added", n_add, ss["sp_window"].lower(), "g"), ("Removed", n_rem, ss["sp_index"], "r")]),
        "liqlead": [("Liquid Leaders", n_ll, "pass both groups", "g")],
        "rsscore": [("Strong quadrant", n_strong, "leading on week and month", "g"),
                    ("Improving", n_impr, "leading this week", "b")],
    }.get(mode_key, [])
    mk = market_mood(MARKETS[universe]["bench"], MARKETS[universe]["bench_name"]) if universe in MARKETS else market_mood()
    ts = ss.get("downloaded_ts")
    age = (f"{int((time.time() - ts) // 60)} min ago" if ts and time.time() - ts < 3600 * 20 else ss.get("downloaded_at", ""))
    cards = [("Matches", f"{len(res):,}", f"of {len(metrics):,} scanned · {n_active} filters on", "n")] + \
            [(a, f"{b:,}", c, d) for a, b, c, d in mode_cards] + \
            ([("Market", mk[0], mk[1], mk[2])] if mk else []) + [("Prices", age, ss.get("downloaded_at", ""), "n")]
    render_cards(cards)
    left = right = st.container()          # table on top, chart + stock card underneath
    with left:
        try:
            tab = pwidget(st.segmented_control, "tab", "Setup", "Columns", TAB_CHOICES, label_visibility="collapsed",
                          format_func=lambda t: f"⚡ {setup_tab}" if t == "Setup" else t)
        except Exception:
            tab = pwidget(st.radio, "tab", "Setup", "Columns", TAB_CHOICES, horizontal=True, label_visibility="collapsed",
                          format_func=lambda t: f"⚡ {setup_tab}" if t == "Setup" else t)
        tab = tab or "Setup"
        eff_tab = setup_tab if tab == "Setup" else tab
        if is_sp_mode() and ss["sp_type"] != "Inclusion candidates" and len(sp_changes):
            back = SP_WINDOWS.get(ss["sp_window"], 31)
            dd = (sp_changes["Change date"] - pd.Timestamp(date.today())).dt.days
            want = sp_pick_index(sp_changes[(dd >= -back) & (dd <= 400)], ss["sp_index"])
            if ss["sp_type"] != "Added or removed":
                want = want[want["S&P change"] == ss["sp_type"].upper()]
            latest = sp_changes["Change date"].max()
            last_ann = sp_changes["Announced"].max() if "Announced" in sp_changes else pd.NaT
            idx_name = "S&P index" if ss["sp_index"] == "Any S&P index" else ss["sp_index"]
            st.caption(f"🏛 {len(want)} {idx_name} changes in this window before your filters · latest effective date listed: "
                       f"**{latest:%b %d, %Y}**" + (f" · latest S&P announcement: **{last_ann:%b %d, %Y}**" if pd.notna(last_ann) else
                       " — S&P's newsroom unreachable and Wikipedia may lag; try a longer window."
                       if (pd.Timestamp(date.today()) - latest).days > 45 else ""))
            no_px = want[~want["Symbol"].isin(metrics.index)]
            if len(no_px):
                st.caption("No price data (not trading yet, acquired/delisted, or press SCAN to fetch): " + " · ".join(
                    (f"{r['Symbol'] or r.get('Company') or '?'} ({r['S&P change'].lower()} {r['Change date']:%b %d})"
                     if pd.notna(r['Change date']) else f"{r['Symbol'] or r.get('Company')}")
                    for _, r in no_px.head(30).iterrows()))

        if tab == "Fundamentals":
            todo = [t for t in res["Symbol"] if t not in ss.get("fund", {})]
            cap = 300
            lab = (f"Load fundamentals for {min(len(todo), cap)} stocks" if todo else "Fundamentals loaded ✓")
            if st.button(lab, disabled=not todo,
                         help="Fetches P/E, growth, short interest and earnings date from Yahoo. "
                              "Narrow your filters first — about 1 second per 8 stocks."):
                bar = st.progress(0.0, text="Loading fundamentals …")
                todo = todo[:cap]
                out = {}
                with ThreadPoolExecutor(8) as ex:
                    for i, (t, f) in enumerate(zip(todo, ex.map(fetch_fundamentals, todo))):
                        out[t] = f
                        bar.progress((i + 1) / len(todo), text=f"Loading fundamentals … {i + 1}/{len(todo)}")
                ss.setdefault("fund", {}).update(out)
                st.rerun()

        if res.empty:
            if is_record_mode():
                st.warning("No stocks match. Try a shorter window (e.g. 3 months), raise **Record set within the "
                           "last N days**, or lower the × average setting in Record-volume settings.")
            elif is_rs_mode():
                st.warning("No stocks pass the RS rules. Lower **Minimum Score**, allow more **Stages**, untick "
                           "the top-half industries rule, or scan a bigger list.")
            elif is_rc_mode():
                st.warning("No 50-day reclaims. Show **TRIGGER, SETUP or DIP**, try **Any** for the SMA direction, "
                           "more **Pop above ≤ days**, fewer **Under the SMA ≥ closes**, a deeper **Dip ≤ %**, "
                           "**Show TRIGGERs from the last N days** = 3–5, or a bigger list.")
            elif is_f5_mode():
                st.warning("No first closes above the 50-day. Lower **Under the 50-day for ≥ N sessions** (e.g. 20 or 10), "
                           "show **TRIGGERs from the last N days** = 3–5, widen **SETUP within % under**, or scan a "
                           "bigger list.")
            elif is_sh_mode():
                st.warning("No swing shorts. In a strong market there are few — try **Both** setups, a smaller "
                           "**Bounce ≥ %**, turn **RS rating ≤** off (0), more **Fails within N days**, or scan a "
                           "bigger list.")
            elif is_htf_mode():
                st.warning("No high tight flags — they are rare (a handful a year). Try the **Loose** preset, a lower "
                           "**Pole ≥ %**, a deeper flag, or scan **All US stocks** in a strong market.")
            elif is_mac_mode():
                st.warning("No MA consolidations. Try fewer **Last N days**, a wider **Range**, a bigger **Allow % "
                           "under the line**, **Every close** instead of every low, or untick the uptrend rule.")
            elif is_vcp_mode():
                st.warning("No VCPs with these settings. Try a smaller **Swing ≥ %**, a longer base, a deeper "
                           "**First pullback**, a looser **Last pullback** or dry-up, untick **Each pullback smaller**, "
                           "or turn off the Trend Template. Real VCPs are rare — scan a big universe.")
            elif is_so_mode():
                st.warning("No shakeout +3 setups. Try a longer base, allow the shakeout further back (**Shakeout "
                           "within N days**), widen the undercut range, lower the prior-uptrend minimum, or pick "
                           "**Any stage**.")
            elif is_gl_mode():
                st.warning("No green-line setups. Breakouts to all-time highs cluster in strong markets — try a bigger "
                           "universe, widen **Green line set … months ago**, raise **NEAR = within % below**, or untick "
                           "**Must be the all-time high**.")
            elif is_xb_mode():
                st.warning("No EMA crossbacks right now. Try a lower rally or RS minimum, widen the touch tolerance, "
                           "allow more days since the tag, untick a strength check, or scan a bigger universe.")
            elif is_ll_mode():
                st.warning("No stocks pass Liquid Leaders. It's strict — try **All US stocks** as the universe, switch Group 2 "
                           "to ANY, or loosen a value (e.g. $ volume ≥ $200M).")
            elif is_sp_mode():
                st.warning("No S&P index changes match. Try a longer **Change date** window (see the latest change date "
                           "above), press **SCAN** if you haven't since choosing this scan, or clear the Price / Avg volume "
                           "filters (removed stocks are often small or thinly traded).")
            elif is_pb_mode():
                st.warning("No parabolic runs with these settings. Try **Any parabolic run**, a lower minimum run-up "
                           "or % above the 10 EMA, or a bigger universe (small caps go parabolic far more often).")
            elif is_ep_mode():
                st.warning("No episodic pivots with these settings. Widen **within the last N days**, lower the gap or "
                           "volume minimum, or scan a bigger universe. EPs cluster around earnings season.")
            elif is_earn_mode():
                st.warning("No stocks in this list have earnings in that window. Try another window, set "
                           "Report time to Any, or clear some filters.")
            else:
                st.warning("No stocks match these filters. Clear some filters, or try the **Loose** pattern preset "
                           "in the sidebar.")
            st.stop()

        MAX_ROWS = 1500
        view = res.head(MAX_ROWS)
        cols = [c for c in COLUMN_SETS[eff_tab] if c in view.columns]
        if eff_tab == "Earnings":   # hide result columns that don't apply (e.g. actual EPS for upcoming reports)
            cols = [c for c in cols if c not in ("EPS actual", "Surprise %", "Earnings gap %", "Reaction %",
                                                 "Since earnings %", "Report time") or view[c].notna().any()]
        view = view[cols].copy()
        also_txt, also_n = also_in(res.head(MAX_ROWS), mode_key)
        view.insert(cols.index("Name") + 1 if "Name" in cols else 1, "Also in", also_txt.to_numpy())
        if "Group" in res:
            view.insert(list(view.columns).index("Also in") + 1, "Group", res.head(MAX_ROWS)["Group"].to_numpy())
        if "Theme" in res and "Theme" not in view:
            at_ = list(view.columns).index("Sector") + 1 if "Sector" in view else len(view.columns)
            view.insert(at_, "Theme", res.head(MAX_ROWS)["Theme"].to_numpy())
        ex_map = get_exchange_map()
        tv_link = lambda syms: "https://www.tradingview.com/chart/?symbol=" + syms.map(lambda t: tv_symbol(t, ex_map))
        if "Name" in view:
            view["Name"] = view["Name"].where(view["Name"].notna(), "").astype(str).replace("nan", "")
        if len(view) <= 600 and "panels" in ss:      # 3-month mini price line next to each symbol
            closes = ss["panels"]["Close"]
            view.insert(1, "3M", [closes[t].iloc[-63:].dropna().round(2).tolist() if t in closes else []
                                  for t in view["Symbol"]])


        def color_pos_neg(v):
            if isinstance(v, (int, float)) and v == v:
                return "color: #26a69a" if v > 0 else ("color: #ef5350" if v < 0 else "")
            return ""


        def color_pattern(v):
            return {"TRIGGER": "color: #26a69a; font-weight: 700", "SETUP": "color: #90caf9",
                    "CRACK": "color: #ef5350; font-weight: 700", "EXTENDED": "color: #ffb74d", "FADING": "color: #9e9e9e",
                    "HOLDING": "color: #26a69a; font-weight: 700", "FAILED": "color: #ef5350",
                    "ADDED": "color: #26a69a; font-weight: 700", "BREAKOUT": "color: #26a69a; font-weight: 700",
                    "RETEST": "color: #90caf9; font-weight: 700", "NEAR": "color: #9e9e9e",
                    "RECLAIMED": "color: #90caf9; font-weight: 700", "UNDERCUT": "color: #ffb74d", "REMOVED": "color: #ef5350; font-weight: 700",
                    "CANDIDATE": "color: #90caf9", "Strong": "color: #3CC47C; font-weight: 700",
                    "Improving": "color: #6fa8ff", "Weakening": "color: #F5C542", "Weak": "color: #E74C4C",
                    **{sg: f"color: {c}; font-weight: 700" for sg, (_, c) in RS_STAGES.items() if sg != "?"}}.get(v, "")


        signed = [c for c in view.columns if c.startswith("Perf") or c in ("Chg %", "Gap %", "vs EMA21 %", "vs SMA20 %", "vs SMA50 %",
                                                                             "vs SMA200 %", "EPS growth %", "Rev growth %",
                                                                             "Record day chg %", "Surprise %", "Earnings gap %", "Chg 5d %", "Chg 20d %", "vs EMA10 %", "vs SMA20 %", "vs SPY 1M pts",
                                                                             "Short vs bench %", "Long vs bench %", "Day %",
                                                                             "EP gap %", "EP day chg %", "Since EP %", "Prior 3M %",
                                                                             "Since change %", "Since announced %",
                                                                             "Reaction %", "Since earnings %")]
        sty = view.style
        if len(view) > 400:            # colouring hundreds of rows is slow; big lists show plain numbers
            signed, sty = [], view
        if signed:
            sty = sty.map(color_pos_neg, subset=signed)
        status_cols = [c for c in ("Pattern", "Parabolic", "EP", "S&P change", "Quadrant", "Stage", "Crossback", "GLB", "SO+3", "VCP", "MAC", "HTF", "Short", "Reclaim", "First50") if c in view]
        if status_cols and len(view) <= 400:
            sty = sty.map(color_pattern, subset=status_cols)

        pct = lambda c: st.column_config.NumberColumn(c, format="%.2f%%")
        cfg = {
            "Chart": st.column_config.LinkColumn("", display_text="📈", width="small"),
            "Price": st.column_config.NumberColumn(format=(MARKETS[universe]["cur"] if universe in MARKETS else "$") + "%.2f"),
            "Mkt cap $B": st.column_config.NumberColumn("Mkt cap", format="$%.2fB"),
            "Volume": st.column_config.NumberColumn(format="compact"),
            "Avg vol 50d": st.column_config.NumberColumn(format="compact"),
            "$ Vol M": st.column_config.NumberColumn("$ Vol", format="$%.1fM"),
            "Rel vol": st.column_config.NumberColumn(format="%.2fx"),
            "Vol ratio": st.column_config.NumberColumn(format="%.2f"),
            "RS": st.column_config.ProgressColumn("RS", format="%d", min_value=1, max_value=99,
                                                  help="Relative strength rank 1-99 vs. the stocks in this scan"),
            "3M": st.column_config.LineChartColumn("3M", width="small", help="Price over the last 3 months"),
            "RSI 14": st.column_config.NumberColumn(format="%.1f"),
            "Days since high": st.column_config.NumberColumn(format="%d"),
            "Earnings in (days)": st.column_config.NumberColumn(format="%d"),
            "P/E": st.column_config.NumberColumn(format="%.1f"),
            "Fwd P/E": st.column_config.NumberColumn(format="%.1f"),
            "Trend template": st.column_config.CheckboxColumn("Stage 2"),
            "Vol record": st.column_config.TextColumn("Record", help="Longest window this volume is the highest of"),
            "Record date": st.column_config.DateColumn(format="MMM D, YYYY"),
            "Record vol": st.column_config.NumberColumn(format="compact"),
            "Record × avg": st.column_config.NumberColumn(format="%.2fx", help="Record-day volume ÷ prior 50-day average"),
            "Earnings date": st.column_config.DateColumn(format="ddd, MMM D"),
            "S&P change": st.column_config.TextColumn("Change"),
            "Score": st.column_config.NumberColumn(format="%.0f", help="Average of Week RS and Month RS (0-100)"),
            "Week RS": st.column_config.NumberColumn(format="%.0f", help="Rank of the short-lookback return vs. the benchmark"),
            "Month RS": st.column_config.NumberColumn(format="%.0f", help="Rank of the long-lookback return vs. the benchmark"),
            "COMP": st.column_config.NumberColumn(format="%.0f", help="Average rank across week, month, 3M, 6M and 1Y"),
            "Stage": st.column_config.TextColumn(help="Cycle-of-price-action stage estimated from 10/20 EMAs and 50/200 SMAs"),
            "Crossback": st.column_config.TextColumn("Crossback", help="TRIGGER = broke out today · SETUP = at the EMA"),
            "Also in": st.column_config.TextColumn("Also in", help="Other scans this stock matches right now: 🚩 Tight flag · "
                                                   "🔁 EMA crossback · ⛳ High tight flag · 🌀 VCP · 📏 MA consolidation · 🟢 Green line · 🪤 Shakeout +3 · 🚀 Episodic "
                                                   "pivot · 📉 Parabolic · 🏛 S&P change · 📊 3-month volume record"),
            "EMA tagged": st.column_config.TextColumn("Line tagged", help="Which line the low touched recently: 10 = the "
                                                      "10-day EMA · 20 = the 20-day SMA"),
            "Theme": st.column_config.TextColumn("Theme", help="The stock's theme: well-known pure plays (quantum, "
                                                 "cybersecurity, crypto, space, nuclear, AI power, solar, EVs …) and "
                                                 "industries that are a theme (solar, uranium, lithium, renewables …)"),
            "Group": st.column_config.TextColumn("Group", help="The stock's industry ETF (or sector ETF) and its status "
                                                 "on the Sectors in play tab: 🔥 in play · 🌱 emerging · ⚠️ cooling · "
                                                 "❄️ weak"),
            "vs EMA10 %": st.column_config.NumberColumn(format="%+.2f%%"),
            "vs SMA20 %": st.column_config.NumberColumn(format="%+.2f%%", help="Close vs the 20-day SMA"),
            "EMA gap %": st.column_config.NumberColumn("10E/20S gap %", format="%.2f%%",
                                                       help="Gap between the 10-day EMA and the 20-day SMA"),
            "Rally %": st.column_config.NumberColumn("Rally off lows", format="%+.0f%%"),
            "Pullback from high %": st.column_config.NumberColumn("Off 20d high", format="%.1f%%"),
            "vs SPY 1M pts": st.column_config.NumberColumn("vs index 1M", format="%+.1f pts",
                                                           help="1-month return minus the market index's (S&P 500, "
                                                                "Hang Seng, SET or Straits Times)"),
            "Tight (3d ÷ ATR)": st.column_config.NumberColumn("Tightness", format="%.2f",
                                                              help="Last 3 days' average range ÷ 20-day ATR (lower = tighter)"),
            "Up/Down vol": st.column_config.NumberColumn(format="%.2fx", help="Avg volume on up days ÷ down days (10d)"),
            "Entry above": st.column_config.NumberColumn(format="$%.2f", help="Last 5 days' high — buy on a close above it"),
            "Stop": st.column_config.NumberColumn(format="$%.2f", help="Lowest low of the last 6 days"),
            "Risk %": st.column_config.NumberColumn(format="%.1f%%", help="Distance from price to the stop"),
            "Avg $ Vol 50d M": st.column_config.NumberColumn("Avg $ Vol 50d", format="$%.0fM",
                                                             help="Average daily dollar volume (price × shares), last 50 days"),
            "ADR % 14d": st.column_config.NumberColumn("ADR % 14d", format="%.2f%%",
                                                       help="Average daily range: average of (high ÷ low − 1), last 14 days"),
            "Under 50d days": st.column_config.NumberColumn("Days under 50d", format="%d",
                                                            help="Sessions in a row it closed at/below its 50-day SMA before "
                                                                 "today's close back above it (blank = not above it today)"),
            "Chg 5d %": st.column_config.NumberColumn("% Chg 5d", format="%+.2f%%"),
            "Chg 20d %": st.column_config.NumberColumn("% Chg 20d", format="%+.2f%%"),
            "Off 52W high %": st.column_config.NumberColumn("% Off 52W high", format="%.2f%%"),
            "Change date": st.column_config.DateColumn("Effective", format="ddd, MMM D, YYYY",
                                                       help="Date the change takes effect (announcements come ~1 week earlier)"),
            "In S&P since": st.column_config.DateColumn("In S&P 500 since", format="MMM D, YYYY",
                                                        help="Inclusion date — when it joined the S&P 500"),
            "Index move": st.column_config.TextColumn(help="Same-day move between S&P indices, e.g. from S&P MidCap 400"),
            "Days to change": st.column_config.NumberColumn("Days", format="%+d", help="+ upcoming, − already effective"),
            "Replaced": st.column_config.TextColumn("Swapped with", help="The stock it replaced (or that replaced it)"),
            "Change reason": st.column_config.TextColumn("Reason", width="medium"),
            "Since change %": st.column_config.NumberColumn(format="%+.2f%%", help="Latest close vs. the close before the effective date"),
            "Announced": st.column_config.DateColumn(format="ddd, MMM D, YYYY", help="Date S&P Dow Jones Indices announced it (usually after the close)"),
            "Since announced %": st.column_config.NumberColumn(format="%+.2f%%", help="Latest close vs. the close on the announcement day"),
            "Source": st.column_config.LinkColumn("Source", display_text="📰 open", help="S&P's press release (or Wikipedia)"),
            "Parabolic": st.column_config.TextColumn("Status", help="CRACK = first red day after the run (short trigger)"),
            "Run %": st.column_config.NumberColumn(format="%+.0f%%", help="Peak (last 3 days) vs. the low of the run"),
            "Run days": st.column_config.NumberColumn(format="%d", help="Trading days from the low to the peak"),
            "Up days in a row": st.column_config.NumberColumn(format="%d"),
            "Above EMA10 %": st.column_config.NumberColumn(format="%+.1f%%", help="Most stretched close vs. the 10 EMA, last 3 days"),
            "Off high %": st.column_config.NumberColumn(format="%.1f%%", help="Below the peak high"),
            "EP": st.column_config.TextColumn("Status", help="HOLDING = above the gap day's low"),
            "EP date": st.column_config.DateColumn(format="ddd, MMM D"),
            "Days since EP": st.column_config.NumberColumn("Days ago", format="%d"),
            "EP vol × avg": st.column_config.NumberColumn(format="%.1fx"),
            "EP low": st.column_config.NumberColumn("EP-day low", format="$%.2f", help="Common stop level"),
            "Days to earnings": st.column_config.NumberColumn("Days", format="%+d",
                                                              help="Days from today: + upcoming, − already reported"),
            "EPS actual": st.column_config.NumberColumn(format="$%.2f", help="Reported EPS"),
            "Surprise %": st.column_config.NumberColumn(format="%+.1f%%", help="Actual vs. estimated EPS"),
            "Earnings gap %": st.column_config.NumberColumn("Gap %", format="%+.2f%%",
                                                            help="Open after the report vs. the close before it"),
            "Reaction %": st.column_config.NumberColumn(format="%+.2f%%",
                                                        help="Close on the first trading day after the report vs. the close before it"),
            "Since earnings %": st.column_config.NumberColumn(format="%+.2f%%",
                                                              help="Latest close vs. the close before the report"),
            "EPS est": st.column_config.NumberColumn(format="$%.2f", help="Analysts' average EPS forecast"),
            "Last yr EPS": st.column_config.NumberColumn(format="$%.2f", help="EPS in the same quarter last year"),
            "# Ests": st.column_config.NumberColumn(format="%d", help="Number of analyst estimates"),
            "Name": st.column_config.TextColumn(width="small"),
            "GLB": st.column_config.TextColumn("Status", help="BREAKOUT · RETEST · NEAR"),
            "SO+3": st.column_config.TextColumn("Status", help="TRIGGER · RECLAIMED · UNDERCUT"),
            "HTF": st.column_config.TextColumn("Status", help="TRIGGER = closed above the flag high today · SETUP = in the flag"),
            "Short": st.column_config.TextColumn("Status", help="TRIGGER = closed below the sell level today · "
                                                                "SETUP = armed: sell-stop at 'Sell below' tomorrow"),
            "Short setup": st.column_config.TextColumn("Setup"),
            "Reclaim": st.column_config.TextColumn("Status", help="TRIGGER = closed back above the 50-day SMA today · "
                                                                  "SETUP = in the dip with a higher low · DIP = in the dip, "
                                                                  "no higher low yet"),
            "50 SMA": st.column_config.NumberColumn(format="%.2f", help="Buy on a close above this"),
            "vs 50 SMA %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Low before pop": st.column_config.NumberColumn(format="%.2f", help="Lowest low of the N days before "
                                                                              "the pop — the dip must stay above it"),
            "Higher low": st.column_config.NumberColumn(format="%.2f", help="The swing low after the dip's lowest low"),
            "Dip low": st.column_config.NumberColumn(format="%.2f", help="The lowest low of the dip under the SMA"),
            "Dip depth %": st.column_config.NumberColumn(format="%.1f%%", help="Deepest low under the SMA in the dip"),
            "Days since pop": st.column_config.NumberColumn(format="%d"),
            "Trigger days ago": st.column_config.NumberColumn("Trig. ago", format="%d",
                                                              help="0 = closed back above the SMA today"),
            "Reclaim stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "Reclaim risk %": st.column_config.NumberColumn("Risk %", format="%.1f%%"),
            "Reclaim vol ×": st.column_config.NumberColumn("Vol ×", format="%.2fx"),
            "First50": st.column_config.TextColumn("Status", help="TRIGGER = first close above the 50-day SMA after N+ "
                                                                  "sessions under it · SETUP = still under it after N+ "
                                                                  "sessions, within a few % — buy-stop at the SMA"),
            "F50 SMA": st.column_config.NumberColumn("50 SMA", format="%.2f", help="Buy on a close above this"),
            "F50 vs SMA %": st.column_config.NumberColumn("vs 50 SMA %", format="%+.1f%%"),
            "F50 days under": st.column_config.NumberColumn("Days under", format="%d",
                                                            help="Sessions in a row at/below the 50-day SMA before the first close above"),
            "F50 trig ago": st.column_config.NumberColumn("Trig. ago", format="%d", help="0 = closed above the SMA today"),
            "F50 stop": st.column_config.NumberColumn("Stop", format="%.2f", help="Lowest low of the last 10 sessions"),
            "F50 risk %": st.column_config.NumberColumn("Risk %", format="%.1f%%"),
            "F50 vol ×": st.column_config.NumberColumn("Vol ×", format="%.2fx"),
            "Sell below": st.column_config.NumberColumn(format="%.2f", help="Short trigger: a trade below this"),
            "To sell level %": st.column_config.NumberColumn("To trigger %", format="%.1f%%",
                                                             help="How far price must fall to the sell level"),
            "Short stop": st.column_config.NumberColumn("Stop", format="%.2f", help="Buy-stop ABOVE — the bounce "
                                                        "high (bear flag) or the failed high (failed breakout)"),
            "Short risk %": st.column_config.NumberColumn("Risk %", format="%.1f%%", help="Entry to stop"),
            "Bounce %": st.column_config.NumberColumn(format="+%.1f%%", help="Bounce off the 15-day low"),
            "Bounce vol ×": st.column_config.NumberColumn(format="%.2fx", help="5-day volume ÷ 50-day avg"),
            "Breakout days ago": st.column_config.NumberColumn("Breakout", format="%d d ago"),
            "Short vol ×": st.column_config.NumberColumn("Vol ×", format="%.2fx", help="Today's volume ÷ 50-day avg"),
            "Pole %": st.column_config.NumberColumn(format="+%.0f%%", help="The run-up before the flag"),
            "Flag depth %": st.column_config.NumberColumn(format="%.1f%%", help="Flag low vs the flag high"),
            "Flag days": st.column_config.NumberColumn(format="%d", help="Days since the high"),
            "Top-2 days %": st.column_config.NumberColumn(format="%.0f%%", help="Share of the pole made by its 2 "
                                                          "biggest days — high = a gap-driven pole"),
            "Flag high": st.column_config.NumberColumn(format="%.2f", help="Buy above this"),
            "To flag high %": st.column_config.NumberColumn(format="%.1f%%"),
            "Flag low": st.column_config.NumberColumn("Stop", format="%.2f", help="Low of the flag"),
            "HTF risk %": st.column_config.NumberColumn("Risk %", format="%.1f%%", help="Flag high to flag low"),
            "Tightness": st.column_config.NumberColumn(format="%.2f", help="Last 5 days' range ÷ the pole's"),
            "HTF dry-up": st.column_config.NumberColumn("Dry-up", format="%.2fx", help="10-day vol ÷ 50-day avg"),
            "HV1 in pole": st.column_config.TextColumn("HV1", help="A highest-volume-in-a-year day in the pole"),
            "HTF vol ×": st.column_config.NumberColumn("Vol today", format="%.2fx"),
            "MAC": st.column_config.TextColumn("Status", help="TRIGGER = closed above the box high today · SETUP = "
                                               "consolidating on the line"),
            "MA held": st.column_config.TextColumn(help="Which SMA the consolidation held: 20, 50 or both"),
            "vs line %": st.column_config.NumberColumn(format="%+.1f%%", help="Close vs that SMA"),
            "Box range %": st.column_config.NumberColumn(format="%.1f%%", help="Range of the consolidation"),
            "Box high": st.column_config.NumberColumn(format="%.2f", help="Buy above this"),
            "Box low": st.column_config.NumberColumn("Stop", format="%.2f", help="Low of the consolidation"),
            "To box high %": st.column_config.NumberColumn(format="%.1f%%"),
            "Box risk %": st.column_config.NumberColumn("Risk %", format="%.1f%%", help="Box high to box low"),
            "Box vol ×": st.column_config.NumberColumn("Vol today", format="%.2fx", help="Today's volume ÷ 50-day avg"),
            "VCP": st.column_config.TextColumn("Status", help="TRIGGER = broke out above the pivot today · SETUP = "
                                               "just under the pivot"),
            "Contractions": st.column_config.TextColumn(help="Depth of each pullback in the base, oldest first"),
            "Last contraction %": st.column_config.NumberColumn("Last pullback", format="%.1f%%"),
            "Pivot": st.column_config.NumberColumn(format="%.2f", help="Buy above this (top of the last contraction)"),
            "To pivot %": st.column_config.NumberColumn(format="%.1f%%", help="How far below the pivot"),
            "VCP stop": st.column_config.NumberColumn("Stop", format="%.2f", help="Low of the last contraction"),
            "VCP risk %": st.column_config.NumberColumn("Risk %", format="%.1f%%", help="Pivot to stop"),
            "Vol dry-up": st.column_config.NumberColumn(format="%.2fx", help="Last 10 days' volume ÷ 50-day avg"),
            "Breakout vol ×": st.column_config.NumberColumn("Vol today", format="%.2fx", help="Today's volume ÷ "
                                                            "50-day avg"),
            "First low": st.column_config.NumberColumn(format="%.2f", help="The W's first low"),
            "Shakeout low": st.column_config.NumberColumn(format="%.2f", help="The undercut (second) low"),
            "Undercut %": st.column_config.NumberColumn("Undercut", format="%.1f%%", help="How far below the first low"),
            "+3 level": st.column_config.NumberColumn("+3 level", format="%.2f", help="The buy level"),
            "To +3 %": st.column_config.NumberColumn("To +3", format="%+.1f%%", help="+ below the level · − above it"),
            "Days since shakeout": st.column_config.NumberColumn("Days since", format="%d"),
            "Shakeout vol ×": st.column_config.NumberColumn("Shakeout vol", format="%.1fx",
                                                            help="Volume on the shakeout day ÷ 50-day average"),
            "SO depth %": st.column_config.NumberColumn("Depth", format="%.0f%%", help="Base high to shakeout low"),
            "SO stop": st.column_config.NumberColumn("Stop", format="%.2f", help="Just under the shakeout low"),
            "Green line": st.column_config.NumberColumn(format="%.2f", help="The all-time high = the entry level"),
            "To green line %": st.column_config.NumberColumn("To line", format="%+.1f%%",
                                                             help="+ below the line · − above it (broken out)"),
            "Line date": st.column_config.DateColumn("Line set", format="MMM D, YYYY"),
            "Line age (months)": st.column_config.NumberColumn("Months", format="%.1f",
                                                               help="How long the green line has stood"),
            "GLB vol ×": st.column_config.NumberColumn("Vol ×", format="%.1fx", help="Today's volume ÷ 50-day average"),
            "ATH": st.column_config.TextColumn("ATH", help="✓ = confirmed all-time high on monthly history back to IPO"),
            "GLB stop": st.column_config.NumberColumn("Stop", format="%.2f",
                                                      help="Just below the green line, or the breakout candle's low"),
        }
        for c in view.columns:
            if c.endswith("%") and c not in cfg:
                cfg[c] = pct(c)

        # click a cell (e.g. the Symbol) to chart that stock — no checkbox column
        try:
            cfg["Symbol"] = st.column_config.TextColumn("Symbol", help="Click a symbol to show its chart", pinned=True)
        except TypeError:                     # older Streamlit: no pinned columns
            cfg["Symbol"] = st.column_config.TextColumn("Symbol", help="Click a symbol to show its chart")
        table_kw = dict(use_container_width=True, hide_index=True, column_config=cfg,
                        height=min(35 * (len(view) + 1) + 3, 660), on_select="rerun", key=f"tbl_{eff_tab}")
        try:
            event = st.dataframe(sty, selection_mode="single-cell", **table_kw)
        except Exception:                     # older Streamlit without cell selection → row checkboxes
            event = st.dataframe(sty, selection_mode="single-row", **table_kw)
        if len(res) > MAX_ROWS:
            st.caption(f"Showing the first {MAX_ROWS:,} rows — add filters to narrow it down.")

        try:
            export_row = st.container(key="exportrow")
        except TypeError:
            export_row = st.container()
        d1, d2, d3, _ = export_row.columns([1, 1, 1, 4], vertical_alignment="center")
        full = res.copy()
        full.insert(1, "Chart", tv_link(full["Symbol"]))
        d1.download_button("⬇ CSV", full.to_csv(index=False).encode(),
                           file_name=f"screener_{date.today():%Y-%m-%d}.csv", mime="text/csv", use_container_width=True)
        tv_res = res.head(1000)
        list_name = f"Screener {date.today():%b %d}"
        d2.download_button("⬇ TV list", tv_watchlist_text(tv_res, ex_map).encode(),
                           file_name=f"{list_name}.txt", mime="text/plain", use_container_width=True,
                           help="In TradingView: open the Watchlist panel → click the list name → "
                                "'Import list…' → pick this file. It appears as a new watchlist.")
        with d3.popover("📋 Tickers", use_container_width=True):
            st.code(",".join(tv_symbol(t, ex_map) for t in tv_res["Symbol"]), language=None)
            st.caption("Or paste into a TradingView watchlist's 'Add symbol' box.")
        if len(res) > 1000:
            st.caption("TradingView watchlists hold up to 1,000 symbols — only the first 1,000 are exported.")

    with right:
        # ---- chart of the selected stock ----
        sel = []
        if event is not None and hasattr(event, "selection"):
            cells = event.selection.get("cells", []) if hasattr(event.selection, "get") else getattr(event.selection, "cells", [])
            if cells:
                c0 = cells[0]
                sel = [c0[0] if isinstance(c0, (list, tuple)) else c0.get("row")]
            else:
                sel = list(getattr(event.selection, "rows", []) or [])
        if sel and sel[0] is not None and 0 <= sel[0] < len(view):
            ss["chart_symbol"] = view["Symbol"].iloc[sel[0]]
        tick_list = view["Symbol"].tolist()
        default_t = ss.get("chart_symbol") if ss.get("chart_symbol") in tick_list else tick_list[0]

        # chart templates: the controls below are loaded from the chosen template; Save writes them back
        TPL_KEYS = {f: f"tpl_{f}" for f in TPL_FIELDS}
        TPL_KEYS.update(interval="chart_int", height="chart_h")


        def apply_template(name=None):
            tpls, active = load_templates()
            name = name or ss.get("tpl_name") or active
            name = name if name in tpls else "Default"
            ss["tpl_name"] = name
            for f, k in TPL_KEYS.items():
                ss[k] = tpls[name][f]


        def current_template():
            return {f: ss[k] for f, k in TPL_KEYS.items()}


        def tpl_save():
            tpls, _ = load_templates()
            tpls[ss["tpl_name"]] = current_template()
            if save_templates(tpls, ss["tpl_name"]):
                ss["tpl_msg"] = f"✅ Saved “{ss['tpl_name']}”."


        def tpl_save_as():
            name = (ss.get("tpl_newname") or "").strip()
            if not name:
                ss["tpl_msg"] = "⚠️ Type a name first."
                return
            tpls, _ = load_templates()
            tpls[name] = current_template()
            if save_templates(tpls, name):
                ss["tpl_name"], ss["tpl_newname"] = name, ""
                ss["tpl_msg"] = f"✅ Saved new template “{name}”."


        def tpl_delete():
            name = ss["tpl_name"]
            if name == "Default":
                ss["tpl_msg"] = "The Default template can be changed but not deleted."
                return
            tpls, _ = load_templates()
            tpls.pop(name, None)
            save_templates(tpls, "Default")
            apply_template("Default")
            ss["tpl_msg"] = f"🗑 Deleted “{name}”."


        def tpl_switch():
            apply_template(ss["tpl_pick"])
            tpls, _ = load_templates()
            save_templates(tpls, ss["tpl_name"])          # remember the last template used


        if any(k not in ss for k in TPL_KEYS.values()):
            apply_template()
        for k in TPL_KEYS.values():      # keep unsaved edits when the chart controls are hidden (e.g. Built-in chart)
            ss[k] = ss[k]
        ss.setdefault("chart_src", "Built-in")
        if not ss.get("_chart_src_v2"):                  # new default view: Built-in (once, then your choice sticks)
            ss["chart_src"], ss["_chart_src_v2"] = "Built-in", True

        # the stock card and chart come first, their settings row below them
        chart_area = st.container()
        c1, c2, c3, c4, c5 = st.columns([1.3, 1.1, 1, 1.4, 1.1], vertical_alignment="bottom")
        pick = c1.selectbox("Chart", tick_list, index=tick_list.index(default_t), help="Or click a symbol in the table.",
                            label_visibility="collapsed")
        chart_src = c2.segmented_control("Chart type", ["Built-in", "TradingView"], key="chart_src",
                                         label_visibility="collapsed") if hasattr(st, "segmented_control") else \
            c2.radio("Chart type", ["Built-in", "TradingView"], key="chart_src", horizontal=True, label_visibility="collapsed")
        chart_src = chart_src or "Built-in"
        tv_sym = tv_symbol(pick, ex_map)
        chart_h = c4.slider("Chart height", 400, 1400, step=50, key="chart_h", label_visibility="collapsed",
                            help="Chart height — drag to make it taller or shorter.")

        if chart_src == "TradingView":
            c3.selectbox("Timeframe", list(TV_INTERVALS), key="chart_int", label_visibility="collapsed")
            with c5.popover(f"⚙️ {ss['tpl_name']}", use_container_width=True, help="Chart template"):
                tpls, _ = load_templates()
                names = list(tpls)
                ss["tpl_pick"] = ss["tpl_name"] if ss["tpl_name"] in names else "Default"
                st.selectbox("Template", names, key="tpl_pick", on_change=tpl_switch)
                st.text_input("Moving averages", key="tpl_mas", placeholder="EMA 10, EMA 21, SMA 50, SMA 200",
                              help="Any mix of EMA / SMA / WMA with a length, separated by commas (up to 8).")
                st.multiselect("Other indicators", list(TV_EXTRAS), key="tpl_extras", placeholder="None")
                a1, a2 = st.columns(2)
                a1.selectbox("Candle style", list(TV_STYLES), key="tpl_style")
                a2.selectbox("Theme", ["Dark", "Light"], key="tpl_theme")
                st.checkbox("Show volume", key="tpl_volume")
                st.checkbox("Show drawing toolbar", key="tpl_toolbar")
                st.checkbox("Show scan results as a watchlist inside the chart", key="tpl_watchlist",
                            help="Click through your matches without leaving the chart.")
                st.text_input("My TradingView layout link (optional)", key="tpl_layout_url",
                              placeholder="https://www.tradingview.com/chart/AbCd1234/",
                              help="Open your own saved chart on tradingview.com, copy the link from the address bar and "
                                   "paste it here. 'Open on TradingView' will then open stocks in YOUR layout, "
                                   "with your own saved indicators and drawings.")
                st.caption("Timeframe and height (next to the chart) are saved with the template too.")
                b1, b2 = st.columns(2)
                b1.button("💾 Save", on_click=tpl_save, use_container_width=True, type="primary")
                b2.button("🗑 Delete", on_click=tpl_delete, use_container_width=True, disabled=ss["tpl_name"] == "Default")
                n1, n2 = st.columns([2, 1])
                n1.text_input("Save as new template", key="tpl_newname", placeholder="e.g. Swing, Day trade",
                              label_visibility="collapsed")
                n2.button("Save as", on_click=tpl_save_as, use_container_width=True)
                if ss.get("tpl_msg"):
                    st.caption(ss.pop("tpl_msg"))

            with chart_area:
                detail_card(pick, metrics, mode_key)
                tpl = current_template()
                watch = [tv_symbol(t, ex_map) for t in tick_list[:100]]
                tradingview_chart(tv_sym, tpl, watch)
            st.caption(f"Live chart from TradingView · template **{ss['tpl_name']}** · the trigger and stop are in the card "
                       f"above — switch to **Built-in** to see them drawn as lines · scroll to zoom, drag to pan, drag "
                       f"the price axis to stretch · [Open {tv_sym} on TradingView ↗]({tv_open_url(tv_sym, tpl['layout_url'])})"
                       + (" in your layout" if tpl["layout_url"] else ""))
        else:
            df = ss["prices"].get(pick)
            if df is not None:
                days = c3.selectbox("Show", [60, 120, 160, 250, 330], index=2, key="chart_days",
                                    format_func=lambda n: f"Last {n} days", label_visibility="collapsed")
                with chart_area:
                    detail_card(pick, metrics, mode_key)
                    fig = price_chart(pick, df, days=days)
                    for lvl, nm, colr in setup_levels(pick, metrics, mode_key)[1]:
                        trig_line = nm.startswith("Trigger")
                        fig.add_hline(y=lvl, line_dash="dot" if trig_line else "dash", line_color=colr,
                                      line_width=1.6 if trig_line else 1.2, row=1, col=1,
                                      annotation_text=f"{nm} {(market_of(pick) or {}).get('cur', '$')}{lvl:,.2f}", annotation_position="top left",
                                      annotation_font_color=colr)
                    # "Signal" marker when the scan triggered today
                    row_ = metrics.loc[pick] if metrics is not None and pick in metrics.index else {}
                    fired = {"pattern": ("Pattern", "TRIGGER"), "xback": ("Crossback", "TRIGGER"), "glb": ("GLB", "BREAKOUT"),
                             "so3": ("SO+3", "TRIGGER"), "vcp": ("VCP", "TRIGGER"), "mac": ("MAC", "TRIGGER"), "htf": ("HTF", "TRIGGER"),
                             "parabolic": ("Parabolic", "CRACK"), "swshort": ("Short", "TRIGGER"),
                             "reclaim": ("Reclaim", "TRIGGER"), "f50scan": ("First50", "TRIGGER")}.get(mode_key)
                    if fired and row_.get(fired[0]) == fired[1]:
                        last_day = df.dropna(subset=["Close"]).index[-1]
                        fig.add_vline(x=last_day, line_dash="dot", line_color="rgba(128,128,128,.6)", line_width=1)
                        fig.add_annotation(x=last_day, y=1, yref="paper", text="Signal", showarrow=False, yanchor="bottom",
                                           font=dict(size=11, color="rgba(160,160,160,1)"))
                    fig.update_layout(height=chart_h)
                    st.plotly_chart(fig, use_container_width=True)
                st.caption("Green dotted line = the trigger (breakout level) · red dashed = the stop · grey dotted "
                           "'Signal' = the scan triggered today. Dotted blue box = the last 7 days' range.")
