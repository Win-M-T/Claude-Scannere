"""
Stock Screener + Tight Flag Scanner — Streamlit app
---------------------------------------------------
One-time install (Command Prompt):
    py -m pip install streamlit plotly yfinance pandas numpy requests lxml

Start it:
    double-click "Start Streamlit Scanner.bat"
    (or:  py -m streamlit run scanner_app.py)

How it works
    1. Pick the stocks to scan in the sidebar and press SCAN (downloads prices once).
    2. Use the filter buttons at the top — just like TradingView's screener.
       Every filter updates the table instantly; no re-download needed.
"""


import html as html_lib
import io
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")


ss = st.session_state
for k, v in PATTERN_PRESETS["Default"].items():
    ss.setdefault(k, v)


for k, v in dict(vr_days=1, vr_min_ratio=1.0, vr_dir="Any", vr_full=False,
                 vr_window="Highest in 3 months", earn_window="Next 1 week", earn_time="Any",
                 scan_mode=M_TF, **SCAN_EXTRA_DEFAULTS).items():
    ss.setdefault(k, v)
# Only one scan's settings is shown at a time. Re-saving these keys stops Streamlit from
# forgetting the hidden scan's settings when you switch back and forth.
for k in list(PATTERN_PRESETS["Default"]) + ["vr_days", "vr_min_ratio", "vr_dir", "vr_full",
                                            "vr_window", "earn_window", "earn_time", "scan_mode"] + list(SCAN_EXTRA_DEFAULTS):
    ss[k] = ss[k]
if ss.get("xb_ema") not in XB_EMAS:                      # older saved choice (the 20-day line is now an SMA)
    ss["xb_ema"] = {"20-day EMA": XB_20, "10-day EMA": "10-day EMA"}.get(ss.get("xb_ema"), XB_EMAS[0])
if ss["earn_window"] not in EARN_WINDOWS:
    ss["earn_window"] = "Next 1 week"
if ss.get("v_f_earnwin", "Any") not in list(EARN_WINDOWS) + ["Any"]:
    ss["v_f_earnwin"] = ss["earn_window"] if ss["scan_mode"] == M_EARN else "Any"
ss["scan_mode"] = OLD_SCAN_NAMES.get(ss["scan_mode"], ss["scan_mode"])
if ss["scan_mode"] not in SCAN_MODES:
    ss["scan_mode"] = M_TF


# Filter choices are stored under "v_<name>" and copied into the widget ("w_<name>") on every run.


for key in F:
    ss.setdefault(f"v_f_{key}", DEFAULT_CHOICE.get(key, "Any"))
ss.setdefault("v_visible", [F[k]["label"] for k in DEFAULT_VISIBLE])



ss.setdefault("page", PAGES[0])
if ss.get("page") not in PAGES:
    ss["page"] = PAGES[0]
RSP_DEFAULTS = dict(etf_kind="All", etf_view="Ranking",rsp_tickers="NVDA, AMD, SMCI, TSLA, PLTR, COIN", rsp_bench="SPY", rsp_short=5, rsp_long=21,
                    rsp_mbench="SPY", rsp_mshort=5, rsp_mlong=21, rsp_mmin=3, rsp_mstages=RS_STAGE_ORDER[:-1],
                    rsp_mtop=True, rsp_mstrong=False, rsp_wsrc="Stock Scanner results", rsp_wtext="",
                    pp_pool="Same sector", pp_n=10, pp_look=90, pp_minsim=0.3, pp_mincap=1.0, pp_pool_n=150,
                    pp_ticker="", pp_stocks_only=True,
                    sm_view="🏦 Famous investors (13F)", sm_funds=SM_DEFAULT_FUNDS, sm_ciks="", sm_email="",
                    sm_mode="Consensus", sm_sort="Most held", sm_days=30, sm_chamber="Both", sm_type="All",
                    sm_amount="Any amount", sm_member="", sm_tick="", sm_only_tick=True, sm_house_max=150,
                    sm_cview="Latest trades",
                    bt_strategy="🚩 Tight Flag Pattern", bt_combo=["🔁 EMA Crossback", "📈 Shakeout +3"], bt_combo_mode="Signals from any of them", bt_combo_win=10, bt_universe="S&P 500", bt_tickers="", bt_period="1 year",
                    bt_entry="Next day's open", bt_stop="The setup's stop", bt_acct=100_000, bt_risk=1.0, bt_maxpos=100, bt_max_open=10, bt_priority="Highest RS first",
                    bt_compound=True, bt_stop_pct=8.0, bt_target="None",
                    bt_target_r=3.0, bt_target_pct=20.0, bt_adr_x=2.0, bt_min_stop=1.0, bt_max_stop=0.0, bt_pause=0, bt_pause_ind=0, bt_cut_days=0, bt_cut_r=1.0, bt_pyr=False, bt_pyr_step=1.5, bt_pyr_max=5, bt_pyr_stop=PYR_STOPS[0], bt_pyr_unit=2.0, bt_pyr_rule=PYR_RULES[0], bt_pyr_profit=True, bt_groups="All groups", bt_minpx=0.0, bt_minvol=0, bt_mindvol=0.0, bt_minrs=0, bt_grp_rs=0, bt_grp_corr=0.3, bt_stage_size=False, bt_stage_bench="QQQE", bt_f50_days=30, bt_fn_ind_top=0, bt_fn_ind_min=3, bt_fn_lead=95, bt_fn_corr=0.3, bt_fn_stages=[],
                    **{f"bt_stg_{k}": v for k, v in STAGE_SIZE_DEFAULT.items()}, bt_grp_sec=True, bt_grp_ind=False, bt_grp_thm=False, bt_market="Off", bt_mkt_above=["21 EMA"], bt_mkt_below=[], bt_atr_x=10.0, bt_atr_ma=50, bt_trail="20-day SMA", bt_max_days=60, bt_max_stocks=1500,
                    bt_cmin=None, bt_cmax=None)
for k, v in {**RSP_DEFAULTS, **FN_DEFAULTS}.items():
    ss.setdefault(k, v)
# settings of the page you're NOT on would otherwise be forgotten by Streamlit — re-save them each run
for k in [k for k in list(ss.keys()) if str(k).startswith(("rsp_", "pp_", "sm_", "bt_", "etf_", "fn_")) and k not in ("bt_tbl", "etf_tbl", "etf_cache", "etf_asof") and k not in ("rsp_upload", "rsp_wupload")] + \
        ["page"] + (["universe"] if "universe" in ss else []):
    try:
        ss[k] = ss[k]
    except Exception:            # buttons / download buttons / uploaders can't be re-saved — nothing to keep
        pass
try:
    _tabs_box = st.container(key="pagetabs")
except TypeError:
    _tabs_box = st.container()
with _tabs_box:
    try:
        st.segmented_control("Page", PAGES, key="page", label_visibility="collapsed")
    except Exception:
        st.radio("Page", PAGES, key="page", horizontal=True, label_visibility="collapsed")
_page_fn = dict(zip(PAGES[1:], [etf_page, rs_page, peers_page, smart_money_page, backtest_page, positions_page, funnel_page]))
if (ss.get("page") or PAGES[0]) in _page_fn:
    _page_fn[ss["page"]]()
    st.stop()


# ---- saved scan presets (scan type + its settings + filters) ----


with st.sidebar:
    st.header("📂 Stocks to scan")
    def universe_changed():
        """US and Asian markets need different starting filters (prices differ a lot between markets)."""
        asia = ss["universe"] in MARKETS
        if asia != ss.get("_asia", False):
            ss["pre_pmin"], ss["pre_cmin"] = (None, 0.5) if asia else (5.0, 0.3)
            ss["pre_sectors"], ss["pre_industries"] = [], []
            ss["v_f_price"] = "Any" if asia else DEFAULT_CHOICE["price"]
        ss["_asia"] = asia

    universe = st.selectbox("Universe", ["S&P 500 (~500 · 1–2 min)", "Nasdaq-listed (~3,500)",
                                         "All US stocks (~6,000+)", *MARKETS, "My own tickers"],
                            key="universe", on_change=universe_changed,
                            help="Hong Kong, Thailand and Singapore prices come from Yahoo Finance (tickers like "
                                 "0700.HK, PTT.BK, D05.SI) and are shown in the local currency.")
    mkt = MARKETS.get(universe)
    my_text = ""
    if universe == "My own tickers":
        up = st.file_uploader("Upload a watchlist (.txt / .csv)", type=["txt", "csv"])
        my_text = st.text_area("…or type tickers", placeholder="NTRA, NOW, NVDA", height=80)
        if up is not None:
            my_text += "\n" + up.getvalue().decode("utf-8", errors="ignore")

    base_err = None
    try:
        with st.spinner("Loading stock list …"):
            base_tickers, base_meta = base_universe(universe, my_text)
    except Exception as e:
        base_tickers, base_meta, base_err = [], None, str(e)

    pre = None
    has_meta = base_meta is not None and len(base_meta) and universe != "My own tickers"
    # ---- 🏷 Sectors: tick the ones to scan ----
    ss.setdefault("sec_excl", list(DEFAULT_SECTOR_EXCL))
    try:
        # only the sectors of the stocks in the chosen list (the S&P list and Nasdaq's list name sectors differently)
        if base_meta is not None and len(base_meta) and "Sector" in base_meta and base_tickers:
            sec_src = base_meta.reindex(base_tickers)
        elif universe == "My own tickers" and parse_tickers(my_text):
            sec_src = sector_lookup(parse_tickers(my_text))
        else:
            sec_src = get_market_meta(universe) if mkt else get_screener_meta()
        sec_names = sorted({str(x).strip() for x in sec_src["Sector"].dropna() if str(x).strip()})
    except Exception:
        sec_names = []
    if has_meta:
        def _pre_summary():
            bits = []
            lo, hi = ss.get("pre_pmin"), ss.get("pre_pmax")
            if lo or hi:
                bits.append(f"${lo:g}+" if lo and not hi else f"≤ ${hi:g}" if hi and not lo else f"${lo:g}–{hi:g}")
            lo, hi = ss.get("pre_cmin"), ss.get("pre_cmax")
            if lo or hi:
                bits.append(f"cap {lo:g}B+" if lo and not hi else f"cap ≤ {hi:g}B" if hi and not lo
                            else f"cap {lo:g}–{hi:g}B")
            if ss.get("pre_vol", "Any") != "Any":
                bits.append(f"vol {ss['pre_vol']}")
            if ss.get("pre_sectors"):
                bits.append(f"{len(ss['pre_sectors'])} sector" + ("s" if len(ss["pre_sectors"]) > 1 else ""))
            if ss.get("pre_industries"):
                bits.append(f"{len(ss['pre_industries'])} industr" + ("ies" if len(ss["pre_industries"]) > 1 else "y"))
            if ss.get("pre_us"):
                bits.append("US only")
            n_out = sum(n.lower() in {x.lower() for x in ss.get("sec_excl", [])} for n in sec_names)
            if n_out:
                bits.append(f"{len(sec_names) - n_out} of {len(sec_names)} sectors")
            return " · ".join(bits) or "no limits"
        ss.setdefault("pre_pmin", 5.0)
        ss.setdefault("pre_cmin", 0.3)
        st.caption(f"🔎 Pre-filter: {_pre_summary()}")
        with st.expander("Edit pre-filter", expanded=False, icon=":material/tune:"):
            st.caption("Only stocks that pass these are downloaded — fewer stocks = faster scan. " +
                       (f"Uses today's snapshot from TradingView's {mkt['country']} screener. **Price and market cap "
                        f"here are in US$** (converted from {mkt['cur']}), so one setting works for every market."
                        if mkt else "Uses today's snapshot from Nasdaq's stock list."))
            # blank box = no limit (defaults: price from $5, market cap from $0.3B)
            ss.setdefault("pre_pmin", 5.0)
            ss.setdefault("pre_cmin", 0.3)
            c1, c2 = st.columns(2)
            pmin = c1.number_input("Price from $", min_value=0.0, value=None, step=1.0, key="pre_pmin",
                                   placeholder="No limit")
            pmax = c2.number_input("to $", min_value=0.0, value=None, step=10.0, key="pre_pmax",
                                   placeholder="No limit", help="Leave blank for no limit")
            c1, c2 = st.columns(2)
            cmin = c1.number_input("Mkt cap from $B", min_value=0.0, value=None, step=0.1, key="pre_cmin",
                                   placeholder="No limit")
            cmax = c2.number_input("to $B", min_value=0.0, value=None, step=1.0, key="pre_cmax",
                                   placeholder="No limit", help="Leave blank for no limit")
            vol_opts = {"Any": 0, "100K+": 1e5, "300K+": 3e5, "500K+": 5e5, "1M+": 1e6, "5M+": 5e6}
            vsel = st.select_slider("Volume today at least", list(vol_opts), value="Any", key="pre_vol",
                                    help="Shares traded in the latest session (a quick liquidity check).")
            bm = base_meta.reindex(base_tickers)
            if sec_names:
                sector_picker(sec_names, "secchk_", "Tick the sectors to scan. The Backtest tab uses the same "
                                                    "choice.", inline=True)
            sectors = []                                  # the ticked sectors are applied by drop_excluded()
            ss["pre_sectors"] = []
            ind_src = bm[[not is_excluded_sector(x) for x in bm["Sector"].fillna("")]]
            ind_opts = sorted(ind_src["Industry"].dropna().unique())
            ss["pre_industries"] = [x for x in ss.get("pre_industries", []) if x in ind_opts]
            industries = st.multiselect("Industries", ind_opts, key="pre_industries", placeholder="All industries")
            us_only = st.checkbox("US-based companies only", value=False, key="pre_us") if not mkt else False
            pre = dict(pmin=pmin, pmax=pmax, cmin=cmin, cmax=cmax, vmin=vol_opts[vsel],
                       sectors=sectors, industries=industries, us_only=us_only)
    elif universe != "My own tickers" and not base_err:
        st.caption(f"Couldn't load the full {mkt['country']} stock list, so the {mkt['bench_name']} index members "
                   "will be scanned instead (no pre-filter)." if mkt else
                   "Pre-filters unavailable — couldn't load Nasdaq's stock list, so all stocks will be downloaded.")

    if sec_names and not has_meta:                        # no pre-filter panel (own tickers …): a button instead
        sector_picker(sec_names, "secchk_", "Tick the sectors to scan. The Backtest tab uses the same choice.")
    scan_tickers = drop_excluded(prefilter(base_tickers, base_meta, pre), base_meta)
    if base_err:
        st.error(f"Couldn't load the stock list: {base_err}")
    elif scan_tickers:
        secs = len(scan_tickers) * 0.15
        eta = "under a minute" if secs < 60 else f"about {round(secs / 60)} min"
        st.markdown(f"**{len(scan_tickers):,}** of {len(base_tickers):,} stocks will be downloaded · {eta}")
        if ss.get("sec_excl"):
            st.caption("Left out: " + ", ".join(ss["sec_excl"]))
    elif universe != "My own tickers":
        st.warning("No stocks pass the pre-filter. Loosen it.")

    SCAN_HELP = ("Tight-flag = basing / breaking-out stocks. Record volume = heaviest volume in 3M/6M/1Y/since IPO. "
                  "Earnings = upcoming or recent reports. Parabolic short = huge fast run-ups, and the first crack. "
                  "Episodic pivot = big gap up on huge volume, usually on earnings or news. "
                  "S&P index changes = stocks added to / removed from the S&P 500, 100, MidCap 400 or SmallCap 600. "
                  "Liquid Leaders = very liquid, volatile stocks with strong recent momentum (two filter groups). "
                  "EMA crossback = a leading stock pulls back to its rising 10 EMA / 20 SMA, holds, then breaks out. "
                  "Green line breakout = a stock clears its all-time high after 3+ months of going sideways below it. "
                  "Shakeout +3 = a W-bottom: price undercuts the base's first low, then rallies back above it. "
                  "RS score = the strongest stocks vs the market, ranked like the Relative Strength tab. "
                  "50-day reclaim = a pop above the 50-day SMA fails, a shallow dip makes a higher low, then it "
                  "closes back above the line. "
                  "First close above 50-day = the first close above the 50-day SMA after N+ sessions at or below it. "
                  "Swing shorts = bear flags (a weak stock bounces into a falling 20/50-day SMA and rolls over) and "
                  "failed breakouts (a breakout that closes back under the old high) — traded short.")

    def pick_scan(m):
        """Click a scan's name: select it (its settings stay closed)."""
        if ss["scan_mode"] != m:
            ss["scan_mode"] = m
            scan_mode_changed()
            ss["scan_open"] = False

    def toggle_scan(m):
        """Click the arrow: open / close that scan's settings (selecting it if needed)."""
        if ss["scan_mode"] != m:
            ss["scan_mode"] = m
            scan_mode_changed()
            ss["scan_open"] = True
        else:
            ss["scan_open"] = not ss.get("scan_open", False)

    ss.setdefault("scan_open", False)

    # "Scan for" is one list: click a scan to select it — its options open right there
    try:
        scan_list = st.container(key="scanlist")
    except TypeError:                               # older Streamlit: no keyed containers
        scan_list = st.container()
    ss.setdefault("scan_open", False)
    scan_list.markdown("**Scan for**", help=SCAN_HELP)
    cur = SCAN_MODES.index(ss["scan_mode"])
    def scan_row(i, m):
        """Name = select the scan · arrow = open / close its settings."""
        on = i == cur
        try:
            row = scan_list.container(key=f"scanrow_{'on' if on else i}")
        except TypeError:
            row = scan_list.container()
        c1, c2 = row.columns([5, 1], gap="small", vertical_alignment="center")
        c1.button(m, key=f"scanpick_{i}", on_click=pick_scan, args=(m,), use_container_width=True)
        try:
            c2.button("", key=f"scanarrow_{i}", on_click=toggle_scan, args=(m,), use_container_width=True,
                      icon=":material/keyboard_arrow_right:")
        except TypeError:                           # older Streamlit: no button icons
            c2.button("›", key=f"scanarrow_{i}", on_click=toggle_scan, args=(m,), use_container_width=True)

    for i, m in enumerate(SCAN_MODES[:cur]):
        scan_row(i, m)
    # ---- one dropdown for the chosen scan: its choices first, then its settings ----
    full_hist = bool(ss["vr_full"])
    sp_changes, sp_err, sp_extra = pd.DataFrame(), None, []
    # the chosen scan is an expander: its arrow opens / closes the settings right in the browser (no reload),
    # its name does nothing (CSS below makes only the arrow clickable)
    with scan_list.expander(ss["scan_mode"], expanded=bool(ss.get("scan_open", False))):
        if ss["scan_mode"] == M_TF:
            st.caption("Presets")
            b1, b2, b3 = st.columns(3)
            b1.button("Default", on_click=load_pattern_preset, args=("Default",), use_container_width=True)
            b2.button("Strict", on_click=load_pattern_preset, args=("Strict",), use_container_width=True)
            b3.button("Loose", on_click=load_pattern_preset, args=("Loose",), use_container_width=True)
            try:
                svg, note = tight_flag_sketch(ss["max_range"], ss["pull_range"], ss["min_bars"], ss["max_ema_dist"],
                                              ss["max_ema_gap"], ss["dry_up"], ss["breakout_vol"])
                st.markdown(f'<div style="margin:.2rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Example with your settings — redraws as you change them. " + note)
            except Exception:
                pass
            st.slider("Max 7-day range %", 3.0, 15.0, step=0.5, key="max_range", help="How tight the base must be.")
            st.slider("Pullback % from 20-day high", 0.0, 30.0, step=0.5, key="pull_range")
            st.slider("Min days since the high", 1, 15, step=1, key="min_bars")
            st.slider("Max % above 20 SMA", 0.5, 8.0, step=0.5, key="max_ema_dist")
            st.slider("Max gap 10 EMA / 20 SMA %", 0.5, 5.0, step=0.25, key="max_ema_gap")
            st.slider("Max volume ratio (5d / 50d)", 0.3, 1.2, step=0.05, key="dry_up")
            st.slider("Breakout volume × avg", 1.0, 3.0, step=0.1, key="breakout_vol")
        if is_record_mode():
            if V("f_volrec") in VOLREC_OPTS:         # keep in step with the filter button above the table
                ss["vr_window"] = V("f_volrec")
            st.selectbox("Highest volume in", VOLREC_OPTS, key="vr_window", on_change=vr_window_changed,
                         format_func=lambda x: x.replace("Highest in ", "").replace("Highest ", "").capitalize())
            if ss["vr_window"] == "Highest since IPO" and not ss["vr_full"]:
                st.caption("⚠️ Only stocks listed in the last ~2 years can match until you tick "
                           "**Download full price history** below and SCAN again.")

            st.divider()
            st.slider("Record set within the last N days", 1, 10, step=1, key="vr_days",
                      help="1 = today's session must be the record. Raise it to catch records from earlier this week.")
            st.slider("…and at least × 50-day avg volume", 1.0, 5.0, step=0.25, key="vr_min_ratio",
                      help="Skips 'records' that are barely above normal volume.")
            st.radio("Price on the record day", ["Any", "Up", "Down"], key="vr_dir", horizontal=True,
                     help="Up-day records often mean buying (accumulation); down-day records often mean selling.")
            full_hist = st.checkbox("Download full price history (for 'Since IPO')", key="vr_full",
                                    help="Without it, prices go back 2 years — enough for 3M/6M/1Y, and for "
                                         "'Since IPO' on stocks listed in the last 2 years. Older stocks need "
                                         "full history (slower download). Press SCAN after ticking.")
        if is_earn_mode():
            if V("f_earnwin") in EARN_DAYS:          # keep in step with the filter button above the table
                ss["earn_window"] = V("f_earnwin")
            st.selectbox("Earnings reported", list(EARN_WINDOWS), key="earn_window", on_change=earn_window_changed,
                         help="Today / next week = upcoming reports. Previous = companies that already reported — "
                              "shows actual EPS, surprise and how the stock reacted.")

            st.divider()
            st.radio("Report time", ["Any", "Before open", "After close"], key="earn_time", horizontal=True)
            st.caption("Dates come from Nasdaq's earnings calendar (confirmed and expected dates). "
                       "Dates further out are more often estimates and can move.")
        pb_opts = [p[0] for p in F["parabolic"]["presets"]]
        ep_opts = [p[0] for p in F["ep"]["presets"]]
        if is_pb_mode():
            if V("f_parabolic") in pb_opts:
                ss["pb_status"] = V("f_parabolic")
            st.selectbox("Show", pb_opts, key="pb_status", on_change=pb_status_changed)
            st.divider()
            st.slider("Run-up measured over the last N days", 3, 30, step=1, key="pb_days")
            st.slider("Minimum run-up %", 20, 500, step=10, key="pb_gain",
                      help="Rule of thumb: large caps ~50–100% in days/weeks; small caps 200–300%+.")
            st.slider("Minimum up days in a row", 1, 8, step=1, key="pb_up")
            st.slider("Minimum % above the 10 EMA (at the peak)", 5, 100, step=5, key="pb_ext",
                      help="How stretched the stock got above its 10-day EMA.")
            st.caption("**CRACK** = first red day after the run (the usual short trigger) · "
                       "**EXTENDED** = still rising · **FADING** = already rolling over.")
        if is_ep_mode():
            if V("f_ep") in ep_opts:
                ss["ep_status"] = V("f_ep")
            st.selectbox("Show", ep_opts, key="ep_status", on_change=ep_status_changed)

            st.divider()
            st.slider("Gap happened within the last N days", 1, 20, step=1, key="ep_days")
            st.slider("Minimum gap up %", 3.0, 40.0, step=1.0, key="ep_gap",
                      help="Open vs. the previous close.")
            st.slider("Minimum volume × 50-day average", 1.5, 15.0, step=0.5, key="ep_volx")
            st.checkbox("Must close strong (green, upper half of the day's range)", key="ep_strong")
            st.checkbox("Only neglected stocks", key="ep_neglect_on",
                        help="Skip stocks that already ran up a lot in the 3 months before the gap.")
            if ss["ep_neglect_on"]:
                st.slider("…prior 3-month gain at most %", -50, 200, step=5, key="ep_neglect")
            st.caption("**HOLDING** = still above the gap day's low (the usual stop) · **FAILED** = broke it. "
                       "Check the news/earnings behind the gap — the catalyst is what makes an EP.")
        if is_sp_mode():
            sp_opts = [p[0] for p in F["spchg"]["presets"]]
            if V("f_spchg") in sp_opts:
                ss["sp_type"] = V("f_spchg")
            st.selectbox("Index", SP_INDEXES, key="sp_index")
            st.selectbox("Show", sp_opts, key="sp_type", on_change=sp_type_changed)
            if ss["sp_type"] == "Inclusion candidates" and ss["sp_index"] != "S&P 500":
                st.caption("Inclusion candidates are worked out for the **S&P 500** only.")
            if ss["sp_type"] != "Inclusion candidates":
                st.selectbox("Change date", list(SP_WINDOWS), key="sp_window",
                             help="Every choice also includes changes that are announced but not yet effective.")
            try:
                sp_changes = sp_changes_all()
            except Exception as e:
                sp_err = str(e)
            # these stocks are downloaded too, even if they're outside the chosen universe / pre-filter
            recent = sp_changes[sp_changes["Change date"] >= pd.Timestamp(date.today()) - pd.Timedelta(days=366)] \
                if len(sp_changes) else sp_changes
            if ss["sp_type"] != "Inclusion candidates":
                recent = sp_pick_index(recent, ss["sp_index"])
            try:
                members, _ = get_sp500()
            except Exception:
                members = []
            cands = sp_candidates(get_screener_meta(), members, ss["sp_mincap"]) \
                if ss["sp_index"] == "S&P 500" or ss["sp_type"] == "Inclusion candidates" else pd.DataFrame()
            sp_extra = [t for t in list(recent["Symbol"]) + list(cands.index) if t and t not in set(scan_tickers)]
            sp_extra = drop_excluded(list(dict.fromkeys(sp_extra)))
            scan_tickers = scan_tickers + sp_extra
            # already-downloaded ones show up right away (no need to press SCAN again)
            if ss.get("scan_set") is not None:
                ss["scan_set"] = ss["scan_set"] | {t for t in sp_extra if t in ss.get("prices", {})}
            if len(sp_changes) and not bool(sp_changes["Official"].iloc[0]):
                st.caption("⚠️ Couldn't reach S&P's newsroom (press.spglobal.com) — using Wikipedia only.")
            elif len(sp_changes):
                st.caption("Source: S&P Dow Jones Indices announcements, plus Wikipedia for older changes.")
            if sp_err:
                st.error(f"Couldn't load S&P 500 changes: {sp_err}")
            elif sp_extra:
                pending = [t for t in sp_extra if t not in ss.get("prices", {}) and t not in ss.get("tried", set())]
                st.caption(f"+ {len(pending)} recent S&P changes / candidates outside your list — press **SCAN** "
                           "to download them." if pending else
                           f"Includes {len(sp_extra)} recent S&P changes / candidates outside your list.")

            st.divider()
            st.number_input("Candidates: minimum market cap $B", 1.0, 200.0, step=0.5, key="sp_mincap",
                            help="S&P's size bar for new members (it was raised to $22.7B in July 2025 — "
                                 "S&P updates it from time to time).")
            st.caption("Changes come from S&P Dow Jones Indices' own press releases (press.spglobal.com — "
                       "announcement + effective dates), with Wikipedia filling in older changes. Announcements "
                       "usually come about a week before the effective date. Candidates are a size check only — S&P also "
                       "requires positive earnings, enough trading and float, and picks at its discretion.")

        if is_xb_mode():
            xb_opts = [p[0] for p in F["xback"]["presets"]]
            if V("f_xback") in xb_opts:
                ss["xb_status"] = V("f_xback")
            st.selectbox("Show", xb_opts, key="xb_status", on_change=xb_status_changed)
            try:
                svg, note = xb_sketch(ss["xb_rally"], ss["xb_rally_days"], ss["xb_ema"], ss["xb_tol"], ss["xb_days"],
                                      ss["xb_pull_max"], ss["xb_tight"], ss["xb_tight_max"], ss["xb_vol"], ss["xb_candle"])
                st.markdown(f'<div style="margin:.2rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Example TRIGGER with your settings — redraws as you change them."
                           + (f"\n\n{note}" if note else ""))
            except Exception:
                pass
            st.divider()
            st.markdown("**1 · Leading stock**")
            c1, c2 = st.columns(2)
            c1.number_input("Rally off the lows ≥ %", 0, 300, step=5, key="xb_rally",
                            help="Low-to-high run-up, with the high in the last N days.")
            c2.number_input("…high within N days", 10, 120, step=5, key="xb_rally_days")
            st.slider("Minimum RS rating", 1, 99, key="xb_rs")
            st.checkbox(f"Beating the {mkt['bench_name'] if mkt else 'S&P 500'} over the last month", key="xb_beat",
                        help="Relative strength while the market lags — the guide's early-mover sign.")
            st.markdown("**2 · Pullback to the 10 EMA / 20 SMA**")
            if ss.get("xb_ema") not in XB_EMAS:                 # older saved choices
                ss["xb_ema"] = {"20-day EMA": XB_20, "10-day EMA": "10-day EMA"}.get(ss.get("xb_ema"), XB_EMAS[0])
            st.radio("EMA", XB_EMAS, key="xb_ema", horizontal=True, label_visibility="collapsed")
            c1, c2 = st.columns(2)
            c1.number_input("Touch tolerance %", 0.0, 5.0, step=0.25, key="xb_tol",
                            help="How close the day's low must come to the EMA (the close must hold above it).")
            c2.number_input("Tagged within N days", 1, 10, key="xb_days")
            st.slider("Max pullback from 20-day high %", 3, 40, key="xb_pull_max")
            st.caption("Both EMAs must be rising, and price above the 50-day.")
            st.markdown("**3 · Signs of strength**")
            st.checkbox("Tight price bars", key="xb_tight")
            if ss["xb_tight"]:
                st.slider("…last 3 days' range ÷ ATR at most", 0.4, 1.5, step=0.05, key="xb_tight_max")
            st.checkbox("More volume on up days than down days (10d)", key="xb_vol")
            st.checkbox("Bullish candle today (hammer / engulfing)", key="xb_candle")
            st.caption("**4 · Stop** = lowest low of the last 6 days · **5 · Entry** = a close above the last 5 days' "
                       "high. **TRIGGER** = that breakout happened today after tagging the EMA.")

        if is_gl_mode():
            gl_opts = [p[0] for p in F["glb"]["presets"]]
            if V("f_glb") in gl_opts:
                ss["gl_status"] = V("f_glb")
            st.selectbox("Show", gl_opts, key="gl_status", on_change=gl_status_changed)
            try:
                svg, note = green_line_sketch(ss["gl_months"], ss["gl_vol"], ss["gl_near"], ss["gl_retest_days"],
                                              ss["gl_tol"], ss["gl_stop"])
                st.markdown(f'<div style="margin:.2rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Example with your settings — redraws as you change them. " + note)
            except Exception:
                pass
            st.slider("Green line set … months ago", 1, 15, key="gl_months",
                      help="The all-time high (the green line) must be at least this old — 3+ months of sideways "
                           "action under it — and not older than the upper value.")
            st.checkbox("Must be the all-time high", key="gl_ath",
                        help="Downloads monthly history back to IPO for the matches to confirm nothing was higher "
                             "before. Untick to accept the highest high of the last ~15 months.")
            st.divider()
            c1, c2 = st.columns(2)
            c1.number_input("Breakout volume ≥ × avg", 1.0, 5.0, step=0.1, key="gl_vol",
                            help="Breakout day's volume vs the 50-day average — 'above-average volume'.")
            c2.number_input("NEAR = within % below", 0.5, 20.0, step=0.5, key="gl_near")
            c1, c2 = st.columns(2)
            c1.number_input("Retest: broke out within N days", 3, 40, key="gl_retest_days")
            c2.number_input("…and dipped within % of line", 0.0, 10.0, step=0.5, key="gl_tol")
            st.number_input("Stop: % below the green line", 0.0, 10.0, step=0.5, key="gl_stop",
                            help="Or the breakout candle's low, whichever is lower.")
            st.caption("**Green line** = the all-time high, left untouched for 3+ months. **BREAKOUT** = first close "
                       "above it today on above-average volume · **RETEST** = broke out, pulled back to the line and "
                       "holding it (the second-chance entry) · **NEAR** = just below it (watchlist). Consider "
                       "selling part after a 20–25% gain; trail with the 21-day EMA or 10-week line.")

        if is_so_mode():
            so_opts = [p[0] for p in F["so3"]["presets"]]
            if V("f_so3") in so_opts:
                ss["so_status"] = V("f_so3")
            st.selectbox("Show", so_opts, key="so_status", on_change=so_status_changed)
            try:
                svg, note = shakeout_sketch(ss["so_rule"], ss["so_pct"], ss["so_base"], ss["so_recent"], ss["so_under"],
                                            ss["so_mid"], ss["so_depth"], ss["so_trend"], ss["so_vol"], ss["so_stop"],
                                            entry=ss["so_entry"], fast=int(ss["so_fast"]), shvol=float(ss["so_shvol"]))
                st.markdown(f'<div style="margin:.2rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Example with your settings — redraws as you change them. " + note)
            except Exception:
                pass
            st.radio("Buy on", SO_ENTRIES, key="so_entry",
                     help="**+3 level** — the close above the +3 level (the article's rule). **Reclaim** — the first "
                          "close back above the first low after the shakeout: earlier, with the stop much closer "
                          "(just under the shakeout low).")
            if ss["so_entry"] != SO_RECLAIM:
                st.selectbox("+3 buy level", ["10% above the first low", "Livermore $3 ($6 above $60)"], key="so_rule",
                         help="The article's two versions: 10% above the first low, or Jesse Livermore's $3 above "
                              "it ($6 for stocks above $60).")
                if ss["so_rule"] == "10% above the first low":
                    st.slider("% above the first low", 3.0, 20.0, step=0.5, key="so_pct")
            st.markdown("**A real shakeout**")
            c1, c2 = st.columns(2)
            c1.number_input("Buy within N days of the low", 0, 30, key="so_fast",
                            help="A real shakeout is a quick flush that fails: the buy (reclaim or +3) must come "
                                 "within this many days of the shakeout low. 0 = off.")
            c2.number_input("Shakeout-day vol ≥ × avg", 0.0, 5.0, step=0.1, key="so_shvol",
                            help="Heavy volume on the day of the shakeout low = sellers flushed out. 0 = off.")
            st.divider()
            st.markdown("**The W-bottom**")
            c1, c2 = st.columns(2)
            c1.number_input("Base length (days)", 30, 200, step=5, key="so_base",
                            help="How far back to look for the first low")
            c2.number_input("Shakeout within N days", 3, 40, key="so_recent",
                            help="The undercut must be recent")
            st.slider("Undercut of the first low, %", 0.0, 30.0, step=0.5, key="so_under",
                      help="How far the shakeout dips below the first low")
            c1, c2 = st.columns(2)
            c1.number_input("Rally between lows ≥ %", 0.0, 40.0, step=1.0, key="so_mid",
                            help="The middle of the W — a bounce between the two lows")
            c2.number_input("Base depth ≤ %", 5.0, 70.0, step=1.0, key="so_depth",
                            help="From the base's high to the shakeout low")
            st.markdown("**Prior uptrend**")
            st.number_input("Base high ≥ % above the low of the 12 months before", 0.0, 300.0, step=5.0, key="so_trend",
                            help="'After a prolonged uptrend when the stock consolidates its gains'")
            st.checkbox("Base high above the 200-day average", key="so_above200")
            st.number_input("RS rating ≥", 0, 99, step=5, key="so_rs",
                            help="Only stocks stronger than this share of the others over the last year (1–99). "
                                 "In your backtests, shakeouts in RS 50+ stocks won far more often. 0 = off.")
            st.markdown("**Trigger & stop**")
            c1, c2 = st.columns(2)
            c1.number_input("Trigger volume ≥ × avg", 0.0, 5.0, step=0.1, key="so_vol",
                            help="0 or 1 = any volume; higher = demand confirmation")
            c2.number_input("Stop % below shakeout low", 0.0, 10.0, step=0.5, key="so_stop")
            st.caption("**UNDERCUT** = shaken out below the first low (watch) · **RECLAIMED** = back above the first "
                       "low (early entry zone) · **TRIGGER** = closed above the +3 level today. Stop = just under "
                       "the shakeout low.")

        if is_htf_mode():
            htf_opts = [p[0] for p in F["htf"]["presets"]]
            if V("f_htf") in htf_opts:
                ss["htf_status"] = V("f_htf")
            st.selectbox("Show", htf_opts, key="htf_status", on_change=htf_status_changed)
            st.caption("Presets")
            p1, p2, p3 = st.columns(3)
            for col_, nm, lbl in zip((p1, p2, p3), HTF_PRESETS, ("Strict", "Bulkowski", "Loose")):
                col_.button(lbl, key=f"htfp_{lbl}", on_click=htf_load_preset, args=(nm,), use_container_width=True,
                            help=nm + ": pole ≥ " + f"{HTF_PRESETS[nm]['htf_pole']:g}% · flag "
                                 f"{HTF_PRESETS[nm]['htf_depth'][0]:g}–{HTF_PRESETS[nm]['htf_depth'][1]:g}% deep")
            try:
                svg, note = htf_sketch(ss["htf_pole"], ss["htf_depth"], ss["htf_flag_min"], ss["htf_flag_max"],
                                       ss["htf_vol"], ss["htf_dry"])
                st.markdown(f'<div style="margin:.2rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Example with your settings — redraws as you change them. " + note)
            except Exception:
                pass
            st.markdown("**The pole**")
            c1, c2 = st.columns(2)
            c1.number_input("Pole ≥ %", 20.0, 500.0, step=10.0, key="htf_pole",
                            help="The flag's top close vs the lowest close of the days before it. O'Neil / Stockbee: "
                                 "+100%. Bulkowski: +90%. Half-size HTF: +50%.")
            c2.number_input("…within N days", 10, 80, step=5, key="htf_pole_days", help="40 ≈ 8 weeks")
            st.number_input("Top 2 days ≤ % of the pole", 20.0, 100.0, step=5.0, key="htf_share",
                            help="Skips poles made mostly of 1–2 giant gap days (blow-offs). 100 = off.")
            st.checkbox("Pole has a highest-volume-in-a-year day (HV1)", key="htf_hv",
                        help="The strongest poles start with a record-volume day — big funds buying.")
            st.markdown("**The flag**")
            st.slider("Flag depth, % under the high", 0.0, 50.0, step=1.0, key="htf_depth",
                      help="O'Neil 10–25% (best 10–20%) · Bulkowski 10–34%.")
            c1, c2 = st.columns(2)
            c1.number_input("Flag ≥ days", 2, 30, key="htf_flag_min", help="One week of weakness is not a flag.")
            c2.number_input("Flag ≤ days", 5, 60, key="htf_flag_max", help="≈ 2–5 weeks = 10–25 trading days")
            c1, c2 = st.columns(2)
            c1.number_input("Tightness ≤ × pole range", 0.2, 3.0, step=0.1, key="htf_tight",
                            help="Last 5 days' average daily range ÷ the pole's. Under 1 = calmer than the run-up.")
            c2.number_input("Dry-up: 10d vol ≤ × avg", 0.0, 3.0, step=0.1, key="htf_dry", help="0 = off")
            st.checkbox("No more than 1 heavy-volume red day in the flag", key="htf_nored",
                        help="A down day on 1.5× average volume = someone big selling.")
            st.checkbox("10-day SMA above the 20-day", key="htf_ma", help="The flag holds high, surfing the short MAs")
            st.markdown("**Breakout & liquidity**")
            c1, c2 = st.columns(2)
            c1.number_input("Breakout vol ≥ × avg", 0.0, 5.0, step=0.1, key="htf_vol",
                            help="And above the day before (Stockbee). 0 = off.")
            c2.number_input("Close in top % of range", 0.0, 100.0, step=5.0, key="htf_close_top",
                            help="Breakout day closes near its high. 30 = top 30% of the day's range. 0 = off.")
            c1, c2 = st.columns(2)
            c1.number_input("Price ≥ $", 0.0, 100.0, step=1.0, key="htf_minpx")
            c2.number_input("Avg \\$ vol ≥ \\$M", 0.0, 500.0, step=1.0, key="htf_mindvol",
                            help="50-day average of price × volume. Thin, cheap names don't count (Stockbee).")
            st.caption("**SETUP** = resting in the flag (buy-stop over the flag high) · **TRIGGER** = closed above the "
                       "flag high today. Stop = the flag low — often 10–20% away, so size down. Real HTFs are rare: "
                       "a handful a year.")

        if is_rc_mode():
            rc_opts = [p[0] for p in F["reclaim"]["presets"]]
            if V("f_reclaim") in rc_opts:
                ss["rc_status"] = V("f_reclaim")
            st.selectbox("Show", rc_opts, key="rc_status", on_change=rc_status_changed)
            try:
                svg, note = reclaim_sketch(ss["rc_trend"], ss["rc_dip_max"], int(ss["rc_min_dip"]), int(ss["rc_max_days"]),
                                           ss["rc_hl"], ss["rc_stop"] == RC_STOPS[0], ss["rc_prior"])
                st.markdown(f'<div style="margin:.2rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Example with your settings — redraws as you change them. " + note)
            except Exception:
                pass
            st.radio("50-day SMA at the pop", RC_TRENDS, key="rc_trend",
                     help="**Turnaround** — the SMA was still falling (a stock coming out of a decline). "
                          "**Pullback** — the SMA was rising (a stock in an uptrend that dipped under it). "
                          "Direction = the SMA vs 20 days earlier.")
            st.markdown("**The pop**")
            c1, c2 = st.columns(2)
            c1.number_input("Under the SMA ≥ closes before", 1, 60, key="rc_below",
                            help="Closes under the 50-day SMA before the pop — the stock was below the line.")
            c2.number_input("…of the last (that + N) closes", 0, 20, key="rc_slack",
                            help="Allows a few brief pokes above the SMA before the pop. Example: 10 + 2 → at least "
                                 "10 of the last 12 closes under the SMA. 0 = strictly in a row.")
            c1, c2 = st.columns(2)
            c1.number_input("Pop above ≤ days", 1, 30, key="rc_pop_max",
                            help="Closes above the SMA during the pop before it falls back under (a 1-day dip that "
                                 "pops right back up still counts as the same pop).")
            st.markdown("**The dip**")
            c1, c2 = st.columns(2)
            c1.number_input("Dip ≤ % under the SMA", 1.0, 30.0, step=0.5, key="rc_dip_max",
                            help="Every low of the dip stays within this % under the 50-day SMA — not far away.")
            c2.number_input("Dip ≥ closes under", 1, 20, key="rc_min_dip")
            c1, c2 = st.columns(2)
            c1.number_input("Pop → reclaim ≤ days", 3, 120, key="rc_max_days")
            st.checkbox("Higher low inside the dip (after the pop)", key="rc_hl",
                        help="After the dip's lowest low, price bounces and pulls back to a swing low that stays "
                             "above it — a higher low — before closing back above the SMA.")
            st.selectbox("Stop", RC_STOPS, key="rc_stop")
            c1, c2 = st.columns([1.3, 1], vertical_alignment="bottom")
            c1.checkbox("Dip low above the low before the pop", key="rc_prior",
                        help="The dip's lowest low (and so the higher low) must stay above the lowest low of the N "
                             "days before the pop — the low after the pop is higher than the low before it. If the "
                             "dip undercuts that low, the setup is dropped.")
            c2.number_input("…low of the last N days", 5, 120, key="rc_base")
            st.markdown("**Trigger & quality**")
            c1, c2 = st.columns(2)
            c1.number_input("Reclaim vol ≥ × avg", 0.0, 5.0, step=0.1, key="rc_vol", help="0 = off")
            c2.number_input("RS rating ≥", 0, 99, step=5, key="rc_rs", help="0 = off")
            c1, c2 = st.columns(2)
            c1.number_input("Show TRIGGERs from the last N days", 1, 10, key="rc_recent",
                            help="1 = today only. Higher = also list stocks that reclaimed in the last N days and are "
                                 "still above the SMA and the stop (see 'Trigger days ago').")
            c1, c2 = st.columns(2)
            c1.number_input("Price ≥ $", 0.0, 500.0, step=1.0, key="rc_minpx")
            c2.number_input("Avg \\$ vol ≥ \\$M", 0.0, 2000.0, step=1.0, key="rc_mindvol")
            st.caption("**DIP** = in the dip under the 50-day SMA, higher low not formed yet (early watchlist) · "
                       "**SETUP** = in the dip with a higher low (buy-stop at the SMA) · **TRIGGER** = closed back "
                       "above the SMA. Stop = under the higher low (or the dip's lowest low).")

        if is_f5_mode():
            f5_opts = [p[0] for p in F["f50scan"]["presets"]]
            if V("f_f50scan") in f5_opts:
                ss["f5_status"] = V("f_f50scan")
            st.selectbox("Show", f5_opts, key="f5_status", on_change=f5_status_changed)
            st.markdown("**The first close**")
            c1, c2 = st.columns(2)
            c1.number_input("Under the 50-day for ≥ N sessions", 1, 120, key="f5_days",
                            help="Closes in a row at or below the 50-day SMA right before today's close above it. "
                                 "30 = the first close above the line in 30+ sessions.")
            c2.number_input("SETUP within % under", 0.0, 20.0, step=0.5, key="f5_near",
                            help="Stocks still under the SMA after N+ sessions and within this % of it are the "
                                 "watchlist (SETUP). 0 = no SETUPs.")
            c1, c2 = st.columns(2)
            c1.number_input("Show TRIGGERs from the last N days", 1, 10, key="f5_recent",
                            help="1 = today only. Higher = also list stocks that closed above in the last N days and "
                                 "are still above the SMA and the stop.")
            c2.number_input("Trigger vol ≥ × avg", 0.0, 5.0, step=0.1, key="f5_vol", help="0 = off")
            c1, c2 = st.columns(2)
            c1.number_input("Price ≥ $", 0.0, 500.0, step=1.0, key="f5_minpx")
            c2.number_input("Avg \\$ vol ≥ \\$M", 0.0, 2000.0, step=1.0, key="f5_mindvol")
            st.caption("**TRIGGER** = closed above the 50-day SMA after N+ sessions at/below it · **SETUP** = still "
                       "under it after N+ sessions and close to it (buy-stop at the SMA). Stop = lowest low of the "
                       "last 10 sessions. The 🔃 First Close Above 50-day strategy in the Backtest tests the same idea.")

        if is_sh_mode():
            sh_opts = [p[0] for p in F["swshort"]["presets"]]
            if V("f_swshort") in sh_opts:
                ss["sh_status"] = V("f_swshort")
            st.selectbox("Show", sh_opts, key="sh_status", on_change=sh_status_changed)
            st.radio("Setups", SH_SETUPS, key="sh_which", horizontal=True)
            show_bf, show_fb = ss["sh_which"] != SH_SETUPS[2], ss["sh_which"] != SH_SETUPS[1]
            try:
                sk = []
                if show_bf:
                    sk.append(("🐻 Bear flag", bear_flag_sketch(ss["sh_bf_ma"], ss["sh_bf_tol"], ss["sh_bf_bounce"],
                                                               ss["sh_bf_dry"], int(ss["sh_bf_days"]))))
                if show_fb:
                    sk.append(("🪤 Failed breakout", failed_breakout_sketch(int(ss["sh_fb_len"]), int(ss["sh_fb_days"]),
                                                                          ss["sh_fb_vol"])))
                for ttl, svg in sk:
                    st.markdown(f'<div style="font-size:.8rem;opacity:.8">{ttl}</div>'
                                  f'<div style="margin:.1rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Examples with your settings — they redraw as you change them. Short = sell first, buy "
                           "back lower; the stop sits ABOVE the entry.")
            except Exception:
                pass
            if show_bf:
                st.markdown("**🐻 Bear flag** — a weak stock bounces into a falling moving average and rolls over")
                st.selectbox("Bounce tags the", SH_MAS, key="sh_bf_ma")
                c1, c2 = st.columns(2)
                c1.number_input("High within % under it", 0.0, 5.0, step=0.25, key="sh_bf_tol",
                                help="The day's high gets within this % of the SMA (or above it), but the close stays "
                                     "below it.")
                c2.number_input("Bounce ≥ % off 15d low", 0.0, 30.0, step=0.5, key="sh_bf_bounce")
                c1, c2 = st.columns(2)
                c1.number_input("Bounce vol ≤ × avg (5d)", 0.0, 3.0, step=0.1, key="sh_bf_dry",
                                help="Light volume on the way up = no real buyers. 0 = off.")
                c2.number_input("RS rating ≤", 0, 99, key="sh_bf_rs",
                                help="Only laggards (weaker than most stocks over the last year). 0 = off.")
                st.number_input("Armed for N days", 1, 10, key="sh_bf_days",
                                help="After the tag of the SMA, a sell-stop sits at the lowest low since then for "
                                     "this many days.")
                st.caption("Always: 50-day SMA under the 200-day and falling, close under the 50-day SMA. "
                           "**Sell below** = the lowest low since the SMA tag · **stop** = the bounce high.")
            if show_fb:
                st.markdown("**🪤 Failed breakout** — a breakout above a base that closes back inside it (bull trap)")
                c1, c2 = st.columns(2)
                c1.number_input("Breakout above the N-day high", 10, 120, step=5, key="sh_fb_len",
                                help="A close above the highest high of the prior N days.")
                c2.number_input("…that stood ≥ days", 1, 60, key="sh_fb_base",
                                help="The old high was set at least this many days before — a real base, not "
                                     "yesterday's high in a stock already running.")
                c1, c2 = st.columns(2)
                c1.number_input("Fails within N days", 1, 10, key="sh_fb_days",
                                help="A close back under the old high within this many days of the breakout.")
                c2.number_input("Failure-day vol ≥ × avg", 0.0, 5.0, step=0.1, key="sh_fb_vol", help="0 = off")
                st.checkbox("Only under the 200-day SMA", key="sh_fb_200",
                            help="Failed breakouts in a stock that's already in a long-term downtrend.")
                st.caption("**Sell below** = the old high (the pivot) · **stop** = the highest high since the "
                           "breakout.")
            st.markdown("**Liquidity**")
            c1, c2 = st.columns(2)
            c1.number_input("Price ≥ $", 0.0, 500.0, step=1.0, key="sh_minpx")
            c2.number_input("Avg \\$ vol ≥ \\$M", 0.0, 2000.0, step=5.0, key="sh_mindvol",
                            help="50-day average of price × volume. Shorts need liquid names that are easy to borrow.")
            st.caption("**SETUP** = armed: place a sell-stop at **Sell below** for tomorrow · **TRIGGER** = closed "
                       "below it today. Shorts work best when the market itself is weak (SPY under its 50-day). "
                       "Check that shares are available to borrow.")

        if is_mac_mode():
            mac_opts = [p[0] for p in F["mac"]["presets"]]
            if V("f_mac") in mac_opts:
                ss["mac_status"] = V("f_mac")
            st.selectbox("Show", mac_opts, key="mac_status", on_change=mac_status_changed)
            try:
                svg, note = mac_sketch(ss["mac_ma"], ss["mac_days"], ss["mac_tol"], ss["mac_range"], ss["mac_near"],
                                       ss["mac_vol"], ss["mac_hold"])
                st.markdown(f'<div style="margin:.2rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Example with your settings — redraws as you change them. " + note)
            except Exception:
                pass
            st.markdown("**The line**")
            st.radio("Consolidating above", MAC_LINES, key="mac_ma", horizontal=True)
            c1, c2 = st.columns(2)
            c1.radio("Must hold", ["Every close", "Every low"], key="mac_hold",
                     help="Every close = the closes stayed above the line (intraday dips allowed). Every low = "
                          "stricter, not even the lows went under.")
            c2.number_input("Allow % under the line", 0.0, 5.0, step=0.25, key="mac_tol",
                            help="A small undercut still counts as holding.")
            st.checkbox("The line must be rising", key="mac_rising", help="Higher than 10 days ago.")
            st.markdown("**The box**")
            c1, c2 = st.columns(2)
            c1.number_input("Last N days", 5, 40, key="mac_days", help="How long the consolidation has lasted.")
            c2.number_input("Range ≤ %", 2.0, 30.0, step=0.5, key="mac_range",
                            help="(highest high − lowest low) ÷ lowest low over those days.")
            st.number_input("Close ≤ % above the line", 0.5, 20.0, step=0.5, key="mac_near",
                            help="Resting on the line, not extended above it.")
            st.markdown("**Trend & trigger**")
            st.checkbox("Uptrend: above the 200-day, 50-day above the 200-day", key="mac_trend")
            c1, c2 = st.columns(2)
            c1.number_input("RS rating ≥", 0, 99, key="mac_rs", help="0 = off")
            c2.number_input("Breakout vol ≥ × avg", 0.0, 5.0, step=0.1, key="mac_vol",
                            help="TRIGGER day volume vs the 50-day average. 0 = off.")
            st.caption("**SETUP** = consolidating on the line now (buy-stop above the box high) · **TRIGGER** = "
                       "closed above the box high today. Stop = the box low.")

        if is_vcp_mode():
            vcp_opts = [p[0] for p in F["vcp"]["presets"]]
            if V("f_vcp") in vcp_opts:
                ss["vcp_status"] = V("f_vcp")
            st.selectbox("Show", vcp_opts, key="vcp_status", on_change=vcp_status_changed)
            try:
                svg, note = vcp_sketch(ss["vcp_swing"], ss["vcp_min_c"], ss["vcp_first_max"], ss["vcp_last_max"],
                                       ss["vcp_dry"], ss["vcp_vol"], ss["vcp_near"], ss["vcp_base"])
                st.markdown(f'<div style="margin:.2rem 0 .1rem 0">{svg}</div>', unsafe_allow_html=True)
                st.caption("Example with your settings — redraws as you change them. " + note)
            except Exception:
                pass
            st.markdown("**The base**")
            c1, c2 = st.columns(2)
            c1.number_input("Base = last N days", 20, 250, step=5, key="vcp_base",
                            help="The base starts at the highest high of these days (at least 15 days ago).")
            c2.number_input("Swing ≥ %", 1.0, 10.0, step=0.5, key="vcp_swing",
                            help="A pullback / rally must be at least this big to count as a swing. Smaller = more, "
                                 "tinier contractions detected.")
            c1, c2 = st.columns(2)
            c1.number_input("Min contractions", 2, 5, key="vcp_min_c", help="Minervini: usually 2–4 (up to 6).")
            c2.number_input("Stop risk ≤ %", 2.0, 25.0, step=0.5, key="vcp_risk",
                            help="Pivot to the last contraction's low. Minervini keeps it under ~8–10%.")
            c1, c2 = st.columns(2)
            c1.number_input("First pullback ≤ %", 5.0, 60.0, step=1.0, key="vcp_first_max")
            c2.number_input("Last pullback ≤ %", 2.0, 25.0, step=0.5, key="vcp_last_max",
                            help="The tightest, final contraction right before the pivot.")
            st.checkbox("Each pullback smaller than the one before", key="vcp_shrink")
            st.markdown("**Volume & trigger**")
            c1, c2 = st.columns(2)
            c1.number_input("Dry-up: 10d vol ≤ × avg", 0.0, 2.0, step=0.05, key="vcp_dry",
                            help="Last 10 days' average volume vs the 50-day average. 0 = off.")
            c2.number_input("Breakout vol ≥ × avg", 0.0, 5.0, step=0.1, key="vcp_vol",
                            help="Volume on the TRIGGER day vs the 50-day average. 0 = off.")
            st.number_input("SETUP = within % under the pivot", 1.0, 20.0, step=0.5, key="vcp_near")
            st.markdown("**Trend Template**")
            c1, c2 = st.columns([1.3, 1])
            c1.checkbox("Must pass the Trend Template", key="vcp_tt",
                        help="Price above the 50/150/200-day MAs · 50 > 150 > 200 · 200-day rising for a month · "
                             "≥ 25% above the 52-week low · within 25% of the 52-week high · RS ≥ the minimum.")
            c2.number_input("RS ≥", 1, 99, key="vcp_rs")
            st.caption("**SETUP** = a valid VCP just under its pivot (watchlist, buy-stop above the pivot) · "
                       "**TRIGGER** = closed above the pivot today on volume. Stop = the last contraction's low.")

        if is_rs_mode():
            rss_opts = [p[0] for p in F["rsscore"]["presets"]]
            if V("f_rsscore") in rss_opts:
                ss["rss_status"] = V("f_rsscore")
            st.selectbox("Show", rss_opts, key="rss_status", on_change=rss_status_changed)
            st.divider()
            st.text_input("Benchmark", key="rs_bench", help="Returns are compared with this — SPY, QQQ, IWM …")
            c1, c2 = st.columns(2)
            c1.number_input("Short (days) · Week RS", 2, 60, key="rs_short")
            c2.number_input("Long (days) · Month RS", 5, 250, key="rs_long")
            st.caption("5 days = 1 week, 21 days = 1 month, 63 days = 3 months.")
            st.slider("Minimum Score", 0, 100, key="rss_min",
                      help="Score = average of Week RS and Month RS (0–100, ranked against the stocks in this scan).")
            c1, c2 = st.columns(2)
            c1.slider("Min Week RS", 0, 100, key="rss_wk")
            c2.slider("Min Month RS", 0, 100, key="rss_mo")
            st.slider("Minimum COMP", 0, 100, key="rss_comp", help="COMP = average rank of week, month, 3M, 6M and 1Y.")
            st.multiselect("Stages to include", RS_STAGE_ORDER[:-1], key="rss_stages",
                           help=" · ".join(f"{sg} = {RS_STAGES[sg][0]}" for sg in RS_STAGE_ORDER[:-1]))
            st.checkbox('Only stocks in top-half industries ("one rule")', key="rss_top",
                        help="The stock's industry must rank in the top half (industry Score ≥ 50).")
            if ss["rss_top"]:
                st.slider("…min stocks per industry", 1, 20, key="rss_mingrp")
            st.caption("Same numbers as the Relative Strength tab's table. Ranks are against every stock in this "
                       "scan, so scan a big list (e.g. S&P 500 or All US stocks) for meaningful ranks.")

        if is_ll_mode():
            ll_opts = [p[0] for p in F["liqlead"]["presets"]]
            if V("f_liqlead") in ll_opts:
                ss["ll_status"] = V("f_liqlead")
            st.selectbox("Show", ll_opts, key="ll_status", on_change=ll_status_changed)
            for g, (title, conds, extra) in enumerate([("Group 1", LL_G1, True), ("Group 2", LL_G2, False)], start=1):
                with st.container(border=True):
                    st.markdown(f"**{title}** · must match")
                    if hasattr(st, "segmented_control"):
                        st.segmented_control("Match", ["ALL", "ANY"], key=f"ll_g{g}_mode", label_visibility="collapsed")
                    else:
                        st.radio("Match", ["ALL", "ANY"], key=f"ll_g{g}_mode", horizontal=True,
                                 label_visibility="collapsed")
                    for k, (lab, col, dflt, unit, step) in conds.items():
                        r1, r2 = st.columns([1.6, 1], vertical_alignment="center")
                        r1.checkbox(lab, key=f"ll_on_{k}")
                        r2.number_input(lab, step=step, key=f"ll_v_{k}", label_visibility="collapsed",
                                        disabled=not ss[f"ll_on_{k}"], format="%g")
                    if extra:
                        st.caption("ETF / ETN: **No** — the stock lists already leave out ETFs and ETNs.")
            st.caption("A stock must pass **Group 1 and Group 2**. Inside a group, **ALL** = every ticked condition, "
                       "**ANY** = at least one. Works best on a big universe (e.g. All US stocks).")

    for i, m in enumerate(SCAN_MODES[cur + 1:], start=cur + 1):
        scan_row(i, m)

    def start_scan():
        ss["scanning"], ss["scan_req"] = True, True
        ss.pop("scan_stopped", None)
        ss.pop("stop_toasted", None)

    def stop_scan():
        ss["scanning"] = False
        ss.pop("scan_req", None)
        ss["scan_stopped"] = True

    try:
        scan_bar = st.container(key="scanbar")
    except TypeError:
        scan_bar = st.container()
    with scan_bar:
        r1, r2 = st.columns([3.2, 1], gap="small", vertical_alignment="center")
        if ss.get("scanning"):
            with r1.container(key="scanstop"):
                st.button("■  STOP", type="primary", use_container_width=True, on_click=stop_scan,
                          help="Stop the scan. Results from your last finished scan stay on screen.")
        else:
            with r1.container(key="scango"):
                st.button("▶  SCAN", type="primary", use_container_width=True, on_click=start_scan)
        scan = bool(ss.pop("scan_req", False))
        refresh = st.checkbox("Re-download prices", value=False,
                              help="Prices are kept for the session. Tick to fetch fresh ones.")
        with r2.popover("⭐", use_container_width=True, help="Saved scans — save this scan and its filters to reuse later"):
            presets = load_scan_presets()
            if presets:
                ss.setdefault("preset_pick", next(iter(presets)))
                if ss["preset_pick"] not in presets:
                    ss["preset_pick"] = next(iter(presets))
                st.selectbox("Saved scans", list(presets), key="preset_pick")
                p1, p2 = st.columns(2)
                p1.button("Load", on_click=apply_scan_preset, use_container_width=True, type="primary")
                p2.button("🗑 Delete", on_click=delete_scan_preset, use_container_width=True)
            else:
                st.caption("No saved scans yet.")
            n1, n2 = st.columns([2, 1], vertical_alignment="bottom")
            n1.text_input("Save current scan as", key="preset_new", placeholder="e.g. Morning EP scan")
            n2.button("Save", on_click=save_scan_preset, use_container_width=True)
            if ss.get("preset_msg"):
                st.caption(ss.pop("preset_msg"))


pp = build_pp(universe)

# ============================================================================
# 8. SCAN = download prices (only when needed)
# ============================================================================
data_key = (universe, date.today().isoformat(), bool(full_hist))
if ss.get("scan_warn"):
    st.warning(ss.pop("scan_warn"))
    st.stop()
if ss.get("scan_err"):
    st.error(ss.pop("scan_err"))
    st.stop()

if scan:
    if universe == "My own tickers" and not parse_tickers(my_text):
        ss["scanning"], ss["scan_warn"] = False, "Type some tickers or upload a watchlist first."
        st.rerun()
    if not scan_tickers:
        ss["scanning"], ss["scan_warn"] = False, "No stocks to download — check the pre-filter or the stock list."
        st.rerun()
    ss["scan_set"] = set(scan_tickers)
    have = ss.get("prices", {}) if ss.get("data_key", (None,))[:3] == data_key and not refresh else None
    todo = [t for t in scan_tickers if t not in (have or {}) and t not in ss.get("tried", set())]
    if have is not None and todo:
        # same universe/day: only fetch the stocks we don't have yet, then merge
        with st.status(f"Downloading {len(todo):,} more stocks …", expanded=True) as status:
            msg, bar = st.empty(), st.progress(0.0)
            new = download_all(todo, bar, msg, period="max" if full_hist else "2y")
            ss["tried"] = ss.get("tried", set()) | set(todo)
            if new:
                panels = build_panels({**have, **new})
                vh_new = volume_history(new, panels["Close"].index[0], full_hist)
                if full_hist:
                    new = {t: df.tail(520) for t, df in new.items()}
                ss["prices"] = {**have, **new}
                ss["panels"] = panels
                ss["vol_hist"] = pd.concat([ss.get("vol_hist", pd.DataFrame()), vh_new])
                ss["meta"] = base_meta.combine_first(ss["meta"]) if base_meta is not None and len(ss.get("meta", [])) \
                    else (base_meta if base_meta is not None else ss.get("meta"))
                ss["data_key"] = data_key + (len(ss["prices"]),)
                ss["downloaded_ts"] = time.time()
                ss.pop("metrics_key", None)
            status.update(label=f"Added {len(new):,} of {len(todo):,} stocks", state="complete", expanded=False)
        st.rerun()
    if refresh or have is None:
        with st.status("Scanning …", expanded=True) as status:
            msg = st.empty()
            tickers, meta = scan_tickers, base_meta if base_meta is not None else pd.DataFrame()
            bar = st.progress(0.0)
            prices = download_all(tickers, bar, msg, period="max" if full_hist else "2y")
            if not prices:
                status.update(label="No price data downloaded", state="error")
                ss["scanning"] = False
                ss["scan_err"] = "Nothing downloaded. Check your internet connection or ticker symbols."
                st.rerun()
            msg.write("Calculating indicators …")
            panels = build_panels(prices)
            ss["vol_hist"] = volume_history(prices, panels["Close"].index[0], full_hist)
            if full_hist:  # keep only recent bars for charts — the old volume max is already saved
                prices = {t: df.tail(520) for t, df in prices.items()}
            ss["prices"] = prices
            ss["panels"] = panels
            ss["meta"] = meta
            ss["data_key"] = data_key + (len(prices),)
            ss["tried"] = set(tickers)
            ss["fund"] = {}
            ss["downloaded_at"] = datetime.now().strftime("%b %d, %I:%M %p")
            ss["downloaded_ts"] = time.time()
            ss.pop("metrics_key", None)
            status.update(label=f"Downloaded {len(prices):,} of {len(tickers):,} stocks", state="complete",
                          expanded=False)
        st.rerun()

# ---- metrics table (recomputed only when data or pattern settings change) ----
metrics = None
if ss.get("scanning") and "panels" not in ss:
    ss["scanning"] = False
    st.rerun()
if ss.get("scan_stopped") and not ss.get("stop_toasted"):
    st.toast("⏹ Scan stopped")
    ss["stop_toasted"] = True
if "panels" in ss:
    mkey = (ss["data_key"], tuple(sorted(pp.items())))
    if ss.get("metrics_key") != mkey and ss.get("scan_stopped") and "metrics" in ss:
        st.info("⏹ Scan stopped before it finished — click **SCAN** to run it again.")
        st.stop()
    if ss.get("metrics_key") != mkey:
        n_stk = ss["panels"]["Close"].shape[1]
        calc_bar = st.progress(0.0, text=f"Calculating indicators for {n_stk:,} stocks …")
        ckey = (ss["data_key"], tuple((k, pp[k]) for k in CORE_KEYS))
        if ss.get("core_key") != ckey:
            m = compute_metrics(ss["panels"], pp, ss.get("vol_hist"))
            meta = ss.get("meta")
            if meta is not None and len(meta):
                m = m.join(meta[["Name", "Sector", "Industry", "Mkt cap $B"]], how="left")
            for c in ["Name", "Sector", "Industry", "Mkt cap $B"]:
                if c not in m:
                    m[c] = np.nan
            ss["core"], ss["core_key"] = m, ckey
        m = ss["core"]
        parts = ss.setdefault("scan_parts", {})
        for i, (prefix, label, fn) in enumerate(SCAN_PARTS):
            pkey = (ss["data_key"], tuple(sorted((k, v) for k, v in pp.items() if k.startswith(prefix))))
            if parts.get(prefix, (None,))[0] != pkey:
                calc_bar.progress((i + 1) / (len(SCAN_PARTS) + 1), text=f"Finding {label} in {n_stk:,} stocks …")
                parts[prefix] = (pkey, fn(ss["panels"], m, pp, ss.get("vol_hist")))
            m = m.join(parts[prefix][1])
        calc_bar.empty()
        ss["metrics"] = m
        ss["metrics_key"] = mkey
    if ss.get("scanning"):                    # scan finished: turn STOP back into SCAN
        ss["scanning"] = False
        st.rerun()
    metrics = ss["metrics"].copy()
    if universe in MARKETS:                   # $ volume columns in US$ so the same thresholds work everywhere
        _fx = market_fx(MARKETS[universe]["fx"], MARKETS[universe]["fx_fallback"])
        for c in ["$ Vol M", "Avg $ Vol 50d M"]:
            if c in metrics:
                metrics[c] = metrics[c] * _fx
    fund = ss.get("fund", {})
    fdf = pd.DataFrame.from_dict(fund, orient="index") if fund else pd.DataFrame()
    for c in FUND_COLS:
        metrics[c] = fdf[c].reindex(metrics.index) if c in fdf else np.nan
    if "_mcap" in fdf:
        metrics["Mkt cap $B"] = metrics["Mkt cap $B"].fillna(fdf["_mcap"].reindex(metrics.index))
    for t, v in ss.get("mcap_fill", {}).items():
        if t in metrics.index and pd.isna(metrics.at[t, "Mkt cap $B"]):
            metrics.at[t, "Mkt cap $B"] = v

    # only the stocks in the current scan list (earlier downloads may hold more)
    if ss.get("scan_set"):
        metrics = metrics[metrics.index.isin(ss["scan_set"])]

    # ---- Relative strength (ranks vs. every stock in this scan) ----
    RS_COLS = ["Week RS", "Month RS", "Quadrant", "Score", "COMP", "Stage", "Short vs bench %", "Long vs bench %",
               "Day %", "Week %", "Month %", "Quarter %", "6 Month %", "Year %", "Bench"] + [tf for tf, _, _ in RS_TIMEFRAMES]
    for c in RS_COLS:
        metrics[c] = np.nan
    metrics["Quadrant"] = ""
    metrics["RS pass"] = False
    if is_rs_mode() or V("f_rsq", "Any") != "Any" or V("f_rsscore", "Any") != "Any":
        bench = (ss["rs_bench"] or "SPY").strip().upper()
        rkey = (ss.get("data_key"), tuple(sorted(metrics.index)), bench, int(ss["rs_short"]), int(ss["rs_long"]))
        if ss.get("rs_key") != rkey:
            with st.spinner("Ranking relative strength …"):
                pnl = {k: v[[c for c in metrics.index if c in v.columns]] for k, v in ss["panels"].items()}
                ss["rs_table"] = rs_stock_table(pnl, bench, int(ss["rs_short"]), int(ss["rs_long"]))
                ss["rs_key"] = rkey
        rt = ss["rs_table"].reindex(metrics.index)
        for c in RS_COLS:
            if c in rt:
                metrics[c] = rt[c]
        metrics["Quadrant"] = metrics["Quadrant"].fillna("")
        metrics["Stage"] = metrics["Stage"].fillna("?")
        # RS score scan: the same Score / ranks / stage / "one rule" as the Relative Strength page
        ok = ((metrics["Score"].fillna(-1) >= float(ss["rss_min"]))
              & (metrics["Week RS"].fillna(-1) >= float(ss["rss_wk"]))
              & (metrics["Month RS"].fillna(-1) >= float(ss["rss_mo"]))
              & (metrics["COMP"].fillna(-1) >= float(ss["rss_comp"]))
              & metrics["Stage"].isin(ss["rss_stages"] or RS_STAGE_ORDER[:-1]))
        if ss["rss_top"] and "Industry" in metrics and metrics["Industry"].notna().any():
            gdf = metrics.assign(Industry=metrics["Industry"].fillna("Unknown").astype(str),
                                 Sector=metrics["Sector"].fillna("Unknown").astype(str))
            ind = rs_groups(gdf, "Industry", int(ss["rss_mingrp"]))
            keep = set(ind.loc[ind["Score"] >= 50, "Industry"]) if len(ind) else set()
            ok &= gdf["Industry"].isin(keep)
        metrics["RS pass"] = ok

    # ---- Green line: confirm the line really was the all-time high (monthly history, matches only) ----
    metrics["ATH"] = ""
    if "GLB" in metrics:
        # in the Green line scan: always · in other scans (for the 'Also in' column): when there are few matches
        gl_focus = is_gl_mode() or V("f_glb", "Any") != "Any"
        cand = metrics.index[metrics["GLB"] != ""]
        if ss.get("gl_ath", True) and 0 < len(cand) <= (400 if gl_focus else 150):
            with st.spinner(f"Checking all-time highs for {len(cand)} green-line matches …"):
                hist = all_time_highs(tuple(sorted(cand)))
            for t in cand:
                h, line, when = hist.get(t), metrics.at[t, "Green line"], metrics.at[t, "Line date"]
                if h is None or not len(h) or pd.isna(when):
                    metrics.at[t, "ATH"] = "?"
                    continue
                before = h[h.index < pd.Timestamp(when).to_period("M").start_time]
                metrics.at[t, "ATH"] = "✓" if not len(before) or before.max() <= line * 1.005 else "✗"
            metrics.loc[metrics["ATH"] == "✗", "GLB"] = ""
        elif ss.get("gl_ath", True) and len(cand) > 400 and gl_focus:
            st.caption(f"ℹ️ {len(cand)} green-line matches — too many to check all-time highs; narrow the list first.")

    # ---- LiqLead groups ----
    def _ll_group(conds, mode):
        tests = [metrics[col] >= float(ss[f"ll_v_{k}"]) for k, (_, col, *_r) in conds.items() if ss[f"ll_on_{k}"]]
        if not tests:
            return pd.Series(True, index=metrics.index)
        t = pd.concat(tests, axis=1).fillna(False)
        return t.all(axis=1) if (mode or "ALL") == "ALL" else t.any(axis=1)
    metrics["LiqLead G1"] = _ll_group(LL_G1, ss["ll_g1_mode"])
    metrics["LiqLead G2"] = _ll_group(LL_G2, ss["ll_g2_mode"])
    metrics["LiqLead"] = metrics["LiqLead G1"] & metrics["LiqLead G2"]

    # ---- S&P 500 changes ----
    for c in ["S&P change", "Change date", "Days to change", "Replaced", "Change reason", "Since change %",
              "Announced", "Since announced %", "Source", "Index", "Index move"]:
        metrics[c] = np.nan
    metrics["S&P change"] = ""
    if is_sp_mode() or V("f_spchg", "Any") != "Any":
        try:
            chg = sp_changes_all()
        except Exception:
            chg = pd.DataFrame()
        today = pd.Timestamp(date.today())
        if len(chg):
            ch = sp_pick_index(chg, ss["sp_index"])
            ch = ch[ch["Symbol"].isin(metrics.index)]
            ch = ch.assign(_a=ch["S&P change"].eq("ADDED")).sort_values(["Change date", "_a"], ascending=[False, False]) \
                .drop_duplicates("Symbol").set_index("Symbol")
            for c in ["S&P change", "Change date", "Replaced", "Change reason", "Announced", "Source", "Index",
                      "Index move"]:
                metrics[c] = ch[c].reindex(metrics.index)
            metrics["Change date"] = pd.to_datetime(metrics["Change date"])
            metrics["Announced"] = pd.to_datetime(metrics["Announced"])
            metrics["Days to change"] = (metrics["Change date"] - today).dt.days
            C = ss["panels"]["Close"]
            for t, d in ch["Change date"].items():
                if t in C.columns and d <= today:
                    i = C.index.searchsorted(d) - 1          # close before the effective date
                    if 0 <= i < len(C) and C[t].iloc[i]:
                        metrics.at[t, "Since change %"] = (C[t].iloc[-1] / C[t].iloc[i] - 1) * 100
            for t, a in ch["Announced"].dropna().items():
                if t in C.columns:
                    i = C.index.searchsorted(pd.Timestamp(a), side="right") - 1   # announced after the close
                    if 0 <= i < len(C) and C[t].iloc[i]:
                        metrics.at[t, "Since announced %"] = (C[t].iloc[-1] / C[t].iloc[i] - 1) * 100
        try:
            members, _ = get_sp500()
        except Exception:
            members = []
        cands = sp_candidates(get_screener_meta(), members, ss["sp_mincap"])
        metrics["S&P change"] = metrics["S&P change"].astype(object).where(metrics["S&P change"].notna(), "")
        is_cand = metrics.index.isin(cands.index) & (metrics["S&P change"] == "")
        metrics.loc[is_cand, "S&P change"] = "CANDIDATE"
    metrics["S&P change"] = metrics["S&P change"].fillna("")
    if is_sp_mode() or V("f_spchg", "Any") != "Any" or ss.get("v_tab") == "S&P changes":
        try:
            metrics["In S&P since"] = sp_member_dates().reindex(metrics.index)
        except Exception:
            metrics["In S&P since"] = pd.NaT
    else:
        metrics["In S&P since"] = pd.NaT

    # ---- earnings (only fetched when the Earnings scan or filter is in use) ----
    EARN_COLS = ["Earnings date", "Days to earnings", "Report time", "EPS est", "EPS actual", "Surprise %",
                 "Last yr EPS", "# Ests", "Earnings gap %", "Reaction %", "Since earnings %"]
    for c in EARN_COLS:
        metrics[c] = np.nan
    if is_earn_mode() or V("f_earnwin", "Any") != "Any":
        lo, hi = EARN_WINDOWS.get(V("f_earnwin"), EARN_WINDOWS[ss["earn_window"]])
        with st.spinner("Loading earnings calendar …"):
            cal = None if universe in MARKETS else earnings_calendar(lo, hi)
        if cal is None:
            # Nasdaq unreachable → Yahoo, one stock at a time (slower; capped at 600 stocks)
            todo = [t for t in metrics.index if t not in ss.get("yearn", {})][:600]
            if todo:
                bar = st.progress(0.0, text="Loading earnings from Yahoo …")
                out = {}
                with ThreadPoolExecutor(8) as ex:
                    for i, (t, f) in enumerate(zip(todo, ex.map(yahoo_earnings, todo))):
                        out[t] = f
                        bar.progress((i + 1) / len(todo), text=f"Loading earnings from Yahoo … {i + 1}/{len(todo)}")
                ss.setdefault("yearn", {}).update(out)
                st.rerun()
            rows = [r for t in metrics.index for r in ss.get("yearn", {}).get(t, [])]
            cal = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Symbol"] + CAL_COLS)
            today = pd.Timestamp(date.today())
            dd = (pd.to_datetime(cal["Earnings date"]) - today).dt.days if len(cal) else pd.Series(dtype=float)
            cal = cal[dd.between(lo, hi)] if len(cal) else cal
            cal = _nearest_per_stock(cal) if len(cal) else pd.DataFrame(columns=CAL_COLS)
            st.caption("ℹ️ Earnings dates come from Yahoo for this market." if universe in MARKETS else
                       "ℹ️ Nasdaq's earnings calendar couldn't be reached, so earnings come from Yahoo.")
        if len(cal):
            cal = cal.reindex(metrics.index)
            for c in CAL_COLS:
                if c in cal and c in metrics:
                    metrics[c] = cal[c]
            metrics["Earnings date"] = pd.to_datetime(cal["Earnings date"])
            metrics["Days to earnings"] = (metrics["Earnings date"] - pd.Timestamp(date.today())).dt.days
            metrics["Earnings in (days)"] = metrics["Earnings in (days)"].fillna(
                metrics["Days to earnings"].where(metrics["Days to earnings"] >= 0))
            past = metrics["Days to earnings"] <= 0
            if past.any():
                rx = earnings_reaction(metrics.loc[past, "Earnings date"], metrics.loc[past, "Report time"],
                                       ss["panels"])
                for c in rx:
                    metrics.loc[past, c] = rx[c]



results_area()

