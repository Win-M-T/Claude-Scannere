"""Constants: user-agent, page CSS data, option lists, presets, filter definitions."""

import re
from pathlib import Path

import plotly.express as px
import streamlit as st

from scanner.backtest import _bt_bf, _bt_ep, _bt_f50, _bt_fb, _bt_glb, _bt_htf, _bt_mac, _bt_para, _bt_rc, _bt_so3, _bt_tight, _bt_vcp, _bt_xback
from scanner.data.fundamentals import between, is_true, perf_filter


UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
      "Accept": "application/json, text/plain, */*"}





# ---- S&P Dow Jones Indices official announcements (press.spglobal.com) ----
SPDJI_BASE = "https://press.spglobal.com"


_TAG = re.compile(r"<[^>]+>")


SP_INDEX_NAMES = ["S&P 500", "S&P 100", "S&P MidCap 400", "S&P SmallCap 600"]


# ---- Asian markets: Hong Kong, Thailand, Singapore ----
MARKETS = {
    "🇭🇰 Hong Kong (HKEX)": dict(tv="hongkong", suffix=".HK", tv_prefix="HKEX", fx="HKDUSD=X", fx_fallback=0.128,
                                 bench="^HSI", bench_name="Hang Seng", cur="HK$", country="Hong Kong"),
    "🇹🇭 Thailand (SET)": dict(tv="thailand", suffix=".BK", tv_prefix="SET", fx="THBUSD=X", fx_fallback=0.03,
                              bench="^SET.BK", bench_name="SET index", cur="฿", country="Thailand"),
    "🇸🇬 Singapore (SGX)": dict(tv="singapore", suffix=".SI", tv_prefix="SGX", fx="SGDUSD=X", fx_fallback=0.77,
                               bench="^STI", bench_name="Straits Times", cur="S$", country="Singapore"),
}


# If the full stock list can't be loaded, fall back to each market's main index
MARKET_FALLBACK = {
    ".HK": "0001 0002 0003 0005 0006 0011 0012 0016 0017 0027 0066 0101 0175 0241 0267 0285 0288 0291 0316 0322 0386 "
           "0388 0669 0688 0700 0762 0823 0836 0857 0868 0881 0883 0939 0941 0960 0968 0981 0992 1024 1038 1044 1088 "
           "1093 1099 1109 1113 1177 1211 1299 1378 1398 1810 1876 1928 1997 2015 2020 2269 2313 2318 2319 2331 2382 "
           "2388 2628 2688 2899 3690 3692 3968 3988 6618 6690 6862 9618 9633 9888 9901 9961 9988 9999",
    ".BK": "ADVANC AOT AWC BANPU BBL BDMS BEM BGRIM BH BJC BTS CBG CENTEL COM7 CPALL CPF CPN CRC DELTA EA EGCO GLOBAL "
           "GPSC GULF HMPRO IVL KBANK KCE KTB KTC LH MINT MTC OR OSP PTT PTTEP PTTGC RATCH SAWAD SCB SCC SCGP TIDLOR "
           "TISCO TOP TRUE TTB WHA",
    ".SI": "D05 O39 U11 Z74 C38U A17U C6L BN4 Y92 S68 G13 V03 U96 F34 H78 C07 D01 J36 BS6 9CI S63 N2IU M44U ME8U AJBU "
           "U14 5E2 CJLU C09 S58",
}


VOL_WINDOWS = [("3M", 63), ("6M", 126), ("1Y", 252)]


VOL_REC_NAME = {"IPO": "Since IPO", "1Y": "1 year", "6M": "6 months", "3M": "3 months"}


VOL_REC_RANK = {"Since IPO": 4, "1 year": 3, "6 months": 2, "3 months": 1}


XB_20 = "20-day SMA"


XB_EMAS = ["10 EMA or 20 SMA", "10-day EMA", XB_20]


SO_RECLAIM = "Reclaim of the first low (earlier)"


SO_ENTRIES = ["+3 level (the original)", SO_RECLAIM]


MAC_LINES = ["20 or 50-day SMA", "20-day SMA", "50-day SMA"]


HTF_PRESETS = {
    "Strict (O'Neil / Stockbee)": dict(htf_pole=100.0, htf_depth=(10.0, 25.0)),
    "Bulkowski": dict(htf_pole=90.0, htf_depth=(10.0, 34.0)),
    "Loose (half-size HTF)": dict(htf_pole=50.0, htf_depth=(3.0, 15.0)),
}


# ---------------------------------------------------------------------------------------------------------------
# 🐻 Swing shorts: bear flag into a falling 20/50-day SMA · failed breakout (bull trap)
# ---------------------------------------------------------------------------------------------------------------
SH_SETUPS = ["Both", "🐻 Bear flag", "🪤 Failed breakout"]


SH_MAS = ["20- or 50-day SMA", "20-day SMA", "50-day SMA"]


# ---------------------------------------------------------------------------------------------------------------
# 🔂 50-day reclaim: a pop above the 50-day SMA fails, a shallow dip makes a higher low, then a close back above
# ---------------------------------------------------------------------------------------------------------------
RC_TRENDS = ["Any", "Turnaround (50-day SMA falling)", "Pullback in an uptrend (50-day SMA rising)"]


RC_STOPS = ["Under the higher low", "Under the dip's lowest low"]


# ---- Earnings calendar (Nasdaq, one request per weekday; Yahoo as fallback) ----
EARN_TIME = {"time-pre-market": "Before open", "time-after-hours": "After close"}


CAL_COLS = ["Earnings date", "Report time", "EPS est", "EPS actual", "Surprise %", "Last yr EPS", "# Ests", "Quarter"]


FUND_COLS = ["P/E", "Fwd P/E", "EPS growth %", "Rev growth %", "Profit margin %",
             "Short float %", "Earnings in (days)", "Analyst"]


PERF_PRESETS = [("Up", "Positive return", between, (0.0001, None)),
                ("Above 10%", "", between, (10, None)), ("Above 20%", "", between, (20, None)),
                ("Above 50%", "Big winners", between, (50, None)), ("Above 100%", "Doubled", between, (100, None)),
                ("Down", "Negative return", between, (None, -0.0001)),
                ("Below −10%", "", between, (None, -10)), ("Below −25%", "Big losers", between, (None, -25))]


SP_INDEXES = ["S&P 500", "S&P 100", "S&P MidCap 400", "S&P SmallCap 600", "Any S&P index"]


def sp_pick_index(df, idx):
    return df if idx == "Any S&P index" or not len(df) else df[df["Index"] == idx]


SP_WINDOWS = {"Upcoming only": 0, "Last 2 weeks": 14, "Last 1 month": 31, "Last 3 months": 92,
              "Last 6 months": 183, "Last 1 year": 366}


DEFAULT_VISIBLE = ["adr"]     # switched-on filters always show as buttons; "＋ Filter" adds more


DEFAULT_CHOICE = {"pattern": "SETUP or TRIGGER", "price": "Above 10", "avgvol": "Above 500K"}


COLUMN_SETS = {
    "Overview": ["Symbol", "Name", "Price", "Chg %", "Rel vol", "Avg vol 50d", "ADR % 14d", "Under 50d days", "RS", "Perf 3M %",
                 "Mkt cap $B", "Sector"],
    "Performance": ["Symbol", "Pattern", "Price", "Chg %", "Perf 1W %", "Perf 1M %", "Perf 3M %",
                    "Perf 6M %", "Perf YTD %", "Perf 1Y %", "RS"],
    "Technicals": ["Symbol", "Pattern", "Price", "RSI 14", "ATR %", "vs EMA21 %", "vs SMA20 %", "vs SMA50 %",
                   "vs SMA200 %", "Below 52W high %", "Above 52W low %", "Gap %", "Trend template"],
    "Tight flag": ["Symbol", "Pattern", "Price", "Pullback %", "Days since high", "7d range %",
                   "10d range %", "vs SMA20 %", "EMA gap %", "Vol ratio", "Perf 3M %"],
    "Volume": ["Symbol", "Pattern", "Price", "Volume", "Avg vol 50d", "Rel vol", "$ Vol M", "Vol ratio"],
    "Record volume": ["Symbol", "Name", "Vol record", "Record date", "Record vol", "Record × avg",
                      "Record day chg %", "Price", "Chg %", "Avg vol 50d", "Mkt cap $B", "Sector"],
    "Parabolic": ["Symbol", "Name", "Parabolic", "Run %", "Run days", "Up days in a row", "Above EMA10 %",
                  "Off high %", "Price", "Chg %", "Gap %", "Rel vol", "Mkt cap $B", "Sector"],
    "Episodic pivot": ["Symbol", "Name", "EP", "EP date", "Days since EP", "EP gap %", "EP day chg %", "EP vol × avg",
                       "Since EP %", "Prior 3M %", "EP low", "Price", "Chg %", "Mkt cap $B", "Sector"],
    "Relative strength": ["Symbol", "Name", "Quadrant", "Score", "Week RS", "Month RS", "COMP", "Stage",
                          "Short vs bench %", "Long vs bench %", "Day %", "Industry", "Sector", "Mkt cap $B"],
    "Shakeout +3": ["Symbol", "Name", "SO+3", "Price", "Chg %", "First low", "Shakeout low", "Undercut %", "+3 level",
                    "To +3 %", "Days since shakeout", "Shakeout vol ×", "SO depth %", "SO stop", "RS", "Mkt cap $B",
                    "Sector"],
    "High tight flag": ["Symbol", "Name", "HTF", "Price", "Chg %", "Pole %", "Flag depth %", "Flag days", "Top-2 days %",
                        "Flag high", "To flag high %", "Flag low", "HTF risk %", "Tightness", "HTF dry-up", "HV1 in pole",
                        "HTF vol ×", "RS", "Mkt cap $B", "Sector"],
    "50-day reclaim": ["Symbol", "Name", "Reclaim", "Price", "Chg %", "50 SMA", "vs 50 SMA %", "Low before pop", "Dip low",
                       "Higher low", "Dip depth %", "Days since pop", "Trigger days ago", "Reclaim stop", "Reclaim risk %",
                       "Reclaim vol ×", "RS",
                       "Mkt cap $B", "Sector"],
    "First close above 50": ["Symbol", "Name", "First50", "Price", "Chg %", "F50 SMA", "F50 vs SMA %", "F50 days under",
                             "F50 trig ago", "F50 stop", "F50 risk %", "F50 vol ×", "Rel vol", "ADR % 14d", "RS",
                             "Mkt cap $B", "Sector"],
    "Swing shorts": ["Symbol", "Name", "Short", "Short setup", "Price", "Chg %", "Sell below", "To sell level %",
                     "Short stop", "Short risk %", "vs SMA20 %", "vs SMA50 %", "vs SMA200 %", "Bounce %",
                     "Bounce vol ×", "Breakout days ago", "Short vol ×", "RS", "Mkt cap $B", "Sector"],
    "MA consolidation": ["Symbol", "Name", "MAC", "Price", "Chg %", "MA held", "vs line %", "Box range %", "Box high",
                         "To box high %", "Box low", "Box risk %", "Box vol ×", "RS", "Mkt cap $B", "Sector"],
    "VCP": ["Symbol", "Name", "VCP", "Price", "Chg %", "Contractions", "# contractions", "Last contraction %", "Pivot",
            "To pivot %", "VCP stop", "VCP risk %", "Vol dry-up", "Breakout vol ×", "Base days", "RS", "Mkt cap $B",
            "Sector"],
    "Green line": ["Symbol", "Name", "GLB", "Price", "Chg %", "Green line", "To green line %", "Line date",
                   "Line age (months)", "GLB vol ×", "ATH", "GLB stop", "RS", "Perf 1Y %", "Mkt cap $B", "Sector"],
    "EMA crossback": ["Symbol", "Name", "Crossback", "Price", "Chg %", "EMA tagged", "vs EMA10 %", "vs SMA20 %",
                      "Rally %", "Pullback from high %", "RS", "vs SPY 1M pts", "Tight (3d ÷ ATR)", "Up/Down vol",
                      "Candle", "Entry above", "Stop", "Risk %", "Sector"],
    "Liquid Leaders": ["Symbol", "Name", "Price", "Chg %", "Avg $ Vol 50d M", "ADR % 14d", "Chg 5d %", "Chg 20d %",
                "Off 52W high %", "RS", "Mkt cap $B", "Sector"],
    "S&P changes": ["Symbol", "Name", "Index", "S&P change", "Announced", "Change date", "Days to change",
                    "Index move", "In S&P since", "Replaced", "Since announced %", "Change reason", "Source",
                    "Since change %", "Price", "Chg %", "Perf 1W %", "Perf 1M %", "Rel vol", "Mkt cap $B", "Sector"],
    "Earnings": ["Symbol", "Name", "Earnings date", "Days to earnings", "Report time", "EPS est", "EPS actual",
                 "Surprise %", "Last yr EPS", "# Ests", "Earnings gap %", "Reaction %", "Since earnings %",
                 "Price", "Chg %", "Perf 3M %", "RS", "Mkt cap $B", "Sector"],
    "Fundamentals": ["Symbol", "Name", "Mkt cap $B", "Sector", "Industry", "P/E", "Fwd P/E",
                     "EPS growth %", "Rev growth %", "Profit margin %", "Short float %",
                     "Earnings in (days)", "Analyst"],
}


# ============================================================================
# 6. Chart
# ============================================================================
TV_INTERVALS = {"Daily": "D", "Weekly": "W", "Monthly": "M", "4 hour": "240", "1 hour": "60",
                "30 min": "30", "15 min": "15", "5 min": "5"}


TV_STYLES = {"Candles": "1", "Hollow candles": "9", "Heikin Ashi": "8", "Bars": "0", "Line": "2", "Area": "3"}


TV_EXTRAS = {"RSI": "RSI@tv-basicstudies", "MACD": "MACD@tv-basicstudies", "Bollinger Bands": "BB@tv-basicstudies",
             "VWAP": "VWAP@tv-basicstudies", "ATR": "ATR@tv-basicstudies", "Stochastic RSI": "StochasticRSI@tv-basicstudies",
             "Stochastic": "Stochastic@tv-basicstudies", "Pivot points": "PivotPointsStandard@tv-basicstudies"}


TV_MA_IDS = {"EMA": "MAExp@tv-basicstudies", "SMA": "MASimple@tv-basicstudies", "WMA": "MAWeighted@tv-basicstudies"}


# Chart templates live in a small file next to this app, so they survive restarts.
TPL_FILE = Path(__file__).parent.parent.with_name("chart_templates.json")


DEFAULT_TPL = dict(style="Candles", interval="Daily", height=720, mas="EMA 10, EMA 21, SMA 50, SMA 200",
                   extras=[], volume=True, theme="Dark", toolbar=True, watchlist=True, layout_url="")


TPL_FIELDS = list(DEFAULT_TPL)


# ============================================================================
# 6a. Relative Strength scanner (ported from the RS Scanner app)
#     Week RS (x) vs Month RS (y), ranked 0-100; groups by sector / industry; stages
# ============================================================================
RS_SECTOR_ETFS = {
    "XLE": "Energy", "XLK": "Technology", "XLF": "Financial", "XLV": "Healthcare",
    "XLY": "Consumer Cyclical", "XLP": "Consumer Defensive", "XLI": "Industrials",
    "XLB": "Basic Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
}


RS_STAGES = {
    "1A": ("Upward Pivot", "#F0A36B"), "1B": ("Mean Reversion", "#F5E663"),
    "2A": ("Bullish Trend", "#A8E6A1"), "2B": ("Breakout Confirm", "#3CC47C"),
    "2C": ("Extended Bullish", "#FFFFFF"), "3A": ("Bullish Fade", "#8FD3F4"),
    "3B": ("Fade Confirmation", "#4A90D9"), "4A": ("Bearish Trend", "#F7A8A8"),
    "4B": ("Breakdown Confirm", "#E74C4C"), "4C": ("Extended Bearish", "#F07BF0"),
    "?": ("Unknown", "#777777"),
}


RS_STAGE_ORDER = list(RS_STAGES)


RS_PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24


# (label, return column, trading days)
RS_TIMEFRAMES = [("DAY", "Day %", 1), ("WK", "Week %", 5), ("MTH", "Month %", 21), ("QTR", "Quarter %", 63),
                 ("6M", "6 Month %", 126), ("1Y", "Year %", 252)]


RS_VIEWS = ["Stocks (quadrant)", "Leading groups - table", "Leading groups - cards", "Industries (quadrant)",
            "Sectors (quadrant)", "Stage analysis", "Sector ETFs (quadrant)"]


RS_CSS = """<style>
.rs-wrap{overflow-x:auto}
table.rs{border-collapse:collapse;font-size:12px;width:100%;color:#ddd}
table.rs th{color:#8a93a6;font-weight:600;text-align:center;padding:6px 4px;border-bottom:1px solid #333;font-size:11px}
table.rs td{padding:5px 4px;border-bottom:1px solid #262b36;text-align:center;vertical-align:middle}
table.rs td.name{text-align:left;white-space:nowrap;font-weight:600}
table.rs td.chips{text-align:left}
.chip{display:inline-block;background:#1b2130;border-left:3px solid #888;padding:2px 6px;margin:2px 3px 2px 0;border-radius:3px;font-weight:700;font-size:11px;white-space:nowrap}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.card{background:#151a24;border-radius:6px;overflow:hidden;font-size:12px;color:#ddd}
.card .hd{display:flex;justify-content:space-between;padding:5px 8px;font-weight:700;color:#fff}
.card .sub{display:flex;justify-content:space-between;padding:3px 8px;color:#8a93a6;font-size:10px}
.row{display:flex;align-items:center;justify-content:space-between;padding:3px 8px;margin:2px 4px;background:#1b2130;border-left:3px solid #888;border-radius:3px}
.badge{display:inline-block;border:1px solid;border-radius:4px;padding:0 4px;font-size:10px;margin-left:4px;font-weight:700}
.circ{display:inline-block;border:1.5px solid;border-radius:50%;width:22px;height:22px;line-height:19px;text-align:center;font-size:10px;margin-left:4px;font-weight:700}
.stagecol{background:#151a24;border-radius:6px;padding:0 0 6px 0;min-width:0}
.stagecol .hd{padding:6px;text-align:center;color:#111;font-weight:700;border-radius:6px 6px 0 0}
</style>"""


# ============================================================================
# 7. Sidebar — what to scan + pattern settings
# ============================================================================
PATTERN_PRESETS = {
    "Default": dict(max_range=8.0, pull_range=(3.0, 15.0), min_bars=5, max_ema_dist=3.0,
                    max_ema_gap=2.0, dry_up=0.8, breakout_vol=1.5),
    "Strict": dict(max_range=6.0, pull_range=(3.0, 12.0), min_bars=5, max_ema_dist=2.5,
                   max_ema_gap=1.5, dry_up=0.7, breakout_vol=1.5),
    "Loose": dict(max_range=10.0, pull_range=(2.0, 18.0), min_bars=4, max_ema_dist=4.0,
                  max_ema_gap=3.0, dry_up=0.9, breakout_vol=1.3),
}


M_SH = "🐻 Swing Shorts"


M_RC = "🔂 50-Day Reclaim"


M_F5 = "🔃 First Close Above 50-day"


SCAN_MODES = [M_RS, M_LL, M_EARN, M_VOL, M_EP, M_PB, M_SH, M_SP, M_GL, M_TF, M_HTF, M_VCP, M_MAC, M_RC, M_F5, M_SO, M_XB]


# earlier names (saved scans, open sessions) -> today's names
OLD_SCAN_NAMES = {"🚩 Tight-flag pattern": M_TF, "📊 Record volume": M_VOL, "📅 Earnings": M_EARN,
                  "📉 Parabolic short": M_PB, "🚀 Episodic pivot": M_EP, "🏛 S&P index changes": M_SP,
                  "💧 LiqLead": M_LL, "🔁 EMA crossback": M_XB, "🟢 Green line breakout": M_GL,
                  "🪤 Shakeout +3": M_SO, "🧭 RS score": M_RS}


SCAN_EXTRA_DEFAULTS = dict(pb_status="First crack (short trigger)", pb_days=20, pb_gain=50, pb_up=3, pb_ext=20,
                           ep_status="Any EP", ep_days=5, ep_gap=10.0, ep_volx=3.0, ep_strong=True,
                           ep_neglect_on=False, ep_neglect=30,
                           sp_type="Added", sp_window="Last 3 months", sp_mincap=22.7,
                           sp_index="S&P 500",
                           xb_status="SETUP or TRIGGER", xb_ema="10 EMA or 20 SMA", xb_tol=1.5, xb_days=3, xb_rally=20,
                           xb_rally_days=40, xb_pull_max=15, xb_rs=70, xb_beat=True, xb_tight=True, xb_tight_max=1.0,
                           xb_vol=True, xb_candle=False,
                           gl_status="Breakout, retest or near", gl_months=(3, 12), gl_vol=1.5, gl_near=5.0,
                           gl_retest_days=10, gl_tol=2.0, gl_stop=1.0, gl_ath=True,
                           so_status="Trigger or reclaimed", so_base=60, so_recent=15, so_under=(0.5, 15.0),
                           so_mid=5.0, so_depth=35.0, so_trend=30.0, so_above200=True,
                           so_rule="10% above the first low", so_pct=10.0, so_vol=1.0, so_stop=0.5, so_rs=0,
                           so_entry="+3 level (the original)", so_fast=0, so_shvol=0.0,
                           htf_status="SETUP or TRIGGER", htf_pole=100.0, htf_pole_days=40, htf_flag_min=7,
                           htf_flag_max=25, htf_depth=(10.0, 25.0), htf_share=60.0, htf_tight=1.0, htf_dry=1.0,
                           htf_nored=True, htf_ma=True, htf_hv=False, htf_minpx=5.0, htf_mindvol=5.0, htf_vol=1.5,
                           htf_close_top=30.0,
                           rc_status="TRIGGER, SETUP or DIP", rc_trend=RC_TRENDS[0], rc_below=10, rc_slack=2, rc_recent=1, rc_pop_max=10,
                           rc_dip_max=8.0, rc_min_dip=2, rc_max_days=30, rc_base=20, rc_hl=True, rc_prior=True, rc_stop="Under the higher low", rc_vol=0.0, rc_rs=0,
                           rc_minpx=5.0, rc_mindvol=5.0,
                           f5_status="TRIGGER or SETUP", f5_days=30, f5_near=3.0, f5_recent=1, f5_vol=0.0, f5_minpx=5.0, f5_mindvol=5.0,
                           sh_status="SETUP or TRIGGER", sh_which="Both", sh_bf_ma=SH_MAS[0], sh_bf_tol=1.0,
                           sh_bf_bounce=3.0, sh_bf_dry=1.0, sh_bf_rs=50, sh_bf_days=3, sh_fb_len=20, sh_fb_base=5,
                           sh_fb_days=3, sh_fb_vol=1.0, sh_fb_200=False, sh_minpx=10.0, sh_mindvol=20.0,
                           mac_status="SETUP or TRIGGER", mac_ma=MAC_LINES[0], mac_days=10, mac_hold="Every close",
                           mac_tol=1.0, mac_range=10.0, mac_near=6.0, mac_rising=True, mac_trend=True, mac_rs=0,
                           mac_vol=1.2,
                           vcp_status="SETUP or TRIGGER", vcp_base=65, vcp_swing=3.0, vcp_min_c=2,
                           vcp_first_max=35.0, vcp_last_max=10.0, vcp_shrink=True, vcp_dry=0.8, vcp_near=5.0,
                           vcp_vol=1.4, vcp_tt=True, vcp_rs=70, vcp_risk=10.0,
                           ll_g1_mode="ALL", ll_g2_mode="ANY", ll_status="Passes all groups",
                           rs_view="Stocks (quadrant)", rs_bench="SPY", rs_short=5, rs_long=21, rs_min=3,
                           rs_stages=RS_STAGE_ORDER[:-1], rs_top=True, rs_strong=False, rs_quad="Any",
                           rss_status="Passes the RS rules", rss_min=70, rss_wk=0, rss_mo=0, rss_comp=0,
                           rss_stages=RS_STAGE_ORDER[:-1], rss_top=True, rss_mingrp=3,
                           **{f"ll_on_{k}": True for k in list(LL_G1) + list(LL_G2)},
                           **{f"ll_v_{k}": v[2] for k, v in {**LL_G1, **LL_G2}.items()})


VOLREC_OPTS = [p[0] for p in F["volrec"]["presets"]]


# scan -> (its filter button, where its choice lives, its results tab)
MODE_SPEC = {M_TF: ("pattern", None, "Overview"), M_VOL: ("volrec", "vr_window", "Record volume"),
             M_EARN: ("earnwin", "earn_window", "Earnings"), M_PB: ("parabolic", "pb_status", "Parabolic"),
             M_EP: ("ep", "ep_status", "Episodic pivot"), M_SP: ("spchg", "sp_type", "S&P changes"),
             M_LL: ("liqlead", "ll_status", "Liquid Leaders"),
             M_XB: ("xback", "xb_status", "EMA crossback"),
             M_GL: ("glb", "gl_status", "Green line"),
             M_SO: ("so3", "so_status", "Shakeout +3"),
             M_VCP: ("vcp", "vcp_status", "VCP"),
             M_MAC: ("mac", "mac_status", "MA consolidation"),
             M_HTF: ("htf", "htf_status", "High tight flag"),
             M_SH: ("swshort", "sh_status", "Swing shorts"),
             M_RC: ("reclaim", "rc_status", "50-day reclaim"),
             M_F5: ("f50scan", "f5_status", "First close above 50"),
             M_RS: ("rsscore", "rss_status", "Relative strength")}


WATCH_SOURCES = ["Stock Scanner results", "My own watchlist (file / tickers)"]


# ============================================================================
# 7b. Page 3 — Similar stocks (peer group strength)
#     Type a ticker → the stocks that trade most like it (correlation of daily returns),
#     with stage, RS, similarity and today's moves.
# ============================================================================
PEER_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
             "SMH", "SOXX", "IGV", "CIBR", "HACK", "BUG", "SKYY", "WCLD", "ARKK", "XBI", "IBB", "KRE", "KBE", "XHB",
             "ITB", "XRT", "JETS", "TAN", "ICLN", "URA", "GDX", "SIL", "XME", "XOP", "OIH", "IYT", "PAVE", "ITA",
             "BOTZ", "FINX", "IPO", "MTUM", "SPHB"]


PEER_BULL = {"1A", "1B", "2A", "2B", "2C"}


PEER_CSS = """<style>
.peer{border:1px solid rgba(127,127,127,.22);border-radius:14px;overflow:hidden;background:rgba(127,127,127,.04)}
.peer .top{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid rgba(127,127,127,.18);flex-wrap:wrap}
.peer .top .t{font-weight:700;font-size:17px}
.peer .top .tick{background:#26a69a;color:#06120f;font-weight:800;border-radius:8px;padding:3px 10px;font-size:15px}
.peer .top .asof{color:#8a93a6;font-size:13px}
table.pt{border-collapse:collapse;width:100%;font-size:14px;border:none !important;margin:0}
table.pt th, table.pt td{border-left:none !important;border-right:none !important;border-top:none !important}
table.pt th{color:#8a93a6;font-weight:600;text-align:left;padding:8px 10px;border-bottom:1px solid rgba(127,127,127,.18);vertical-align:bottom}
table.pt td{padding:7px 10px;border-bottom:1px solid rgba(127,127,127,.12)}
table.pt tr.me td{background:rgba(38,166,154,.10)}
table.pt a{color:inherit;font-weight:700;text-decoration:none;border-bottom:2px solid rgba(127,127,127,.35)}
table.pt .sub{display:block;color:#8a93a6;font-size:11px;font-weight:400;max-width:230px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pill{display:inline-block;border-radius:999px;padding:2px 10px;border:1px solid rgba(127,127,127,.3);font-variant-numeric:tabular-nums}
.pill.g{background:rgba(38,166,154,.14);border-color:rgba(38,166,154,.45);color:#4cd9a3}
.pill.r{background:rgba(239,83,80,.14);border-color:rgba(239,83,80,.45);color:#ff7b78}
.pill.n{background:rgba(127,127,127,.10)}
.cnt{display:inline-block;margin-top:4px;border-radius:999px;padding:2px 10px;background:rgba(38,166,154,.14);border:1px solid rgba(38,166,154,.45);color:#4cd9a3;font-size:13px}
.cnt.r{background:rgba(239,83,80,.14);border-color:rgba(239,83,80,.45);color:#ff7b78}
</style>"""


# ============================================================================
# Smart Money: 13F portfolios of famous investors + stock trades by Congress
# ============================================================================
FAMOUS_13F = {   # label -> SEC CIK (the fund's filer ID on EDGAR)
    "Warren Buffett · Berkshire Hathaway": 1067983,
    "Bill Ackman · Pershing Square": 1336528,
    "Stanley Druckenmiller · Duquesne": 1536411,
    "David Tepper · Appaloosa": 1656456,
    "Michael Burry · Scion": 1649339,
    "Seth Klarman · Baupost": 1061768,
    "Li Lu · Himalaya Capital": 1709323,
    "Mohnish Pabrai · Dalal Street": 1549575,
    "Howard Marks · Oaktree": 949509,
    "Ray Dalio · Bridgewater": 1350694,
    "George Soros · Soros Fund": 1029160,
    "Carl Icahn · Icahn Enterprises": 921669,
    "Dan Loeb · Third Point": 1040273,
    "David Einhorn · Greenlight": 1079114,
    "Chase Coleman · Tiger Global": 1167483,
    "Philippe Laffont · Coatue": 1135730,
    "Cathie Wood · ARK Invest": 1697748,
    "Chuck Akre · Akre Capital": 1112520,
    "Terry Smith · Fundsmith": 1569205,
    "Tom Gayner · Markel": 1096343,
    "Andreas Halvorsen · Viking Global": 1103804,
    "Stephen Mandel · Lone Pine": 1061165,
    "Jim Simons · Renaissance Technologies": 1037389,
    "Bill & Melinda Gates Foundation Trust": 1166559,
}


SM_DEFAULT_FUNDS = ["Warren Buffett · Berkshire Hathaway", "Bill Ackman · Pershing Square",
                    "Stanley Druckenmiller · Duquesne", "David Tepper · Appaloosa", "Seth Klarman · Baupost",
                    "Li Lu · Himalaya Capital", "Howard Marks · Oaktree", "Chuck Akre · Akre Capital"]


CUSIP_FILE = Path(__file__).parent.parent.with_name("cusip_tickers.json")


SM_NAME_DROP = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "HLDGS", "HLDG", "HOLDINGS",
                "HOLDING", "GROUP", "GRP", "CL", "CLASS", "COM", "NEW", "THE", "SA", "NV", "AG", "LP", "ADR", "SPONSORED",
                "SPON", "ADS", "SHS", "ORD", "A", "B", "C", "DEL", "INCORPORATED", "TR", "TRUST", "N", "V", "SE", "LLC"}


# ---- Congress ----
SM_AMOUNT_MIN = {"Any amount": 0, "$15K+": 15001, "$50K+": 50001, "$100K+": 100001, "$250K+": 250001,
                 "$1M+": 1000001}


HOUSE_ROW = re.compile(
    r"\(([A-Z][A-Z0-9.\-/]{0,7})\)\s*\[(ST|OP|EF|ET|AB)\]\s*"
    r"(P|S \(partial\)|S|E)\s+(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}/\d{1,2}/\d{4})\s*"
    r"(\$[\d,]+\s*-\s*\$[\d,]+|Over \$[\d,]+|\$[\d,]+ \+|Spouse/DC Over \$[\d,]+)")


CHANGE_STYLE = {"NEW": "color:#22a355;font-weight:700", "ADD": "color:#22a355", "TRIM": "color:#e0a106",
                "SOLD": "color:#e5484d;font-weight:700", "Purchase": "color:#22a355;font-weight:700",
                "Sale": "color:#e5484d;font-weight:700", "D": "color:#3b82f6;font-weight:700",
                "R": "color:#e5484d;font-weight:700"}


BT_STRATEGIES = {
    "🚩 Tight Flag Pattern": (260, "long", _bt_tight, "TRIGGER days of the Tight-flag scan"),
    "🔁 EMA Crossback": (160, "long", _bt_xback, "TRIGGER days of the EMA crossback scan"),
    "✳️ Green Line Breakout": (330, "long", _bt_glb, "BREAKOUT days of the Green line scan (all-time-high check "
                                                    "is skipped here — it uses the last ~15 months)"),
    "📈 Shakeout +3": (260, "long", _bt_so3, "TRIGGER days of the Shakeout +3 scan (close above the +3 level)"),
    "⛳ High Tight Flag": (330, "long", _bt_htf, "TRIGGER days of the High Tight Flag scan (a +100% pole, a tight "
                                              "10–25% flag, then a close above the flag high on volume)"),
    "📏 MA Consolidation": (260, "long", _bt_mac, "TRIGGER days of the MA Consolidation scan (close above the "
                                                "box high after a tight base holding the 20/50-day SMA)"),
    "🌀 VCP (Minervini)": (280, "long", _bt_vcp, "TRIGGER days of the VCP scan (close above the pivot on volume, "
                                                "Trend Template passed)"),
    "🚀 Episodic Pivot": (120, "long", _bt_ep, "The day of each episodic pivot (gap up on huge volume)"),
    "📉 Parabolic Short": (120, "short", _bt_para, "CRACK days of the Parabolic scan — traded SHORT"),
    "🔃 First Close Above 50-day": (200, "long", _bt_f50, "The first close back above the 50-day SMA after N straight "
                                                       "closes at or below it (set N below). Stop = the lowest low of "
                                                       "the last 10 sessions"),
    "🔂 50-Day Reclaim": (200, "long", _bt_rc, "TRIGGER days of the 50-day reclaim scan (a failed pop above the "
                                            "50-day SMA, a shallow dip with a higher low, then a close back above)"),
    "🐻 Bear Flag (short)": (260, "short", _bt_bf, "TRIGGER days of the Swing Shorts scan's bear flag (a light-volume "
                                                 "bounce into a falling 20/50-day SMA, then a close below the sell "
                                                 "level) — traded SHORT"),
    "🪤 Failed Breakout (short)": (260, "short", _bt_fb, "TRIGGER days of the Swing Shorts scan's failed breakout (a "
                                                       "close back under the old high within a few days of "
                                                       "breaking out) — traded SHORT"),
}


BT_COMBINED = "🧩 Combined (several strategies, one account)"


BT_AGREE = "✅ All agree"


BT_TRAILS = ["None", "10-day SMA", "20-day SMA", "50-day SMA"]


BT_SCALE_OUT = "Scale out: ⅓ · ⅓ · trail the rest"


BT_TARGETS = ["None", "R-multiple", "Percent", BT_SCALE_OUT]


BT_ENTRY_TRIGGER = "Trigger day, at the breakout price"


BT_ENTRY_330 = "Trigger day at 3:30 pm (above the trigger & up on the day)"


BT_ENTRIES = ["Next day's open", "Signal day's close", BT_ENTRY_TRIGGER, BT_ENTRY_330]


is_trig_entry = lambda e: e in (BT_ENTRY_TRIGGER, BT_ENTRY_330)       # setups armed the day before


BT_ATR_STOPS = {"ATR × 0.5": 0.5, "ATR × 0.8": 0.8, "ATR × 1": 1.0}


BT_STOPS = ["The setup's stop", "Low of the entry day", *BT_ATR_STOPS, "Fixed %"]


BT_UNIVERSES = ["S&P 500", "Nasdaq-listed", "All US stocks", *MARKETS, "Stock Scanner results", "My own tickers"]


PYR_STOPS = ["Raise to the previous buy (ladder)", "Raise to breakeven (average cost)", "Keep the trade's stop"]


PYR_RULES = ["Each day it still qualifies (SETUP or TRIGGER)", "Price ladder: every N × ATR higher",
             "Each day the trend is stacked (close ≥ EMA10 ≥ EMA21 ≥ EMA50)"]


STAGE_BENCHES = ["QQQE", "SPY", "QQQ", "RSP", "IWM"]


STAGE_SIZE_DEFAULT = {"1A": 50, "1B": 50, "2A": 100, "2B": 100, "2C": 75, "3A": 50, "3B": 25, "4A": 25, "4B": 0,
                      "4C": 0}


BT_LIST_UNIVERSES = {"S&P 500": "S&P 500 (~500 · 1–2 min)", "Nasdaq-listed": "Nasdaq-listed (~3,500)",
                     "All US stocks": "All US stocks (~6,000+)", **{m: m for m in MARKETS}}


BT_VOL = {"Any": 0, "100K+": 1e5, "300K+": 3e5, "500K+": 5e5, "1M+": 1e6, "2M+": 2e6, "5M+": 5e6}


MKT_PULLBACK = "SPY pullback only (below 21 EMA, above 50 EMA)"


MKT_CUSTOM = "Custom: pick the lines"


BT_MARKET_FILTERS = ["Off", "SPY in an uptrend", "SPY not in a downtrend", MKT_PULLBACK, MKT_CUSTOM]


MKT_LINES = ["10 EMA", "21 EMA", "50 EMA", "50 SMA", "200 SMA"]


# sectors the user left out (Stock Scanner sidebar → Sectors); used by every scan and backtest
DEFAULT_SECTOR_EXCL = ["Real Estate", "Utilities"]


BT_PRIORITY = ["Highest RS first", "Tightest stop first", "Alphabetical"]


OLD_BT_NAMES = {'🚩 Tight-flag breakout': '🚩 Tight Flag Pattern', '🔁 EMA crossback': '🔁 EMA Crossback', '🟢 Green line breakout': '✳️ Green Line Breakout', '🪤 Shakeout +3': '📈 Shakeout +3', '🚀 Episodic pivot': '🚀 Episodic Pivot', '📉 Parabolic short': '📉 Parabolic Short'}


# ---- page switcher at the top: Stock Scanner | Relative Strength | Similar stocks | Smart Money ----
# ============================================================================
# Sectors in play: sector / industry ETF scanner
# ============================================================================
ETF_MARKET = [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("IWM", "Russell 2000 (small caps)")]


# (ticker, name, sector it belongs to, "Sector" / "Industry")
ETF_LIST = [
    ("XLK", "Technology", "Technology", "Sector"),
    ("XLC", "Communication Services", "Communication Services", "Sector"),
    ("XLY", "Consumer Discretionary", "Consumer Discretionary", "Sector"),
    ("XLP", "Consumer Staples", "Consumer Staples", "Sector"),
    ("XLV", "Health Care", "Health Care", "Sector"),
    ("XLF", "Financials", "Financials", "Sector"),
    ("XLI", "Industrials", "Industrials", "Sector"),
    ("XLE", "Energy", "Energy", "Sector"),
    ("XLB", "Materials", "Materials", "Sector"),
    ("XLU", "Utilities", "Utilities", "Sector"),
    ("XLRE", "Real Estate", "Real Estate", "Sector"),
    ("SMH", "Semiconductors", "Technology", "Industry"),
    ("IGV", "Software", "Technology", "Industry"),
    ("CIBR", "Cybersecurity", "Technology", "Theme"),
    ("SKYY", "Cloud computing", "Technology", "Theme"),
    ("BOTZ", "Robotics & AI", "Technology", "Theme"),
    ("QTUM", "Quantum computing & ML", "Technology", "Theme"),
    ("FDN", "Internet", "Communication Services", "Industry"),
    ("XRT", "Retail", "Consumer Discretionary", "Industry"),
    ("IBUY", "Online retail", "Consumer Discretionary", "Theme"),
    ("ITB", "Home builders", "Consumer Discretionary", "Industry"),
    ("DRIV", "Autos & EVs", "Consumer Discretionary", "Theme"),
    ("PEJ", "Leisure & entertainment", "Consumer Discretionary", "Industry"),
    ("PBJ", "Food & beverage", "Consumer Staples", "Industry"),
    ("XBI", "Biotech (equal weight)", "Health Care", "Industry"),
    ("IBB", "Biotech (large)", "Health Care", "Industry"),
    ("XPH", "Pharma", "Health Care", "Industry"),
    ("IHI", "Medical devices", "Health Care", "Industry"),
    ("IHF", "Health care providers", "Health Care", "Industry"),
    ("KRE", "Regional banks", "Financials", "Industry"),
    ("KBE", "Banks", "Financials", "Industry"),
    ("IAI", "Brokers & exchanges", "Financials", "Industry"),
    ("KIE", "Insurance", "Financials", "Industry"),
    ("FINX", "Fintech", "Financials", "Theme"),
    ("IBIT", "Bitcoin", "Financials", "Theme"),
    ("ITA", "Aerospace & defense", "Industrials", "Industry"),
    ("UFO", "Space", "Industrials", "Theme"),
    ("IYT", "Transports", "Industrials", "Industry"),
    ("JETS", "Airlines", "Industrials", "Industry"),
    ("PAVE", "Infrastructure", "Industrials", "Theme"),
    ("GRID", "Power grid", "Industrials", "Theme"),
    ("XOP", "Oil & gas producers", "Energy", "Industry"),
    ("OIH", "Oil services", "Energy", "Industry"),
    ("URA", "Uranium", "Energy", "Theme"),
    ("NLR", "Nuclear energy", "Utilities", "Theme"),
    ("TAN", "Solar", "Energy", "Theme"),
    ("ICLN", "Clean energy", "Energy", "Theme"),
    ("XME", "Metals & mining", "Materials", "Industry"),
    ("GDX", "Gold miners", "Materials", "Industry"),
    ("SIL", "Silver miners", "Materials", "Industry"),
    ("COPX", "Copper miners", "Materials", "Industry"),
    ("LIT", "Lithium & batteries", "Materials", "Theme"),
    ("SLX", "Steel", "Materials", "Industry"),
    # ---- more industries ----
    ("XSD", "Semiconductors (equal weight)", "Technology", "Industry"),
    ("XSW", "Software & services (equal weight)", "Technology", "Industry"),
    ("XTL", "Telecom", "Communication Services", "Industry"),
    ("SOCL", "Social media", "Communication Services", "Industry"),
    ("ESPO", "Video games & esports", "Communication Services", "Industry"),
    ("XHB", "Homebuilders & building products", "Consumer Discretionary", "Industry"),
    ("BJK", "Casinos & gaming", "Consumer Discretionary", "Industry"),
    ("XHE", "Health care equipment", "Health Care", "Industry"),
    ("XHS", "Health care services", "Health Care", "Industry"),
    ("KBWB", "Large banks", "Financials", "Industry"),
    ("KCE", "Capital markets", "Financials", "Industry"),
    ("REM", "Mortgage REITs", "Real Estate", "Industry"),
    ("XAR", "Aerospace & defense (equal weight)", "Industrials", "Industry"),
    ("XTN", "Transportation (equal weight)", "Industrials", "Industry"),
    ("XES", "Oil & gas equipment & services", "Energy", "Industry"),
    ("FCG", "Natural gas", "Energy", "Industry"),
    ("AMLP", "Pipelines & MLPs", "Energy", "Industry"),
    ("GDXJ", "Junior gold miners", "Materials", "Industry"),
    ("SILJ", "Junior silver miners", "Materials", "Industry"),
    ("MOO", "Agribusiness", "Materials", "Industry"),
    ("WOOD", "Timber & forestry", "Materials", "Industry"),
    # ---- more themes ----
    ("AIQ", "Artificial intelligence", "Technology", "Theme"),
    ("DTCR", "Data centers", "Technology", "Theme"),
    ("ARKK", "Disruptive innovation (ARK)", "Technology", "Theme"),
    ("ARKG", "Genomics", "Health Care", "Theme"),
    ("WGMI", "Bitcoin miners", "Financials", "Theme"),
    ("BLOK", "Blockchain", "Financials", "Theme"),
    ("SHLD", "Defense tech", "Industrials", "Theme"),
    ("REMX", "Rare earths", "Materials", "Theme"),
    ("HYDR", "Hydrogen", "Energy", "Theme"),
    ("FAN", "Wind energy", "Energy", "Theme"),
    ("PHO", "Water", "Utilities", "Theme"),
    ("KWEB", "China internet", "Communication Services", "Theme"),
    ("BETZ", "Sports betting", "Consumer Discretionary", "Theme"),
    ("MJ", "Cannabis", "Health Care", "Theme"),
]


ETF_STATUS = {"🔥 In play": "#22c55e", "🌱 Emerging": "#3b82f6", "⚠️ Cooling": "#f59e0b", "❄️ Weak": "#ef4444"}


ETF_TICKERS = [r[0] for r in ETF_LIST]


ETF_NAME = {r[0]: r[1] for r in ETF_LIST}


ETF_OK = {"🔥 In play + 🌱 Emerging": ("🔥 In play", "🌱 Emerging"), "🔥 In play only": ("🔥 In play",)}


GROUP_CHOICES = ["All groups", *ETF_OK]


# a stock's sector (any naming: Nasdaq, GICS, Yahoo) -> sector ETF
ETF_SECTOR_OF = {
    "technology": "XLK", "information technology": "XLK", "tech": "XLK",
    "communication services": "XLC", "telecommunications": "XLC", "communications": "XLC",
    "consumer discretionary": "XLY", "consumer cyclical": "XLY",
    "consumer staples": "XLP", "consumer defensive": "XLP",
    "health care": "XLV", "healthcare": "XLV",
    "financials": "XLF", "financial services": "XLF", "finance": "XLF", "financial": "XLF",
    "industrials": "XLI", "energy": "XLE", "materials": "XLB", "basic materials": "XLB",
    "utilities": "XLU", "real estate": "XLRE",
}


# a stock's industry -> industry ETF (first match wins; checked before the sector)
ETF_INDUSTRY_RULES = [(re.compile(p, re.I), t) for p, t in [
    (r"semiconductor", "SMH"), (r"software|prepackaged", "IGV"), (r"biotech|biological products", "XBI"),
    (r"pharma|drug manufacturers", "XPH"),
    (r"medical (devices|instruments)|medical/dental|surgical|orthopedic|electromedical", "IHI"),
    (r"health ?care plans|medical care facilities|hospital|health care providers|medical specialities|nursing",
     "IHF"),
    (r"banks? ?- ?regional|regional bank|savings institutions", "KRE"), (r"\bbank", "KBE"),
    (r"insur", "KIE"), (r"capital markets|broker|exchanges|investment bankers|financial data", "IAI"),
    (r"uranium", "URA"), (r"solar", "TAN"),
    (r"oil ?& ?gas (e&p|production|integrated)|integrated oil|crude", "XOP"),
    (r"oil.*(equipment|services|drilling)|oilfield", "OIH"),
    (r"\bgold|precious metals", "GDX"), (r"silver", "SIL"), (r"copper", "COPX"), (r"steel|iron ore", "SLX"),
    (r"lithium", "LIT"), (r"mining|metals", "XME"),
    (r"aerospace|defense|military", "ITA"), (r"airline|air transport", "JETS"),
    (r"trucking|railroad|freight|delivery|marine shipping|transportation", "IYT"),
    (r"residential construction|homebuilding", "ITB"),
    (r"internet retail|catalog|direct marketing|online retail", "IBUY"),
    (r"retail|department stores|apparel stores|discount stores", "XRT"),
    (r"internet", "FDN"), (r"auto manufacturers|auto parts|motor vehicles|auto & truck", "DRIV"),
    (r"electronic gaming|video game|interactive home entertainment", "ESPO"),
    (r"casino|gambling", "BJK"), (r"resorts|leisure|lodging|restaurants|travel", "PEJ"),
    (r"telecom|wireless|communications equipment", "XTL"),
    (r"mortgage reit|reit.*mortgage|reit - mortgage", "REM"),
    (r"oil ?& ?gas midstream|pipeline", "AMLP"),
    (r"agricultur|farm products|fertilizer", "MOO"), (r"lumber|wood|paper|timber", "WOOD"),
    (r"packaged foods|beverages|food|confectioner", "PBJ"),
    # themes (a stock can have an industry ETF AND a theme ETF)
    (r"renewable", "ICLN"), (r"\bwater\b", "PHO"), (r"independent power", "NLR"), (r"electrical equipment", "GRID"),
    (r"engineering ?& ?construction|infrastructure operations|construction materials|building products", "PAVE"),
]]


GROUP_LEVELS = ["Sector", "Industry", "Theme"]


ETF_LEVEL = {r[0]: r[3] for r in ETF_LIST}


# Themes: well-known pure plays by ticker (a hand-made list — themes can't be read from industry names),
# then the industry-name rules above. theme -> (its ETF or None, tickers)
THEMES = {
    "Quantum computing": ("QTUM", "IONQ RGTI QBTS QUBT ARQQ"),
    "Cybersecurity": ("CIBR", "CRWD PANW ZS FTNT S OKTA CYBR TENB QLYS RPD CHKP VRNS NET"),
    "Crypto & bitcoin": ("IBIT", "COIN MSTR MARA RIOT CLSK HUT BITF CIFR WULF IREN BTBT GLXY"),
    "Space": ("UFO", "RKLB ASTS LUNR PL RDW BKSY SPCE MNTS FLY"),
    "Nuclear & uranium": ("URA", "CCJ UEC NXE DNN UUUU LEU OKLO SMR NNE BWXT URG EU"),
    "AI power & grid": ("GRID", "VRT ETN PWR GEV POWL HUBB NVT CEG VST TLN"),
    "AI data centers & servers": ("DTCR", "SMCI DELL ANET NBIS APLD CRWV CORZ"),
    "Drones & defense tech": ("SHLD", "AVAV KTOS RCAT UMAC ONDS DPRO"),
    "Solar": ("TAN", "FSLR ENPH RUN SEDG NXT ARRY CSIQ JKS SHLS"),
    "EVs": ("DRIV", "TSLA RIVN LCID NIO XPEV LI"),
    "Lithium & batteries": ("LIT", "ALB SQM LAC PLL SGML QS SLDP ENVX EOSE FLNC"),
    "Fintech": ("FINX", "SOFI AFRM UPST PYPL XYZ HOOD NU TOST"),
    "Cloud software": ("SKYY", "SNOW DDOG MDB ESTC CFLT GTLB"),
}


THEME_OF_TICKER = {t: name for name, (_, tks) in THEMES.items() for t in tks.split()}


THEME_ETF = {name: etf for name, (etf, _) in THEMES.items()}


ETF_THEME_NAME = {"TAN": "Solar", "URA": "Nuclear & uranium", "NLR": "Nuclear & uranium", "LIT": "Lithium & batteries",
                  "ICLN": "Clean energy", "GRID": "AI power & grid", "PAVE": "Infrastructure", "DRIV": "EVs",
                  "IBUY": "Online retail", "PHO": "Water"}


GRP_CORR_DAYS = 60


# ---------------------------------------------------------------------------------------------------------------
# 💼 Positions: pilot + add to winners — the daily action list for your open positions
# ---------------------------------------------------------------------------------------------------------------
POS_FILE = Path(__file__).parent.parent.with_name("positions.json")


POS_COLS = ["Symbol", "Pilot date", "Buys", "Stop", "Note"]


# ---------------------------------------------------------------------------------------------------------------
# 🔻 Funnel: market → leading groups → leading stocks → setup today → pilot buys
# ---------------------------------------------------------------------------------------------------------------
# scan key -> (statuses that are a buy today, statuses that are still a watchlist setup)
FN_SCANS = {"pattern": (("TRIGGER",), ("SETUP",)), "xback": (("TRIGGER",), ("SETUP",)),
            "glb": (("BREAKOUT",), ("RETEST", "NEAR")), "so3": (("TRIGGER",), ("RECLAIMED", "UNDERCUT")),
            "vcp": (("TRIGGER",), ("SETUP",)), "mac": (("TRIGGER",), ("SETUP",)),
            "htf": (("TRIGGER",), ("SETUP",)), "reclaim": (("TRIGGER",), ("SETUP", "DIP"))}


FN_F50 = ["Off", "Also count it as a setup", "Only these stocks"]


FN_DEFAULTS = dict(fn_f50=FN_F50[0], fn_f50_days=30, fn_minpx=10.0, fn_mindv=20.0, fn_adr=3.0, fn_mkt=True, fn_ind_top=30, fn_ind_min=3, fn_lead_rs=95,
                   fn_corr=0.3, fn_etf="Off", fn_rs=85, fn_stages=["1A", "1B", "2A", "2B"],
                   fn_scans=[F[k]["label"] for k in FN_SCANS])


FN_STEPS = ["1 · All stocks (liquid)", "2 · In a leading group", "3 · Leading stock", "4 · Setup today",
            "5 · Pilot buys (TRIGGER)"]


PAGES = ["📈 Stock Scanner", "🗂 Sectors in play", "🧭 Relative Strength", "🔗 Similar stocks", "💰 Smart Money",
         "🧪 Backtest", "💼 Positions", "🔻 Funnel"]


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


# ---- saved scan presets (scan type + its settings + filters) ----
PRESET_FILE = Path(__file__).parent.parent.with_name("scan_presets.json")

# the Stock Scanner page body lives in scanner/main.py (extracted from the old monolith)
from scanner.main import stock_scanner_page  # noqa: E402


PRESET_PREFIXES = ("pb_", "ep_", "sp_", "xb_", "gl_", "so_", "vcp_", "mac_", "htf_", "sh_", "rc_", "f5_", "ll_", "rss_", "rs_", "vr_", "earn_", "v_f_", "v_fmin_", "v_fmax_", "v_fcat_")


# Each scan's own calculations, cached separately: changing one scan's setting only re-runs that scan.
CORE_KEYS = ("max_range", "min_pull", "max_pull", "min_bars", "max_ema_dist", "max_ema_gap", "dry_up", "breakout_vol")


F = {}
F["pattern"] = dict(label="Tight flag", col="Pattern", kind="preset", presets=[
    ("SETUP or TRIGGER", "Basing, or breaking out today", lambda d: d["Pattern"].isin(["SETUP", "TRIGGER"])),
    ("TRIGGER only", "Broke above the base today on volume", lambda d: d["Pattern"] == "TRIGGER"),
    ("SETUP only", "Still basing — watchlist candidates", lambda d: d["Pattern"] == "SETUP")])
F["volrec"] = dict(label="Record volume", col="Vol record", kind="preset", presets=[
    ("Highest in 3 months", "Heaviest day in ~63 sessions", is_true("VolHi 3M")),
    ("Highest in 6 months", "Heaviest day in ~126 sessions", is_true("VolHi 6M")),
    ("Highest in 1 year", "Heaviest day in ~252 sessions", is_true("VolHi 1Y")),
    ("Highest since IPO", "Heaviest day ever traded", is_true("VolHi IPO"))])
F["parabolic"] = dict(label="Parabolic", col="Parabolic", kind="preset", presets=[
    ("First crack (short trigger)", "First red day after the run", lambda d: d["Parabolic"] == "CRACK"),
    ("Any parabolic run", "Extended, cracking or fading", lambda d: d["Parabolic"].fillna("") != ""),
    ("Still extending", "Up again today, near the high", lambda d: d["Parabolic"] == "EXTENDED"),
    ("Fading", "Rolling over from the peak", lambda d: d["Parabolic"] == "FADING")])
F["ep"] = dict(label="Episodic pivot", col="EP", kind="preset", presets=[
    ("Any EP", "Gap up on huge volume in the window", lambda d: d["EP"].fillna("") != ""),
    ("Holding above EP-day low", "Still valid", lambda d: d["EP"] == "HOLDING"),
    ("Today only", "Gapped up today", lambda d: (d["EP"].fillna("") != "") & (d["Days since EP"] == 0)),
    ("Failed (below EP-day low)", "", lambda d: d["EP"] == "FAILED")])
F["spchg"] = dict(label="S&P index change", col="S&P change", kind="preset", presets=[
    ("Added", "Joined (or joining) the index", lambda d: d["S&P change"] == "ADDED"),
    ("Removed", "Left (or leaving) the index", lambda d: d["S&P change"] == "REMOVED"),
    ("Added or removed", "", lambda d: d["S&P change"].isin(["ADDED", "REMOVED"])),
    ("Inclusion candidates", "S&P 500: largest US non-members", lambda d: d["S&P change"] == "CANDIDATE")])
SP_WINDOWS = {"Upcoming only": 0, "Last 2 weeks": 14, "Last 1 month": 31, "Last 3 months": 92,
              "Last 6 months": 183, "Last 1 year": 366}
F["rsq"] = dict(label="RS quadrant", col="Quadrant", kind="preset", presets=[
    ("Strong", "Leading on both week and month", lambda d: d["Quadrant"] == "Strong"),
    ("Improving", "Strong this week, weak over the month", lambda d: d["Quadrant"] == "Improving"),
    ("Weakening", "Weak this week, strong over the month", lambda d: d["Quadrant"] == "Weakening"),
    ("Weak", "Lagging on both", lambda d: d["Quadrant"] == "Weak"),
    ("Strong or Improving", "", lambda d: d["Quadrant"].isin(["Strong", "Improving"]))])
F["xback"] = dict(label="EMA crossback", col="Crossback", kind="preset", presets=[
    ("SETUP or TRIGGER", "At the EMA, or breaking out", lambda d: d["Crossback"].isin(["SETUP", "TRIGGER"])),
    ("TRIGGER only", "Broke above the consolidation high today", lambda d: d["Crossback"] == "TRIGGER"),
    ("SETUP only", "Pulled back to the EMA, holding — watchlist", lambda d: d["Crossback"] == "SETUP")])
F["glb"] = dict(label="Green line", col="GLB", kind="preset", presets=[
    ("Breakout, retest or near", "All green-line setups", lambda d: d["GLB"].isin(["BREAKOUT", "RETEST", "NEAR"])),
    ("BREAKOUT only", "First close above the green line today, on volume", lambda d: d["GLB"] == "BREAKOUT"),
    ("RETEST only", "Broke out, pulled back to the line and holding it", lambda d: d["GLB"] == "RETEST"),
    ("NEAR only", "Just below the green line — watchlist", lambda d: d["GLB"] == "NEAR"),
    ("Breakout or retest", "Actionable now", lambda d: d["GLB"].isin(["BREAKOUT", "RETEST"]))])
F["so3"] = dict(label="Shakeout +3", col="SO+3", kind="preset", presets=[
    ("Trigger or reclaimed", "Crossed the +3 level today, or back above the first low",
     lambda d: d["SO+3"].isin(["TRIGGER", "RECLAIMED"])),
    ("TRIGGER only", "Closed above the +3 level today", lambda d: d["SO+3"] == "TRIGGER"),
    ("RECLAIMED only", "Back above the first low — early entry zone", lambda d: d["SO+3"] == "RECLAIMED"),
    ("UNDERCUT only", "Just shook out below the first low — watchlist", lambda d: d["SO+3"] == "UNDERCUT"),
    ("Any stage", "All three", lambda d: d["SO+3"].isin(["TRIGGER", "RECLAIMED", "UNDERCUT"]))])
F["htf"] = dict(label="High tight flag", col="HTF", kind="preset", presets=[
    ("SETUP or TRIGGER", "In the flag, or broke out of it today", lambda d: d["HTF"].isin(["TRIGGER", "SETUP"])),
    ("TRIGGER only", "Closed above the flag high today on volume", lambda d: d["HTF"] == "TRIGGER"),
    ("SETUP only", "Resting in the flag — watchlist", lambda d: d["HTF"] == "SETUP")])
F["reclaim"] = dict(label="50-day reclaim", col="Reclaim", kind="preset", presets=[
    ("TRIGGER, SETUP or DIP", "Everything in the pattern: reclaimed, armed, or still dipping (higher low not formed yet)",
     lambda d: d["Reclaim"].isin(["TRIGGER", "SETUP", "DIP"])),
    ("SETUP or TRIGGER", "In the dip under the 50-day SMA, or closed back above it today",
     lambda d: d["Reclaim"].isin(["TRIGGER", "SETUP"])),
    ("TRIGGER only", "Closed back above the 50-day SMA (today, or within the last N days and still above)",
     lambda d: d["Reclaim"] == "TRIGGER"),
    ("SETUP only", "Dipping under the 50-day SMA with a higher low — buy-stop at the SMA", lambda d: d["Reclaim"] == "SETUP"),
    ("DIP only", "Dipping under the 50-day SMA, higher low not formed yet — early watchlist",
     lambda d: d["Reclaim"] == "DIP")])
F["f50scan"] = dict(label="First-close status", col="First50", kind="preset", presets=[
    ("TRIGGER or SETUP", "Closed above the 50-day SMA for the first time in N+ sessions, or still under it and close",
     lambda d: d["First50"].isin(["TRIGGER", "SETUP"])),
    ("TRIGGER only", "First close above the 50-day SMA (today, or within the last N days and still above)",
     lambda d: d["First50"] == "TRIGGER"),
    ("SETUP only", "Still under the 50-day SMA after N+ sessions, within a few % of it — watchlist",
     lambda d: d["First50"] == "SETUP")])
F["swshort"] = dict(label="Swing short", col="Short", kind="preset", presets=[
    ("SETUP or TRIGGER", "Armed for a sell-stop tomorrow, or triggered today",
     lambda d: d["Short"].isin(["TRIGGER", "SETUP"])),
    ("TRIGGER only", "Closed below the sell level today", lambda d: d["Short"] == "TRIGGER"),
    ("SETUP only", "Armed — sell-stop order for tomorrow", lambda d: d["Short"] == "SETUP")])
F["mac"] = dict(label="MA consolidation", col="MAC", kind="preset", presets=[
    ("SETUP or TRIGGER", "Consolidating on the line, or broke out of the box today",
     lambda d: d["MAC"].isin(["TRIGGER", "SETUP"])),
    ("TRIGGER only", "Closed above the box high today", lambda d: d["MAC"] == "TRIGGER"),
    ("SETUP only", "Still consolidating on the line — watchlist", lambda d: d["MAC"] == "SETUP")])
F["vcp"] = dict(label="VCP", col="VCP", kind="preset", presets=[
    ("SETUP or TRIGGER", "Just under the pivot, or broke out today", lambda d: d["VCP"].isin(["TRIGGER", "SETUP"])),
    ("TRIGGER only", "Closed above the pivot today on volume", lambda d: d["VCP"] == "TRIGGER"),
    ("SETUP only", "A valid VCP just under its pivot — watchlist", lambda d: d["VCP"] == "SETUP")])
F["rsscore"] = dict(label="RS score", col="RS pass", kind="preset", presets=[
    ("Passes the RS rules", "Score, ranks, stage and industry rules", is_true("RS pass")),
    ("Strong quadrant", "Passes, and leading on week and month", lambda d: is_true("RS pass")(d) & (d["Quadrant"] == "Strong")),
    ("Strong or Improving", "Passes, and leading this week", lambda d: is_true("RS pass")(d)
     & d["Quadrant"].isin(["Strong", "Improving"]))])
F["liqlead"] = dict(label="Liquid Leaders", col="LiqLead", kind="preset", presets=[
    ("Passes all groups", "Liquid + volatile + strong momentum", is_true("LiqLead")),
    ("Group 1 only", "Liquid and volatile (ignore momentum)", is_true("LiqLead G1")),
    ("Group 2 only", "Momentum only (ignore liquidity)", is_true("LiqLead G2"))])


F["price"] = dict(label="Price", col="Price", kind="num", unit="$", presets=[
    ("Above 100", "Fractional shares time", between("Price", 100)),
    ("Above 10", "Skip low-priced stocks", between("Price", 10)),
    ("10 to 100", "Mid-priced", between("Price", 10, 100)),
    ("10 and below", "Not quite penny stocks", between("Price", None, 10)),
    ("5 and below", "Penny stocks", between("Price", None, 5)),
    ("Above EMA 21", "Short-term uptrend", between("vs EMA21 %", 0.0001)),
    ("Above SMA 50", "Uptrend", between("vs SMA50 %", 0.0001)),
    ("Above SMA 200", "Long-term uptrend", between("vs SMA200 %", 0.0001)),
    ("Below SMA 50", "Downtrend", between("vs SMA50 %", None, -0.0001))])
F["chg"] = dict(label="Chg %", col="Chg %", kind="num", unit="%", presets=[
    ("Up", "Green today", between("Chg %", 0.0001)), ("Up more than 3%", "", between("Chg %", 3)),
    ("Up more than 5%", "Big movers", between("Chg %", 5)), ("Down", "Red today", between("Chg %", None, -0.0001)),
    ("Down more than 3%", "", between("Chg %", None, -3)), ("Down more than 5%", "", between("Chg %", None, -5))])
F["mcap"] = dict(label="Mkt cap", col="Mkt cap $B", kind="num", unit="$B", presets=[
    ("Mega", "200B and above", between("Mkt cap $B", 200)), ("Large", "10B to 200B", between("Mkt cap $B", 10, 200)),
    ("Mid", "2B to 10B", between("Mkt cap $B", 2, 10)), ("Small", "300M to 2B", between("Mkt cap $B", 0.3, 2)),
    ("Micro", "Below 300M", between("Mkt cap $B", None, 0.3)), ("Above 2B", "Mid caps and bigger", between("Mkt cap $B", 2)),
    ("Above 10B", "Large caps and bigger", between("Mkt cap $B", 10))])
F["avgvol"] = dict(label="Avg volume", col="Avg vol 50d", kind="num", unit="shares", presets=[
    ("Above 100K", "", between("Avg vol 50d", 1e5)), ("Above 500K", "Liquid", between("Avg vol 50d", 5e5)),
    ("Above 1M", "Very liquid", between("Avg vol 50d", 1e6)), ("Above 2M", "", between("Avg vol 50d", 2e6)),
    ("Above 5M", "", between("Avg vol 50d", 5e6)), ("Above 10M", "Mega liquid", between("Avg vol 50d", 1e7))])
F["adr"] = dict(label="ADR %", col="ADR % 14d", kind="num", unit="% daily range", presets=[
    ("Below 2%", "Slow movers", between("ADR % 14d", None, 2)), ("2% to 4%", "Normal", between("ADR % 14d", 2, 4)),
    ("Above 3%", "Moves enough to swing trade", between("ADR % 14d", 3)),
    ("Above 4%", "Volatile", between("ADR % 14d", 4)), ("Above 5%", "Very volatile", between("ADR % 14d", 5)),
    ("Above 8%", "Explosive", between("ADR % 14d", 8))])
F["under50"] = dict(label="First close above 50-day", col="Under 50d days", kind="num", unit="days under it", presets=[
    ("After 30+ days under", "Closed above the 50-day SMA today, first time in 30+ sessions", between("Under 50d days", 30)),
    ("After 20+ days under", "…first time in 20+ sessions", between("Under 50d days", 20)),
    ("After 10+ days under", "…first time in 10+ sessions", between("Under 50d days", 10)),
    ("After 5+ days under", "…first time in 5+ sessions", between("Under 50d days", 5)),
    ("Above it today (any)", "Closed above the 50-day SMA today", between("Under 50d days", 0))])
F["dollarvol"] = dict(label="$ Volume", col="$ Vol M", kind="num", unit="$M/day", presets=[
    ("Above $10M", "Traded per day", between("$ Vol M", 10)), ("Above $50M", "", between("$ Vol M", 50)),
    ("Above $100M", "Institutional size", between("$ Vol M", 100))])
F["relvol"] = dict(label="Rel volume", col="Rel vol", kind="num", unit="x", presets=[
    ("Above 1.5", "Busier than usual", between("Rel vol", 1.5)), ("Above 2", "Heavy volume", between("Rel vol", 2)),
    ("Above 3", "Unusual volume", between("Rel vol", 3)), ("Below 0.75", "Quiet", between("Rel vol", None, 0.75)),
    ("Below 0.5", "Very quiet (dry-up)", between("Rel vol", None, 0.5))])
for key, lbl in [("perf1w", "1W"), ("perf1m", "1M"), ("perf3m", "3M"), ("perf6m", "6M"),
                 ("perfytd", "YTD"), ("perf1y", "1Y")]:
    F[key] = perf_filter(f"Perf {lbl} %", f"Perf {lbl}")
F["rs"] = dict(label="RS rating", col="RS", kind="num", unit="1–99", presets=[
    ("90 and above", "Top 10% leaders", between("RS", 90)), ("80 and above", "Top 20%", between("RS", 80)),
    ("70 and above", "", between("RS", 70)), ("50 and above", "Better than average", between("RS", 50)),
    ("Below 50", "Laggards", between("RS", None, 49.9))])
F["sector"] = dict(label="Sector", col="Sector", kind="cat")
F["industry"] = dict(label="Industry", col="Industry", kind="cat")
F["theme"] = dict(label="Theme", col="Theme", kind="cat")
F["trend"] = dict(label="Moving averages", col=None, kind="preset", presets=[
    ("Price > SMA 50 > SMA 200", "Classic uptrend", is_true("Stack 50>200")),
    ("Stage 2 trend template", "Minervini: 50>150>200, 200 rising, near highs", is_true("Trend template")),
    ("Price > EMA 21, EMA 10 > 21 > SMA 50", "Short-term momentum stack", is_true("EMA stack")),
    ("Price above SMA 200", "Long-term uptrend", between("vs SMA200 %", 0.0001)),
    ("Price below SMA 200", "Long-term downtrend", between("vs SMA200 %", None, -0.0001))])
F["hi52"] = dict(label="52W high", col="Below 52W high %", kind="num", unit="% below", presets=[
    ("New 52-week high today", "Breaking to new highs", is_true("New 52W high")),
    ("Within 5% of high", "Right at the highs", between("Below 52W high %", None, 5)),
    ("Within 10% of high", "", between("Below 52W high %", None, 10)),
    ("Within 25% of high", "", between("Below 52W high %", None, 25)),
    ("More than 25% below high", "Beaten down", between("Below 52W high %", 25))])
F["lo52"] = dict(label="52W low", col="Above 52W low %", kind="num", unit="% above", presets=[
    ("30% above low", "Minervini minimum", between("Above 52W low %", 30)),
    ("100% above low", "Doubled off the low", between("Above 52W low %", 100)),
    ("Within 10% of low", "Near the lows", between("Above 52W low %", None, 10))])
F["rsi"] = dict(label="RSI", col="RSI 14", kind="num", unit="", presets=[
    ("Overbought", "Above 70", between("RSI 14", 70)), ("Strong", "50 to 70", between("RSI 14", 50, 70)),
    ("Weak", "30 to 50", between("RSI 14", 30, 50)), ("Oversold", "Below 30", between("RSI 14", None, 30))])
F["atr"] = dict(label="ATR %", col="ATR %", kind="num", unit="% of price", presets=[
    ("Below 2%", "Calm movers", between("ATR %", None, 2)), ("2% to 4%", "Normal", between("ATR %", 2, 4)),
    ("Above 4%", "Volatile", between("ATR %", 4)), ("Above 6%", "Very volatile", between("ATR %", 6))])
F["gap"] = dict(label="Gap %", col="Gap %", kind="num", unit="%", presets=[
    ("Gap up", "Opened above yesterday's close", between("Gap %", 0.0001)),
    ("Gap up more than 2%", "", between("Gap %", 2)), ("Gap up more than 5%", "Big gap", between("Gap %", 5)),
    ("Gap down", "", between("Gap %", None, -0.0001)), ("Gap down more than 2%", "", between("Gap %", None, -2))])
F["tight"] = dict(label="7d range", col="7d range %", kind="num", unit="%", presets=[
    ("Below 5%", "Very tight", between("7d range %", None, 5)), ("Below 8%", "Tight", between("7d range %", None, 8)),
    ("Below 12%", "Fairly tight", between("7d range %", None, 12))])
F["ema21"] = dict(label="vs EMA 21", col="vs EMA21 %", kind="num", unit="%", presets=[
    ("Within ±2%", "Hugging the 21 EMA", between("vs EMA21 %", -2, 2)),
    ("Within ±5%", "", between("vs EMA21 %", -5, 5)), ("Above", "", between("vs EMA21 %", 0.0001)),
    ("More than 10% above", "Extended", between("vs EMA21 %", 10)), ("Below", "", between("vs EMA21 %", None, -0.0001))])
F["sma20"] = dict(label="vs SMA 20", col="vs SMA20 %", kind="num", unit="%", presets=[
    ("Within ±2%", "Hugging the 20-day line", between("vs SMA20 %", -2, 2)),
    ("0–5% above", "Just above the 20-day line", between("vs SMA20 %", 0, 5)),
    ("Above", "", between("vs SMA20 %", 0.0001)), ("Below", "", between("vs SMA20 %", None, -0.0001))])
F["sma50"] = dict(label="vs SMA 50", col="vs SMA50 %", kind="num", unit="%", presets=[
    ("Above", "", between("vs SMA50 %", 0.0001)), ("Within ±5%", "Near the 50", between("vs SMA50 %", -5, 5)),
    ("More than 20% above", "Extended", between("vs SMA50 %", 20)), ("Below", "", between("vs SMA50 %", None, -0.0001))])
F["pe"] = dict(label="P/E", col="P/E", kind="num", unit="", fund=True, presets=[
    ("Profitable", "P/E above 0", between("P/E", 0.0001)), ("Below 15", "Cheap", between("P/E", 0.0001, 15)),
    ("15 to 30", "", between("P/E", 15, 30)), ("Above 30", "Growth priced", between("P/E", 30))])
F["epsg"] = dict(label="EPS growth", col="EPS growth %", kind="num", unit="% yoy", fund=True, presets=[
    ("Positive", "", between("EPS growth %", 0.0001)), ("Above 25%", "Strong", between("EPS growth %", 25)),
    ("Above 50%", "Very strong", between("EPS growth %", 50))])
F["revg"] = dict(label="Revenue growth", col="Rev growth %", kind="num", unit="% yoy", fund=True, presets=[
    ("Positive", "", between("Rev growth %", 0.0001)), ("Above 20%", "Strong", between("Rev growth %", 20)),
    ("Above 50%", "Very strong", between("Rev growth %", 50))])
F["earn"] = dict(label="Earnings date", col="Earnings in (days)", kind="num", unit="days away", fund=True, presets=[
    ("Next 7 days", "Reports this week", between("Earnings in (days)", 0, 7)),
    ("Next 14 days", "", between("Earnings in (days)", 0, 14)),
    ("Not in next 14 days", "Avoid earnings risk", lambda d: ~d["Earnings in (days)"].between(0, 14))])


# window -> (from, to) in calendar days from today; negative = past
EARN_WINDOWS = {"Today": (0, 0), "Next 1 week": (0, 7), "Previous 1 week": (-7, -1),
                "Previous 2 weeks": (-14, -1), "Previous 1 month": (-30, -1), "Previous 2 months": (-61, -1)}
EARN_DAYS = EARN_WINDOWS   # (name kept for older code)
EARN_DAYS = EARN_WINDOWS   # (name kept for older code)
F["earnwin"] = dict(label="Earnings", col="Days to earnings", kind="preset", presets=[
    (k, {"Today": "Reporting today", "Next 1 week": "Reporting in the next 7 days"}.get(k, f"Reported in the {k.lower()}"),
     between("Days to earnings", lo, hi)) for k, (lo, hi) in EARN_WINDOWS.items()])
F["short"] = dict(label="Short float", col="Short float %", kind="num", unit="%", fund=True, presets=[
    ("Above 10%", "Heavily shorted", between("Short float %", 10)),
    ("Above 20%", "Squeeze candidates", between("Short float %", 20)),
    ("Below 5%", "", between("Short float %", None, 5))])


LL_G1 = {"dvol": ("Avg \\$ vol 50d ≥ (\\$M)", "Avg $ Vol 50d M", 500.0, "$M", 50.0),
         "adr": ("ADR % 14d ≥", "ADR % 14d", 3.0, "%", 0.25)}
LL_G2 = {"c5": ("% chg 5d ≥", "Chg 5d %", 15.0, "%", 1.0),
         "c20": ("% chg 20d ≥", "Chg 20d %", 20.0, "%", 1.0),
         "off": ("% off 52W high ≥", "Off 52W high %", -25.0, "%", 1.0)}


M_RS, M_LL, M_EARN, M_VOL = "🧭 RS Score", "💧 Liquid Leaders", "📅 Earnings", "📊 Record Volume"
M_EP, M_PB, M_SP = "🚀 Episodic Pivot", "📉 Parabolic Short", "🏛 S&P Index Changes"
M_GL, M_TF, M_SO, M_XB = "✳️ Green Line Breakout", "🚩 Tight Flag Pattern", "📈 Shakeout +3", "🔁 EMA Crossback"
M_VCP, M_MAC, M_HTF = "🌀 VCP (Minervini)", "📏 MA Consolidation", "⛳ High Tight Flag"
M_SH = "🐻 Swing Shorts"
M_RC = "🔂 50-Day Reclaim"
M_F5 = "🔃 First Close Above 50-day"




BT_ANY, BT_ALL = "Signals from any of them", "Only stocks where all of them agree"
