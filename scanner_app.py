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

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
      "Accept": "application/json, text/plain, */*"}

st.markdown("""
<style>
/* compact TradingView-like filter pills */
div[data-testid="stPopover"] > div > button, div[data-testid="stPopover"] button {
    border-radius: 6px; padding: 2px 10px; min-height: 32px; font-size: 13px;
}
div[data-testid="stPopoverBody"] { min-width: 290px; }
/* table export row: CSV / TV list / Tickers all the same shape */
.st-key-exportrow button { min-height: 2.5rem !important; height: 2.5rem; border-radius: 8px !important;
  padding: 0 12px !important; font-size: 14px !important; width: 100%; }
.block-container { padding-top: 3.9rem !important; }   /* just below Streamlit's header strip (Deploy / menu) */
/* style-only blocks take no space, so the page tabs are the first thing on the page */
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] > style:only-child) { display: none; }
[data-testid="stSidebarHeader"] { height: 2.4rem; min-height: 0; padding-top: .4rem; padding-bottom: 0; margin-bottom: 0; }
[data-testid="stSidebarUserContent"] { padding-top: .2rem; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
/* sidebar option lists (e.g. "Scan for"): each option is a row, the chosen one is highlighted
   (selectors cover both older and newer Streamlit versions) */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 5px 10px; margin: 2px 0; border-radius: 8px; border: 1px solid transparent;
    transition: background .15s, border-color .15s;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(127,127,127,.10); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {
    background: rgba(38,166,154,.22); border-color: #26a69a;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p { font-weight: 700; }
/* "Scan for" list */
.st-key-scanlist { gap: 4px !important; }
.st-key-scanlist [class*="st-key-scanpick_"] button {
    justify-content: flex-start !important; text-align: left; background: transparent !important;
    border: 1px solid transparent !important; padding: 6px 14px !important; min-height: 42px; gap: 10px;
}
.st-key-scanlist [class*="st-key-scanpick_"] button > div,
.st-key-scanlist [class*="st-key-scanpick_"] button [data-testid="stMarkdownContainer"] {
    justify-content: flex-start !important; text-align: left !important; width: 100%;
}
.st-key-scanlist [class*="st-key-scanpick_"] button:hover, .st-key-scanlist [class*="st-key-scanpick_"] button:focus,
.st-key-scanlist [class*="st-key-scanpick_"] button:active {
    background: transparent !important; border-color: transparent !important; box-shadow: none !important;
}
.st-key-scanlist [class*="st-key-scanpick_"] button p { font-size: 15px; }
/* each row: the scan's name (select) + an arrow button (select and open its settings) */
.st-key-scanlist [class*="st-key-scanrow_"] { border: 1px solid transparent; border-radius: 10px;
    transition: background .12s, border-color .12s; }
.st-key-scanlist [class*="st-key-scanrow_"]:hover { background: rgba(127,127,127,.12); border-color: rgba(127,127,127,.3); }
.st-key-scanlist [class*="st-key-scanrow_"]:active { background: rgba(38,166,154,.22); border-color: #26a69a; }
.st-key-scanlist [class*="st-key-scanrow_"] [data-testid="stHorizontalBlock"] { gap: 4px !important; }
.st-key-scanlist [class*="st-key-scanarrow_"] button {
    background: transparent !important; border: 1px solid transparent !important; min-height: 38px;
    padding: 0 !important; justify-content: center !important; }
.st-key-scanlist [class*="st-key-scanarrow_"] button:hover, .st-key-scanlist [class*="st-key-scanarrow_"] button:focus,
.st-key-scanlist [class*="st-key-scanarrow_"] button:active {
    background: transparent !important; border-color: transparent !important; box-shadow: none !important; }
/* the chosen scan (an expander): arrow on the right, and only the arrow opens / closes it */
.st-key-scanlist [data-testid="stExpander"] summary { pointer-events: none; padding-right: 6px; min-height: 42px; }
.st-key-scanlist [data-testid="stExpander"] summary > span { flex-direction: row-reverse; justify-content: space-between;
    width: 100%; }
.st-key-scanlist [data-testid="stExpander"] summary > span > span:first-child,
.st-key-scanlist [data-testid="stExpander"] summary svg {
    pointer-events: auto; cursor: pointer; padding: 8px 10px; margin: -8px -4px -8px 0; border-radius: 8px; }
.st-key-scanlist [data-testid="stExpander"]:hover details { background: rgba(38,166,154,.18); }
/* buttons inside a scan's options (e.g. Default / Strict / Loose) stay compact */
.st-key-scanlist [data-testid="stExpanderDetails"] .stButton button { padding: 4px 6px !important; min-height: 34px; }
.st-key-scanlist [data-testid="stExpanderDetails"] .stButton button p { font-size: 13px; white-space: nowrap; }
.st-key-scanlist [data-testid="stExpander"] details {
    border: 1px solid #26a69a !important; background: rgba(38,166,154,.10); border-radius: 8px;
}
.st-key-scanlist [data-testid="stExpander"] summary p { font-weight: 700; font-size: 15px; }
/* let option labels wrap instead of being cut off with "…" */
.st-key-scanlist [data-testid="stCheckbox"] label, .st-key-scanlist [data-testid="stCheckbox"] label * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
/* smooth opening of the chosen scan's options */
@keyframes scanOpen { from { opacity: 0; transform: translateY(-6px); max-height: 0; }
                      to   { opacity: 1; transform: none; max-height: 1600px; } }
.st-key-scanlist [data-testid="stExpanderDetails"] { animation: scanOpen .35s ease-out; overflow: hidden; }
.st-key-scanlist [data-testid="stExpander"] details { transition: background .25s, border-color .25s; }
/* don't dim the page while the app reloads after a click (that looked like lag) */
[data-stale="true"] { opacity: 1 !important; transition: none !important; }
/* SCAN button stays in view at the bottom of the sidebar */
section[data-testid="stSidebar"] div:has(.st-key-scanbar) { background-color: inherit; }
section[data-testid="stSidebar"] div:has(> .st-key-scanbar) { position: sticky; bottom: 0; z-index: 20; }
section[data-testid="stSidebar"] .st-key-scanbar { position: sticky; bottom: 0; z-index: 20;
  padding: .7rem .15rem .35rem .15rem; margin-top: .6rem; border-top: 1px solid rgba(128,128,128,.25);
  background-color: inherit;
  border-radius: 14px 14px 0 0; }
section[data-testid="stSidebar"] .st-key-scanbar button[kind="primary"] { font-weight: 700; letter-spacing: .06em;
  padding: .55rem 0; }
/* SCAN = green · STOP (while scanning) = red */
.st-key-scango button { background-color: #16a34a !important; border-color: #16a34a !important; color: #fff !important; }
.st-key-scango button:hover { background-color: #15803d !important; border-color: #15803d !important; }
.st-key-scanstop button { background-color: #dc2626 !important; border-color: #dc2626 !important; color: #fff !important;
  animation: stopPulse 1.4s ease-in-out infinite; }
.st-key-scanstop button:hover { background-color: #b91c1c !important; border-color: #b91c1c !important; }
@keyframes stopPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,.45); } 50% { box-shadow: 0 0 0 5px rgba(220,38,38,0); } }

/* ---------- smooth + rounded look ---------- */
html { scroll-behavior: smooth; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stIFrame"], .rs-wrap, .cards {
    animation: fadeUp .28s ease-out;
}
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid rgba(127,127,127,.18); }
[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
iframe { border-radius: 12px; }
[data-testid="stExpander"] details { border-radius: 12px !important; }
.stButton button, [data-testid="stDownloadButton"] button, [data-testid="stPopover"] button,
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
    border-radius: 10px !important; transition: background .15s, border-color .15s, transform .08s;
}
.stButton button:active, [data-testid="stDownloadButton"] button:active { transform: scale(.98); }
div[data-testid="stPopover"] > div > button, div[data-testid="stPopover"] button { border-radius: 18px !important; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="textarea"], [data-testid="stNumberInput"] > div > div {
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] { border-radius: 12px; }
div[data-testid="stPopoverBody"] { border-radius: 12px; }
/* page tabs at the top */
.st-key-pagetabs { margin-bottom: .35rem; min-height: 46px; position: relative; z-index: 1000001; }
.st-key-pillrow { margin-top: .2rem; margin-bottom: .3rem; align-items: center; }
.st-key-pagetabs button { padding: 8px 13px !important; font-size: 15px !important; min-height: 42px; }
.st-key-pagetabs [data-testid="stButtonGroup"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 1. Stock lists + company info (sector, industry, market cap)
# ============================================================================
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_screener_meta():
    """All US stocks with name/sector/industry/market cap from Nasdaq's public screener (one request)."""
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks"
                         "?tableonly=true&limit=25000&offset=0&download=true",
                         headers=UA, timeout=40)
        rows = r.json()["data"]["rows"]
        df = pd.DataFrame(rows)
        df["Symbol"] = df["symbol"].astype(str).str.strip().str.replace("/", "-", regex=False)
        df = df[~df["Symbol"].str.contains(r"[\^\s\$]", regex=True) & (df["Symbol"] != "")]

        def num(s):
            return pd.to_numeric(s.astype(str).str.replace(r"[\$,%]", "", regex=True), errors="coerce")

        out = pd.DataFrame({
            "Symbol": df["Symbol"],
            "Name": df["name"].astype(str).str.replace(r"\s+(Common Stock|Class [A-C] Common Stock|"
                                                       r"Ordinary Shares|American Depositary Shares).*$",
                                                       "", regex=True),
            "Sector": df["sector"].replace("", np.nan),
            "Industry": df["industry"].replace("", np.nan),
            "Mkt cap $B": num(df["marketCap"]) / 1e9,
            "Country": df["country"].replace("", np.nan),
            "_last": num(df["lastsale"]),
            "_vol": num(df["volume"]),
        }).drop_duplicates("Symbol").set_index("Symbol")
        out.loc[out["Mkt cap $B"] == 0, "Mkt cap $B"] = np.nan
        return out
    except Exception:
        return pd.DataFrame(columns=["Name", "Sector", "Industry", "Mkt cap $B", "Country", "_last", "_vol"])


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _sp500_wiki_html():
    return requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": UA["User-Agent"]}, timeout=30).text


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_sp500():
    html = _sp500_wiki_html()
    t = pd.read_html(io.StringIO(html))[0]
    t["Symbol"] = t["Symbol"].astype(str).str.replace(".", "-", regex=False)
    meta = pd.DataFrame({"Symbol": t["Symbol"], "Name": t["Security"],
                         "Sector": t["GICS Sector"], "Industry": t["GICS Sub-Industry"]}).set_index("Symbol")
    return t["Symbol"].tolist(), meta


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_sp_changes():
    """Wikipedia's 'Selected changes to the list of S&P 500 components' table:
    one row per ticker added or removed, with the effective date (future dates = announced, not yet done)."""
    tables = pd.read_html(io.StringIO(_sp500_wiki_html()))
    tab = next((t for t in tables if isinstance(t.columns, pd.MultiIndex)
                and {"Added", "Removed"} <= set(t.columns.get_level_values(0))), None)
    if tab is None:
        return pd.DataFrame(columns=["Symbol", "S&P change", "Change date", "Replaced", "Change reason", "Company"])
    clean = lambda x: re.sub(r"\[.*?\]", "", str(x)).strip() if pd.notna(x) else ""
    tick = lambda x: clean(x).replace(".", "-").upper()
    top = tab.columns.get_level_values(0)
    col = lambda a, b=None: next(c for c in tab.columns if c[0] == a and (b is None or c[1] == b))
    dcol = next(c for c in tab.columns if "date" in str(c[0]).lower())
    rcol = next((c for c in tab.columns if "reason" in str(c[0]).lower()), None)
    rows = []
    for _, r in tab.iterrows():
        d = pd.to_datetime(clean(r[dcol]), errors="coerce")
        if pd.isna(d):
            continue
        add_t, rem_t = tick(r[col("Added", "Ticker")]), tick(r[col("Removed", "Ticker")])
        add_n, rem_n = clean(r[col("Added", "Security")]), clean(r[col("Removed", "Security")])
        why = clean(r[rcol]) if rcol is not None else ""
        if add_t and add_t != "NAN":
            rows.append(dict(Symbol=add_t, **{"S&P change": "ADDED", "Change date": d, "Replaced": rem_t,
                                             "Change reason": why, "Company": add_n}))
        if rem_t and rem_t != "NAN":
            rows.append(dict(Symbol=rem_t, **{"S&P change": "REMOVED", "Change date": d, "Replaced": add_t,
                                             "Change reason": why, "Company": rem_n}))
    out = pd.DataFrame(rows)
    if len(out):
        out["Index"] = "S&P 500"
    return out.sort_values("Change date", ascending=False) if len(out) else out


def sp_candidates(meta, members, min_cap_b, n=60):
    """Largest US companies NOT in the S&P 500 above the market-cap bar (size check only)."""
    if meta is None or not len(meta) or "Mkt cap $B" not in meta:
        return pd.DataFrame(columns=["Mkt cap $B"])
    m = meta[(meta["Country"] == "United States") & (meta["Mkt cap $B"] >= min_cap_b)]
    m = m[~m.index.isin(set(members))]
    # skip second share classes of members (e.g. BRK-A when BRK-B is in)
    roots = {t.split("-")[0] for t in members}
    m = m[~m.index.map(lambda t: t.split("-")[0] in roots)]
    return m.sort_values("Mkt cap $B", ascending=False).head(n)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def sp_member_dates():
    """Current S&P 500 members -> the date each was added to the index ('Date added' on Wikipedia)."""
    t = pd.read_html(io.StringIO(_sp500_wiki_html()))[0]
    dcol = next((c for c in t.columns if "date" in str(c).lower() and "add" in str(c).lower()), None)
    sym = t["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()
    if dcol is None:
        return pd.Series(pd.NaT, index=sym, dtype="datetime64[ns]")
    txt = t[dcol].astype(str).str.replace(r"\[.*?\]|\(.*?\)", "", regex=True).str.strip()
    d = txt.map(lambda x: pd.to_datetime(x, errors="coerce"))   # one at a time: the page mixes date formats
    return pd.Series(d.values, index=sym.values).groupby(level=0).first()


# ---- S&P Dow Jones Indices official announcements (press.spglobal.com) ----
SPDJI_BASE = "https://press.spglobal.com"
_TAG = re.compile(r"<[^>]+>")


def _txt(h):
    return re.sub(r"\s+", " ", html_lib.unescape(_TAG.sub(" ", h or ""))).strip()


def _slug_title(slug):
    t = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", slug).replace("S-P", "S&P").replace("-", " ")
    return re.sub(r"\s+", " ", t).strip()


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def spdji_release_list(pages=2):
    """Index-change press releases from S&P Global's newsroom (100 per page, newest first)."""
    found = {}
    for p in range(pages):
        try:
            h = requests.get(f"{SPDJI_BASE}/index.php?s=2429&l=100&o={p * 100}",
                             headers={"User-Agent": UA["User-Agent"]}, timeout=25).text
        except Exception:
            if p == 0:
                return None
            break
        for m in re.finditer(r'<a\b[^>]*href=["\'](?:https?://press\.spglobal\.com)?/(\d{4}-\d{2}-\d{2}-[^"\'#?\s]+)'
                             r'["\'][^>]*>(.*?)</a>', h, re.S | re.I):
            slug, title = m.group(1), _txt(m.group(2))
            if len(title) < len(found.get(slug, "")):
                continue
            found[slug] = title or _slug_title(slug)
    out = []
    for slug, title in found.items():
        # index-change releases: "... Set to Join S&P 500", "... to Join S&P MidCap 400", "... Replace ..."
        if re.search(r"\bjoin\b|replace|S&P (?:500|100)\b|MidCap|SmallCap|S-P-(?:500|100)\b", title + " " + slug, re.I):
            out.append({"url": f"{SPDJI_BASE}/{slug}", "title": title, "announced": pd.Timestamp(slug[:10])})
    return out


SP_INDEX_NAMES = ["S&P 500", "S&P 100", "S&P MidCap 400", "S&P SmallCap 600"]


def _norm_index(x):
    x = re.sub(r"\s+", " ", str(x).replace("\u00a0", " ")).strip()
    if re.search(r"small\s*cap\s*600", x, re.I):
        return "S&P SmallCap 600"
    if re.search(r"mid\s*cap\s*400", x, re.I):
        return "S&P MidCap 400"
    if re.fullmatch(r"S&P\s*100", x, re.I):
        return "S&P 100"
    if re.fullmatch(r"S&P\s*500", x, re.I):
        return "S&P 500"
    return None


def _parse_date(s, year_hint):
    s = re.sub(r"\bSept\b", "Sep", str(s)).replace(".", "").strip()
    d = pd.to_datetime(s, errors="coerce")
    if pd.notna(d) and not re.search(r"\d{4}", s):
        d = d.replace(year=year_hint)
    return d


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def spdji_release_changes(url, announced_iso):
    """S&P 500 additions/deletions in one release. Reads the change table; falls back to the
    older 'X will replace Y in the S&P 500 effective ...' sentences."""
    try:
        h = requests.get(url, headers={"User-Agent": UA["User-Agent"]}, timeout=25).text
    except Exception:
        return None
    ann = pd.Timestamp(announced_iso)
    rows = []
    # 1) tables: find the header row that has Action + Ticker, then read rows under it
    for tbl in re.findall(r"<table\b.*?</table>", h, re.S | re.I):
        cols = None
        for tr in re.findall(r"<tr\b.*?</tr>", tbl, re.S | re.I):
            cells = [_txt(c) for c in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]
            low = [c.lower() for c in cells]
            if any("action" == c or c.startswith("action") for c in low) and any("ticker" in c for c in low):
                cols = {k: next((i for i, c in enumerate(low) if k in c), None)
                        for k in ["effective", "index", "action", "company", "ticker", "sector"]}
                continue
            if not cols or cols["action"] is None or len(cells) <= max(v for v in cols.values() if v is not None):
                continue
            get = lambda k: cells[cols[k]] if cols.get(k) is not None else ""
            idx = _norm_index(get("index"))
            if not idx:
                continue
            act = get("action").lower()
            kind = "ADDED" if act.startswith("add") else "REMOVED" if act.startswith(("del", "remov", "drop")) else None
            if not kind:
                continue
            tick = re.sub(r"[^A-Za-z0-9.\-]", "", get("ticker")).replace(".", "-").upper().strip("-")
            rows.append({"Symbol": tick, "S&P change": kind, "Change date": _parse_date(get("effective"), ann.year),
                         "Company": get("company"), "Sector": get("sector"), "Index": idx})
    # 2) older sentence style
    if not rows:
        text = _txt(h)
        eff = re.search(r"effective prior to the open(?:ing)? of trading on (?:\w+day, )?([A-Z][a-z]+\.? \d{1,2}(?:, \d{4})?)", text)
        eff_d = _parse_date(eff.group(1), ann.year) if eff else pd.NaT
        ex = r"\((?:NYSE|NASD|NASDAQ|Nasdaq|Cboe|NYSE American|NYSE Arca|BATS)[^:)]*:\s*([A-Z][A-Z.\-]*)\)"
        for m in re.finditer(r"([A-Z][\w&.,' \-]{1,80}?)\s*" + ex + r"\s+will replace\s+([A-Z][\w&.,' \-]{1,80}?)\s*" + ex
                             + r"\s+in the (S&P (?:500|100|MidCap 400|SmallCap 600))", text):
            idx = _norm_index(m.group(5))
            rows.append({"Symbol": m.group(2).replace(".", "-"), "S&P change": "ADDED", "Change date": eff_d,
                         "Company": m.group(1).strip(), "Sector": "", "Index": idx})
            rows.append({"Symbol": m.group(4).replace(".", "-"), "S&P change": "REMOVED", "Change date": eff_d,
                         "Company": m.group(3).strip(), "Sector": "", "Index": idx})
    # pair additions with deletions on the same date (the usual 1-for-1 swap)
    for idx in SP_INDEX_NAMES:
        adds = [r for r in rows if r["S&P change"] == "ADDED" and r["Index"] == idx]
        dels = [r for r in rows if r["S&P change"] == "REMOVED" and r["Index"] == idx]
        for a, d in zip(adds, dels):
            if a["Change date"] == d["Change date"]:
                a["Replaced"], d["Replaced"] = d["Symbol"] or d["Company"], a["Symbol"] or a["Company"]
    for r in rows:
        r.setdefault("Replaced", "")
        r["Announced"], r["Source"] = ann, url
        r["Change reason"] = "S&P Dow Jones Indices announcement"
    return rows


def spdji_changes(pages=2):
    """S&P 500 / 100 / MidCap 400 / SmallCap 600 adds & deletes from the newsroom's last ~`pages`×100 releases."""
    lst = spdji_release_list(pages)
    if lst is None:
        return None
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda r: spdji_release_changes(r["url"], r["announced"].date().isoformat()), lst))
    rows = [x for r in res if r for x in r]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Symbol", "S&P change", "Change date", "Company", "Sector", "Replaced", "Announced", "Source",
                 "Change reason", "Index"])


def _merge_changes(base, more):
    """Add rows from `more` that `base` doesn't already have (same stock + action within 10 days)."""
    if more is None or not len(more):
        return base
    if not len(base):
        return more.copy()
    keep = []
    for _, r in more.iterrows():
        same = base[(base["Symbol"] == r["Symbol"]) & (base["S&P change"] == r["S&P change"])
                    & (base["Index"] == r.get("Index", "S&P 500"))]
        if len(same) and ((same["Change date"] - r["Change date"]).abs() <= pd.Timedelta(days=10)).any():
            continue
        keep.append(r)
    return pd.concat([base, pd.DataFrame(keep)], ignore_index=True) if keep else base


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def sp_changes_all():
    """S&P 500 changes, best source first:
    1) S&P Dow Jones Indices' own announcements (with announcement date),
    2) Wikipedia's change table, 3) Wikipedia's member list 'Date added'."""
    official = spdji_changes()
    chg = official if official is not None else pd.DataFrame()
    if len(chg):
        chg["Source name"] = "S&P DJI"
    try:
        wiki = get_sp_changes()
        wiki["Source name"] = "Wikipedia"
        wiki["Source"] = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#Selected_changes_to_the_list_of_S&P_500_components"
    except Exception:
        wiki = pd.DataFrame()
    chg = _merge_changes(chg, wiki)
    if not len(chg):
        chg = pd.DataFrame(columns=["Symbol", "S&P change", "Change date", "Replaced", "Change reason", "Company",
                                    "Announced", "Source", "Source name"])
    try:
        added = sp_member_dates()
    except Exception:
        added = pd.Series(dtype="datetime64[ns]")
    recent = added[added >= pd.Timestamp(date.today()) - pd.Timedelta(days=400)]
    extra = []
    for t, d in recent.items():
        same = chg[(chg["Symbol"] == t) & (chg["S&P change"] == "ADDED") & (chg["Index"] == "S&P 500")] if len(chg) else chg
        if len(same) and ((same["Change date"] - d).abs() <= pd.Timedelta(days=10)).any():
            continue                                    # already in the change table
        extra.append({"Symbol": t, "S&P change": "ADDED", "Change date": d, "Replaced": "",
                      "Change reason": "From the member list (Date added)", "Company": "",
                      "Source name": "Wikipedia", "Source": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                      "Index": "S&P 500"})
    if extra:
        chg = pd.concat([chg, pd.DataFrame(extra)], ignore_index=True)
    for c in ["Announced", "Source", "Source name", "Company", "Index"]:
        if c not in chg:
            chg[c] = np.nan
    chg["Index"] = chg["Index"].fillna("S&P 500")
    # moves between indices on the same day, e.g. joins S&P 500 and leaves S&P MidCap 400
    chg["Index move"] = ""
    for i, r in chg.iterrows():
        if not r["Symbol"] or pd.isna(r["Change date"]):
            continue
        other = chg[(chg["Symbol"] == r["Symbol"]) & (chg["Index"] != r["Index"]) & (chg["S&P change"] != r["S&P change"])
                    & ((chg["Change date"] - r["Change date"]).abs() <= pd.Timedelta(days=3))]
        if len(other):
            o = other.iloc[0]["Index"]
            chg.at[i, "Index move"] = f"from {o}" if r["S&P change"] == "ADDED" else f"to {o}"
    chg["Official"] = official is not None
    return chg.sort_values("Change date", ascending=False) if len(chg) else chg


def _nasdaqtrader(url, sym_col):
    text = requests.get(url, headers={"User-Agent": UA["User-Agent"]}, timeout=30).text
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    df = df[~df[sym_col].str.startswith("File Creation")]
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    syms = df[sym_col].str.strip()
    return syms[(syms != "") & ~syms.str.contains(r"[\$\^\.\s]", regex=True)].tolist()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_nasdaq_listed():
    return _nasdaqtrader("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_all_us():
    other = _nasdaqtrader("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol")
    return sorted(set(get_nasdaq_listed() + other))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_exchange_map():
    """Symbol -> TradingView exchange prefix (NASDAQ / NYSE / AMEX / CBOE)."""
    ex = {}
    try:
        text = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                            headers={"User-Agent": UA["User-Agent"]}, timeout=30).text
        df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
        ex.update({s.strip(): "NASDAQ" for s in df["Symbol"]})
    except Exception:
        pass
    try:
        text = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                            headers={"User-Agent": UA["User-Agent"]}, timeout=30).text
        df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
        code = {"N": "NYSE", "A": "AMEX", "P": "AMEX", "Z": "CBOE"}
        for sym, e in zip(df["ACT Symbol"], df["Exchange"]):
            if e in code:
                ex[sym.strip().replace(".", "-")] = code[e]
    except Exception:
        pass
    return ex


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


def market_of(t):
    """The MARKETS entry for a Yahoo ticker like 0700.HK / PTT.BK / D05.SI (None for US stocks)."""
    for spec in MARKETS.values():
        if str(t).endswith(spec["suffix"]):
            return spec
    return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def market_fx(fx_ticker, fallback):
    """US$ per 1 unit of local currency."""
    try:
        v = rs_prices((fx_ticker,), "1mo")[fx_ticker].dropna().iloc[-1]
        return float(v) if 0.001 < v < 10 else fallback
    except Exception:
        return fallback


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_market_meta(name):
    """Every listed stock in an Asian market from TradingView's public screener: name, sector, industry,
    market cap (US$), price (US$) and volume — indexed by Yahoo ticker."""
    spec = MARKETS[name]
    body = {"markets": [spec["tv"]], "symbols": {"query": {"types": []}, "tickers": []}, "options": {"lang": "en"},
            "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
            "columns": ["name", "description", "close", "volume", "market_cap_basic", "sector", "industry"],
            "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"}, "range": [0, 5000]}
    hdr = {"User-Agent": UA["User-Agent"], "Origin": "https://www.tradingview.com", "Referer": "https://www.tradingview.com/"}
    url = f"https://scanner.tradingview.com/{spec['tv']}/scan"
    # main listings only (skips e.g. Intel's HK line or Tencent's RMB counter) — fall back if the field is refused
    prim = {**body, "filter": body["filter"] + [{"left": "is_primary", "operation": "equal", "right": True}]}
    r = requests.post(url, json=prim, timeout=30, headers=hdr)
    if r.status_code != 200 or not (r.json().get("data") or []):
        r = requests.post(url, json=body, timeout=30, headers=hdr)
    r.raise_for_status()
    rows = r.json().get("data") or []
    fx = market_fx(spec["fx"], spec["fx_fallback"])
    out = []
    num = lambda v: float(v) if isinstance(v, (int, float)) else np.nan
    for x in rows:
        sym = str(x.get("s", "")).split(":")[-1]
        d = x.get("d") or []
        if not sym or len(d) < 7:
            continue
        if num(d[2]) * num(d[3]) < 50_000:                    # (almost) untraded line, e.g. a dual listing
            continue
        if spec["suffix"] == ".HK":
            if not sym.isdigit() or int(sym) > 9999:          # 5-digit codes = RMB counters / structured products
                continue
            yt = f"{int(sym):04d}.HK"
        else:
            if re.search(r"[-./ ]", sym):         # skip NVDRs / foreign boards / warrants (e.g. PTT-R, PTT-F)
                continue
            yt = sym + spec["suffix"]
        out.append(dict(Symbol=yt, Name=str(d[1] or d[0]).title(), _last=num(d[2]) * fx, _vol=num(d[3]),
                        **{"Mkt cap $B": num(d[4]) * fx / 1e9}, Sector=d[5] or np.nan, Industry=d[6] or np.nan,
                        Country=spec["country"]))
    meta = pd.DataFrame(out)
    return meta.drop_duplicates("Symbol").set_index("Symbol") if len(meta) else meta


def tv_symbol(t, ex_map):
    """Yahoo-style ticker -> TradingView symbol, e.g. BRK-B -> NYSE:BRK.B · 0700.HK -> HKEX:700 · PTT.BK -> SET:PTT"""
    spec = market_of(t)
    if spec:
        base = t[: -len(spec["suffix"])]
        return f"{spec['tv_prefix']}:{int(base) if base.isdigit() and spec['suffix'] == '.HK' else base}"
    s = t.replace("-", ".")
    e = ex_map.get(t)
    return f"{e}:{s}" if e else s


def tv_watchlist_text(res, ex_map):
    """TradingView import format: comma-separated symbols, '###Name' starts a section."""
    parts = []
    for status, title in [("TRIGGER", "###TRIGGER - breaking out"), ("SETUP", "###SETUP - basing"),
                          ("", "###Other matches")]:
        syms = [tv_symbol(t, ex_map) for t in res.loc[res["Pattern"] == status, "Symbol"]]
        if syms:
            parts.append(title)
            parts.extend(syms)
    return ",".join(parts)


def parse_tickers(text):
    out = []
    for tok in re.split(r"[\s,;]+", text or ""):
        tok = tok.strip().upper()
        if tok and not tok.startswith("#") and tok not in out:
            out.append(tok)
    return out


# ============================================================================
# 2. Price download
# ============================================================================
def _fetch(chunk, threads, period="2y"):
    try:
        return yf.download(chunk, period=period, interval="1d", group_by="ticker",
                           auto_adjust=True, threads=threads, progress=False)
    except Exception:
        return None


def _split(data, chunk):
    got, missing = {}, []
    for t in chunk:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(how="all")
            if df.empty or df["Close"].notna().sum() == 0:
                raise KeyError
            got[t] = df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            missing.append(t)
    return got, missing


def download_all(tickers, bar, msg, period="2y"):
    prices, failed = {}, []
    n = len(tickers)
    for i in range(0, n, 150):
        chunk = tickers[i:i + 150]
        msg.write(f"Downloading prices {i + 1:,}–{i + len(chunk):,} of {n:,} …")
        data = _fetch(chunk, threads=True, period=period)
        if data is None:
            failed += chunk
        else:
            got, missing = _split(data, chunk)
            prices.update(got)
            failed += missing
        bar.progress(min(1.0, (i + len(chunk)) / n))
    if failed and len(failed) <= 400:
        msg.write(f"Retrying {len(failed)} tickers that failed …")
        for i in range(0, len(failed), 50):
            msg.write(f"Retrying tickers that failed … {i + 1:,}–{min(i + 50, len(failed)):,} of {len(failed):,}")
            data = _fetch(failed[i:i + 50], threads=False, period=period)
            if data is not None:
                got, _ = _split(data, failed[i:i + 50])
                prices.update(got)
    return prices


def build_panels(prices, rows=330):
    """Wide tables (dates x tickers) for each price field — lets us compute every stock at once."""
    panels = {}
    for f in ["Open", "High", "Low", "Close", "Volume"]:
        panels[f] = pd.concat({t: df[f] for t, df in prices.items()}, axis=1).sort_index()
    idx = panels["Close"].index[-rows:]
    for f in panels:
        panels[f] = panels[f].loc[idx]
    for f in ["Open", "High", "Low", "Close"]:
        panels[f] = panels[f].ffill(limit=5)
    panels["Volume"] = panels["Volume"].fillna(0)
    # drop stocks that stopped trading (no price in the last 5 days)
    alive = panels["Close"].tail(5).notna().any()
    for f in panels:
        panels[f] = panels[f].loc[:, alive]
    return panels


def volume_history(prices, cutoff, full):
    """Per stock: highest volume BEFORE the panel window, and whether we hold its whole history.
    History is complete if full history was downloaded, or if the stock started trading after
    the download window began (i.e. it IPO'd within the last ~2 years)."""
    firsts = {t: df["Close"].first_valid_index() for t, df in prices.items()}
    starts = [d for d in firsts.values() if d is not None]
    dl_start = min(starts) if starts else None
    rows = {}
    for t, df in prices.items():
        old = df.loc[df.index < cutoff, "Volume"].dropna()
        first = firsts[t]
        young = first is not None and dl_start is not None and first > dl_start + pd.Timedelta(days=10)
        rows[t] = {"old_max": float(old.max()) if len(old) else 0.0,
                   "complete": bool(full or young), "first": first}
    return pd.DataFrame.from_dict(rows, orient="index")


# ============================================================================
# 3. Indicators for every stock (vectorized)
# ============================================================================
def _ago(X, n):
    return X.iloc[-1 - n] if len(X) > n else pd.Series(np.nan, index=X.columns)


def _bars_since_high(H, end_offset, window=20):
    """Days since the highest high in the `window` bars ending `end_offset` bars ago."""
    arr = H.to_numpy(dtype=float)
    stop = len(arr) - end_offset
    w = arr[stop - window:stop][::-1]
    w = np.where(np.isnan(w), -np.inf, w)
    return pd.Series(np.argmax(w, axis=0).astype(float), index=H.columns)


VOL_WINDOWS = [("3M", 63), ("6M", 126), ("1Y", 252)]
VOL_REC_NAME = {"IPO": "Since IPO", "1Y": "1 year", "6M": "6 months", "3M": "3 months"}
VOL_REC_RANK = {"Since IPO": 4, "1 year": 3, "6 months": 2, "3 months": 1}


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


XB_20 = "20-day SMA"
XB_EMAS = ["10 EMA or 20 SMA", "10-day EMA", XB_20]


def ema_crossback(O, H, L, C, V, rs, pp):
    """EMA crossback: a leading stock pulls back to its rising 10-day EMA / 20-day SMA, shows strength, then breaks out.
    SETUP = at the EMA and holding · TRIGGER = broke above its recent consolidation high today."""
    n = len(C)
    e10, e20 = C.ewm(span=10, adjust=False).mean(), C.rolling(20).mean()     # e20 = the 20-day SMA
    s50 = C.rolling(50).mean()
    tol = pp["xb_tol"] / 100
    T = int(pp["xb_days"])
    with np.errstate(all="ignore"):
        # 1) leading stock: sharp rally off the lows, above the 50-day, strong RS, beating the market
        win_hi = int(pp["xb_rally_days"])
        Ha, La = H.to_numpy(float), L.to_numpy(float)
        cols = np.arange(C.shape[1])
        hi_w = np.where(np.isnan(Ha[-win_hi:]), -np.inf, Ha[-win_hi:])
        hi_i = n - win_hi + np.argmax(hi_w, axis=0)
        lo_w = np.where(np.isnan(La[-(win_hi + 40):]), np.inf, La[-(win_hi + 40):])
        lo_i = n - (win_hi + 40) + np.argmin(lo_w, axis=0)
        rally = (Ha[hi_i, cols] / La[lo_i, cols] - 1) * 100
        rally = np.where(lo_i < hi_i, rally, 0.0)                      # the low must come before the high
        c = C.iloc[-1]
        above50 = c > s50.iloc[-1]
        bt = pp.get("xb_bench", "SPY")                              # SPY, or the local index for Asian markets
        bench = rs_prices((bt,), "5y")
        spy = bench[bt].dropna() if bt in bench else pd.Series(dtype=float)
        spy = spy[spy.index <= C.index[-1]]
        spy_1m = (spy.iloc[-1] / spy.iloc[-22] - 1) * 100 if len(spy) > 22 else 0.0
        vs_spy = (c / C.iloc[-22] - 1) * 100 - spy_1m if n > 22 else pd.Series(np.nan, index=C.columns)
        lead = (rally >= pp["xb_rally"]) & above50 & (rs >= pp["xb_rs"])
        if pp["xb_beat"]:
            lead &= vs_spy > 0
        # 2) pullback to the rising 10 EMA / 20 SMA — the low tags the line, the close holds it
        t10 = (L <= e10 * (1 + tol)) & (C >= e10 * (1 - tol))
        t20 = (L <= e20 * (1 + tol)) & (C >= e20 * (1 - tol))
        which = pp["xb_ema"]
        touch = t10 if which == "10-day EMA" else t20 if which == XB_20 else (t10 | t20)
        touched_now = touch.iloc[-T:].any()                           # within the last T days (incl. today)
        touched_before = touch.iloc[-(T + 1):-1].any()                # before today (for a breakout today)
        rising = (e10.iloc[-1] > e10.iloc[-6]) & (e20.iloc[-1] > e20.iloc[-6])
        hi20 = H.iloc[-20:].max()
        pull = (hi20 - c) / hi20 * 100
        pb_ok = pull <= pp["xb_pull_max"]
        # 3) signs of strength
        pc = C.shift(1)
        tr = np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy())
        atr = pd.DataFrame(tr, index=C.index, columns=C.columns).rolling(20).mean().iloc[-1]
        rng3 = (H - L).iloc[-3:].mean()
        tight_ratio = rng3 / atr
        up = (C > pc).iloc[-10:]
        vv = V.iloc[-10:]
        upv, dnv = vv.where(up).mean(), vv.where(~up).mean()
        vol_ratio = upv / dnv
        o, h, l, cl = O.iloc[-1], H.iloc[-1], L.iloc[-1], C.iloc[-1]
        body = (cl - o).abs()
        hammer = ((np.minimum(o, cl) - l) >= 2 * body) & ((h - np.maximum(o, cl)) <= body.clip(lower=(h - l) * 0.1)) \
            & ((cl - l) >= 0.66 * (h - l)) & ((h - l) > 0)
        po, pcl = O.iloc[-2], C.iloc[-2]
        engulf = (cl > o) & (pcl < po) & (cl >= po) & (o <= pcl)
        candle = np.where(engulf, "Engulfing", np.where(hammer, "Hammer", ""))
        strong = pd.Series(True, index=C.columns)
        if pp["xb_tight"]:
            strong &= tight_ratio <= pp["xb_tight_max"]
        if pp["xb_vol"]:
            strong &= vol_ratio >= 1.0
        if pp["xb_candle"]:
            strong &= pd.Series(candle != "", index=C.columns)
        # 4) stop = recent swing low · 5) entry = break above the recent consolidation high
        stop = L.iloc[-6:].min()
        range_hi = H.iloc[-6:-1].max()
        risk = (c - stop) / c * 100
        breakout = (c > range_hi) & (c > C.iloc[-2])
    base = lead & rising & pb_ok & (C.iloc[-1] > s50.iloc[-1])
    trigger = base & touched_before & breakout
    if pp["xb_vol"]:                      # buyers confirmed: more volume on up days, and a real volume push today
        trigger &= (vol_ratio >= 1.0) & (V.iloc[-1] >= V.iloc[-21:-1].mean())
    setup = base & touched_now & strong & ~breakout & (c >= np.minimum(e10.iloc[-1], e20.iloc[-1]) * (1 - tol))
    out = pd.DataFrame(index=C.columns)
    out["Crossback"] = np.where(trigger, "TRIGGER", np.where(setup, "SETUP", ""))
    has = out["Crossback"] != ""
    last10, last20 = t10.iloc[-T:].any(), t20.iloc[-T:].any()
    out["EMA tagged"] = np.where(has, np.where(last10 & last20, "10 & 20", np.where(last10, "10", np.where(last20, "20", ""))), "")
    out["vs EMA10 %"] = (c / e10.iloc[-1] - 1) * 100
    out["Rally %"] = rally
    out["Pullback from high %"] = pull
    out["vs SPY 1M pts"] = vs_spy
    out["Tight (3d ÷ ATR)"] = tight_ratio
    out["Up/Down vol"] = vol_ratio
    out["Candle"] = candle
    out["Stop"] = stop
    out["Risk %"] = risk
    out["Entry above"] = range_hi
    return out


def _xb_path(rally, n_up, use20, tol, T, tight, tight_max, vol, candle, seed=7):
    """Made-up daily bars: quiet base -> rally -> pullback that tags the EMA -> tight bars -> breakout."""
    rnd = np.random.default_rng(seed)
    B = dict(O=[], H=[], L=[], C=[], V=[], E10=[], E20=[])
    st_ = {"e10": None, "e20": None}

    def add(o, h, l, c, v):
        for k, x in zip("OHLCV", (o, h, l, c, v)):
            B[k].append(x)
        st_["e10"] = c if st_["e10"] is None else st_["e10"] + (c - st_["e10"]) * 2 / 11
        last20 = B["C"][-20:]
        st_["e20"] = sum(last20) / len(last20)                   # the 20-day SMA
        B["E10"].append(st_["e10"]); B["E20"].append(st_["e20"])

    px = 100.0
    for i in range(24):                                   # long enough for the EMAs to settle
        c = px * (1 + rnd.normal(0, 0.006)); add(px, max(px, c) * 1.008, min(px, c) * 0.992, c, 1.0); px = c
    lo_i = len(B["C"]) - 1
    lo_px = B["L"][lo_i] = min(B["L"][lo_i], px * 0.988)
    step = (lo_px * (1 + max(rally, 5) / 100 * 1.03) / px) ** (1 / n_up)
    for i in range(n_up):
        c = px * step * (1 + rnd.normal(0, 0.005))
        add(px, max(px, c) * 1.006, min(px, c) * 0.994, c, (1.5 if vol else 1.1) * (1 + rnd.normal(0, .12)))
        px = c
    atr = float(np.mean(np.array(B["H"][-10:]) - np.array(B["L"][-10:])))
    ema = lambda: B["E20"][-1] if use20 else B["E10"][-1]
    for _ in range(40):                                   # pull back until the low is at the EMA
        if px - atr * 0.5 <= ema() * (1 + tol / 100):
            break
        c = px - max(atr * 0.6, (px - ema()) * 0.3)
        add(px, px + atr * 0.15, c - atr * 0.25, c, 0.65 if vol else 1.0)
        px = c
    tag_low = ema() * (1 - tol / 100 * 0.3)
    c = max(ema() * 1.004, tag_low + atr * (0.8 if candle else 0.5))
    add(c - atr * (0.05 if candle else 0.2), c + atr * 0.1, tag_low, c, 0.75)
    tag_i = len(B["C"]) - 1
    rb = atr * (tight_max if tight else 1.3) * 0.8
    for _ in range(max(0, int(T) - 1)):
        c = max(B["C"][-1], ema() * 1.012) * (1 + rnd.normal(0, 0.0015))
        o = B["C"][-1]
        add(o, max(o, c) + rb * .3, min(o, c) - rb * .45, c, 0.65 if vol else 1.0)
    entry, stop = max(B["H"][-5:]), min(B["L"][-6:])
    c = entry + atr * 0.8
    add(B["C"][-1], c + atr * 0.12, B["C"][-1] - atr * 0.1, c, 2.0 if vol else 1.3)
    hi_i = int(np.argmax(B["H"][:tag_i + 1]))
    return B, lo_i, hi_i, tag_i, len(B["C"]) - 1, entry, stop, atr


def xb_sketch(rally, rally_days, ema, tol, T, pull_max, tight, tight_max, vol, candle):
    """Minimal line sketch of an EMA-crossback TRIGGER, drawn with the current settings."""
    use20 = ema == XB_20
    T = int(T)
    R = _xb_path(rally, int(np.clip(round(rally_days * 0.6), 6, 70)), use20, tol, T, tight, tight_max, vol, candle)
    rb, rhi, rtag, rtrig = R[0], R[2], R[3], R[4]
    pull_pct = (rb["H"][rhi] - min(rb["L"][rtag:rtrig])) / rb["H"][rhi] * 100
    B, lo_i, hi_i, tag_i, trig_i, entry, stop, atr = _xb_path(
        float(np.clip(rally, 12, 40)), int(np.clip(round(rally_days * 0.3), 6, 16)), use20, tol, T, tight, tight_max,
        vol, candle)
    C, L, H = B["C"], B["L"], B["H"]
    E = B["E20"] if use20 else B["E10"]
    n = len(C)
    start = max(0, lo_i - 4)
    W, Hh, top, pb, right = 300, 130, 16, 112, 262
    split = 8 + (right - 8) * 0.38

    def xs(i):
        if i <= hi_i:
            return 8 + (i - start) / max(1, hi_i - start) * (split - 8)
        return split + (i - hi_i) / max(1, n - 1 - hi_i) * (right - split)
    # price line: closes, but the tag day uses its low so the touch shows
    P = list(C)
    P[tag_i] = L[tag_i]
    lo_y, hi_y = min(min(P[start:]), stop, min(E[start:])) - atr, max(max(P[start:]), entry) + atr * 1.8
    ys = lambda v: top + (hi_y - v) / (hi_y - lo_y) * (pb - top)
    g, r, blue, ec = "#26a69a", "#ef5350", "#3b82f6", ("#a855f7" if use20 else "#f59e0b")
    path = lambda arr: " ".join(f"{xs(i):.1f},{ys(arr[i]):.1f}" for i in range(start, n))
    o = [f'<svg viewBox="0 0 {W} {Hh}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         'style="font:9px sans-serif;fill:currentColor;display:block">',
         f'<polyline points="{path(E)}" fill="none" stroke="{ec}" stroke-width="1.4" opacity=".9"/>',
         f'<polyline points="{path(P)}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>']
    xe = xs(trig_i - 5)
    for y, colr, lbl in [(entry, g, "entry"), (stop, r, "stop")]:
        o.append(f'<line x1="{xe:.1f}" x2="{right + 4}" y1="{ys(y):.1f}" y2="{ys(y):.1f}" stroke="{colr}" '
                 f'stroke-width="1" stroke-dasharray="3 2"/>')
    ye, yst = ys(entry) + 3, ys(stop) + 3
    if yst - ye < 10:
        ye, yst = (ye + yst) / 2 - 5, (ye + yst) / 2 + 5
    o.append(f'<text x="{W - 2}" y="{ye:.1f}" text-anchor="end" fill="{g}">entry</text>'
             f'<text x="{W - 2}" y="{yst:.1f}" text-anchor="end" fill="{r}">stop</text>')
    o.append(f'<text x="{right + 6}" y="{ys(E[-1]) + 3:.1f}" fill="{ec}">{"SMA20" if use20 else "EMA10"}</text>'
             if abs(ys(E[-1]) - ys(stop)) > 9 and abs(ys(E[-1]) - ys(entry)) > 9 else "")
    # marks
    o.append(f'<circle cx="{xs(tag_i):.1f}" cy="{ys(P[tag_i]):.1f}" r="3.2" fill="{blue}"/>')
    o.append(f'<text x="{xs(tag_i):.1f}" y="{ys(P[tag_i]) + 13:.1f}" text-anchor="middle" fill="{blue}">tag ±{tol:g}%</text>')
    o.append(f'<circle cx="{xs(trig_i):.1f}" cy="{ys(C[trig_i]):.1f}" r="3.2" fill="{g}"/>')
    o.append(f'<text x="{min(xs(trig_i), right - 16):.1f}" y="{ys(C[trig_i]) - 7:.1f}" text-anchor="middle" fill="{g}" '
             f'style="font-weight:700">TRIGGER</text>')
    o.append(f'<text x="{xs(hi_i):.1f}" y="{ys(H[hi_i]) - 6:.1f}" text-anchor="middle" opacity=".7">+{rally:g}% in ≤{rally_days:g}d</text>')
    o.append(f'<text x="{xs(lo_i):.1f}" y="{ys(P[lo_i]) + 12:.1f}" text-anchor="middle" opacity=".55">low</text>')
    # the "tagged within N days" span
    bx1, bx2 = xs(trig_i - T), xs(trig_i - 1)
    o.append(f'<path d="M{bx1:.1f},{pb + 4} H{max(bx2, bx1 + 2):.1f}" stroke="currentColor" opacity=".4"/>'
             f'<text x="{(bx1 + bx2) / 2:.1f}" y="{pb + 14}" text-anchor="middle" opacity=".6">≤ {T}d</text>')
    o.append("</svg>")
    note = ""
    if pull_pct > pull_max:
        note = (f"⚠️ After a +{rally:g}% rally, a dip to the {'20' if use20 else '10'}-day EMA is usually about "
                f"{pull_pct:.0f}% — more than your {pull_max:g}% max pullback, so few stocks will pass.")
    return "".join(o), note


def _sk_path(knots, amp=0.005):
    """Straight segments between (x, price) knots with a small zig-zag so it reads like a price line."""
    out = []
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        for x in range(int(x0), int(x1)):
            y = y0 + (y1 - y0) * (x - x0) / max(1, x1 - x0)
            out.append((x, y * (1 + (amp if x % 2 else -amp) * (0 if x == x0 else 1))))
    out.append(knots[-1])
    return out


def _sk_ema(path, span):
    e, out = None, []
    for x, y in path:
        e = y if e is None else e + (y - e) * 2 / (span + 1)
        out.append((x, e))
    return out


def _sk_render(price, x_from=0, ema=None, hl=(), dots=(), texts=(), band=None, brackets=(), vols=None, hot_col=None):
    """Tiny minimal line chart (SVG). Everything is given in (bar, price) units."""
    W, top, pb, right = 300, 16, 106, 262
    vol_h = 16 if vols else 0
    br_h = 14 if brackets else 0
    Hh = pb + br_h + vol_h + 6
    pts = [p for p in price if p[0] >= x_from]
    ys_all = [y for _, y in pts] + [h[0] for h in hl] + ([y for x, y in ema[0] if x >= x_from] if ema else [])
    if band:
        ys_all += [band[2], band[3]]
    lo, hi = min(ys_all), max(ys_all)
    pad = (hi - lo) * 0.12 or 1
    lo, hi = lo - pad * 0.6, hi + pad * 1.4
    x_to = max(x for x, _ in pts)
    xs = lambda x: 8 + (x - x_from) / max(1, x_to - x_from) * (right - 8)
    ys = lambda v: top + (hi - v) / (hi - lo) * (pb - top)
    poly = lambda arr: " ".join(f"{xs(x):.1f},{ys(y):.1f}" for x, y in arr if x >= x_from)
    g, r = "#22c55e", "#ef5350"
    o = [f'<svg viewBox="0 0 {W} {Hh}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         'style="font:9px sans-serif;fill:currentColor;display:block">']
    if band:
        x0, x1, y0, y1, colr, lbl = band
        o.append(f'<rect x="{xs(x0):.1f}" y="{ys(max(y0, y1)):.1f}" width="{xs(x1) - xs(x0):.1f}" '
                 f'height="{abs(ys(y0) - ys(y1)):.1f}" fill="{colr}" opacity=".16"/>')
        if lbl:
            o.append(f'<text x="{xs(x0) + 2:.1f}" y="{ys(min(y0, y1)) + 9:.1f}" fill="{colr}" opacity=".9">{lbl}</text>')
    if ema:
        pts_e, colr, lbl = ema
        o.append(f'<polyline points="{poly(pts_e)}" fill="none" stroke="{colr}" stroke-width="1.3" opacity=".9"/>')
        if lbl:
            o.append(f'<text x="{right + 6}" y="{ys(pts_e[-1][1]) + 3:.1f}" fill="{colr}">{lbl}</text>')
    labels_r = []
    for y, x0, x1, colr, dash, lbl in hl:
        o.append(f'<line x1="{xs(x0):.1f}" x2="{right + 4 if x1 is None else xs(x1):.1f}" y1="{ys(y):.1f}" '
                 f'y2="{ys(y):.1f}" stroke="{colr}" stroke-width="1.1" stroke-dasharray="{dash}"/>')
        if lbl:
            labels_r.append([ys(y) + 3, colr, lbl])
    labels_r.sort()
    for i in range(1, len(labels_r)):                      # keep right-hand labels from overlapping
        labels_r[i][0] = max(labels_r[i][0], labels_r[i - 1][0] + 10)
    for yv, colr, lbl in labels_r:
        o.append(f'<text x="{W - 2}" y="{yv:.1f}" text-anchor="end" fill="{colr}">{lbl}</text>')
    o.append(f'<polyline points="{poly(price)}" fill="none" stroke="currentColor" stroke-width="1.6" '
             f'stroke-linejoin="round"/>')
    for x, y, colr, lbl, dy, bold in dots:
        o.append(f'<circle cx="{xs(x):.1f}" cy="{ys(y):.1f}" r="3.2" fill="{colr}"/>')
        if lbl:
            fw = "700" if bold else "400"
            o.append(f'<text x="{min(max(xs(x), 20), right - 14):.1f}" y="{ys(y) + dy:.1f}" text-anchor="middle" '
                     f'fill="{colr}" style="font-weight:{fw}">{lbl}</text>')
    for x, y, txt, anchor, colr, op in texts:
        o.append(f'<text x="{xs(x):.1f}" y="{ys(y):.1f}" text-anchor="{anchor}" fill="{colr}" opacity="{op}">{txt}</text>')
    by = pb + 5
    for x0, x1, lbl in brackets:
        o.append(f'<path d="M{xs(x0):.1f},{by - 2} V{by} H{xs(x1):.1f} V{by - 2}" fill="none" stroke="currentColor" '
                 f'opacity=".45"/><text x="{(xs(x0) + xs(x1)) / 2:.1f}" y="{by + 9}" text-anchor="middle" '
                 f'opacity=".65">{lbl}</text>')
    if vols:
        vb = Hh - 3
        vmax = max(v for _, v, _ in vols)
        bw = max(1.2, (right - 8) / max(1, x_to - x_from) * 0.6)
        for x, v, hot in vols:
            if x >= x_from:
                h = v / vmax * vol_h
                o.append(f'<rect x="{xs(x) - bw / 2:.1f}" y="{vb - h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                         f'fill="{(hot_col or g) if hot else "currentColor"}" opacity="{.9 if hot else .25}"/>')
    o.append("</svg>")
    return "".join(o)


def tight_flag_sketch(max_range, pull_range, min_bars, max_ema_dist, max_ema_gap, dry_up, breakout_vol):
    """Minimal line sketch of a Tight-flag TRIGGER with the current settings."""
    pull = float(np.clip((pull_range[0] + pull_range[1]) / 2, 1.5, 30))
    days = int(max(min_bars, 4)) + 1
    hi_x = 26
    flag_lo = 100 * (1 - pull / 100)
    flag_top = flag_lo * (1 + min(max_range, pull) * 0.55 / 100)
    knots = [(0, 72), (8, 73), (hi_x, 100), (hi_x + 2, flag_lo)]
    x = hi_x + 2
    for k in range(days - 2):
        x += 1
        knots.append((x, flag_top if k % 2 == 0 else flag_lo * 1.006))
    path = _sk_path(knots, 0.004)
    trig = max(y for xx, y in path if xx >= x - 6)
    brk = x + 1
    path.append((brk, trig * 1.025))
    path.append((brk + 1, trig * 1.035))
    stop = min(y for xx, y in path if brk - 7 <= xx < brk) * 0.995
    e21 = [(x_, sum(y for _, y in path[max(0, i - 19):i + 1]) / len(path[max(0, i - 19):i + 1]))
           for i, (x_, _) in enumerate(path)]                              # 20-day SMA
    vols = [(xx, (dry_up if hi_x < xx < brk else 1.0), False) for xx, _ in path[:-2]] + \
           [(brk, breakout_vol, True), (brk + 1, 1.1, False)]
    rng7 = (trig - min(y for xx, y in path if brk - 7 <= xx < brk)) / trig * 100
    svg = _sk_render(
        path, x_from=15, ema=(e21, "#a855f7", ""),
        hl=[(trig, brk - 7, None, "#22c55e", "3 2", "trigger"), (stop, brk - 7, None, "#ef5350", "3 2", "stop")],
        dots=[(hi_x, 100, "currentColor", "20-day high", -7, False),
              (brk, trig * 1.025, "#22c55e", "TRIGGER", -8, True)],
        texts=[(hi_x - 1, flag_lo * 0.985, f"pullback {pull_range[0]:g}–{pull_range[1]:g}%", "end",
                "currentColor", .7), (15, e21[15][1] * 0.975, "SMA20", "start", "#a855f7", .9)],
        brackets=[(hi_x, brk - 1, f"≥ {min_bars}d since high · range ≤ {max_range:g}%")],
        vols=vols)
    notes = [f"Flag hugs the 20-day SMA (≤ {max_ema_dist:g}% above) · volume dries up (≤ {dry_up:g}× avg), "
             f"then the breakout comes on ≥ {breakout_vol:g}× volume."]
    if pull_range[0] > max_range and min_bars < 7:
        notes.append(f"⚠️ A pullback of at least {pull_range[0]:g}% can't fit in a 7-day range of {max_range:g}% "
                     f"when the high is less than 7 days back — few stocks will pass.")
    return svg, " ".join(notes)


def green_line_sketch(months, vol_x, near, retest_days, tol, stop_pct):
    """Minimal line sketch of a Green line BREAKOUT (and RETEST) with the current settings."""
    months = f"{months[0]:g}–{months[1]:g}" if isinstance(months, (tuple, list)) else f"≥ {months:g}"
    line = 100.0
    knots = [(0, 68), (12, line), (20, 80), (28, 91), (35, 82), (45, 96.5), (50, 97.2), (53, 104),
             (57, line * (1 + tol / 200)), (62, 110)]
    path = _sk_path(knots, 0.006)
    stop = line * (1 - stop_pct / 100)
    vols = [(x, 1.0, False) for x, _ in path]
    vols[53] = (53, vol_x, True)
    svg = _sk_render(
        path, x_from=0,
        hl=[(line, 12, None, "#22c55e", "none", "green line"), (stop, 50, None, "#ef5350", "3 2", "stop")],
        band=(38, 53, line * (1 - near / 100), line, "#3b82f6", ""),
        dots=[(12, line, "#22c55e", "all-time high", -7, False),
              (53, 104, "#22c55e", "BREAKOUT", -8, True),
              (57, line * (1 + tol / 200), "#3b82f6", "", 0, False)],
        texts=[(37, line * (1 - near / 100) * 1.004, f"NEAR ≤ {near:g}%", "end", "#3b82f6", .95),
               (59, line * 0.935, f"RETEST ±{tol:g}%", "middle", "#3b82f6", .95)],
        brackets=[(12, 52, f"line set {months} months ago"), (53, 58, f"≤ {retest_days:g}d")],
        vols=vols)
    note = (f"BREAKOUT = first close above the line on ≥ {vol_x:g}× average volume · RETEST = back to the line "
            f"within {retest_days:g} days and holding · stop {stop_pct:g}% under the line.")
    return svg, note


def shakeout_sketch(rule, pct, base_days, recent, under, mid, depth, trend, vol_x, stop_pct, entry="", fast=0,
                    shvol=0.0):
    """Minimal line sketch of a Shakeout +3 TRIGGER with the current settings."""
    if isinstance(under, (tuple, list)):
        under_lbl = f"{under[0]:g}–{under[1]:g}"
        under = (under[0] + under[1]) / 2
    else:
        under_lbl = f"{under:g}"
    first = 100.0
    shake = first * (1 - under / 100)
    level = first + (6 if first > 60 else 3) if rule.startswith("Livermore") else first * (1 + pct / 100)
    mid_top = first * (1 + mid / 100)
    base_hi = max(shake / (1 - depth * 0.85 / 100), mid_top * 1.03, level * 1.03)
    true_depth = (base_hi - shake) / base_hi * 100
    start = base_hi / (1 + max(trend, 5) / 100)
    knots = [(0, start), (14, base_hi), (24, first), (30, mid_top), (38, shake), (41, first * 1.01),
             (45, level * 1.012), (47, level * 1.03)]
    path = _sk_path(knots, 0.005)
    stop = shake * (1 - stop_pct / 100)
    lvl_lbl = "+3 = +$6" if rule.startswith("Livermore") else f"+3 = +{pct:g}%"
    reclaim = entry.startswith("Reclaim")
    tx = 41 if reclaim else 45
    vols = [(x, 1.0, False) for x, _ in path]
    vols[tx] = (tx, max(vol_x, 1.0) * 1.3, True)
    if shvol > 0:
        vols[38] = (38, max(shvol, 1.0) * 1.3, False)
    hl = [(first, 24, None, "#3b82f6", "3 2", "first low" + (" = buy" if reclaim else "")),
          (stop, 34, None, "#ef5350", "3 2", "stop")]
    if not reclaim:
        hl.insert(1, (level, 24, None, "#22c55e", "3 2", lvl_lbl))
    svg = _sk_render(
        path, x_from=0,
        hl=hl,
        dots=[(24, first, "#3b82f6", "", 0, False), (38, shake, "#ef5350", "", 0, False),
              (tx, first * 1.01 if reclaim else level * 1.012, "#22c55e", "TRIGGER", -8, True)],
        texts=[(30, mid_top * 1.012, f"+{mid:g}%", "middle", "currentColor", .7),
               (36.5, shake * 0.975, f"shakeout −{under_lbl}%", "end", "#ef5350", .95),
               (14, base_hi * 1.015, f"base high (+{trend:g}% uptrend)", "middle", "currentColor", .7)],
        brackets=[(14, 38, f"base ~{base_days:g}d · depth ≤ {depth:g}%")] + ([(38, tx, f"≤ {fast:g}d")] if fast > 0 else []),
        vols=vols)
    note = (f"W-bottom: a first low, a bounce, then a shakeout {under_lbl}% under the first low"
            f"{f' on ≥ {shvol:g}× volume' if shvol > 0 else ''} that quickly reclaims it. TRIGGER = "
            + ("the first close back above the first low" if reclaim else "close above the +3 level")
            + f"{f' on ≥ {vol_x:g}× volume' if vol_x > 1 else ''}"
            + (f", within {fast:g} days of the shakeout low" if fast > 0 else "")
            + f" · stop {stop_pct:g}% under the shakeout low.")
    if true_depth > depth + 0.5:
        note += (f" ⚠️ With a +{mid:g}% bounce and a {under:.0f}% undercut the base is about {true_depth:.0f}% deep — "
                 f"deeper than your {depth:g}% max, so few stocks will pass.")
    return svg, note


def htf_sketch(pole, depth, fmin, fmax, vol_x, dry):
    """Minimal line sketch of a High Tight Flag: a steep pole, a shallow quiet flag, a breakout."""
    base, top = 50.0, 50.0 * (1 + pole / 100)
    dep = (depth[0] + depth[1]) / 2
    lo = top * (1 - dep / 100)
    fl = int(np.clip((fmin + fmax) / 2, 6, 25))
    knots = [(0, base * 0.97), (8, base), (38, top)]
    x, i = 38, 0
    while x < 38 + fl:                                     # a few gentle swings inside the flag
        x = min(38 + fl, x + 3)
        knots.append((x, lo if i == 1 else top * (1 - dep / 100 * (0.3 if i % 2 == 0 else 0.7))))
        i += 1
    knots += [(x + 2, top * 1.03), (x + 3, top * 1.06)]
    path = _sk_path(knots, 0.004)
    vols = [(xx, 1.6 if 8 < xx <= 38 else (min(dry, 1.0) * 0.8 if 38 < xx <= x else 1.0), False) for xx, _ in path]
    vols[x + 2] = (x + 2, max(vol_x, 1.0) * 1.4, True)
    svg = _sk_render(path, x_from=0,
                     hl=[(top, 38, None, "#22c55e", "3 2", "flag high"), (lo, 38, None, "#ef5350", "3 2", "flag low / stop")],
                     dots=[(x + 2, top * 1.03, "#22c55e", "TRIGGER", -8, True)],
                     texts=[(22, (base + top) / 2, f"pole +{pole:g}%", "end", "currentColor", .8),
                            (38 + fl / 2, lo * 0.92, f"−{depth[0]:g}–{depth[1]:g}%", "middle", "#ef5350", .95)],
                     brackets=[(8, 38, "≤ 8 weeks"), (38, x, f"{fmin}–{fmax}d flag")], vols=vols)
    note = (f"A stock up ≥ {pole:g}% in about 8 weeks rests in a flag {depth[0]:g}–{depth[1]:g}% under its high for "
            f"{fmin}–{fmax} days on drying volume · TRIGGER = close above the flag high"
            f"{f' on ≥ {vol_x:g}× volume' if vol_x > 0 else ''} · stop = the flag low.")
    return svg, note


def reclaim_sketch(trend, dip_max, min_dip, max_days, hl, stop_hl=True, prior=True):
    """Minimal line sketch of a 50-day reclaim: under the SMA, a pop above it that fails, a shallow dip whose
    pullback makes a higher low, then a close back above."""
    rising = trend.startswith("Pullback")
    falling = trend.startswith("Turnaround")
    sma = (lambda x: 100 + 0.12 * (x - 40)) if rising else (lambda x: 100 - 0.12 * (x - 40)) if falling else \
        (lambda x: 100.0)
    dd = min(max(dip_max, 2.0), 12.0) / 100
    lo1, lo2 = sma(35) * (1 - dd * 0.75), sma(39) * (1 - dd * 0.42)
    lo0 = min(sma(18) * 0.915, lo1 * 0.965)                 # the low before the pop — under the dip low
    knots = [(0, sma(0) * 0.9), (10, sma(10) * 0.95), (18, lo0), (26, sma(26) * 0.97),
             (30, sma(30) * 1.012), (32, sma(32) * 1.005), (35, lo1), (37, sma(37) * (1 - dd * 0.15)),
             (39, lo2) if hl else (39, sma(39) * (1 - dd * 0.3)), (42, sma(42) * 0.99), (43, sma(43) * 1.012),
             (46, sma(46) * 1.04), (48, sma(48) * 1.055)]
    path = _sk_path(knots, 0.002)
    line = [(x, sma(x)) for x, _ in path]
    stop_y = (lo2 if hl and stop_hl else lo1) * 0.992
    vols = [(x, 1.0, False) for x, _ in path]
    vols[43] = (43, 1.4, True)
    dots = [(30, sma(30) * 1.012, "#f59e0b", "pop", -8, False), (35, lo1, "#94a3b8", "dip low", 14, False),
            (43, sma(43) * 1.012, "#22c55e", "TRIGGER", -9, True)]
    if hl:
        dots.append((39, lo2, "#3b82f6", "higher low", 14, False))
    lines = [(stop_y, 38 if hl and stop_hl else 34, None, "#ef5350", "3 2", "stop")]
    if prior:
        dots.append((18, lo0, "#a855f7", "low before pop", 14, False))
        lines.append((lo0, 18, None, "#a855f7", "2 3", ""))
    svg = _sk_render(path, x_from=10, ema=(line, "#cddc39", ""),
                     hl=lines,
                     dots=dots,
                     texts=[(11, sma(11) * 1.02, "SMA50" + (" rising" if rising else " falling" if falling else ""),
                             "start", "#cddc39", .95)],
                     brackets=[(30, 43, f"≤ {max_days}d · dip ≤ {dip_max:g}% under")], vols=vols)
    note = (f"Under the 50-day SMA, a pop closes above it and fails; the dip stays within {dip_max:g}% of the line"
            + (" and above the low before the pop" if prior else "")
            + (", bounces and pulls back to a higher low above its lowest low" if hl else "")
            + f" (at least {min_dip} closes under) · TRIGGER = the first close back above, within {max_days} days "
            f"of the pop · stop = under the {'higher low' if hl and stop_hl else 'dip low'}.")
    return svg, note


def bear_flag_sketch(ma, tol, bounce, dry, days):
    """Minimal line sketch of a bear flag: a downtrend, a light-volume bounce into the falling SMA, then a break."""
    use50 = ma == "50-day SMA"
    span = 50 if use50 else 20
    x0 = 55 if use50 else 25                                 # start drawing once the SMA exists
    lo_x, x_t = x0 + 45, x0 + 53                             # the low, then the bounce tags the SMA
    start, low = 150.0, 100.0
    sma_of = lambda path, i: sum(y for _, y in path[max(0, i - span + 1):i + 1]) / len(path[max(0, i - span + 1):i + 1])
    top = low * 1.08
    for _ in range(4):                                       # the bounce top sits right at the SMA (within tol)
        knots = [(0, start + (start - low) * x0 / 45), (x0, start), (lo_x, low), (x_t, top)]
        path0 = _sk_path(knots, 0.0)
        top = sma_of(path0, x_t) * (1 - tol / 200)
    knots = [(0, start + (start - low) * x0 / 45), (x0, start), (lo_x, low), (x_t, top),
             (x_t + 2, top * 0.975), (x_t + 3, top * 0.955), (x_t + 6, top * 0.91), (x_t + 10, top * 0.86),
             (x_t + 16, top * 0.8)]
    path = _sk_path(knots, 0.003)
    line = [(x_, sma_of(path, i)) for i, (x_, _) in enumerate(path)]
    sell = top * 0.962
    vols = [(xx, 1.2 if xx <= lo_x else (min(dry, 1.0) * 0.6 if xx <= x_t else 1.0), False) for xx, _ in path]
    vols[x_t + 4] = (x_t + 4, 1.6, True)
    x0 = lo_x - 22
    svg = _sk_render(path, x_from=x0, ema=(line, "#a855f7", ""),
                     hl=[(top * 1.012, x_t - 1, None, "#ef5350", "3 2", "stop = bounce high"),
                         (sell, x_t, None, "#f97316", "3 2", "sell below")],
                     dots=[(x_t, top, "#a855f7", "", -8, False),
                           (x_t + 4, top * 0.935, "#ef5350", "TRIGGER", 13, True)],
                     texts=[(x0 + 2, line[x0 + 2][1] * 1.015, f"falling {'SMA50' if use50 else 'SMA20'}", "start",
                             "#a855f7", .9),
                            (x_t - 1, top * 1.035, "tags the SMA", "end", "#a855f7", .9)],
                     brackets=[(lo_x, x_t, f"bounce ≥ {bounce:g}%, light volume")], vols=vols,
                     hot_col="#ef5350")
    return svg


def failed_breakout_sketch(n, days, vol_x):
    """Minimal line sketch of a failed breakout: a base, a close above its high, then a close back under it."""
    knots = [(0, 92.0), (6, 100.0), (12, 94.0), (18, 99.5), (24, 95.0), (30, 99.0), (33, 103.5), (34, 104.5),
             (35, 102.0), (36, 98.0), (38, 95.0), (42, 90.0), (48, 84.0)]
    path = _sk_path(knots, 0.003)
    vols = [(xx, 0.9, False) for xx, _ in path]
    vols[33] = (33, 1.4, False)
    vols[36] = (36, max(vol_x, 1.0) * 1.5, True)
    svg = _sk_render(path, x_from=0,
                     hl=[(100.0, 6, None, "#f97316", "3 2", f"{n}-day high = sell below"),
                         (104.8, 33, None, "#ef5350", "3 2", "stop (failed high)")],
                     dots=[(33, 103.5, "#22c55e", "", -7, False),
                           (36, 98.0, "#ef5350", "TRIGGER", 12, True)],
                     texts=[(31.5, 104.2, "breakout", "end", "#22c55e", .95)],
                     brackets=[(0, 32, f"base ({n}-day high)"), (33, 33 + days, f"≤ {days}d")], vols=vols,
                     hot_col="#ef5350")
    return svg


def mac_sketch(ma, days, tol, rng, near, vol_x, hold):
    """Minimal line sketch: an uptrend, then a tight sideways box resting on the rising 20/50-day SMA."""
    use50 = ma == "50-day SMA"
    span = 50 if use50 else 20
    d = int(np.clip(days, 6, 30))
    run = 70 if use50 else 45
    box_lo = 100.0 * (1 - min(rng, 20) / 100 * 0.75)
    knots = [(0, 55.0 if use50 else 75.0), (run, 100.0)]
    x, k = run, 0
    while x < run + d:                                   # a few gentle swings inside the box
        x = min(run + d, x + 3)
        knots.append((x, box_lo * 1.01 if k % 2 == 0 else 99.3))
        k += 1
    knots += [(x + 2, 103.0), (x + 3, 104.2)]
    path = _sk_path(knots, 0.002)
    line = [(x_, sum(y for _, y in path[max(0, i - span + 1):i + 1]) / len(path[max(0, i - span + 1):i + 1]))
            for i, (x_, _) in enumerate(path)]
    vols = [(xx, 0.7 if run < xx <= x else 1.0, False) for xx, _ in path]
    vols[x + 2] = (x + 2, max(vol_x, 1.0) * 1.3, True)
    x0 = run - 25
    svg = _sk_render(path, x_from=x0, ema=(line, "#a855f7", ""),
                     hl=[(100.0, run, None, "#22c55e", "3 2", "box high"), (box_lo, run, None, "#ef5350", "3 2", "stop")],
                     dots=[(x + 2, 103.0, "#22c55e", "TRIGGER", -8, True)],
                     texts=[(x0 + 1, line[x0 + 1][1] * 0.975, "SMA50" if use50 else "SMA20", "start", "#a855f7", .9)],
                     brackets=[(run, x, f"{d}d box · range ≤ {rng:g}%")], vols=vols)
    note = (f"Every {'low' if hold == 'Every low' else 'close'} of the last {days:g} days held above the rising "
            f"{ma if ma != MAC_LINES[0] else '20- or 50-day SMA'} (up to {tol:g}% under it allowed), range ≤ {rng:g}%, "
            f"close ≤ {near:g}% above the line · TRIGGER = close above the box high"
            f"{f' on ≥ {vol_x:g}× volume' if vol_x > 0 else ''} · stop = the box low.")
    return svg, note


def vcp_sketch(swing, min_c, first_max, last_max, dry, vol_x, near, base):
    """Minimal line sketch of a VCP with the current settings: shrinking pullbacks, then a breakout."""
    nc = max(2, min(int(min_c) + 1, 4))
    d = [first_max * 0.8]
    while len(d) < nc:
        d.append(d[-1] * 0.5)
    d[-1] = min(d[-1], last_max * 0.8)
    knots, x, top = [(0, 70.0), (12, 100.0)], 12, 100.0
    highs = []
    for i, dep in enumerate(d):
        lo = top * (1 - dep / 100)
        x += 9 - i
        knots.append((x, lo))
        nxt = top * (1 - dep / 100 * 0.25) if i < len(d) - 1 else top * (1 - d[-1] / 100 * 0.15)
        x += 8 - i
        knots.append((x, nxt))
        highs.append((x - (8 - i), top, dep))
        top = nxt
    pivot = top
    knots += [(x + 3, pivot * 1.03), (x + 5, pivot * 1.05)]
    path = _sk_path(knots, 0.004)
    vols = [(xx, 1.0 - 0.6 * min(1, max(0, xx - 12) / max(1, x - 12)), False) for xx, _ in path]
    vols[x + 3] = (x + 3, max(vol_x, 1.0) * 1.2, True)
    texts = [(hx + 3, top_ * (1 - dep / 100) * 0.975, f"−{dep:.0f}%", "middle", "#ef5350", .95)
             for hx, top_, dep in highs]
    svg = _sk_render(path, x_from=0,
                     hl=[(pivot, x - 6, None, "#22c55e", "3 2", "pivot"),
                         (pivot * (1 - d[-1] / 100), x - 6, None, "#ef5350", "3 2", "stop")],
                     dots=[(x + 3, pivot * 1.03, "#22c55e", "TRIGGER", -8, True)],
                     texts=texts + [(12, 101.5, "base high", "middle", "currentColor", .7)],
                     brackets=[(12, x + 3, f"base ≤ {base:g}d · {len(d)} contractions, each smaller")], vols=vols)
    note = (f"Each pullback smaller than the last (first ≤ {first_max:g}%, last ≤ {last_max:g}%), swings of "
            f"≥ {swing:g}% · volume dries up (≤ {dry:g}× avg) · TRIGGER = close above the pivot on ≥ {vol_x:g}× "
            f"volume · SETUP = within {near:g}% under it · stop = the last contraction's low.")
    return svg, note


def compute_metrics(panels, pp, vh=None):
    O, H, L, C, V = (panels[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    last = lambda X: X.iloc[-1]
    prev = lambda X: X.iloc[-2]
    with np.errstate(all="ignore"):
        e10 = C.ewm(span=10, adjust=False).mean()
        e21 = C.ewm(span=21, adjust=False).mean()
        s20 = C.rolling(20).mean()
        s50 = C.rolling(50).mean()
        s150 = C.rolling(150).mean()
        s200 = C.rolling(200).mean()
        vol50 = V.rolling(50).mean()
        vol5 = V.rolling(5).mean()
        hi20 = H.rolling(20).max()
        hi252 = H.rolling(252, min_periods=200).max()
        lo252 = L.rolling(252, min_periods=200).min()

        d = C.diff()
        g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        lo = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + g / lo)
        pc = C.shift(1)
        tr = pd.DataFrame(np.fmax.reduce([(H - L).to_numpy(), (H - pc).abs().to_numpy(),
                                          (L - pc).abs().to_numpy()]), index=C.index, columns=C.columns)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

        c = last(C)
        m = pd.DataFrame(index=C.columns)
        m["Price"] = c
        m["Chg %"] = (c / prev(C) - 1) * 100
        m["Gap %"] = (last(O) / prev(C) - 1) * 100
        m["Volume"] = last(V)
        m["Avg vol 50d"] = last(vol50)
        prior50 = V.iloc[-51:-1].mean()
        m["Rel vol"] = last(V) / prior50
        m["$ Vol M"] = c * last(vol50) / 1e6
        m["Avg $ Vol 50d M"] = last((C * V).rolling(50, min_periods=20).mean()) / 1e6
        m["ADR % 14d"] = last(((H / L - 1) * 100).rolling(14, min_periods=10).mean())
        # sessions in a row the stock closed at/below its 50-day SMA just before today — only for stocks that closed
        # ABOVE it today ("first close back above the 50-day in N days")
        _s50 = C.rolling(50).mean()
        _ok = _s50.notna().to_numpy()
        _below = (_ok & ~(C > _s50).to_numpy())[:-1][::-1]
        _run = np.where(_below.all(axis=0), _below.shape[0], _below.argmin(axis=0)) if len(_below) else np.zeros(C.shape[1])
        m["Under 50d days"] = np.where((C > _s50).to_numpy()[-1] & _ok[-1], _run, np.nan)
        m["Chg 5d %"] = (c / _ago(C, 5) - 1) * 100
        m["Chg 20d %"] = (c / _ago(C, 20) - 1) * 100
        for lbl, n in [("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]:
            m[f"Perf {lbl} %"] = (c / _ago(C, n) - 1) * 100
        prev_year = C[C.index.year < C.index[-1].year]
        m["Perf YTD %"] = (c / prev_year.iloc[-1] - 1) * 100 if len(prev_year) else np.nan
        # IBD-style relative strength, ranked 1-99 against the scanned stocks
        r = lambda n: c / _ago(C, n) - 1
        rs_raw = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)
        rs_raw = rs_raw.fillna(0.4 * r(63) + 0.6 * r(126))
        m["RS"] = (rs_raw.rank(pct=True) * 98 + 1).round()
        m["RSI 14"] = last(rsi)
        m["ATR %"] = last(atr) / c * 100
        m["vs EMA21 %"] = (c / last(e21) - 1) * 100
        m["vs SMA20 %"] = (c / last(s20) - 1) * 100
        m["vs SMA50 %"] = (c / last(s50) - 1) * 100
        m["vs SMA200 %"] = (c / last(s200) - 1) * 100
        m["Below 52W high %"] = (1 - c / last(hi252)) * 100
        m["Off 52W high %"] = -m["Below 52W high %"]
        m["Above 52W low %"] = (c / last(lo252) - 1) * 100
        m["New 52W high"] = last(H) >= last(hi252)
        m["Stack 50>200"] = (c > last(s50)) & (last(s50) > last(s200))
        m["Trend template"] = ((c > last(s150)) & (c > last(s200)) & (last(s150) > last(s200)) &
                               (last(s200) > _ago(s200, 21)) & (last(s50) > last(s150)) &
                               (c > last(s50)) & (c >= 1.3 * last(lo252)) & (c >= 0.75 * last(hi252)))
        m["EMA stack"] = (c > last(e21)) & (last(e10) > last(e21)) & (last(e21) > last(s50))
        m["7d range %"] = (H.iloc[-7:].max() - L.iloc[-7:].min()) / c * 100
        m["10d range %"] = (H.iloc[-10:].max() - L.iloc[-10:].min()) / c * 100
        m["Pullback %"] = (last(hi20) - c) / last(hi20) * 100
        m["Days since high"] = _bars_since_high(H, 0)
        m["EMA gap %"] = (last(e10) - last(s20)).abs() / last(s20) * 100      # 10 EMA vs 20 SMA
        m["Vol ratio"] = last(vol5) / last(vol50)

        # ---- tight flag pattern: SETUP today / yesterday, TRIGGER today ----
        def setup_at(k):  # k = 0 today, 1 yesterday
            ck = C.iloc[-1 - k]
            s50k, s200k = s50.iloc[-1 - k], s200.iloc[-1 - k]
            e21k = s20.iloc[-1 - k]                       # the flag's line: the 20-day SMA
            pull = (hi20.iloc[-1 - k] - ck) / hi20.iloc[-1 - k] * 100
            bars = _bars_since_high(H, k)
            end = len(C) - k
            rng7 = (H.iloc[end - 7:end].max() - L.iloc[end - 7:end].min()) / ck * 100
            dist = (ck - e21k) / e21k * 100
            gap = (e10.iloc[-1 - k] - e21k).abs() / e21k * 100
            return ((ck > s50k) & (s50k > s200k) & (s50k > s50.iloc[-11 - k]) &
                    (bars >= pp["min_bars"]) & (bars <= 19) &
                    (pull >= pp["min_pull"]) & (pull <= pp["max_pull"]) &
                    (rng7 < pp["max_range"]) & (ck > e21k) & (dist < pp["max_ema_dist"]) &
                    (gap < pp["max_ema_gap"]) & (vol5.iloc[-1 - k] < vol50.iloc[-1 - k] * pp["dry_up"]))

        setup_today, setup_yday = setup_at(0), setup_at(1)
        trigger = setup_yday & (c > H.iloc[-8:-1].max()) & (last(V) > last(vol50) * pp["breakout_vol"])
        m["Pattern"] = np.where(trigger, "TRIGGER", np.where(setup_today, "SETUP", ""))
    m.loc[C.notna().sum() < 60, "Pattern"] = ""
    m.index.name = "Symbol"
    return m


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


SO_RECLAIM = "Reclaim of the first low (earlier)"
SO_ENTRIES = ["+3 level (the original)", SO_RECLAIM]


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


MAC_LINES = ["20 or 50-day SMA", "20-day SMA", "50-day SMA"]


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



HTF_PRESETS = {
    "Strict (O'Neil / Stockbee)": dict(htf_pole=100.0, htf_depth=(10.0, 25.0)),
    "Bulkowski": dict(htf_pole=90.0, htf_depth=(10.0, 34.0)),
    "Loose (half-size HTF)": dict(htf_pole=50.0, htf_depth=(3.0, 15.0)),
}


def _win_view(A, w):
    """(days, stocks, w) window of the last w values ending at each day, newest first (NaN-padded at the start)."""
    pad = np.full((w - 1, A.shape[1]), np.nan)
    Ap = np.vstack([pad, A])
    return np.lib.stride_tricks.sliding_window_view(Ap, w, axis=0)[..., ::-1]


def htf_frames(P, pp):
    """High Tight Flag for every day at once (days x stocks), each day using only data up to that day.
    Pole: the flag's top close is >= pole% above the lowest close of the `pole_days` days before it, and the gain
    isn't mostly 1–2 giant days. Flag: the high was set flag_min..flag_max days ago, price is 10–25% (settable)
    under it, recent ranges are tighter than the pole's, volume is drying up, few heavy-volume red days, above the
    20-day SMA (10 over 20 optional). Liquidity: price and 50-day $ volume minimums.
    TRIGGER = yesterday was a valid flag, today closes above its high on volume (x avg, and above yesterday's),
    closing in the top part of the day's range. Stop = the flag's low."""
    H, L, C, V = (P[k].to_numpy(float) for k in ["High", "Low", "Close", "Volume"])
    O = P["Open"].to_numpy(float)
    idx, cols = P["Close"].index, P["Close"].columns
    n, k = C.shape
    fmin, fmax, pd_ = int(pp["htf_flag_min"]), int(pp["htf_flag_max"]), int(pp["htf_pole_days"])
    dmin, dmax = pp["htf_depth"]
    with np.errstate(all="ignore"):
        Cs = pd.DataFrame(C)
        avg50 = pd.DataFrame(V).rolling(50, min_periods=30).mean().shift(1).to_numpy()   # before today
        dvol = (pd.DataFrame(C * V).rolling(50, min_periods=30).mean() / 1e6).to_numpy()
        s10, s20 = Cs.rolling(10).mean().to_numpy(), Cs.rolling(20).mean().to_numpy()
        # the flag's high and how long ago it was set
        Hn = np.where(np.isnan(H), -np.inf, H)
        f = np.full((n, k), -1)
        W = fmax + 1
        for c0 in range(0, k, 250):                           # chunks keep memory small
            sl = slice(c0, c0 + 250)
            wv = _win_view(Hn[:, sl], W)
            f[:, sl] = np.argmax(np.where(np.isnan(wv), -np.inf, wv), axis=2)
        rows = np.arange(n)[:, None]
        h = rows - f                                          # index of the flag high
        ok_h = (h >= pd_ + 1) & np.isfinite(Hn[np.clip(h, 0, n - 1), np.arange(k)])
        hc = np.clip(h, 0, n - 1)
        g = lambda A: np.take_along_axis(A, hc, axis=0)
        pivot = g(H)
        # pole: highest close around the top vs the lowest close of the pole_days before the high
        top_c = pd.DataFrame(C).rolling(W, min_periods=1).max().to_numpy()
        base = g(pd.DataFrame(C).rolling(pd_ + 1, min_periods=pd_ // 2).min().to_numpy())
        pole = (top_c / base - 1) * 100
        # pole quality: share of the (log) gain made by the 2 biggest up days in the pole window
        lr = np.log(C / np.roll(C, 1, axis=0)); lr[0] = np.nan
        top2 = np.full((n, k), np.nan)
        for c0 in range(0, k, 150):
            sl = slice(c0, c0 + 150)
            wv = _win_view(np.nan_to_num(lr[:, sl], nan=-1.0), pd_)
            part = -np.partition(-wv, 1, axis=2)[..., :2]
            top2[:, sl] = np.clip(part, 0, None).sum(axis=2)
        top2 = g(top2)
        share = top2 / np.log(top_c / base) * 100
        # flag low since the high, and depth
        Ln = np.where(np.isnan(L), np.inf, L)
        flag_lo = np.full((n, k), np.nan)
        for c0 in range(0, k, 250):
            sl = slice(c0, c0 + 250)
            wv = _win_view(Ln[:, sl], W)
            mask = np.arange(W)[None, None, :] <= f[:, sl][..., None]
            flag_lo[:, sl] = np.min(np.where(mask & ~np.isnan(wv), wv, np.inf), axis=2)
        depth = (pivot - flag_lo) / pivot * 100
        # tightness: last 5 days' average range vs the pole's average range
        rng = (H - L) / C
        cs = np.nancumsum(np.nan_to_num(rng), axis=0)
        r5 = (cs - np.vstack([np.zeros((5, k)), cs[:-5]])) / 5
        pole_rng = (g(cs) - np.take_along_axis(cs, np.clip(hc - pd_, 0, n - 1), axis=0)) / pd_
        tight = r5 / pole_rng
        # volume: dry-up and heavy red days inside the flag
        dry = pd.DataFrame(V).rolling(10).mean().to_numpy() / avg50
        red = ((C < np.roll(C, 1, axis=0)) & (V > 1.5 * avg50)).astype(float)
        red[0] = 0
        csr = np.cumsum(red, axis=0)
        red_n = csr - g(csr)
        # volume signature: a highest-volume-in-a-year day inside the pole
        hv1 = (V >= pd.DataFrame(V).rolling(252, min_periods=120).max().to_numpy()).astype(float)
        hv_in = g(pd.DataFrame(hv1).rolling(pd_ + 1, min_periods=1).max().to_numpy()) > 0
        valid = (ok_h & (f >= fmin) & (pole >= pp["htf_pole"]) & (depth >= dmin) & (depth <= dmax)
                 & (C <= pivot) & (C >= s20 * (1 - 0.01)) & (tight <= pp["htf_tight"])
                 & (C >= pp["htf_minpx"]) & (dvol >= pp["htf_mindvol"]))
        if pp["htf_share"] < 100:
            valid &= share <= pp["htf_share"]
        if pp["htf_dry"] > 0:
            valid &= dry <= pp["htf_dry"]
        if pp["htf_nored"]:
            valid &= red_n <= 1
        if pp["htf_ma"]:
            valid &= s10 > s20
        if pp["htf_hv"]:
            valid &= hv_in
        valid &= np.arange(n)[:, None] >= 60
        vy = np.vstack([np.zeros((1, k), bool), valid[:-1]])
        py = np.vstack([np.full((1, k), np.nan), pivot[:-1]])
        sy = np.vstack([np.full((1, k), np.nan), flag_lo[:-1]])
        volx = V / avg50
        close_pos = (C - L) / (H - L)
        trig = vy & (C > py)
        if pp["htf_vol"] > 0:
            trig &= (volx >= pp["htf_vol"]) & (V > np.roll(V, 1, axis=0))
        if pp["htf_close_top"] > 0:
            trig &= close_pos >= 1 - pp["htf_close_top"] / 100
        setup = valid & ~trig
    D = lambda A: pd.DataFrame(A, index=idx, columns=cols)
    return dict(valid=D(valid), trig=D(trig), setup=D(setup), pivot=D(pivot), stop=D(flag_lo), trig_pivot=D(py),
                trig_stop=D(sy), pole=D(pole), depth=D(depth), flag_days=D(f.astype(float)), share=D(share),
                tight=D(tight), dry=D(dry), red=D(red_n), hv=D(hv_in), volx=D(volx), close_pos=D(close_pos))


def htf_scan(P, pp):
    """The High Tight Flag scan on the latest day (one row per stock)."""
    C = P["Close"]
    out = pd.DataFrame(index=C.columns)
    out["HTF"] = ""
    if len(C) < 80:
        return out
    F_ = htf_frames(P, pp)
    last = lambda k: F_[k].iloc[-1]
    prev = lambda k: F_[k].iloc[-2]
    t = last("trig").astype(bool)
    s_ = last("setup").astype(bool)
    pick = lambda k: np.where(t, prev(k), last(k))
    out["HTF"] = np.where(t, "TRIGGER", np.where(s_, "SETUP", ""))
    out["Pole %"] = pick("pole")
    out["Flag depth %"] = pick("depth")
    out["Flag days"] = pick("flag_days")
    out["Top-2 days %"] = pick("share")
    out["Flag high"] = np.where(t, last("trig_pivot"), last("pivot"))
    out["Flag low"] = np.where(t, last("trig_stop"), last("stop"))
    out["To flag high %"] = (out["Flag high"] - C.iloc[-1]) / out["Flag high"] * 100
    out["HTF risk %"] = (out["Flag high"] - out["Flag low"]) / out["Flag high"] * 100
    out["Tightness"] = pick("tight")
    out["HTF dry-up"] = pick("dry")
    out["HV1 in pole"] = np.where(pick("hv").astype(bool), "✓", "")
    out["HTF vol ×"] = last("volx")
    return out


# ---------------------------------------------------------------------------------------------------------------
# 🐻 Swing shorts: bear flag into a falling 20/50-day SMA · failed breakout (bull trap)
# ---------------------------------------------------------------------------------------------------------------
SH_SETUPS = ["Both", "🐻 Bear flag", "🪤 Failed breakout"]
SH_MAS = ["20- or 50-day SMA", "20-day SMA", "50-day SMA"]


def _since(flag):
    """Days since `flag` was last True (0 = today), NaN before the first time. flag: (days x stocks) bool array."""
    n, k = flag.shape
    rows = np.where(flag, np.arange(n)[:, None], np.nan)
    last = pd.DataFrame(rows).ffill().to_numpy()
    return np.arange(n)[:, None] - last


def _ffill_at(flag, A):
    """Value of A on the day `flag` was last True, carried forward."""
    return pd.DataFrame(np.where(flag, A, np.nan)).ffill().to_numpy()


def short_frames(P, pp):
    """Both swing-short setups for every day at once (days x stocks), each day using only data up to that day.

    🐻 Bear flag — a stock in a downtrend (50-day SMA under the 200-day and falling, close under the 50) bounces
    ≥ x% off its 15-day low on light volume and tags a falling 20/50-day SMA (high within tol % of it) but closes
    below it. That touch day arms the setup for `bf_days` days: SELL-STOP at the lowest low since the touch, STOP at
    the high of the bounce. TRIGGER = a close below the armed level. Optional: RS rating ≤ x (a laggard).

    🪤 Failed breakout — a close above the prior N-day high that had stood for ≥ 5 days (a real breakout from a base),
    then within `fb_days` days a close back BELOW that old high (the pivot), optionally on ≥ x× volume and under the
    200-day SMA. Armed while still above the pivot: SELL-STOP at the pivot, STOP at the highest high since the
    breakout. Only the first failure of each breakout counts."""
    H, L, C, V = (P[k].to_numpy(float) for k in ["High", "Low", "Close", "Volume"])
    idx, cols = P["Close"].index, P["Close"].columns
    n, k = C.shape
    D = lambda A: pd.DataFrame(A, index=idx, columns=cols)
    roll = lambda A, w, f: getattr(pd.DataFrame(A).rolling(w, min_periods=w), f)().to_numpy()
    shift1 = lambda A, fill=np.nan: np.vstack([np.full((1, k), fill), A[:-1]])
    with np.errstate(all="ignore"):
        s20, s50 = roll(C, 20, "mean"), roll(C, 50, "mean")
        s200 = pd.DataFrame(C).rolling(200, min_periods=150).mean().to_numpy()
        avg50 = shift1(roll(V, 50, "mean"))                          # average volume BEFORE today
        dvol = roll(C * V, 50, "mean") / 1e6
        liq = (C >= float(pp["sh_minpx"])) & (dvol >= float(pp["sh_mindvol"])) & (np.arange(n)[:, None] >= 60)

        # ---------------- 🐻 bear flag ----------------
        tol = float(pp["sh_bf_tol"]) / 100
        t20 = (H >= s20 * (1 - tol)) & (C < s20)
        t50 = (H >= s50 * (1 - tol)) & (C < s50)
        ma = pp["sh_bf_ma"]
        touch = t20 if ma == "20-day SMA" else t50 if ma == "50-day SMA" else (t20 | t50)
        down = (s50 < s200) & (s50 < np.vstack([np.full((10, k), np.nan), s50[:-10]])) & (C < s50)
        lo15 = roll(L, 15, "min")
        bounce = (H / lo15 - 1) * 100
        dry = roll(V, 5, "mean") / avg50
        bf_touch = touch & down & (bounce >= float(pp["sh_bf_bounce"])) & liq
        if float(pp["sh_bf_dry"]) > 0:
            bf_touch &= dry <= float(pp["sh_bf_dry"])
        rs = None
        if float(pp.get("sh_bf_rs", 0)) > 0:
            rs = rs_matrix(P["Close"]).to_numpy()
            bf_touch &= rs <= float(pp["sh_bf_rs"])
        K = int(pp["sh_bf_days"])
        ds = _since(bf_touch)                                       # days since the last touch
        bf_lvl = np.full((n, k), np.nan)                            # sell-stop for TOMORROW (lowest low since touch)
        bf_stp = np.full((n, k), np.nan)                            # stop (high of the bounce: touch day - 2 .. today)
        for j in range(K):
            m = ds == j
            bf_lvl = np.where(m, roll(L, j + 1, "min"), bf_lvl)
            bf_stp = np.where(m, roll(H, j + 3, "max"), bf_stp)
        armed0 = (ds <= K - 1) & (C < bf_stp) & (C < s50)
        lvl_y, stp_y, arm_y = shift1(bf_lvl), shift1(bf_stp), shift1(armed0, False).astype(bool)
        t0 = arm_y & (C < lvl_y)                                    # closed below yesterday's sell level
        t0p = shift1(t0.astype(float), 0.0)                         # … on the day before
        ds_y = shift1(ds)
        # only the first trigger after each touch: none on the days between that touch and today
        prior = np.zeros((n, k), bool)
        spent = np.zeros((n, k), bool)
        for j in range(1, K + 1):
            prior |= (ds_y == j) & (roll(t0p, j, "max") > 0)
            spent |= (ds == j) & (roll(t0.astype(float), j, "max") > 0)   # triggered since today's touch
        bf_trig = t0 & ~prior
        bf_setup = armed0 & ~spent

        # ---------------- 🪤 failed breakout ----------------
        N, FD = int(pp["sh_fb_len"]), int(pp["sh_fb_days"])
        Hn = np.where(np.isnan(H), -np.inf, H)
        age = np.full((n, k), -1)
        for c0 in range(0, k, 250):
            sl = slice(c0, c0 + 250)
            wv = _win_view(shift1(Hn, -np.inf)[:, sl], N)            # the N days BEFORE today, newest first
            age[:, sl] = np.argmax(np.where(np.isnan(wv), -np.inf, wv), axis=2) + 1
        piv = roll(shift1(H), N, "max")                             # prior N-day high (excl. today)
        qbo = (C > piv) & (age >= int(pp["sh_fb_base"])) & (shift1(C) <= shift1(piv)) & liq
        dsb = _since(qbo)
        pivot = _ffill_at(qbo, piv)
        peak = np.full((n, k), np.nan)
        for j in range(FD + 1):
            peak = np.where(dsb == j, roll(H, j + 1, "max"), peak)
        f0 = (dsb >= 1) & (dsb <= FD) & (C < pivot)
        fprior = np.zeros((n, k), bool)
        f0f = f0.astype(float)
        for j in range(2, FD + 1):
            fprior |= (dsb == j) & (roll(shift1(f0f, 0.0), j - 1, "max") > 0)
        volx = V / avg50
        fb_trig = f0 & ~fprior & liq
        if float(pp["sh_fb_vol"]) > 0:
            fb_trig &= volx >= float(pp["sh_fb_vol"])
        if pp["sh_fb_200"]:
            fb_trig &= C < s200
        fb_trig_stop = peak                                          # highest high from the breakout to today
        fb_setup = (dsb >= 0) & (dsb <= FD - 1) & (C >= pivot) & ~(f0 | fprior) & liq
        if pp["sh_fb_200"]:
            fb_setup &= C < s200
    return dict(bf_trig=D(bf_trig), bf_setup=D(bf_setup), bf_level=D(bf_lvl), bf_stop=D(bf_stp),
                bf_trig_level=D(lvl_y), bf_trig_stop=D(stp_y), bf_bounce=D(bounce), bf_dry=D(dry),
                bf_days=D(ds), fb_trig=D(fb_trig), fb_setup=D(fb_setup), fb_pivot=D(pivot), fb_stop=D(peak),
                fb_trig_stop=D(fb_trig_stop), fb_days=D(dsb), volx=D(volx),
                s20=D(s20), s50=D(s50), s200=D(s200), rs=D(rs) if rs is not None else None)


def short_scan(P, pp):
    """The Swing-short scan on the latest day (one row per stock)."""
    C = P["Close"]
    out = pd.DataFrame(index=C.columns)
    out["Short"] = ""
    if len(C) < 80:
        return out
    F_ = short_frames(P, pp)
    last = lambda k: F_[k].iloc[-1].to_numpy()
    which = pp["sh_which"]
    use_bf, use_fb = which != SH_SETUPS[2], which != SH_SETUPS[1]
    bt, bs = last("bf_trig").astype(bool) & use_bf, last("bf_setup").astype(bool) & use_bf
    ft, fs = last("fb_trig").astype(bool) & use_fb, last("fb_setup").astype(bool) & use_fb
    # one row per stock: a trigger beats a setup; bear flag first when both
    is_bf = bt | (bs & ~ft)
    is_fb = ~is_bf & (ft | fs)
    trig = (is_bf & bt) | (is_fb & ft)
    out["Short"] = np.where(trig, "TRIGGER", np.where(is_bf | is_fb, "SETUP", ""))
    out["Short setup"] = np.where(is_bf, SH_SETUPS[1], np.where(is_fb, SH_SETUPS[2], ""))
    sell = np.where(is_bf, np.where(bt, last("bf_trig_level"), last("bf_level")), last("fb_pivot"))
    stop = np.where(is_bf, np.where(bt, last("bf_trig_stop"), last("bf_stop")),
                    np.where(ft, last("fb_trig_stop"), last("fb_stop")))
    px = C.iloc[-1].to_numpy()
    out["Sell below"] = np.where(is_bf | is_fb, sell, np.nan)
    out["Short stop"] = np.where(is_bf | is_fb, stop, np.nan)
    ref = np.where(trig, px, sell)
    out["Short risk %"] = (out["Short stop"] - ref) / ref * 100
    out["To sell level %"] = np.where(trig, np.nan, (px - sell) / px * 100)
    out["Bounce %"] = np.where(is_bf, last("bf_bounce"), np.nan)
    out["Bounce vol ×"] = np.where(is_bf, last("bf_dry"), np.nan)
    out["Breakout days ago"] = np.where(is_fb, last("fb_days"), np.nan)
    out["Short vol ×"] = last("volx")
    return out


# ---------------------------------------------------------------------------------------------------------------
# 🔂 50-day reclaim: a pop above the 50-day SMA fails, a shallow dip makes a higher low, then a close back above
# ---------------------------------------------------------------------------------------------------------------
RC_TRENDS = ["Any", "Turnaround (50-day SMA falling)", "Pullback in an uptrend (50-day SMA rising)"]


RC_STOPS = ["Under the higher low", "Under the dip's lowest low"]


def rc_frames(P, pp, rs_all=None):
    """The 50-day reclaim for every day at once (days x stocks), each day using only data up to that day.
    POP = the first close above the 50-day SMA after >= `rc_below` closes under it; it may stay above for up to
    `rc_pop_max` days. DIP = then closes back under the SMA, staying shallow (every low within `rc_dip_max` % under
    the SMA). HIGHER LOW = inside the dip, after its lowest low, price bounces and pulls back to a swing low that
    stays ABOVE that lowest low (a low lower than both neighbouring days' lows — confirmed the day after).
    TRIGGER = after >= `rc_min_dip` closes under, the first close back above the SMA within `rc_max_days` of the pop
    (with a higher low in place if required). Stop = under the higher low (or the dip's lowest low).
    SETUP = in a valid dip (buy-stop at the SMA). Optional: the dip's low above the low before the pop, the SMA's
    direction at the pop, reclaim-day volume, RS rating, price / $ volume."""
    H, L, C, V = (P[k].to_numpy(float) for k in ["High", "Low", "Close", "Volume"])
    idx, cols = P["Close"].index, P["Close"].columns
    n, k = C.shape
    D = lambda A: pd.DataFrame(A, index=idx, columns=cols)
    with np.errstate(all="ignore"):
        s50 = pd.DataFrame(C).rolling(50, min_periods=50).mean().to_numpy()
        slope = s50 - np.vstack([np.full((20, k), np.nan), s50[:-20]])       # 20-day change of the SMA
        avg50 = np.vstack([np.full((1, k), np.nan), pd.DataFrame(V).rolling(50, min_periods=30).mean().to_numpy()[:-1]])
        dvol = pd.DataFrame(C * V).rolling(50, min_periods=30).mean().to_numpy() / 1e6
        lowN = pd.DataFrame(L).rolling(int(pp["rc_base"]), min_periods=5).min().to_numpy()
        # closes under the SMA among the last B + slack closes (a brief poke above doesn't reset the count)
        W_ = int(pp["rc_below"]) + int(pp.get("rc_slack", 2))
        under_n = pd.DataFrame(np.isfinite(s50) & (C <= s50)).astype(float).rolling(W_, min_periods=1).sum().to_numpy()
    B, POPMAX, DIPMAX = int(pp["rc_below"]), int(pp["rc_pop_max"]), float(pp["rc_dip_max"]) / 100
    MINDIP, MAXD, HL = int(pp["rc_min_dip"]), int(pp["rc_max_days"]), bool(pp["rc_hl"])
    PRIOR = bool(pp.get("rc_prior", False))
    STOP_HL = pp.get("rc_stop", RC_STOPS[0]) == RC_STOPS[0]
    trend = pp["rc_trend"]
    pop_above = np.zeros(k, int)              # closes above the SMA since the pop (a 1-day whipsaw keeps counting)
    phase = np.zeros(k, int)                  # 0 none · 1 in the pop (above) · 2 in the dip (below)
    pop_i = np.full(k, -1)
    prior_low = np.full(k, np.nan)
    dip_low = np.full(k, np.nan)
    dip_low_i = np.full(k, -1)
    hl_low = np.full(k, np.nan)               # the latest higher low inside the dip (NaN = none yet)
    dip_days = np.zeros(k, int)
    depth_max = np.zeros(k)
    pop_trend_ok = np.zeros(k, bool)
    trig = np.zeros((n, k), bool); setup = np.zeros((n, k), bool)
    stop = np.full((n, k), np.nan); level = np.full((n, k), np.nan)
    o_prior = np.full((n, k), np.nan); o_dip = np.full((n, k), np.nan); o_hl = np.full((n, k), np.nan)
    o_pop = np.full((n, k), np.nan); o_depth = np.full((n, k), np.nan)
    dipw = np.zeros((n, k), bool)              # in a valid dip, but not armed yet (no higher low yet / dip too short)
    liq = lambda d: (C[d] >= float(pp["rc_minpx"])) & (dvol[d] >= float(pp["rc_mindvol"]))
    rs_ok = lambda d: (rs_all[d] >= float(pp["rc_rs"])) if (rs_all is not None and float(pp.get("rc_rs", 0)) > 0) \
        else np.ones(k, bool)
    for d in range(2, n):
        c, sm, lo = C[d], s50[d], L[d]
        valid = np.isfinite(c) & np.isfinite(sm)
        ab = valid & (c > sm)
        bl = valid & (c <= sm)
        # ---- a higher low confirmed today: yesterday's low was a swing low above the dip's lowest low ----
        sw = (phase == 2) & (d - 1 > dip_low_i) & (L[d - 1] <= L[d - 2]) & (L[d - 1] <= lo) & (L[d - 1] > dip_low)
        hl_low = np.where(sw, L[d - 1], hl_low)
        need = np.isfinite(hl_low) if HL else np.ones(k, bool)
        base_ok = (dip_low > prior_low) if PRIOR else np.ones(k, bool)
        stp = np.where(STOP_HL & np.isfinite(hl_low), hl_low, dip_low)
        # ---- a close back above ----
        rec = (phase == 2) & ab
        t_ = rec & (dip_days >= MINDIP) & (d - pop_i <= MAXD) & pop_trend_ok & need & base_ok & liq(d) & rs_ok(d)
        if float(pp["rc_vol"]) > 0:
            t_ &= V[d] >= float(pp["rc_vol"]) * avg50[d]
        trig[d] = t_
        for A, val in ((stop, stp), (level, sm), (o_prior, prior_low), (o_dip, dip_low), (o_hl, hl_low),
                       (o_pop, d - pop_i), (o_depth, depth_max * 100)):
            A[d] = np.where(t_, val, np.nan)
        short = rec & (dip_days < MINDIP) & (d - pop_i <= MAXD)             # a too-short dip: still the pop
        phase = np.where(rec & ~short, 0, np.where(short, 1, phase))
        # ---- a new pop: a close above after >= B of the last B + slack closes were under ----
        was_bl = np.isfinite(s50[d - 1]) & (C[d - 1] <= s50[d - 1])
        newpop = ab & was_bl & (under_n[d - 1] >= B) & (phase == 0)
        pop_above = np.where((phase == 1) & ab & ~newpop, pop_above + 1, pop_above)
        if newpop.any():
            pop_above = np.where(newpop, 1, pop_above)
            pop_i = np.where(newpop, d, pop_i)
            prior_low = np.where(newpop, lowN[d - 1], prior_low)
            up = slope[d] > 0
            tr_ok = (np.ones(k, bool) if trend == RC_TRENDS[0] else ~up if trend == RC_TRENDS[1] else up)
            pop_trend_ok = np.where(newpop, tr_ok & np.isfinite(slope[d]), pop_trend_ok)
            phase = np.where(newpop, 1, phase)
        phase = np.where((phase == 1) & ab & ((pop_above > POPMAX) | (d - pop_i > MAXD)), 0, phase)  # held above too long
        # ---- the dip ----
        start = (phase == 1) & bl
        if start.any():
            phase = np.where(start, 2, phase)
            dip_low = np.where(start, lo, dip_low)
            dip_low_i = np.where(start, d, dip_low_i)
            hl_low = np.where(start, np.nan, hl_low)
            dip_days = np.where(start, 0, dip_days)
            depth_max = np.where(start, 0.0, depth_max)
        indip = (phase == 2) & bl
        newlow = indip & (lo < dip_low)                                     # a new lowest low resets the higher low
        dip_low = np.where(newlow, lo, dip_low)
        dip_low_i = np.where(newlow, d, dip_low_i)
        hl_low = np.where(newlow, np.nan, hl_low)
        dip_days = np.where(indip, dip_days + 1, dip_days)
        depth_max = np.where(indip, np.fmax(depth_max, 1 - lo / sm), depth_max)
        bad = indip & ((lo < sm * (1 - DIPMAX)) | (d - pop_i > MAXD) | ((dip_low <= prior_low) if PRIOR else False))
        phase = np.where(bad, 0, phase)
        # ---- armed: a valid dip, a close above tomorrow would trigger ----
        need = np.isfinite(hl_low) if HL else np.ones(k, bool)
        arm = ((phase == 2) & (dip_days >= MINDIP) & (d - pop_i < MAXD) & pop_trend_ok & need & liq(d) & rs_ok(d))
        setup[d] = arm
        wait = ((phase == 2) & (d - pop_i < MAXD) & pop_trend_ok & liq(d) & rs_ok(d) & ~arm
                & (((dip_low > prior_low) | ~np.isfinite(prior_low)) if PRIOR else True))
        dipw[d] = wait
        stp = np.where(STOP_HL & np.isfinite(hl_low), hl_low, dip_low)
        for A, val in ((stop, stp), (level, sm), (o_prior, prior_low), (o_dip, dip_low), (o_hl, hl_low),
                       (o_pop, d - pop_i), (o_depth, depth_max * 100)):
            A[d] = np.where(arm | wait, val, A[d])
    return dict(trig=D(trig), setup=D(setup), dip=D(dipw), level=D(level), stop=D(stop), prior_low=D(o_prior), dip_low=D(o_dip),
                hl_low=D(o_hl), pop_days=D(o_pop), depth=D(o_depth), s50=D(s50), volx=D(V / avg50))


def rc_scan(P, pp):
    """The 50-day reclaim scan on the latest day (one row per stock)."""
    C = P["Close"]
    out = pd.DataFrame(index=C.columns)
    out["Reclaim"] = ""
    if len(C) < 80:
        return out
    rs_all = rs_matrix(C).to_numpy() if float(pp.get("rc_rs", 0)) > 0 else None
    F_ = rc_frames(P, pp, rs_all)
    # a TRIGGER in the last N days still counts while the close holds above the SMA and the stop
    N_ = max(1, int(pp.get("rc_recent", 1)))
    tr = F_["trig"].iloc[-N_:].to_numpy()
    ago = np.where(tr.any(0), np.argmax(tr[::-1], 0), np.nan)            # days since the latest trigger (0 = today)
    px = C.iloc[-1].to_numpy()
    s50_now = F_["s50"].iloc[-1].to_numpy()
    ok_, rows_ = np.isfinite(ago), len(C) - 1 - np.nan_to_num(ago).astype(int)

    def pick(k):                                                  # each stock's value on its trigger day
        v = np.full(len(ago), np.nan)
        v[ok_] = F_[k].to_numpy(float)[rows_[ok_], np.flatnonzero(ok_)]
        return v
    t_stop = pick("stop")
    with np.errstate(invalid="ignore"):
        t = np.isfinite(ago) & ((ago == 0) | ((px > s50_now) & (px > t_stop)))
    TRIG_K = ("stop", "level", "prior_low", "dip_low", "hl_low", "pop_days", "depth")   # as of the trigger day
    last = lambda k: np.where(t, pick(k), F_[k].iloc[-1].to_numpy()) if k in TRIG_K else F_[k].iloc[-1].to_numpy()
    s_ = F_["setup"].iloc[-1].to_numpy().astype(bool) & ~t
    w_ = F_["dip"].iloc[-1].to_numpy().astype(bool) & ~t & ~s_
    out["Reclaim"] = np.where(t, "TRIGGER", np.where(s_, "SETUP", np.where(w_, "DIP", "")))
    out["Trigger days ago"] = np.where(t, ago, np.nan)
    out["50 SMA"] = last("s50")
    out["vs 50 SMA %"] = (px / last("s50") - 1) * 100
    out["Low before pop"] = last("prior_low")
    out["Higher low"] = last("hl_low")
    out["Dip low"] = last("dip_low")
    out["Dip depth %"] = last("depth")
    out["Days since pop"] = last("pop_days")
    out["Reclaim stop"] = last("stop")
    ref = np.where(t, px, last("s50"))
    out["Reclaim risk %"] = np.where(t | s_ | w_, (ref - last("stop")) / ref * 100, np.nan)
    out["Reclaim vol ×"] = last("volx")
    return out



@st.cache_data(ttl=12 * 3600, show_spinner=False)
def all_time_highs(tickers):
    """Monthly highs over each stock's whole history (only for the few matches — confirms the green line really
    was the all-time high when it was set)."""
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = list(tickers[i:i + 100])
        try:
            df = yf.download(chunk, period="max", interval="1mo", auto_adjust=True, progress=False,
                             group_by="ticker", threads=True)
        except Exception:
            continue
        for t in chunk:
            try:
                h = df[t]["High"] if isinstance(df.columns, pd.MultiIndex) else df["High"]
                out[t] = h.dropna()
            except Exception:
                pass
    return out


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


# Each scan's own calculations, cached separately: changing one scan's setting only re-runs that scan.
CORE_KEYS = ("max_range", "min_pull", "max_pull", "min_bars", "max_ema_dist", "max_ema_gap", "dry_up", "breakout_vol")
SCAN_PARTS = [
    ("vr_", "record volume", lambda P, m, pp, vh: volume_records(P["Close"], P["Volume"], pp, vh)),
    ("pb_", "parabolic runs", lambda P, m, pp, vh: parabolic_scan(P["Open"], P["High"], P["Low"], P["Close"],
                                                                   P["Close"].ewm(span=10, adjust=False).mean(), pp)),
    ("ep_", "episodic pivots", lambda P, m, pp, vh: ep_scan(P["Open"], P["High"], P["Low"], P["Close"], P["Volume"], pp)),
    ("gl_", "green line breakouts", lambda P, m, pp, vh: green_line(P["High"], P["Low"], P["Close"], P["Volume"], pp)),
    ("so_", "shakeout +3 setups", lambda P, m, pp, vh: so_rs_gate(
        shakeout_plus3(P["High"], P["Low"], P["Close"], P["Volume"], pp), m["RS"], pp)),
    ("mac_", "MA consolidations", lambda P, m, pp, vh: mac_scan(P, m["RS"], pp)),
    ("htf_", "high tight flags", lambda P, m, pp, vh: htf_scan(P, pp)),
    ("sh_", "swing shorts", lambda P, m, pp, vh: short_scan(P, pp)),
    ("rc_", "50-day reclaims", lambda P, m, pp, vh: rc_scan(P, pp)),
    ("f5_", "first closes above the 50-day", lambda P, m, pp, vh: f5_scan(P, pp)),
    ("vcp_", "VCP patterns", lambda P, m, pp, vh: vcp_scan(P["High"], P["Low"], P["Close"], P["Volume"], m["RS"], pp)),
    ("xb_", "EMA crossbacks", lambda P, m, pp, vh: ema_crossback(P["Open"], P["High"], P["Low"], P["Close"], P["Volume"],
                                                                 m["RS"], pp)),
]


# ============================================================================
# 4. Fundamentals (loaded on demand for the current results)
# ============================================================================
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def fetch_fundamentals(t):
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        info = {}

    def pct(k):
        v = info.get(k)
        return v * 100 if isinstance(v, (int, float)) else np.nan

    days = np.nan
    for k in ("earningsTimestampStart", "earningsTimestamp"):
        ts = info.get(k)
        if isinstance(ts, (int, float)) and ts > 0:
            d = (datetime.fromtimestamp(ts).date() - date.today()).days
            if d >= -1:
                days = d
                break
    rec = info.get("recommendationKey")
    return {
        "P/E": info.get("trailingPE", np.nan), "Fwd P/E": info.get("forwardPE", np.nan),
        "EPS growth %": pct("earningsQuarterlyGrowth"), "Rev growth %": pct("revenueGrowth"),
        "Profit margin %": pct("profitMargins"), "Short float %": pct("shortPercentOfFloat"),
        "Earnings in (days)": days,
        "Analyst": rec.replace("_", " ").title() if isinstance(rec, str) and rec != "none" else None,
        "_mcap": (info.get("marketCap") or np.nan) / 1e9 if info.get("marketCap") else np.nan,
    }


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_mcap(t):
    try:
        v = yf.Ticker(t).fast_info["marketCap"]
        return float(v) / 1e9 if v else np.nan
    except Exception:
        return np.nan


# ---- Earnings calendar (Nasdaq, one request per weekday; Yahoo as fallback) ----
EARN_TIME = {"time-pre-market": "Before open", "time-after-hours": "After close"}


def _money(x):
    try:
        s = str(x).replace("$", "").replace(",", "").strip()
        neg = s.startswith("(") and s.endswith(")")
        v = float(s.strip("()"))
        return -v if neg else v
    except Exception:
        return np.nan


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def earnings_day(day_iso):
    """All companies Nasdaq lists as reporting on one date (past or future). None if the request failed."""
    try:
        r = requests.get(f"https://api.nasdaq.com/api/calendar/earnings?date={day_iso}", headers=UA, timeout=20)
        rows = ((r.json() or {}).get("data") or {}).get("rows") or []
    except Exception:
        return None
    out = []
    for x in rows:
        sym = str(x.get("symbol", "")).strip().replace("/", "-").replace(".", "-")
        if not sym:
            continue
        est, act = _money(x.get("epsForecast")), _money(x.get("eps"))
        surprise = pd.to_numeric(str(x.get("surprise", "")).replace("%", "").strip(), errors="coerce")
        if pd.isna(surprise) and pd.notna(act) and pd.notna(est) and est != 0:
            surprise = (act - est) / abs(est) * 100
        out.append({"Symbol": sym, "Earnings date": pd.Timestamp(day_iso),
                    "Report time": EARN_TIME.get(x.get("time"), "Not given"),
                    "EPS est": est, "EPS actual": act, "Surprise %": surprise,
                    "Last yr EPS": _money(x.get("lastYearEPS")),
                    "# Ests": pd.to_numeric(str(x.get("noOfEsts", "")).strip(), errors="coerce"),
                    "Quarter": x.get("fiscalQuarterEnding") or None})
    return out


CAL_COLS = ["Earnings date", "Report time", "EPS est", "EPS actual", "Surprise %", "Last yr EPS", "# Ests", "Quarter"]


def _nearest_per_stock(df):
    """One row per stock: the report closest to today (the next one, or the latest past one)."""
    today = pd.Timestamp(date.today())
    df = df.assign(_d=(df["Earnings date"] - today).abs()).sort_values("_d")
    return df.drop_duplicates("Symbol").drop(columns="_d").set_index("Symbol")


def earnings_calendar(lo, hi):
    """Earnings from `lo` to `hi` calendar days from today (negative = past). None if Nasdaq is unreachable."""
    today = pd.Timestamp(date.today())
    days = [d.date().isoformat() for d in pd.bdate_range(today + pd.Timedelta(days=lo), today + pd.Timedelta(days=hi))]
    if not days:
        return pd.DataFrame(columns=CAL_COLS)
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(earnings_day, days))
    ok = [r for r in results if r is not None]
    if not ok:
        return None
    rows = [row for r in ok for row in r]
    return _nearest_per_stock(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=CAL_COLS)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def yahoo_earnings(t):
    """Backup source: recent + upcoming earnings for one stock from Yahoo."""
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=12)
    except Exception:
        return []
    if e is None or not len(e):
        return []
    out = []
    for ts, r in e.iterrows():
        ts = pd.Timestamp(ts)
        hour = ts.hour if ts.tzinfo is None else ts.tz_convert("America/New_York").hour
        out.append({"Symbol": t, "Earnings date": pd.Timestamp(ts.date()),
                    "Report time": "Before open" if hour < 12 else ("After close" if hour >= 16 else "Not given"),
                    "EPS est": r.get("EPS Estimate", np.nan), "EPS actual": r.get("Reported EPS", np.nan),
                    "Surprise %": r.get("Surprise(%)", np.nan)})
    return out


def earnings_reaction(dates, times, panels):
    """Price move around each past report: opening gap, reaction-day move, and move since (vs the close before)."""
    C, O = panels["Close"], panels["Open"]
    idx = C.index
    out = pd.DataFrame(np.nan, index=dates.index, columns=["Earnings gap %", "Reaction %", "Since earnings %"])
    for t, d in dates.dropna().items():
        if t not in C.columns:
            continue
        d = pd.Timestamp(d)
        react = idx.searchsorted(d, side="right" if times.get(t) == "After close" else "left")
        if react <= 0 or react >= len(idx):       # no trading day yet after the report
            continue
        pre = C[t].iloc[react - 1]
        if not pre or pd.isna(pre):
            continue
        out.at[t, "Earnings gap %"] = (O[t].iloc[react] / pre - 1) * 100
        out.at[t, "Reaction %"] = (C[t].iloc[react] / pre - 1) * 100
        out.at[t, "Since earnings %"] = (C[t].iloc[-1] / pre - 1) * 100
    return out


FUND_COLS = ["P/E", "Fwd P/E", "EPS growth %", "Rev growth %", "Profit margin %",
             "Short float %", "Earnings in (days)", "Analyst"]


# ============================================================================
# 5. Filter definitions (TradingView-style)
# ============================================================================
def between(col, lo=None, hi=None):
    def f(d):
        s = d[col]
        mask = s.notna()
        if lo is not None:
            mask &= s >= lo
        if hi is not None:
            mask &= s <= hi
        return mask
    return f


def is_true(col):
    return lambda d: d[col].fillna(False).astype(bool)


PERF_PRESETS = [("Up", "Positive return", between, (0.0001, None)),
                ("Above 10%", "", between, (10, None)), ("Above 20%", "", between, (20, None)),
                ("Above 50%", "Big winners", between, (50, None)), ("Above 100%", "Doubled", between, (100, None)),
                ("Down", "Negative return", between, (None, -0.0001)),
                ("Below −10%", "", between, (None, -10)), ("Below −25%", "Big losers", between, (None, -25))]


def perf_filter(col, label):
    return dict(label=label, col=col, kind="num", unit="%",
                presets=[(a, b, fn(col, *args)) for a, b, fn, args in PERF_PRESETS])


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
SP_INDEXES = ["S&P 500", "S&P 100", "S&P MidCap 400", "S&P SmallCap 600", "Any S&P index"]


def sp_pick_index(df, idx):
    return df if idx == "Any S&P index" or not len(df) else df[df["Index"] == idx]
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
# LiqLead conditions: key -> (label, column, default value, unit, step)
LL_G1 = {"dvol": ("Avg \\$ vol 50d ≥ (\\$M)", "Avg $ Vol 50d M", 500.0, "$M", 50.0),
         "adr": ("ADR % 14d ≥", "ADR % 14d", 3.0, "%", 0.25)}
LL_G2 = {"c5": ("% chg 5d ≥", "Chg 5d %", 15.0, "%", 1.0),
         "c20": ("% chg 20d ≥", "Chg 20d %", 20.0, "%", 1.0),
         "off": ("% off 52W high ≥", "Off 52W high %", -25.0, "%", 1.0)}
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
F["earnwin"] = dict(label="Earnings", col="Days to earnings", kind="preset", presets=[
    (k, {"Today": "Reporting today", "Next 1 week": "Reporting in the next 7 days"}.get(k, f"Reported in the {k.lower()}"),
     between("Days to earnings", lo, hi)) for k, (lo, hi) in EARN_WINDOWS.items()])
F["short"] = dict(label="Short float", col="Short float %", kind="num", unit="%", fund=True, presets=[
    ("Above 10%", "Heavily shorted", between("Short float %", 10)),
    ("Above 20%", "Squeeze candidates", between("Short float %", 20)),
    ("Below 5%", "", between("Short float %", None, 5))])

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
TPL_FILE = Path(__file__).with_name("chart_templates.json")
DEFAULT_TPL = dict(style="Candles", interval="Daily", height=720, mas="EMA 10, EMA 21, SMA 50, SMA 200",
                   extras=[], volume=True, theme="Dark", toolbar=True, watchlist=True, layout_url="")
TPL_FIELDS = list(DEFAULT_TPL)


def load_templates():
    try:
        data = json.loads(TPL_FILE.read_text(encoding="utf-8"))
        tpls = {k: {**DEFAULT_TPL, **v} for k, v in data.get("templates", {}).items()}
        active = data.get("active")
    except Exception:
        tpls, active = {}, None
    tpls.setdefault("Default", dict(DEFAULT_TPL))
    return tpls, active if active in tpls else "Default"


def save_templates(tpls, active):
    try:
        TPL_FILE.write_text(json.dumps({"active": active, "templates": tpls}, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        st.session_state["tpl_msg"] = f"⚠️ Couldn't save templates: {e}"
        return False


def parse_mas(text):
    """'EMA 10, SMA 50' -> [("EMA", 10), ("SMA", 50)]"""
    return [(t.upper(), int(n)) for t, n in re.findall(r"(EMA|SMA|WMA)\s*(\d+)", text or "", flags=re.I)][:8]


def tradingview_chart(tv_sym, tpl, watch=None):
    """TradingView's free Advanced Chart widget, set up from a chart template."""
    height = int(tpl["height"])
    dark = tpl["theme"] == "Dark"
    studies = [{"id": TV_MA_IDS[t], "inputs": {"length": n}} for t, n in parse_mas(tpl["mas"])]
    studies += [TV_EXTRAS[x] for x in tpl["extras"] if x in TV_EXTRAS]
    cfg = {
        "autosize": False, "width": "100%", "height": height, "symbol": tv_sym,
        "interval": TV_INTERVALS.get(tpl["interval"], "D"), "timezone": "America/New_York",
        "theme": "dark" if dark else "light", "style": TV_STYLES.get(tpl["style"], "1"), "locale": "en",
        "allow_symbol_change": True, "withdateranges": True, "hide_side_toolbar": not tpl["toolbar"],
        "hide_volume": not tpl["volume"], "details": False, "calendar": False, "save_image": True,
        "studies": studies, "support_host": "https://www.tradingview.com",
    }
    if dark:
        cfg["backgroundColor"] = "#0e1117"
    if watch and tpl["watchlist"]:
        cfg["watchlist"] = watch
    bg = "#0e1117" if dark else "#ffffff"
    # fixed pixel height (autosize collapses inside Streamlit's frame, which made the chart a thin strip)
    html = f"""
<style>html, body {{ margin:0; padding:0; background:{bg}; height:{height}px; overflow:hidden; }}
       .tradingview-widget-container, .tradingview-widget-container__widget,
       .tradingview-widget-container iframe {{ height:{height}px !important; width:100% !important; }}</style>
<div class="tradingview-widget-container" style="height:{height}px;width:100%">
  <div class="tradingview-widget-container__widget" style="height:{height}px;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
          async>{json.dumps(cfg)}</script>
</div>"""
    components.html(html, height=height + 4)


def tv_open_url(tv_sym, layout_url=""):
    """Link to the full TradingView site — into your own saved layout if you pasted its link."""
    m = re.search(r"tradingview\.com/chart/([A-Za-z0-9]+)", layout_url or "")
    base = f"https://www.tradingview.com/chart/{m.group(1)}/" if m else "https://www.tradingview.com/chart/"
    return f"{base}?symbol={tv_sym}"


def price_chart(t, df, days=160):
    d = df.dropna(subset=["Close"]).copy()
    d["EMA 10"] = d["Close"].ewm(span=10, adjust=False).mean()
    d["SMA 20"] = d["Close"].rolling(20).mean()
    d["SMA 50"] = d["Close"].rolling(50).mean()
    d["SMA 200"] = d["Close"].rolling(200).mean()
    d["Vol 50"] = d["Volume"].rolling(50).mean()
    d = d.tail(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                                 increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
                                 name=t, showlegend=False), row=1, col=1)
    for name, color in [("EMA 10", "#8bc34a"), ("SMA 20", "#e57373"), ("SMA 50", "#cddc39"), ("SMA 200", "#9e9e9e")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[name], name=name, mode="lines",
                                 line=dict(color=color, width=1.2)), row=1, col=1)
    last7 = d.tail(7)
    fig.add_shape(type="rect", x0=last7.index[0], x1=last7.index[-1], y0=last7["Low"].min(),
                  y1=last7["High"].max(), line=dict(color="#4fc3f7", width=1, dash="dot"),
                  fillcolor="rgba(79,195,247,0.08)", row=1, col=1)
    colors = np.where(d["Close"] >= d["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=colors, name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Vol 50"], name="Vol 50d avg", mode="lines",
                             line=dict(color="#ffb74d", width=1)), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=10, r=10, t=80, b=10),
                      title=dict(text=f"{t} — daily", y=0.98, x=0.01), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


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


def rs_add_rank(df, short_col, long_col):
    """Percentile-rank two performance columns into Week RS / Month RS (0 = worst, 100 = best)."""
    df = df.dropna(subset=[short_col, long_col]).copy()
    n = len(df)
    df["Week RS"] = (df[short_col].rank() - 1) / max(n - 1, 1) * 100
    df["Month RS"] = (df[long_col].rank() - 1) / max(n - 1, 1) * 100
    strong_w, strong_m = df["Week RS"] >= 50, df["Month RS"] >= 50
    df["Quadrant"] = "Weak"
    df.loc[strong_w & strong_m, "Quadrant"] = "Strong"
    df.loc[~strong_w & strong_m, "Quadrant"] = "Weakening"
    df.loc[strong_w & ~strong_m, "Quadrant"] = "Improving"
    df["Score"] = (df["Week RS"] + df["Month RS"]) / 2
    return df.sort_values("Score", ascending=False).round(1)


def rs_pct_rank(s):
    n = s.notna().sum()
    return ((s.rank() - 1) / max(n - 1, 1) * 100).round(0)


def rs_classify_stage(p, e10, e20, s50, s200, atr):
    """Approximate stage (1A-4C) from price vs. 10/20 EMAs and 50/200 SMAs.
    A simplified take on the Cycle of Price Action, not Steve Jacobs' exact rules."""
    if any(pd.isna(v) for v in (p, e10, e20, s50)):
        return "?"
    s200 = s50 if pd.isna(s200) else s200
    ext = (p - s50) / atr if pd.notna(atr) and atr > 0 else (p / s50 - 1) * 100 / 3.5  # ~25% = 7 "ATRs"
    if p > e10 and e10 > e20:
        if ext >= 7:
            return "2C"
        if e20 > s50 > s200 and p > s50:
            return "2B"
        return "2A" if p > s50 else "1B"
    if p < e10 and e10 < e20:
        if ext <= -7:
            return "4C"
        if e20 < s50 < s200 and p < s50:
            return "4B"
        return "4A" if p < s50 else "3B"
    if p > e10:                       # price turning up through falling EMAs
        return "2A" if p > s50 else "1A"
    return "3A" if p > s50 else "4A"  # price slipping under rising EMAs


@st.cache_data(ttl=30 * 60, show_spinner=False)
def rs_prices(symbols, period="1y"):
    try:
        df = yf.download(list(symbols), period=period, auto_adjust=True, progress=False)["Close"]
    except Exception:
        return pd.DataFrame()
    if isinstance(df, pd.Series):
        df = df.to_frame(symbols[0])
    return df.dropna(how="all")


def rs_stock_table(panels, bench, short_days, long_days):
    """One row per downloaded stock: returns, vs-benchmark returns, RS ranks, quadrant, score, COMP, stage."""
    O, H, L, C = (panels[k] for k in ["Open", "High", "Low", "Close"])
    n = len(C)
    last = C.iloc[-1]
    ago = lambda k: C.iloc[-1 - k] if n > k else pd.Series(np.nan, index=C.columns)
    out = pd.DataFrame(index=C.columns)
    for _, col, k in RS_TIMEFRAMES:
        out[col] = (last / ago(k) - 1) * 100
    b = rs_prices((bench,), "2y")
    b = b[bench].dropna() if bench in b else pd.Series(dtype=float)
    b = b[b.index <= C.index[-1]]
    b_ret = lambda k: (b.iloc[-1] / b.iloc[-1 - k] - 1) * 100 if len(b) > k else 0.0
    out["Short vs bench %"] = (last / ago(short_days) - 1) * 100 - b_ret(short_days)
    out["Long vs bench %"] = (last / ago(long_days) - 1) * 100 - b_ret(long_days)
    out = rs_add_rank(out, "Short vs bench %", "Long vs bench %")
    for tf, col, _ in RS_TIMEFRAMES:
        out[tf] = rs_pct_rank(out[col])
    comp = [tf for tf, col, _ in RS_TIMEFRAMES[1:] if out[col].notna().any()]
    out["COMP"] = out[comp].mean(axis=1).round(0)
    # stage from the latest EMAs / SMAs / ATR
    e10 = C.ewm(span=10, adjust=False).mean().iloc[-1]
    e20 = C.ewm(span=20, adjust=False).mean().iloc[-1]
    s50 = C.rolling(50).mean().iloc[-1]
    s200 = C.rolling(200, min_periods=150).mean().iloc[-1]
    pc = C.shift(1)
    tr = np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy())
    atr = pd.DataFrame(tr, index=C.index, columns=C.columns).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    out["Stage"] = [rs_classify_stage(last.get(t), e10.get(t), e20.get(t), s50.get(t), s200.get(t), atr.get(t))
                    for t in out.index]
    out["Bench"] = bench if len(b) else ""
    return out


def rs_groups(df, by, min_stocks):
    """Group strength = median performance of its stocks, ranked 0-100 against other groups."""
    d = df.dropna(subset=[by])
    agg = {"Stocks": (by, "size")}
    for _, col, _ in RS_TIMEFRAMES:
        agg[col] = (col, "median")
    agg["Short vs bench %"] = ("Short vs bench %", "median")
    agg["Long vs bench %"] = ("Long vs bench %", "median")
    g = d.groupby(by).agg(**agg)
    if by == "Industry":
        g["Sector"] = d.groupby("Industry")["Sector"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "Unknown")
    g = g[g["Stocks"] >= min_stocks].reset_index()
    if not len(g):
        return g
    g = rs_add_rank(g, "Short vs bench %", "Long vs bench %")
    for tf, col, _ in RS_TIMEFRAMES:
        g[tf] = rs_pct_rank(g[col])
    comp = [tf for tf, col, _ in RS_TIMEFRAMES[1:] if g[col].notna().any()]
    g["COMP"] = g[comp].mean(axis=1).round(0)
    return g


def rs_chart(df, label_col, color_col, hover_cols, height=700):
    show_text = len(df) <= 80
    fig = px.scatter(
        df, x="Week RS", y="Month RS",
        color=color_col if color_col else "Score",
        color_continuous_scale="RdYlGn" if not color_col else None,
        range_color=(0, 100) if not color_col else None,
        text=label_col if show_text else None,
        hover_name=label_col, hover_data={c: True for c in hover_cols if c in df.columns},
    )
    fig.update_traces(marker=dict(size=10 if show_text else 7), textposition="top center", textfont_size=10)
    for x0, x1, y0, y1, color, label in [(50, 100, 50, 100, "rgba(40,160,90,0.10)", "STRONG"),
                                         (0, 50, 50, 100, "rgba(200,150,40,0.08)", "WEAKENING"),
                                         (0, 50, 0, 50, "rgba(200,60,60,0.08)", "WEAK"),
                                         (50, 100, 0, 50, "rgba(60,110,220,0.10)", "IMPROVING")]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 - 3, text=label, showarrow=False, font=dict(color="gray", size=11))
    fig.add_vline(x=50, line_dash="dash", line_color="gray")
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.update_layout(height=height, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      xaxis=dict(title="Week RS", range=[-3, 103]), yaxis=dict(title="Month RS", range=[-3, 103]),
                      legend=dict(orientation="h", y=-0.12, title=None), coloraxis_showscale=False,
                      margin=dict(l=40, r=20, t=30, b=40))
    return fig


def rs_rank_color(v):
    if pd.isna(v):
        return "transparent"
    if v >= 50:
        return f"rgba(46,160,90,{0.15 + (v - 50) / 50 * 0.55:.2f})"
    return f"rgba(210,60,60,{0.15 + (50 - v) / 50 * 0.55:.2f})"


def rs_chg_html(v):
    if pd.isna(v):
        return ""
    c = "#4cd97b" if v >= 0 else "#ff6b6b"
    return f'<span style="color:{c};font-size:10px;margin-left:4px">{v:+.1f}%</span>'


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


def rs_html(s):
    """Render raw HTML (no leading spaces so Markdown doesn't turn it into a code block)."""
    st.markdown(RS_CSS + s, unsafe_allow_html=True)


def rs_score_circle(v):
    c = "#3CC47C" if v >= 70 else "#F5C542" if v >= 50 else "#E74C4C"
    return f'<span class="circ" style="border-color:{c};color:{c}">{v:.0f}</span>'


def rs_stage_badge(sg):
    c = RS_STAGES.get(sg, RS_STAGES["?"])[1]
    return f'<span class="badge" style="border-color:{c};color:{c}">{sg}</span>'


def render_rs_view(stocks, view, opts):
    """Draws the chosen RS view above the results table. `stocks` = filtered results with RS columns.
    Returns the list of symbols the view is showing (the table below follows it)."""
    df = stocks.copy()
    df["Sector"] = df["Sector"].fillna("Unknown").astype(str)
    df["Industry"] = df["Industry"].fillna("Unknown").astype(str)
    has_groups = (df["Industry"] != "Unknown").any()
    sectors_sorted = sorted(df["Sector"].unique())
    sector_color = {s: RS_PALETTE[i % len(RS_PALETTE)] for i, s in enumerate(sectors_sorted)}
    in_stage = df[df["Stage"].isin(opts["stages"])]
    min_stocks = opts["min_stocks"]

    def stock_list(industry, limit, order):
        s = in_stage[in_stage["Industry"] == industry]
        if order == "Largest market cap first" and s["Mkt cap $B"].notna().any():
            s = s.sort_values("Mkt cap $B", ascending=False)
        else:
            s = s.sort_values("Score", ascending=False)
        return s.head(limit)

    if view in ("Leading groups - table", "Leading groups - cards", "Industries (quadrant)", "Sectors (quadrant)") \
            and not has_groups:
        st.info("Sector / industry data isn't available for these stocks, so group views can't be drawn. "
                "Showing the stocks quadrant instead.")
        view = "Stocks (quadrant)"

    if view == "Leading groups - table":
        ind = rs_groups(df, "Industry", min_stocks)
        c1, c2, c3, c4 = st.columns(4)
        n_groups = c1.slider("Industries to show", 5, 60, 20, key="rs_tbl_n")
        n_chips = c2.slider("Stocks per industry", 5, 40, 25, key="rs_tbl_chips")
        order = c3.selectbox("Stock order", ["Largest market cap first", "Strongest first"], key="rs_tbl_order")
        sort_by = c4.radio("Rank industries by", ["Score", "COMP"], horizontal=True, key="rs_tbl_sort")
        if not len(ind):
            st.warning("No industries have enough stocks. Lower **Min stocks per group**.")
            return list(df.index)
        ind = ind.sort_values(sort_by, ascending=False).head(n_groups)
        tfs = [tf for tf, col, _ in RS_TIMEFRAMES if ind[col].notna().any()]
        head = "".join(f"<th>{t}</th>" for t in ["INDUSTRY", "# STK"] + tfs + ["COMP", "SCORE"])
        head += '<th style="text-align:left">STOCKS</th>'
        rows, shown = [], []
        for _, r in ind.iterrows():
            cells = f'<td class="name" style="border-left:3px solid {sector_color.get(r["Sector"], "#888")}">{r["Industry"]}</td>'
            cells += f'<td>{int(r["Stocks"])}</td>'
            for tf in tfs + ["COMP", "Score"]:
                cells += f'<td style="background:{rs_rank_color(r[tf])}">{"" if pd.isna(r[tf]) else int(r[tf])}</td>'
            sl = stock_list(r["Industry"], n_chips, order)
            shown += list(sl.index)
            chips = "".join(
                f'<span class="chip" style="border-left-color:{RS_STAGES[s["Stage"]][1]}" '
                f'title="Stage {s["Stage"]} · Score {s["Score"]:.0f}">{t}{rs_chg_html(s["Day %"])}</span>'
                for t, s in sl.iterrows())
            cells += f'<td class="chips">{chips}</td>'
            rows.append(f"<tr>{cells}</tr>")
        rs_html(f'<div class="rs-wrap"><table class="rs"><tr>{head}</tr>{"".join(rows)}</table></div>')
        st.caption("Chip stripe color = stock stage (see Stage analysis). Numbers are 0-100 ranks vs. other industries.")
        return shown

    if view == "Leading groups - cards":
        ind = rs_groups(df, "Industry", min_stocks)
        c1, c2, c3 = st.columns(3)
        n_groups = c1.slider("Industries to show", 4, 48, 12, key="rs_card_n")
        n_rows = c2.slider("Stocks per card", 3, 30, 15, key="rs_card_rows")
        order = c3.selectbox("Stock order", ["Largest market cap first", "Strongest first"], key="rs_card_order")
        if not len(ind):
            st.warning("No industries have enough stocks. Lower **Min stocks per group**.")
            return list(df.index)
        ind = ind.sort_values("Score", ascending=False).head(n_groups)
        cards, shown = [], []
        for _, r in ind.iterrows():
            sl = stock_list(r["Industry"], n_rows, order)
            shown += list(sl.index)
            body = "".join(
                f'<div class="row" style="border-left-color:{RS_STAGES[s["Stage"]][1]}">'
                f'<span><b>{t}</b>{rs_chg_html(s["Day %"])}</span>'
                f'<span>{rs_stage_badge(s["Stage"])}{rs_score_circle(s["Score"])}</span></div>'
                for t, s in sl.iterrows())
            cards.append(
                f'<div class="card"><div class="hd" style="background:{sector_color.get(r["Sector"], "#555")}">'
                f'<span>{r["Industry"]}</span><span>{r["Score"]:.0f}</span></div>'
                f'<div class="sub"><span>Wk {r["Week RS"]:.0f} · Mth {r["Month RS"]:.0f}</span>'
                f'<span>{len(sl)}/{int(r["Stocks"])}</span></div>{body}</div>')
        rs_html(f'<div class="cards">{"".join(cards)}</div>')
        st.caption("Header color = sector. Badge = stage. Circle = the stock's own RS score (0-100).")
        return shown

    if view == "Stage analysis":
        c1, c2, c3 = st.columns(3)
        scope = c1.radio("Stocks", ["All stocks", "Only top-half industries"], horizontal=True, key="rs_stage_scope")
        per_col = c2.slider("Stocks listed per stage", 5, 100, 25, key="rs_stage_per")
        order = c3.selectbox("Order within each stage", ["Strongest first", "Largest market cap first"], key="rs_stage_order")
        pool = df
        if scope == "Only top-half industries" and has_groups:
            ind = rs_groups(df, "Industry", min_stocks)
            if len(ind):
                pool = pool[pool["Industry"].isin(set(ind.loc[ind["Score"] >= 50, "Industry"]))]
        pool = pool[pool["Stage"] != "?"]
        counts = pool["Stage"].value_counts().reindex(RS_STAGE_ORDER[:-1], fill_value=0)
        bull = counts[["1A", "1B", "2A", "2B", "2C"]].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Stocks", f"{len(pool):,}")
        m2.metric("Bullish stages (1A-2C)", f"{bull / max(len(pool), 1):.0%}")
        m3.metric("Watchable (1A-2B)", f'{counts[["1A", "1B", "2A", "2B"]].sum():,}')
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[RS_STAGES[s][1] for s in counts.index],
                               text=counts.values, textposition="outside", cliponaxis=False))
        fig.update_layout(height=260, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
        cols_html, shown = [], []
        for sg in RS_STAGE_ORDER[:-1]:
            s = pool[pool["Stage"] == sg]
            s = (s.sort_values("Mkt cap $B", ascending=False) if order.startswith("Largest")
                 else s.sort_values("Score", ascending=False)).head(per_col)
            shown += list(s.index)
            name, colr = RS_STAGES[sg]
            rows = "".join(
                f'<div class="row" style="border-left-color:{sector_color.get(x["Sector"], "#888")}">'
                f'<span><b>{t}</b>{rs_chg_html(x["Day %"])}</span>{rs_score_circle(x["Score"])}</div>'
                for t, x in s.iterrows())
            cols_html.append(f'<div class="stagecol"><div class="hd" style="background:{colr}">'
                             f'{counts[sg]}<br><span style="font-size:10px">{name} · {sg}</span></div>{rows}</div>')
        rs_html('<div class="rs-wrap"><div style="display:grid;grid-template-columns:repeat(10,minmax(120px,1fr));'
                'gap:6px;font-size:12px;color:#ddd">' + "".join(cols_html) + "</div></div>")
        st.caption("Stages are estimated from price vs. the 10/20-day EMAs and 50/200-day SMAs "
                   "(a simplified version of the Cycle of Price Action). Stripe color = sector.")
        return shown

    if view == "Industries (quadrant)":
        ind = rs_groups(df, "Industry", min_stocks)
        if len(ind):
            st.plotly_chart(rs_chart(ind, "Industry", "Sector", ["Stocks", "Week %", "Month %"]), use_container_width=True)
            with st.expander("Industry table"):
                st.dataframe(ind, use_container_width=True, hide_index=True)
        return list(df.index)

    if view == "Sectors (quadrant)":
        sec = rs_groups(df, "Sector", min_stocks)
        if len(sec):
            st.plotly_chart(rs_chart(sec, "Sector", "Sector", ["Stocks", "Week %", "Month %"]), use_container_width=True)
            with st.expander("Sector table"):
                st.dataframe(sec, use_container_width=True, hide_index=True)
        return list(df.index)

    if view == "Sector ETFs (quadrant)":
        bench, sd, ld = opts["bench"], opts["short"], opts["long"]
        prices = rs_prices(tuple(sorted(set(list(RS_SECTOR_ETFS) + [bench]))), "1y")
        if bench not in prices.columns or prices[bench].dropna().empty:
            st.error(f"Couldn't load the benchmark {bench} or the sector ETFs. Check your connection.")
            return list(df.index)
        prices = prices.dropna(axis=1, thresh=ld + 2).ffill()
        ret_s = prices.iloc[-1] / prices.iloc[-1 - sd] - 1
        ret_l = prices.iloc[-1] / prices.iloc[-1 - ld] - 1
        e = pd.DataFrame({"Short vs bench %": (ret_s - ret_s[bench]) * 100,
                          "Long vs bench %": (ret_l - ret_l[bench]) * 100}).drop(index=bench)
        e.index.name = "Ticker"
        e = rs_add_rank(e, "Short vs bench %", "Long vs bench %").reset_index()
        e["Name"] = e["Ticker"].map(RS_SECTOR_ETFS).fillna(e["Ticker"])
        st.caption(f"SPDR sector ETFs vs {bench} · prices as of {prices.index[-1]:%b %d, %Y}")
        st.plotly_chart(rs_chart(e, "Name", None, ["Short vs bench %", "Long vs bench %"]), use_container_width=True)
        with st.expander("Sector ETF table"):
            st.dataframe(e, use_container_width=True, hide_index=True)
        return list(df.index)

    # Stocks (quadrant)
    shown = in_stage
    if has_groups and opts["top_only"]:
        ind = rs_groups(df, "Industry", min_stocks)
        if len(ind):
            keep = set(ind.loc[ind["Score"] >= 50, "Industry"])
            shown = shown[shown["Industry"].isin(keep)]
    if opts["strong_only"]:
        shown = shown[shown["Quadrant"] == "Strong"]
    st.caption(f"Showing {len(shown):,} stocks. Ranks are against all {opts.get('total', len(stocks)):,} "
               "stocks in this list.")
    if len(shown):
        plot = shown.reset_index().rename(columns={"index": "Symbol"}) if "Symbol" not in shown.columns else shown
        hover = [c for c in ["Industry", "Stage", "Week %", "Month %", "Mkt cap $B"] if c in plot.columns]
        st.plotly_chart(rs_chart(plot, "Symbol", "Sector" if has_groups else None, hover), use_container_width=True)
    return list(shown.index)


# ============================================================================
# 6b. Universe + pre-filter (decides WHICH stocks get downloaded)
# ============================================================================
def base_universe(universe, my_text):
    if universe in MARKETS:
        spec = MARKETS[universe]
        try:
            meta = get_market_meta(universe)
        except Exception:
            meta = pd.DataFrame()
        if len(meta):
            return meta.index.tolist(), meta
        return [t + spec["suffix"] for t in MARKET_FALLBACK[spec["suffix"]].split()], None
    meta = get_screener_meta()
    if universe.startswith("S&P"):
        tickers, sp_meta = get_sp500()
        meta = sp_meta.combine_first(meta)
    elif universe == "My own tickers":
        tickers = parse_tickers(my_text)
    elif len(meta):
        tickers = meta.index.tolist()
        if universe.startswith("Nasdaq"):
            try:
                listed = set(get_nasdaq_listed())
                tickers = [t for t in tickers if t in listed]
            except Exception:
                pass
    else:
        tickers = get_nasdaq_listed() if universe.startswith("Nasdaq") else get_all_us()
    return tickers, meta


def prefilter(tickers, meta, o):
    if not o or meta is None or not len(meta):
        return tickers
    m = meta.reindex(tickers)
    mask = pd.Series(True, index=m.index)

    def rng(col, lo, hi):
        nonlocal mask
        if lo:
            mask &= m[col] >= lo
        if hi:
            mask &= m[col] <= hi

    rng("_last", o["pmin"], o["pmax"])
    rng("Mkt cap $B", o["cmin"], o["cmax"])
    rng("_vol", o["vmin"], None)
    if o["sectors"]:
        mask &= m["Sector"].isin(o["sectors"])
    if o["industries"]:
        mask &= m["Industry"].isin(o["industries"])
    if o["us_only"]:
        mask &= m["Country"] == "United States"
    return m.index[mask].tolist()


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
ss = st.session_state
for k, v in PATTERN_PRESETS["Default"].items():
    ss.setdefault(k, v)
M_RS, M_LL, M_EARN, M_VOL = "🧭 RS Score", "💧 Liquid Leaders", "📅 Earnings", "📊 Record Volume"
M_EP, M_PB, M_SP = "🚀 Episodic Pivot", "📉 Parabolic Short", "🏛 S&P Index Changes"
M_GL, M_TF, M_SO, M_XB = "✳️ Green Line Breakout", "🚩 Tight Flag Pattern", "📈 Shakeout +3", "🔁 EMA Crossback"
M_VCP, M_MAC, M_HTF = "🌀 VCP (Minervini)", "📏 MA Consolidation", "⛳ High Tight Flag"
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
# This keeps the buttons showing the right choice even after the page reloads during a scan.
def V(name, default=None):
    return ss.get("v_" + name, default)


def pwidget(fn, name, default, *args, on_change=None, **kw):
    vkey, wkey = "v_" + name, "w_" + name
    ss.setdefault(vkey, default)
    ss[wkey] = ss[vkey]

    def _sync():
        ss[vkey] = ss[wkey]
        if on_change:
            on_change()
    return fn(*args, key=wkey, on_change=_sync, **kw)


for key in F:
    ss.setdefault(f"v_f_{key}", DEFAULT_CHOICE.get(key, "Any"))
ss.setdefault("v_visible", [F[k]["label"] for k in DEFAULT_VISIBLE])


def load_pattern_preset(name):
    for k, v in PATTERN_PRESETS[name].items():
        ss[k] = v


VOLREC_OPTS = [p[0] for p in F["volrec"]["presets"]]


def is_record_mode():
    return ss.get("scan_mode") == M_VOL


def is_earn_mode():
    return ss.get("scan_mode") == M_EARN


def is_pb_mode():
    return ss.get("scan_mode") == M_PB


def is_ep_mode():
    return ss.get("scan_mode") == M_EP


def is_rs_mode():
    return ss.get("scan_mode") == M_RS


def rss_status_changed():
    ss["v_f_rsscore"] = ss["rss_status"]


def rs_quad_changed():
    ss["v_f_rsq"] = ss["rs_quad"]


def is_xb_mode():
    return ss.get("scan_mode") == M_XB


def xb_status_changed():
    ss["v_f_xback"] = ss["xb_status"]


def is_gl_mode():
    return ss.get("scan_mode") == M_GL


def gl_status_changed():
    ss["v_f_glb"] = ss["gl_status"]


def is_so_mode():
    return ss.get("scan_mode") == M_SO


def so_status_changed():
    ss["v_f_so3"] = ss["so_status"]


def is_htf_mode():
    return ss.get("scan_mode") == M_HTF


def htf_status_changed():
    ss["v_f_htf"] = ss["htf_status"]


def htf_load_preset(name):
    for k, v in HTF_PRESETS[name].items():
        ss[k] = v


def is_rc_mode():
    return ss.get("scan_mode") == M_RC


def rc_status_changed():
    ss["v_f_reclaim"] = ss["rc_status"]


def is_f5_mode():
    return ss.get("scan_mode") == M_F5


def f5_status_changed():
    ss["v_f_f50scan"] = ss["f5_status"]


def is_sh_mode():
    return ss.get("scan_mode") == M_SH


def sh_status_changed():
    ss["v_f_swshort"] = ss["sh_status"]


def is_mac_mode():
    return ss.get("scan_mode") == M_MAC


def mac_status_changed():
    ss["v_f_mac"] = ss["mac_status"]


def is_vcp_mode():
    return ss.get("scan_mode") == M_VCP


def vcp_status_changed():
    ss["v_f_vcp"] = ss["vcp_status"]


def is_ll_mode():
    return ss.get("scan_mode") == M_LL


def ll_status_changed():
    ss["v_f_liqlead"] = ss["ll_status"]


def is_sp_mode():
    return ss.get("scan_mode") == M_SP


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


def scan_mode_changed():
    """Switch between the scans: set the matching filter, filter button and table tab."""
    mode_keys = [v[0] for v in MODE_SPEC.values()]
    others = [x for x in ss.get("v_visible", []) if x not in [F[k]["label"] for k in mode_keys]]
    for k in mode_keys:
        ss[f"v_f_{k}"] = "Any"
    key, sel, tab = MODE_SPEC[ss["scan_mode"]]
    ss[f"v_f_{key}"] = ss[sel] if sel else "SETUP or TRIGGER"
    ss["v_tab"] = "Setup"
    ss["v_visible"] = others


def pb_status_changed():
    ss["v_f_parabolic"] = ss["pb_status"]


def ep_status_changed():
    ss["v_f_ep"] = ss["ep_status"]


def sp_type_changed():
    ss["v_f_spchg"] = ss["sp_type"]


def earn_window_changed():
    ss["v_f_earnwin"] = ss["earn_window"]


def vr_window_changed():
    ss["v_f_volrec"] = ss["vr_window"]


WATCH_SOURCES = ["Stock Scanner results", "My own watchlist (file / tickers)"]


def watchlist_tickers(text):
    """Tickers from typed text or a TradingView watchlist export ('NASDAQ:AAPL,NYSE:BRK.B,###Section,…')."""
    out = []
    for tok in re.split(r"[\s,;]+", text or ""):
        tok = tok.strip().strip('"').upper()
        if not tok or tok.startswith("#"):
            continue
        tok = tok.split(":")[-1].replace(".", "-")
        if re.fullmatch(r"[A-Z0-9\-]{1,10}", tok) and tok not in out:
            out.append(tok)
    return out


@st.cache_data(ttl=30 * 60, show_spinner=False)
def watchlist_panels(tickers):
    """Daily prices for a watchlist, in the same shape the scanner uses."""
    prices = {}
    for i in range(0, len(tickers), 150):
        chunk = list(tickers[i:i + 150])
        data = _fetch(chunk, threads=True)
        if data is not None:
            got, _ = _split(data, chunk)
            prices.update(got)
    return build_panels(prices) if prices else None


# ============================================================================
# 7a. Page 2 — Relative Strength Scanner (your RS Scanner app, running inside this app)
# ============================================================================
def rs_page():
    """Relative Strength Scanner: Week RS (x) vs Month RS (y), ranked 0-100.
    Data sources: live Yahoo prices, a TradingView screener CSV, or the stocks from your last Stock Scanner scan."""
    # ---------- Ticker lists ----------
    SECTORS = {
        "XLE": "Energy", "XLK": "Technology", "XLF": "Financial", "XLV": "Healthcare",
        "XLY": "Consumer Cyclical", "XLP": "Consumer Defensive", "XLI": "Industrials",
        "XLB": "Basic Materials", "XLU": "Utilities", "XLRE": "Real Estate",
        "XLC": "Communication Services",
    }
    LARGE_CAPS = """AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B JPM LLY V UNH XOM MA
    JNJ PG HD COST ABBV WMT NFLX CRM BAC ORCL CVX KO MRK AMD PEP ADBE TMO LIN ACN MCD
    CSCO WFC ABT GE DHR IBM QCOM CAT TXN INTU AMGN VZ PM ISRG NOW GS SPGI UBER CMCSA
    NEE RTX PFE HON AMAT UNP LOW T BKNG ELV AXP SYK PGR BLK COP MS TJX VRTX LMT ETN
    C BSX MDT SCHW ADP PLD REGN MU PANW CB DE MMC LRCX KLAC ANET SBUX BMY GILD ADI FI
    SO MO DUK SHW CME CRWD PLTR APP""".split()
    UNIVERSES = {
        "Sector ETFs (SPDR)": list(SECTORS),
        "Large-cap stocks (~100)": LARGE_CAPS,
        "My own list": [],
    }
    SOURCE_LIVE = "Live prices (Yahoo)"
    SOURCE_TV = "TradingView screener export (CSV)"
    SOURCE_MYSCAN = "Stocks from watchlist"


    # ---------- Shared helpers ----------
    def add_rank(df: pd.DataFrame, short_col: str, long_col: str) -> pd.DataFrame:
        """Percentile-rank two performance columns into Week RS / Month RS (0 = worst, 100 = best)."""
        df = df.dropna(subset=[short_col, long_col]).copy()
        n = len(df)
        df["Week RS"] = (df[short_col].rank() - 1) / max(n - 1, 1) * 100
        df["Month RS"] = (df[long_col].rank() - 1) / max(n - 1, 1) * 100
        strong_w, strong_m = df["Week RS"] >= 50, df["Month RS"] >= 50
        df["Quadrant"] = "Weak"
        df.loc[strong_w & strong_m, "Quadrant"] = "Strong"
        df.loc[~strong_w & strong_m, "Quadrant"] = "Weakening"
        df.loc[strong_w & ~strong_m, "Quadrant"] = "Improving"
        df["Score"] = (df["Week RS"] + df["Month RS"]) / 2
        return df.sort_values("Score", ascending=False).round(1)


    def make_chart(df: pd.DataFrame, label_col: str, color_col: str | None, hover_cols: list) -> go.Figure:
        show_text = len(df) <= 80
        fig = px.scatter(
            df, x="Week RS", y="Month RS",
            color=color_col if color_col else "Score",
            color_continuous_scale="RdYlGn" if not color_col else None,
            range_color=(0, 100) if not color_col else None,
            text=label_col if show_text else None,
            hover_name=label_col, hover_data={c: True for c in hover_cols},
        )
        fig.update_traces(marker=dict(size=10 if show_text else 7), textposition="top center", textfont_size=10)
        shades = [(50, 100, 50, 100, "rgba(40,160,90,0.10)", "STRONG"),
                  (0, 50, 50, 100, "rgba(200,150,40,0.08)", "WEAKENING"),
                  (0, 50, 0, 50, "rgba(200,60,60,0.08)", "WEAK"),
                  (50, 100, 0, 50, "rgba(60,110,220,0.10)", "IMPROVING")]
        for x0, x1, y0, y1, color, label in shades:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")
            fig.add_annotation(x=(x0 + x1) / 2, y=y1 - 3, text=label, showarrow=False, font=dict(color="gray", size=11))
        fig.add_vline(x=50, line_dash="dash", line_color="gray")
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.update_layout(height=750, template="plotly_dark",
                          xaxis=dict(title="Week RS", range=[-3, 103]),
                          yaxis=dict(title="Month RS", range=[-3, 103]),
                          legend=dict(orientation="h", y=-0.12, title=None),
                          coloraxis_showscale=False,
                          margin=dict(l=40, r=20, t=30, b=40))
        return fig


    # ---------- Sidebar ----------
    st.sidebar.title("🧭 Relative strength")
    source = st.sidebar.radio("Data source", [SOURCE_LIVE, SOURCE_TV, SOURCE_MYSCAN], key="rsp_source")
    st.title("Relative Strength Scanner")


    # =====================================================================
    # 1) LIVE PRICES (Yahoo)
    # =====================================================================
    @st.cache_data(ttl=60 * 30, show_spinner="Downloading prices...")
    def rsp_get_prices(symbols: tuple) -> pd.DataFrame:
        try:
            df = yf.download(list(symbols), period="1y", auto_adjust=True, progress=False)["Close"]
        except Exception:
            return pd.DataFrame()
        if isinstance(df, pd.Series):
            df = df.to_frame(symbols[0])
        return df.dropna(how="all")


    def run_live():
        universe = st.sidebar.selectbox("What to scan", list(UNIVERSES), key="rsp_universe")
        if universe == "My own list":
            raw = st.sidebar.text_area("Tickers (spaces or commas)", key="rsp_tickers")
            tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
        else:
            tickers = UNIVERSES[universe]
        benchmark = (st.sidebar.text_input("Benchmark", key="rsp_bench") or "SPY").strip().upper()
        short_days = int(st.sidebar.number_input("Short lookback (trading days) - X axis", 2, 60, key="rsp_short"))
        long_days = int(st.sidebar.number_input("Long lookback (trading days) - Y axis", 5, 250, key="rsp_long"))
        st.sidebar.caption("5 days = 1 week, 21 days = 1 month, 63 days = 3 months.")

        st.caption(f"Each dot = one ticker. Right = stronger than peers over {short_days} days, "
                   f"up = stronger over {long_days} days (both after subtracting {benchmark}).")
        if not tickers:
            st.info("Add some tickers in the sidebar.")
            st.stop()

        prices = rsp_get_prices(tuple(sorted(set(tickers + [benchmark]))))
        if benchmark not in prices.columns or prices[benchmark].dropna().empty:
            st.error(f"Couldn't load the benchmark {benchmark}. Check the ticker and try again.")
            st.stop()
        prices = prices.dropna(axis=1, thresh=long_days + 2).ffill()
        missing = set(tickers) - set(prices.columns)
        if missing:
            st.warning("No data for: " + ", ".join(sorted(missing)))
        if len(prices) <= long_days + 1:
            st.error("Not enough price history for that lookback.")
            st.stop()

        ret_s = prices.iloc[-1] / prices.iloc[-1 - short_days] - 1
        ret_l = prices.iloc[-1] / prices.iloc[-1 - long_days] - 1
        df = pd.DataFrame({
            "Short vs bench %": (ret_s - ret_s[benchmark]) * 100,
            "Long vs bench %": (ret_l - ret_l[benchmark]) * 100,
        }).drop(index=benchmark)
        df.index.name = "Ticker"
        df = add_rank(df, "Short vs bench %", "Long vs bench %").reset_index()
        df["Name"] = df["Ticker"].map(SECTORS).fillna(df["Ticker"]) if universe.startswith("Sector") else df["Ticker"]

        st.caption(f"Prices as of {prices.index[-1]:%b %d, %Y}")
        st.plotly_chart(make_chart(df, "Name", None, ["Short vs bench %", "Long vs bench %"]), use_container_width=True)
        st.subheader("Leaders (Strong quadrant)")
        st.dataframe(df[df["Quadrant"] == "Strong"], use_container_width=True, hide_index=True)
        with st.expander("All tickers"):
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", df.to_csv(index=False), "rs_scan.csv", "text/csv")


    # =====================================================================
    # 2) TRADINGVIEW SCREENER EXPORT (CSV)
    # =====================================================================
    # Stage colors follow the "Cycle of Price Action" chart
    STAGES = {
        "1A": ("Upward Pivot", "#F0A36B"), "1B": ("Mean Reversion", "#F5E663"),
        "2A": ("Bullish Trend", "#A8E6A1"), "2B": ("Breakout Confirm", "#3CC47C"),
        "2C": ("Extended Bullish", "#FFFFFF"), "3A": ("Bullish Fade", "#8FD3F4"),
        "3B": ("Fade Confirmation", "#4A90D9"), "4A": ("Bearish Trend", "#F7A8A8"),
        "4B": ("Breakdown Confirm", "#E74C4C"), "4C": ("Extended Bearish", "#F07BF0"),
        "?": ("Unknown", "#777777"),
    }
    STAGE_ORDER = list(STAGES)
    PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

    # (key, regex patterns, required?)
    COLUMNS = [
        ("Symbol", [r"^symbol$", r"^ticker$", r"symbol", r"ticker"], True),
        ("Week %", [r"perf.*1\s*w", r"1\s*week", r"perf.*week"], True),
        ("Month %", [r"perf.*1\s*m(?!o)", r"perf.*1\s*month", r"1\s*month"], True),
        ("Day %", [r"^change\s*%$", r"^change\s*%\s*(1\s*d|1d|day)?$", r"^chg\s*%"], False),
        ("Quarter %", [r"perf.*3\s*m(?!o)", r"perf.*3\s*month", r"3\s*month"], False),
        ("6 Month %", [r"perf.*6\s*m(?!o)", r"perf.*6\s*month", r"6\s*month"], False),
        ("Year %", [r"perf.*1\s*y", r"1\s*year", r"perf.*52\s*w"], False),
        ("Sector", [r"^sector$", r"sector"], False),
        ("Industry", [r"^industry$", r"industry"], False),
        ("Market cap", [r"^market\s*cap(italization)?$", r"market\s*cap(?!.*currency)"], False),
        ("Price", [r"^price$", r"^close$", r"^last$", r"^price(?!.*currency)"], False),
        ("EMA 10", [r"exponential moving average\s*\(10\)", r"\bema\s*\(?10\b"], False),
        ("EMA 20", [r"exponential moving average\s*\(20\)", r"\bema\s*\(?20\b", r"exponential moving average\s*\(21\)", r"\bema\s*\(?21\b"], False),
        ("SMA 50", [r"simple moving average\s*\(50\)", r"\bsma\s*\(?50\b"], False),
        ("SMA 200", [r"simple moving average\s*\(200\)", r"\bsma\s*\(?200\b"], False),
        ("ATR", [r"average true range", r"\batr\b"], False),
    ]
    TIMEFRAMES = [("DAY", "Day %"), ("WK", "Week %"), ("MTH", "Month %"), ("QTR", "Quarter %"),
                  ("6M", "6 Month %"), ("1Y", "Year %")]


    def guess_col(cols, patterns):
        for p in patterns:
            for c in cols:
                if re.search(p, str(c), re.IGNORECASE):
                    return c
        return None


    def to_number(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace("−", "-").str.replace(r"[%,\s$]", "", regex=True),
                             errors="coerce")


    def pct_rank(s: pd.Series) -> pd.Series:
        n = s.notna().sum()
        return ((s.rank() - 1) / max(n - 1, 1) * 100).round(0)


    def classify_stage(r) -> str:
        """Approximate stage (1A-4C) from price vs. 10/20 EMAs and 50/200 SMAs.
        A simplified take on the Cycle of Price Action, not Steve Jacobs' exact rules."""
        p, e10, e20, s50, s200 = r["Price"], r["EMA 10"], r["EMA 20"], r["SMA 50"], r["SMA 200"]
        if any(pd.isna(v) for v in (p, e10, e20, s50)):
            return "?"
        s200 = s50 if pd.isna(s200) else s200
        atr = r.get("ATR", float("nan"))
        ext = (p - s50) / atr if pd.notna(atr) and atr > 0 else (p / s50 - 1) * 100 / 3.5  # ~25% = 7 "ATRs"
        bull_short = p > e10 and e10 > e20
        bear_short = p < e10 and e10 < e20
        if bull_short:
            if ext >= 7:
                return "2C"
            if e20 > s50 > s200 and p > s50:
                return "2B"
            return "2A" if p > s50 else "1B"
        if bear_short:
            if ext <= -7:
                return "4C"
            if e20 < s50 < s200 and p < s50:
                return "4B"
            return "4A" if p < s50 else "3B"
        if p > e10:                      # price turning up through falling EMAs
            return "2A" if p > s50 else "1A"
        return "3A" if p > s50 else "4A"  # price slipping under rising EMAs


    def rank_color(v) -> str:
        if pd.isna(v):
            return "transparent"
        if v >= 50:
            a = 0.15 + (v - 50) / 50 * 0.55
            return f"rgba(46,160,90,{a:.2f})"
        a = 0.15 + (50 - v) / 50 * 0.55
        return f"rgba(210,60,60,{a:.2f})"


    def chg_html(v) -> str:
        if pd.isna(v):
            return ""
        c = "#4cd97b" if v >= 0 else "#ff6b6b"
        return f'<span style="color:{c};font-size:10px;margin-left:4px">{v:+.1f}%</span>'


    CSS = """<style>
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


    def html(s: str):
        """Render raw HTML (no leading spaces so Markdown doesn't turn it into a code block)."""
        st.markdown(CSS + s, unsafe_allow_html=True)


    def score_circle(v) -> str:
        c = "#3CC47C" if v >= 70 else "#F5C542" if v >= 50 else "#E74C4C"
        return f'<span class="circ" style="border-color:{c};color:{c}">{v:.0f}</span>'


    def stage_badge(sg) -> str:
        c = STAGES.get(sg, STAGES["?"])[1]
        return f'<span class="badge" style="border-color:{c};color:{c}">{sg}</span>'


    def load_tradingview():
        st.caption("Scan thousands of stocks using TradingView's own data, sectors and industries.")
        with st.expander("How to export from TradingView (do this once a day)", expanded=False):
            st.markdown(
                "1. Open **Screener → Stocks** in TradingView and load your saved screen "
                "(for example: USA, market cap above 1B, no REITs).\n"
                "2. Add these **columns** (column settings button on the right of the table):\n"
                "   - Needed: **Performance % 1 week**, **Performance % 1 month**, **Sector**, **Industry**\n"
                "   - For the table: **Change %**, **Performance % 3 months**, **6 months**, **1 year**, **Market capitalization**\n"
                "   - For stages: **Price**, **Exponential Moving Average (10)**, **Exponential Moving Average (20)**, "
                "**Simple Moving Average (50)**, **Simple Moving Average (200)**, and optionally **Average True Range (14)**\n"
                "3. Click the **export / download** button above the table to save a CSV file.\n"
                "4. Drag that file into the box below.\n\n"
                "Tip: save the screen with these columns once, so tomorrow it's just open → export → upload."
            )
        up = st.file_uploader("TradingView screener CSV", type=["csv"], key="rsp_upload")
        if up is not None:
            ss["rsp_csv"], ss["rsp_csv_name"] = up.getvalue(), up.name
        if not ss.get("rsp_csv"):
            st.info("Upload a TradingView screener export to begin.")
            st.stop()
        if up is None:
            st.caption(f"Using your earlier upload: **{ss.get('rsp_csv_name', 'screener.csv')}** — "
                       "upload a new file to replace it.")

        raw = pd.read_csv(io.BytesIO(ss["rsp_csv"]))
        cols = list(raw.columns)
        guesses = {k: guess_col(cols, pats) for k, pats, _ in COLUMNS}
        required = [k for k, _, req in COLUMNS if req]
        with st.sidebar.expander("Column matching", expanded=any(guesses[k] is None for k in required)):
            chosen = {}
            for key, _, _ in COLUMNS:
                opts = ["(none)"] + cols
                chosen[key] = st.selectbox(key, opts, index=opts.index(guesses[key]) if guesses[key] in cols else 0)
        for key in required:
            if chosen[key] == "(none)":
                st.error(f"Couldn't find the **{key}** column. Pick it under *Column matching* in the sidebar, "
                         "or add it to your TradingView screener and export again.")
                st.stop()

        df = pd.DataFrame({"Ticker": raw[chosen["Symbol"]].astype(str).str.split(":").str[-1]})
        for key, _, _ in COLUMNS[1:]:
            if key in ("Sector", "Industry"):
                df[key] = raw[chosen[key]].fillna("Unknown").astype(str) if chosen[key] != "(none)" else "Unknown"
            else:
                df[key] = to_number(raw[chosen[key]]) if chosen[key] != "(none)" else float("nan")
        df = df.dropna(subset=["Week %", "Month %"]).drop_duplicates("Ticker").reset_index(drop=True)
        df["Market cap ($B)"] = (df["Market cap"] / 1e9).round(1)
        df["Stage"] = df.apply(classify_stage, axis=1)
        return df


    def run_tradingview():
        df = load_tradingview()
        has_groups = (df["Industry"] != "Unknown").any()
        has_stage = (df["Stage"] != "?").any()

        # --- stock-level ranks for every timeframe ---
        stocks = add_rank(df, "Week %", "Month %")
        for tf, col in TIMEFRAMES:
            stocks[tf] = pct_rank(stocks[col])
        comp_cols = [tf for tf, col in TIMEFRAMES[1:] if stocks[col].notna().any()]
        stocks["COMP"] = stocks[comp_cols].mean(axis=1).round(0)

        sectors_sorted = sorted(df["Sector"].unique())
        sector_color = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sectors_sorted)}

        # --- sidebar ---
        st.sidebar.markdown("---")
        views = (["Leading groups - table", "Leading groups - cards", "Industries (quadrant)", "Sectors (quadrant)"]
                 if has_groups else [])
        if has_stage:
            views.append("Stage analysis")
        views.append("Stocks (quadrant)")
        if ss.get("rsp_tvview") not in views:
            ss["rsp_tvview"] = views[0]
        view = st.sidebar.radio("Show", views, key="rsp_tvview")
        min_stocks = st.sidebar.slider("Min stocks per group", 1, 20, 3) if has_groups else 1
        st.sidebar.caption("Group strength = median performance of its stocks, ranked 0-100 against other groups. "
                           "Score = average of Week RS and Month RS. COMP = average of all timeframes.")
        if has_stage:
            stage_pick = st.sidebar.multiselect("Stages to include", STAGE_ORDER[:-1], default=STAGE_ORDER[:-1])
        else:
            stage_pick = STAGE_ORDER

        def groups(by):
            agg = {"Stocks": ("Ticker", "size")}
            for tf, col in TIMEFRAMES:
                agg[col] = (col, "median")
            g = df.groupby(by).agg(**agg)
            if by == "Industry":
                g["Sector"] = df.groupby("Industry")["Sector"].agg(lambda s: s.mode().iat[0])
            g = g[g["Stocks"] >= min_stocks].reset_index()
            g = add_rank(g, "Week %", "Month %")
            for tf, col in TIMEFRAMES:
                g[tf] = pct_rank(g[col])
            g["COMP"] = g[comp_cols].mean(axis=1).round(0)
            return g

        st.caption(f"{len(df):,} stocks loaded from your TradingView export."
                   + ("" if has_stage else "  Add the Price and moving-average columns to your export to see stages."))
        in_stage = stocks[stocks["Stage"].isin(stage_pick)]

        def stock_list(industry, limit, order):
            s = in_stage[in_stage["Industry"] == industry]
            if order == "Largest market cap first" and s["Market cap"].notna().any():
                s = s.sort_values("Market cap", ascending=False)
            else:
                s = s.sort_values("Score", ascending=False)
            return s.head(limit)

        result = stocks

        # ---------------- Leading groups: table ----------------
        if view == "Leading groups - table":
            ind = groups("Industry")
            c1, c2, c3 = st.columns(3)
            n_groups = c1.slider("Industries to show", 5, 60, 20)
            n_chips = c2.slider("Stocks per industry", 5, 40, 25)
            order = c3.selectbox("Stock order", ["Largest market cap first", "Strongest first"])
            sort_by = st.radio("Rank industries by", ["Score", "COMP"], horizontal=True)
            ind = ind.sort_values(sort_by, ascending=False).head(n_groups)
            tfs = [tf for tf, col in TIMEFRAMES if ind[col].notna().any()]
            head = "".join(f"<th>{t}</th>" for t in ["INDUSTRY", "# STK"] + tfs + ["COMP", "SCORE"])
            head += '<th style="text-align:left">STOCKS</th>'
            rows = []
            for _, r in ind.iterrows():
                cells = f'<td class="name" style="border-left:3px solid {sector_color[r["Sector"]]}">{r["Industry"]}</td>'
                cells += f'<td>{int(r["Stocks"])}</td>'
                for tf in tfs + ["COMP", "Score"]:
                    cells += f'<td style="background:{rank_color(r[tf])}">{"" if pd.isna(r[tf]) else int(r[tf])}</td>'
                chips = "".join(
                    f'<span class="chip" style="border-left-color:{STAGES[s["Stage"]][1]}" title="Stage {s["Stage"]} · Score {s["Score"]:.0f}">'
                    f'{s["Ticker"]}{chg_html(s["Day %"])}</span>'
                    for _, s in stock_list(r["Industry"], n_chips, order).iterrows()
                )
                cells += f'<td class="chips">{chips}</td>'
                rows.append(f"<tr>{cells}</tr>")
            html(f'<div class="rs-wrap"><table class="rs"><tr>{head}</tr>{"".join(rows)}</table></div>')
            st.caption("Chip stripe color = stock stage (see Stage analysis). Numbers are 0-100 ranks vs. other industries.")
            result = ind

        # ---------------- Leading groups: cards ----------------
        elif view == "Leading groups - cards":
            ind = groups("Industry")
            c1, c2, c3 = st.columns(3)
            n_groups = c1.slider("Industries to show", 4, 48, 12)
            n_rows = c2.slider("Stocks per card", 3, 30, 15)
            order = c3.selectbox("Stock order", ["Largest market cap first", "Strongest first"])
            ind = ind.sort_values("Score", ascending=False).head(n_groups)
            cards = []
            for _, r in ind.iterrows():
                sl = stock_list(r["Industry"], n_rows, order)
                body = "".join(
                    f'<div class="row" style="border-left-color:{STAGES[s["Stage"]][1]}">'
                    f'<span><b>{s["Ticker"]}</b>{chg_html(s["Day %"])}</span>'
                    f'<span>{stage_badge(s["Stage"])}{score_circle(s["Score"])}</span></div>'
                    for _, s in sl.iterrows()
                )
                cards.append(
                    f'<div class="card"><div class="hd" style="background:{sector_color[r["Sector"]]}">'
                    f'<span>{r["Industry"]}</span><span>{r["Score"]:.0f}</span></div>'
                    f'<div class="sub"><span>Wk {r["Week RS"]:.0f} · Mth {r["Month RS"]:.0f}</span>'
                    f'<span>{len(sl)}/{int(r["Stocks"])}</span></div>{body}</div>'
                )
            html(f'<div class="cards">{"".join(cards)}</div>')
            st.caption("Header color = sector. Badge = stage. Circle = the stock's own RS score (0-100).")
            result = ind

        # ---------------- Stage analysis ----------------
        elif view == "Stage analysis":
            scope = st.radio("Stocks", ["All stocks", "Only top-half industries"], horizontal=True)
            pool = stocks
            if scope == "Only top-half industries" and has_groups:
                ind = groups("Industry")
                pool = pool[pool["Industry"].isin(set(ind.loc[ind["Score"] >= 50, "Industry"]))]
            pool = pool[pool["Stage"] != "?"]
            counts = pool["Stage"].value_counts().reindex(STAGE_ORDER[:-1], fill_value=0)
            bull = counts[["1A", "1B", "2A", "2B", "2C"]].sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("Stocks", f"{len(pool):,}")
            m2.metric("Bullish stages (1A-2C)", f"{bull / max(len(pool), 1):.0%}")
            m3.metric("Watchable (1A-2B)", f'{counts[["1A", "1B", "2A", "2B"]].sum():,}')
            fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[STAGES[s][1] for s in counts.index],
                                   text=counts.values, textposition="outside", cliponaxis=False))
            fig.update_layout(height=260, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20),
                              yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True)
            per_col = st.slider("Stocks listed per stage", 5, 100, 25)
            order = st.selectbox("Order within each stage", ["Strongest first", "Largest market cap first"])
            cols_html = []
            for sg in STAGE_ORDER[:-1]:
                s = pool[pool["Stage"] == sg]
                s = (s.sort_values("Market cap", ascending=False) if order.startswith("Largest")
                     else s.sort_values("Score", ascending=False)).head(per_col)
                name, colr = STAGES[sg]
                rows = "".join(
                    f'<div class="row" style="border-left-color:{sector_color[x["Sector"]]}">'
                    f'<span><b>{x["Ticker"]}</b>{chg_html(x["Day %"])}</span>{score_circle(x["Score"])}</div>'
                    for _, x in s.iterrows()
                )
                cols_html.append(f'<div class="stagecol"><div class="hd" style="background:{colr}">'
                                 f'{counts[sg]}<br><span style="font-size:10px">{name} · {sg}</span></div>{rows}</div>')
            html('<div class="rs-wrap"><div style="display:grid;grid-template-columns:repeat(10,minmax(120px,1fr));gap:6px;font-size:12px;color:#ddd">'
                 + "".join(cols_html) + "</div></div>")
            st.caption("Stages are estimated from price vs. the 10/20-day EMAs and 50/200-day SMAs "
                       "(a simplified version of the Cycle of Price Action). Stripe color = sector.")
            result = pool

        # ---------------- Quadrant views ----------------
        elif view == "Industries (quadrant)":
            ind = groups("Industry")
            st.plotly_chart(make_chart(ind, "Industry", "Sector", ["Stocks", "Week %", "Month %"]), use_container_width=True)
            st.dataframe(ind, use_container_width=True, hide_index=True)
            result = ind

        elif view == "Sectors (quadrant)":
            sec = groups("Sector")
            st.plotly_chart(make_chart(sec, "Sector", "Sector", ["Stocks", "Week %", "Month %"]), use_container_width=True)
            st.dataframe(sec, use_container_width=True, hide_index=True)
            result = sec

        else:  # Stocks (quadrant)
            shown = in_stage
            if has_groups:
                ind = groups("Industry")
                only_top = st.sidebar.checkbox("Only stocks in top-half industries (\"one rule\")", value=True)
                only_strong = st.sidebar.checkbox("Only stocks in the Strong quadrant", value=False)
                if only_top:
                    keep = set(ind.loc[ind["Score"] >= 50, "Industry"])
                    shown = shown[shown["Industry"].isin(keep)].merge(
                        ind[["Industry", "Score"]].rename(columns={"Score": "Industry score"}), on="Industry", how="left")
                if only_strong:
                    shown = shown[shown["Quadrant"] == "Strong"]
            st.caption(f"Showing {len(shown):,} stocks. Ranks are against all {len(stocks):,} stocks in your export.")
            hover = [c for c in ["Industry", "Stage", "Week %", "Month %", "Market cap ($B)"] if c in shown.columns]
            st.plotly_chart(make_chart(shown, "Ticker", "Sector" if has_groups else None, hover), use_container_width=True)
            st.dataframe(shown, use_container_width=True, hide_index=True)
            result = shown

        st.download_button("Download results (CSV)", result.to_csv(index=False), "rs_scan.csv", "text/csv")



    # ---------------- 3) stocks from the last Stock Scanner scan ----------------
    def run_myscan():
        """Stocks from a watchlist: the Stock Scanner's current results, or your own list / TradingView file."""
        wl = st.sidebar.radio("Watchlist", WATCH_SOURCES, key="rsp_wsrc",
                              help="Stock Scanner results = the stocks in the scanner's results table right now "
                                   "(the list its 'TradingView watchlist' button exports).")
        if wl == WATCH_SOURCES[0]:
            syms = [t for t in ss.get("last_results", []) if "panels" in ss and t in ss["panels"]["Close"].columns]
            if not syms:
                st.info("No scanner results yet. Run a scan on the **📈 Stock Scanner** page — the stocks in its "
                        "results table show up here.")
                st.stop()
            panels_src, meta_src = ss["panels"], ss.get("meta")
            src_note = (f"{len(syms):,} stocks from your Stock Scanner results · {ss.get('last_results_label', '')} "
                        f"(prices from {ss.get('downloaded_at', '')})")
        else:
            up = st.sidebar.file_uploader("Upload a watchlist (.txt / .csv)", type=["txt", "csv"], key="rsp_wupload",
                                          help="A TradingView watchlist export works as-is (NASDAQ:AAPL,NYSE:BRK.B,…).")
            if up is not None:
                ss["rsp_wfile"] = up.getvalue().decode("utf-8", errors="ignore")
                ss["rsp_wfile_name"] = up.name
            st.sidebar.text_area("…or type tickers", key="rsp_wtext", placeholder="NVDA, AMD, PLTR", height=90)
            if ss.get("rsp_wfile") and up is None:
                st.sidebar.caption(f"Using **{ss.get('rsp_wfile_name', 'watchlist')}** — upload another to replace it.")
            syms = watchlist_tickers((ss.get("rsp_wfile") or "") + "\n" + (ss.get("rsp_wtext") or ""))
            if not syms:
                st.info("Upload a watchlist file or type some tickers in the sidebar.")
                st.stop()
            with st.spinner(f"Downloading prices for {len(syms)} stocks …"):
                panels_src = watchlist_panels(tuple(syms))
            if panels_src is None:
                st.error("Couldn't download prices for these tickers. Check the symbols and your connection.")
                st.stop()
            missing = [t for t in syms if t not in panels_src["Close"].columns]
            syms = [t for t in syms if t in panels_src["Close"].columns]
            if missing:
                st.warning("No price data for: " + ", ".join(missing[:40]))
            meta_src = get_screener_meta()
            src_note = f"{len(syms):,} stocks from your watchlist"
        if ss.get("rsp_view") not in RS_VIEWS[:-1]:
            ss["rsp_view"] = RS_VIEWS[0]
        view = st.sidebar.radio("Show", RS_VIEWS[:-1], key="rsp_view")
        bench = (st.sidebar.text_input("Benchmark", key="rsp_mbench") or "SPY").strip().upper()
        c1, c2 = st.sidebar.columns(2)
        sd = int(c1.number_input("Short (days) · X", 2, 60, key="rsp_mshort"))
        ld = int(c2.number_input("Long (days) · Y", 5, 250, key="rsp_mlong"))
        st.sidebar.caption("5 days = 1 week, 21 days = 1 month, 63 days = 3 months.")
        min_stocks = st.sidebar.slider("Min stocks per group", 1, 20, key="rsp_mmin")
        stages = st.sidebar.multiselect("Stages to include", RS_STAGE_ORDER[:-1], key="rsp_mstages",
                                        help=" · ".join(f"{sg} = {RS_STAGES[sg][0]}" for sg in RS_STAGE_ORDER[:-1]))
        top_only = strong_only = False
        if view == "Stocks (quadrant)":
            top_only = st.sidebar.checkbox('Only stocks in top-half industries ("one rule")', key="rsp_mtop")
            strong_only = st.sidebar.checkbox("Only stocks in the Strong quadrant", key="rsp_mstrong")
        st.sidebar.caption("Group strength = median performance of its stocks, ranked 0-100 against other groups. "
                           "Score = average of Week RS and Month RS. COMP = average of all timeframes.")
        key = (wl, ss.get("data_key") if wl == WATCH_SOURCES[0] else None, tuple(syms), bench, sd, ld)
        if ss.get("rsp_key") != key:
            with st.spinner("Ranking relative strength …"):
                pnl = {k: v[syms] for k, v in panels_src.items()}
                tbl = rs_stock_table(pnl, bench, sd, ld)
                meta = meta_src
                for c in ["Name", "Sector", "Industry", "Mkt cap $B"]:
                    tbl[c] = meta[c].reindex(tbl.index) if meta is not None and c in meta else np.nan
                ss["rsp_table"], ss["rsp_key"] = tbl, key
        stocks = ss["rsp_table"]
        st.caption(f"{src_note} · returns vs {bench}. Ranks are against the other stocks in this watchlist.")
        shown = render_rs_view(stocks, view, dict(stages=stages or RS_STAGE_ORDER[:-1], min_stocks=min_stocks,
                                                  top_only=top_only, strong_only=strong_only, bench=bench,
                                                  short=sd, long=ld, total=len(stocks)))
        out = stocks.loc[[t for t in shown if t in stocks.index]] if view in (
            "Stocks (quadrant)", "Leading groups - table", "Leading groups - cards", "Stage analysis") else stocks
        cols = [c for c in ["Name", "Quadrant", "Score", "Week RS", "Month RS", "COMP", "Stage", "Short vs bench %",
                            "Long vs bench %", "Day %", "Industry", "Sector", "Mkt cap $B"] if c in out]
        st.dataframe(out[cols].sort_values("Score", ascending=False), use_container_width=True)
        st.download_button("Download results (CSV)", out[cols].to_csv(), "rs_scan.csv", "text/csv")

    if source == SOURCE_LIVE:
        run_live()
    elif source == SOURCE_TV:
        run_tradingview()
    else:
        run_myscan()


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


def peer_pool(target, meta, pool_by, min_cap, max_n, stocks_only):
    """Candidate peers: same industry / sector (or the biggest stocks), biggest companies first."""
    m = meta[meta["Mkt cap $B"] >= min_cap] if meta is not None and len(meta) else pd.DataFrame()
    note = ""
    if target in getattr(meta, "index", []) and len(m):
        row = meta.loc[target]
        if pool_by == "Same industry" and pd.notna(row["Industry"]):
            m, note = m[m["Industry"] == row["Industry"]], f"industry: {row['Industry']}"
        elif pool_by in ("Same industry", "Same sector") and pd.notna(row["Sector"]):
            m, note = m[m["Sector"] == row["Sector"]], f"sector: {row['Sector']}"
        else:
            note = "all stocks"
    else:
        note = "all stocks (sector unknown)"
    pool = list(m.sort_values("Mkt cap $B", ascending=False).index[:max_n]) if len(m) else []
    if not stocks_only:
        pool += PEER_ETFS
    return [t for t in dict.fromkeys([target] + pool)], note


@st.cache_data(ttl=60, show_spinner=False)
def peer_intraday(tickers):
    """Today's 1-minute bars (last price, open, price 10 minutes ago, bar time) for a few tickers."""
    try:
        d = yf.download(list(tickers), period="1d", interval="1m", group_by="ticker", auto_adjust=False,
                        prepost=False, progress=False, threads=True)
    except Exception:
        return {}
    out = {}
    for t in tickers:
        try:
            x = (d[t] if isinstance(d.columns, pd.MultiIndex) else d).dropna(subset=["Close"])
            if len(x) < 2:
                continue
            j = max(0, len(x) - 11)
            out[t] = dict(last=float(x["Close"].iloc[-1]), open=float(x["Open"].iloc[0]),
                          ago10=float(x["Close"].iloc[j]), day=pd.Timestamp(x.index[-1]).date(), ts=x.index[-1])
        except Exception:
            continue
    return out


def peer_table(target, panels, lookback, n_peers, min_sim):
    C, O = panels["Close"], panels["Open"]
    rets = C.pct_change().iloc[-lookback:]
    if target not in rets:
        return None
    sim = rets.corrwith(rets[target]).drop(target, errors="ignore").dropna()
    sim = sim[sim >= min_sim].sort_values(ascending=False).head(n_peers)
    # IBD-style RS rating, ranked 1-99 against the whole pool
    ago = lambda k: C.iloc[-1 - k] if len(C) > k else pd.Series(np.nan, index=C.columns)
    r = lambda k: C.iloc[-1] / ago(k) - 1
    raw = (0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)).fillna(0.4 * r(63) + 0.6 * r(126))
    rs = (raw.rank(pct=True) * 98 + 1).round()
    stage = rs_stock_table({k: v[[target] + list(sim.index)] for k, v in panels.items()}, "SPY", 5, 21)["Stage"]
    rows = [target] + list(sim.index)
    df = pd.DataFrame(index=rows)
    df["Stage"] = stage.reindex(rows)
    df["RS"] = rs.reindex(rows)
    df["Sim."] = [np.nan] + list(sim.values)
    # today's moves: intraday if available, otherwise the latest daily bar
    intra = peer_intraday(tuple(rows))
    last_day = C.index[-1].date()
    for t in rows:
        i = intra.get(t)
        daily_prev = C[t].iloc[-2] if len(C) > 1 else np.nan
        if i:
            prev = C[t][C.index.date < i["day"]].iloc[-1] if (C.index.date < i["day"]).any() else daily_prev
            df.at[t, "CoY"] = (i["last"] / prev - 1) * 100
            df.at[t, "Open"] = (i["open"] / prev - 1) * 100
            df.at[t, "10 Min"] = (i["last"] / i["ago10"] - 1) * 100
        else:
            df.at[t, "CoY"] = (C[t].iloc[-1] / daily_prev - 1) * 100
            df.at[t, "Open"] = (O[t].iloc[-1] / daily_prev - 1) * 100
            df.at[t, "10 Min"] = np.nan
    stamp = max((v["ts"] for v in intra.values()), default=None)
    return df, stamp, last_day


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


def _pct_pill(v):
    if pd.isna(v):
        return '<span class="pill n">–</span>'
    return f'<span class="pill {"g" if v >= 0 else "r"}">{v:+.2f}%</span>'


def peers_page():
    st.sidebar.title("🔗 Similar stocks")
    pool_by = st.sidebar.radio("Look for peers in", ["Same sector", "Same industry", "All stocks"], key="pp_pool",
                               help="Candidates are then ranked by how closely their daily moves track your stock.")
    n_peers = st.sidebar.slider("Peers to show", 3, 30, key="pp_n")
    lookback = st.sidebar.select_slider("Similarity measured over", [20, 40, 60, 90, 120, 180, 250], key="pp_look",
                                        format_func=lambda d: f"{d} trading days")
    min_sim = st.sidebar.slider("Minimum similarity", 0.0, 0.9, step=0.05, key="pp_minsim")
    min_cap = st.sidebar.number_input("Candidates: market cap at least $B", 0.0, 500.0, step=0.5, key="pp_mincap")
    max_pool = st.sidebar.slider("Candidates to compare", 50, 400, step=25, key="pp_pool_n",
                                 help="More candidates = better peers but a slower first search (then it's cached).")
    st.sidebar.caption("**Sim.** = correlation of daily returns (1 = moves exactly alike). **RS** = IBD-style rating "
                       "1-99 vs. all candidates. **CoY** = change vs. yesterday's close · **Open** = today's opening "
                       "gap · **10 Min** = change over the last 10 minutes.")

    st.title("Similar stocks")
    c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
    q = c1.text_input("Stock", key="pp_ticker", placeholder="Type a ticker, e.g. PANW — then press Enter")
    stocks_only = c2.toggle("Stocks only", key="pp_stocks_only", help="Off = also compare against popular ETFs")
    target = watchlist_tickers(q)[0] if watchlist_tickers(q) else ""
    if not target:
        st.info("Type a ticker above to see the stocks that trade most like it — with their stage, relative "
                "strength and how they're moving today.")
        st.stop()

    meta = get_screener_meta()
    pool, note = peer_pool(target, meta, pool_by, float(min_cap), int(max_pool), stocks_only)
    with st.spinner(f"Comparing {target} with {len(pool) - 1} candidates ({note}) …"):
        panels = watchlist_panels(tuple(sorted(pool)))
    if panels is None or target not in panels["Close"].columns:
        st.error(f"Couldn't load prices for **{target}**. Check the ticker.")
        st.stop()
    res = peer_table(target, panels, int(lookback), int(n_peers), float(min_sim))
    if res is None or len(res[0]) <= 1:
        st.warning("No similar stocks found. Lower **Minimum similarity**, or look in **All stocks**.")
        st.stop()
    df, stamp, last_day = res
    peers = df.iloc[1:]
    n = len(peers)
    bull = int(peers["Stage"].isin(PEER_BULL).sum())
    cnt = lambda col: int((peers[col] > 0).sum())
    badge = lambda k: f'<span class="cnt{" r" if k < n / 2 else ""}">{k}/{n}</span>'
    asof = (f"Data as of {pd.Timestamp(stamp).tz_convert('America/New_York'):%I:%M:%S %p} ET"
            if stamp is not None and getattr(pd.Timestamp(stamp), "tzinfo", None) else f"Daily data · {last_day:%b %d, %Y}")
    ex_map = get_exchange_map()
    names = meta["Name"] if meta is not None and "Name" in meta else pd.Series(dtype=str)
    rows = []
    for t, r in df.iterrows():
        tv = tv_symbol(t, ex_map)
        name = names.get(t, "") if t in names.index else ""
        stg = r["Stage"] if isinstance(r["Stage"], str) else "?"
        stg_cls = "g" if stg in PEER_BULL else ("r" if stg != "?" else "n")
        sim_txt = "—" if t == target else f"{r['Sim.']:.2f}"
        rows.append(
            f'<tr class="{"me" if t == target else ""}">'
            f'<td><a href="https://www.tradingview.com/chart/?symbol={tv}" target="_blank">{t}</a>'
            f'<span class="sub">{name}</span></td>'
            f'<td><span class="pill {stg_cls}">{stg}</span></td>'
            f'<td><span class="pill n">{"" if pd.isna(r["RS"]) else int(r["RS"])}</span></td>'
            f'<td><span class="pill n">{sim_txt}</span></td>'
            f'<td>{_pct_pill(r["CoY"])}</td><td>{_pct_pill(r["Open"])}</td><td>{_pct_pill(r["10 Min"])}</td></tr>')
    head = (f'<tr><th>Peer</th><th>Stage<br>{badge(bull)}</th><th>RS</th><th>Sim.</th>'
            f'<th>CoY<br>{badge(cnt("CoY"))}</th><th>Open<br>{badge(cnt("Open"))}</th>'
            f'<th>10 Min<br>{badge(cnt("10 Min")) if peers["10 Min"].notna().any() else ""}</th></tr>')
    st.markdown(PEER_CSS + f'<div class="peer"><div class="top"><span class="t">Group strength</span>'
                f'<span class="tick">{target}</span><span class="asof">{asof} · peers from {note}</span></div>'
                f'<table class="pt">{head}{"".join(rows)}</table></div>', unsafe_allow_html=True)
    st.caption(f"Similarity over the last {lookback} trading days, from {len(pool) - 1} candidates. "
               "Stage badge counts bullish stages (1A–2C); the other badges count peers that are up.")

    # chart any of them
    st.markdown("")
    tpls, active = load_templates()
    if ss.get("pp_chart") not in list(df.index):
        ss["pp_chart"] = target
    pick = st.selectbox("Chart", list(df.index), key="pp_chart")
    tpl = {**DEFAULT_TPL, **tpls.get(ss.get("tpl_name", active), tpls["Default"])}
    tradingview_chart(tv_symbol(pick, ex_map), tpl, [tv_symbol(t, ex_map) for t in df.index])
    st.download_button("Download peers (CSV)", df.round(3).to_csv(), f"peers_{target}.csv", "text/csv")


# ============================================================================
# Page furniture: summary cards, stock detail card, entry/stop levels
# ============================================================================
st.markdown("""<style>
.cardrow{display:flex;flex-wrap:wrap;gap:.6rem;margin:.1rem 0 .7rem 0;animation:fadeUp .25s ease-out}
.scard{flex:1 1 130px;min-width:120px;padding:.55rem .8rem;border-radius:14px;
  border:1px solid rgba(128,128,128,.22);background:rgba(128,128,128,.06);border-left-width:4px}
.scard .l{font-size:.72rem;opacity:.7;text-transform:uppercase;letter-spacing:.04em}
.scard .v{font-size:1.35rem;font-weight:700;line-height:1.35}
.scard .s{font-size:.72rem;opacity:.65;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.scard.g{border-left-color:#16a34a}.scard.b{border-left-color:#2563eb}.scard.r{border-left-color:#dc2626}
.scard.n{border-left-color:rgba(128,128,128,.45)}.scard.y{border-left-color:#d97706}
.dcard{padding:.7rem .9rem;border-radius:14px;border:1px solid rgba(128,128,128,.22);
  background:rgba(128,128,128,.05);margin:.2rem 0 .5rem 0;animation:fadeUp .2s ease-out}
.dcard .top{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;flex-wrap:wrap}
.dcard .sym{font-size:1.25rem;font-weight:800}.dcard .nm{opacity:.75;font-size:.85rem}
.dcard .px{font-size:1.15rem;font-weight:700}.dcard .up{color:#16a34a}.dcard .dn{color:#dc2626}
.dcard .sub{font-size:.75rem;opacity:.65;margin-top:.1rem}
.dcard .chips{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.45rem}
.dcard .chip{font-size:.72rem;padding:.12rem .5rem;border-radius:999px;background:rgba(128,128,128,.13)}
.dcard .chip.g{background:rgba(34,197,94,.16);color:#22a355}.dcard .chip.r{background:rgba(239,68,68,.15);color:#e5484d}
.dcard .chip.b{background:rgba(59,130,246,.15);color:#3b82f6}.dcard .chip.y{background:rgba(245,158,11,.16);color:#d98a06}
.dcard .lv{display:flex;gap:1rem;margin-top:.5rem;font-size:.82rem;flex-wrap:wrap}
.dcard .lv b{font-size:.95rem}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def market_mood(bench="SPY", bench_label="SPY"):
    """Light market read for the summary card: the index vs its 50/200-day (+ the VIX for the US). None if unavailable."""
    try:
        px_ = rs_prices((bench, "^VIX") if bench == "SPY" else (bench,), "1y")
        spy = px_[bench].dropna()
        if len(spy) < 60:
            return None
        c, s50 = spy.iloc[-1], spy.rolling(50).mean().iloc[-1]
        s200 = spy.rolling(200, min_periods=120).mean().iloc[-1]
        vix = px_["^VIX"].dropna().iloc[-1] if "^VIX" in px_ and px_["^VIX"].notna().any() else np.nan
        up50, up200 = c > s50, (c > s200) if pd.notna(s200) else True
        label, tone = (("Uptrend", "g") if up50 and up200 else ("Pullback", "y") if up200 else
                       ("Bounce", "y") if up50 else ("Downtrend", "r"))
        if pd.notna(vix) and vix >= 25:
            tone = "r" if tone != "g" else "y"
        sub = f"{bench_label} {'>' if up50 else '<'} 50d · {'>' if up200 else '<'} 200d" + (f" · VIX {vix:.0f}" if pd.notna(vix) else "")
        return label, sub, tone
    except Exception:
        return None


def render_cards(cards):
    esc = html_lib.escape
    h = "".join(f'<div class="scard {t}"><div class="l">{esc(str(l))}</div><div class="v">{esc(str(v))}</div>'
                f'<div class="s" title="{esc(str(s))}">{esc(str(s))}</div></div>' for l, v, s, t in cards)
    st.markdown(f'<div class="cardrow">{h}</div>', unsafe_allow_html=True)


def _num(row, col):
    try:
        v = row.get(col, np.nan)
        return float(v) if pd.notna(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def setup_levels(sym, metrics, mode_key):
    """(summary text, [(price, name, colour)]) — the entry / stop that fit the chosen scan."""
    if metrics is None or sym not in metrics.index:
        return "", []
    row = metrics.loc[sym]
    price = _num(row, "Price")
    lv = []
    try:
        pn = ss.get("panels", {})
        H, L = pn["High"][sym].dropna(), pn["Low"][sym].dropna()
    except Exception:
        H = L = pd.Series(dtype=float)
    if mode_key == "pattern" and len(H) >= 8:
        base_hi = H.iloc[-8:-1].max() if row.get("Pattern") == "TRIGGER" else H.iloc[-7:].max()
        lv = [(base_hi, "Trigger", "#22c55e"), (L.iloc[-7:].min(), "Stop", "#dc2626")]
    elif mode_key == "so3" and pd.notna(_num(row, "+3 level")):
        lv = [(_num(row, "+3 level"), "Trigger (+3)", "#22c55e"), (_num(row, "First low"), "First low", "#2563eb"),
              (_num(row, "SO stop"), "Stop", "#dc2626")]
    elif mode_key == "reclaim" and pd.notna(_num(row, "Reclaim stop")):
        lv = [(_num(row, "50 SMA"), "Trigger (50-day SMA)", "#22c55e"), (_num(row, "Reclaim stop"), "Stop", "#dc2626")]
        if pd.notna(_num(row, "Dip low")) and _num(row, "Dip low") < _num(row, "Reclaim stop") - 1e-9:
            lv.append((_num(row, "Dip low"), "Dip low", "#2563eb"))
    elif mode_key == "f50scan" and pd.notna(_num(row, "F50 stop")):
        lv = [(_num(row, "F50 SMA"), "Trigger (50-day SMA)", "#22c55e"), (_num(row, "F50 stop"), "Stop", "#dc2626")]
    elif mode_key == "swshort" and pd.notna(_num(row, "Sell below")):
        lv = [(_num(row, "Sell below"), "Short below", "#f97316"),
              (_num(row, "Short stop"), "Stop (above)", "#dc2626")]
    elif mode_key == "htf" and pd.notna(_num(row, "Flag high")):
        lv = [(_num(row, "Flag high"), "Trigger (flag high)", "#22c55e"), (_num(row, "Flag low"), "Stop (flag low)", "#dc2626")]
    elif mode_key == "mac" and pd.notna(_num(row, "Box high")):
        lv = [(_num(row, "Box high"), "Trigger (box high)", "#22c55e"), (_num(row, "Box low"), "Stop", "#dc2626")]
    elif mode_key == "vcp" and pd.notna(_num(row, "Pivot")):
        lv = [(_num(row, "Pivot"), "Trigger (pivot)", "#22c55e"), (_num(row, "VCP stop"), "Stop", "#dc2626")]
    elif mode_key == "glb" and pd.notna(_num(row, "Green line")):
        lv = [(_num(row, "Green line"), "Trigger", "#22c55e"), (_num(row, "GLB stop"), "Stop", "#dc2626")]
    elif mode_key == "xback" and pd.notna(_num(row, "Entry above")):
        lv = [(_num(row, "Entry above"), "Trigger", "#22c55e"), (_num(row, "Stop"), "Stop", "#dc2626")]
    elif mode_key == "ep" and pd.notna(_num(row, "EP low")):
        lv = [(_num(row, "EP low"), "Stop (EP low)", "#dc2626")]
    elif mode_key == "parabolic" and len(H):
        lv = [(L.iloc[-1], "Short below", "#dc2626"), (H.iloc[-1], "Stop", "#d97706")]
    lv = [(float(a), b, c) for a, b, c in lv if pd.notna(a) and a > 0]
    text = ""
    ent = next((a for a, b, _ in lv if b.startswith(("Entry", "Trigger")) or b == "Short below"), None)
    stp = next((a for a, b, _ in lv if b.startswith("Stop")), None)
    if ent and stp:
        text = f"Risk {abs(ent - stp) / ent * 100:.1f}%"
    elif stp and pd.notna(price) and price:
        text = f"{(price - stp) / price * 100:.1f}% above stop"
    return text, lv


def detail_card(sym, metrics, mode_key):
    if metrics is None or sym not in metrics.index:
        return
    esc = html_lib.escape
    row = metrics.loc[sym]
    name = row.get("Name")
    name = name if isinstance(name, str) and name.strip() and name != "nan" else ""
    sec = " · ".join(x for x in [row.get("Sector"), row.get("Industry")] if isinstance(x, str) and x and x != "nan")
    price, chg = _num(row, "Price"), _num(row, "Chg %")
    cap = _num(row, "Mkt cap $B")
    chips = []
    rs = _num(row, "RS")
    if pd.notna(rs):
        chips.append((f"RS {rs:.0f}", "g" if rs >= 80 else "r" if rs < 40 else ""))
    if bool(row.get("Trend template", False)):
        chips.append(("Stage 2 trend", "g"))
    hi = _num(row, "Below 52W high %")
    if pd.notna(hi):
        chips.append(("At 52W high" if hi <= 0.5 else f"{hi:.0f}% below 52W high", "g" if hi <= 5 else ""))
    rv = _num(row, "Rel vol")
    if pd.notna(rv) and rv >= 1.5:
        chips.append((f"Rel vol {rv:.1f}×", "b"))
    adr = _num(row, "ADR % 14d")
    if pd.notna(adr):
        chips.append((f"ADR {adr:.1f}%", ""))
    ed = _num(row, "Days to earnings")
    ed = ed if pd.notna(ed) else _num(row, "Earnings in (days)")
    if pd.notna(ed):
        chips.append(("Earnings today" if ed == 0 else f"Earnings in {ed:.0f}d" if ed > 0 else f"Reported {-ed:.0f}d ago",
                      "y" if 0 <= ed <= 7 else ""))
    for col in ["Pattern", "HTF", "VCP", "MAC", "Crossback", "GLB", "SO+3", "Parabolic", "EP", "Vol record", "S&P change",
                "Short", "Reclaim", "First50"]:
        v = row.get(col)
        if isinstance(v, str) and v:
            if col == "Short":
                chips.append((f"{row.get('Short setup', '')} short {v}".strip(), "r" if v == "TRIGGER" else "b"))
                continue
            chips.append((v, "g" if v in ("TRIGGER", "ADDED", "HOLDING", "BREAKOUT") else "r" if v in ("REMOVED", "CRACK") else "b"))
    text, lv = setup_levels(sym, metrics, mode_key)
    cur = (market_of(sym) or {}).get("cur", "$")
    lv_html = "".join(f'<span>{esc(n)} <b>{cur}{a:,.2f}</b></span>' for a, n, _ in lv) + \
        (f'<span style="opacity:.75">{esc(text)}</span>' if text else "")
    cls = "up" if pd.notna(chg) and chg >= 0 else "dn"
    chips_html = "".join(f'<span class="chip {t}">{esc(c)}</span>' for c, t in chips)
    st.markdown(
        f'<div class="dcard"><div class="top"><div><span class="sym">{esc(sym)}</span> '
        f'<span class="nm">{esc(name)}</span></div>'
        f'<div><span class="px">{"" if pd.isna(price) else f"{cur}{price:,.2f}"}</span> '
        f'<span class="{cls}">{"" if pd.isna(chg) else f"{chg:+.2f}%"}</span></div></div>'
        f'<div class="sub">{esc(sec)}{" · " if sec and pd.notna(cap) else ""}{"" if pd.isna(cap) else f"${cap:,.1f}B"}</div>'
        f'<div class="chips">{chips_html}</div>'
        + (f'<div class="lv">{lv_html}</div>' if lv_html else "") + '</div>', unsafe_allow_html=True)


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
CUSIP_FILE = Path(__file__).with_name("cusip_tickers.json")
SM_NAME_DROP = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "HLDGS", "HLDG", "HOLDINGS",
                "HOLDING", "GROUP", "GRP", "CL", "CLASS", "COM", "NEW", "THE", "SA", "NV", "AG", "LP", "ADR", "SPONSORED",
                "SPON", "ADS", "SHS", "ORD", "A", "B", "C", "DEL", "INCORPORATED", "TR", "TRUST", "N", "V", "SE", "LLC"}


def sec_headers(ua):
    return {"User-Agent": ua or "StockScreener personal research (contact: research@example.com)",
            "Accept-Encoding": "gzip, deflate"}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def sec_13f_filings(cik, ua):
    """(filer name, DataFrame of 13F-HR filings newest first)."""
    r = requests.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", headers=sec_headers(ua), timeout=30)
    r.raise_for_status()
    j = r.json()
    rec = j.get("filings", {}).get("recent", {})
    df = pd.DataFrame({k: rec.get(k, []) for k in ["accessionNumber", "filingDate", "reportDate", "form"]})
    df = df[df["form"].isin(["13F-HR", "13F-HR/A"])].copy()
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
    return j.get("name", str(cik)).title(), df.sort_values(["reportDate", "filingDate"], ascending=False)


def _xml_text(el, name):
    for x in el.iter():
        if x.tag.split("}")[-1] == name:
            return (x.text or "").strip()
    return ""


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def sec_13f_holdings(cik, accession, filed, ua):
    """Holdings in one 13F filing: Issuer, Class, CUSIP, Value $, Shares, Put/Call (options kept separately)."""
    import xml.etree.ElementTree as ET
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    items = requests.get(base + "index.json", headers=sec_headers(ua), timeout=30).json()["directory"]["item"]
    xmls = [i for i in items if str(i.get("name", "")).lower().endswith(".xml")
            and str(i.get("name", "")).lower() != "primary_doc.xml"]
    if not xmls:
        return pd.DataFrame()
    pick = sorted(xmls, key=lambda i: ("info" not in i["name"].lower() and "table" not in i["name"].lower(),
                                       -int(str(i.get("size") or 0).strip() or 0)))[0]
    root = ET.fromstring(requests.get(base + pick["name"], headers=sec_headers(ua), timeout=60).content)
    rows = []
    for it in root.iter():
        if it.tag.split("}")[-1] != "infoTable":
            continue
        rows.append(dict(Issuer=_xml_text(it, "nameOfIssuer"), Class=_xml_text(it, "titleOfClass"),
                         CUSIP=_xml_text(it, "cusip").upper(), Value=pd.to_numeric(_xml_text(it, "value"), errors="coerce"),
                         Shares=pd.to_numeric(_xml_text(it, "sshPrnamt"), errors="coerce"),
                         PutCall=_xml_text(it, "putCall").title()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if pd.Timestamp(filed) < pd.Timestamp("2023-01-03"):      # older filings reported value in $ thousands
        df["Value"] = df["Value"] * 1000
    return df.groupby(["CUSIP", "PutCall"], as_index=False).agg(
        Issuer=("Issuer", "first"), Class=("Class", "first"), Value=("Value", "sum"), Shares=("Shares", "sum"))


def _norm_name(s):
    words = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper()).split()
    return " ".join(w for w in words if w not in SM_NAME_DROP)


def cusip_tickers(df):
    """CUSIP -> ticker. Saved lookups first, then OpenFIGI (free), then matching the company name."""
    try:
        known = json.loads(CUSIP_FILE.read_text(encoding="utf-8"))
    except Exception:
        known = {}
    todo = [c for c in df["CUSIP"].unique() if c and c not in known and c not in ss.get("figi_miss", set())]
    if todo:
        found = {}
        for i in range(0, min(len(todo), 250), 10):       # keyless OpenFIGI: 10 per request, ~25 requests a minute
            batch = todo[i:i + 10]
            try:
                r = requests.post("https://api.openfigi.com/v3/mapping", timeout=20,
                                  json=[{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in batch])
                if r.status_code != 200:
                    break
                for c, res in zip(batch, r.json()):
                    data = res.get("data") or []
                    t = next((d.get("ticker") for d in data if d.get("marketSector") == "Equity" and d.get("ticker")),
                             data[0].get("ticker") if data else None)
                    if t:
                        found[c] = str(t).replace("/", "-").replace(" ", "-")
                    else:
                        ss.setdefault("figi_miss", set()).add(c)
            except Exception:
                break
        if found:
            known.update(found)
            try:
                CUSIP_FILE.write_text(json.dumps(known, indent=0), encoding="utf-8")
            except Exception:
                pass
    out = df["CUSIP"].map(known)
    miss = out.isna()
    if miss.any():                                        # fall back to the company name in Nasdaq's stock list
        try:
            meta = get_screener_meta()
            names = {}
            for sym, nm in zip(meta.index if "Symbol" not in meta else meta["Symbol"], meta["Name"]):
                names.setdefault(_norm_name(nm), sym)
            out[miss] = df.loc[miss, "Issuer"].map(lambda n: names.get(_norm_name(n)))
        except Exception:
            pass
    return out


def fund_portfolio(label, cik, ua):
    """Latest 13F vs. the quarter before: value, weight, and what changed."""
    name, filings = sec_13f_filings(cik, ua)
    main = filings[filings["form"] == "13F-HR"].drop_duplicates("reportDate")
    if main.empty:
        main = filings.drop_duplicates("reportDate")
    if main.empty:
        return None
    cur = main.iloc[0]
    now = sec_13f_holdings(cik, cur["accessionNumber"], cur["filingDate"], ua)
    if now.empty:
        return None
    prev_df, prev_period = pd.DataFrame(columns=now.columns), None
    if len(main) > 1:
        p = main.iloc[1]
        prev_period = p["reportDate"]
        try:
            prev_df = sec_13f_holdings(cik, p["accessionNumber"], p["filingDate"], ua)
        except Exception:
            pass
    key = ["CUSIP", "PutCall"]
    m = now.merge(prev_df[key + ["Value", "Shares"]].rename(columns={"Value": "Prev value", "Shares": "Prev shares"}),
                  on=key, how="outer")
    m["Issuer"] = m["Issuer"].fillna(m["CUSIP"].map(prev_df.set_index("CUSIP")["Issuer"].to_dict()
                                                    if len(prev_df) else {}))
    for c in ["Value", "Shares", "Prev value", "Prev shares"]:
        m[c] = m[c].fillna(0)
    chg = (m["Shares"] / m["Prev shares"].replace(0, np.nan) - 1) * 100
    m["Change"] = np.select([m["Prev shares"] == 0, m["Shares"] == 0, chg > 0.5, chg < -0.5],
                            ["NEW", "SOLD", "ADD", "TRIM"], "—") if prev_period is not None else "—"
    m["Shares chg %"] = np.where(m["Change"].isin(["ADD", "TRIM"]), chg, np.nan)
    total = m["Value"].sum()
    m["Weight %"] = m["Value"] / total * 100 if total else np.nan
    m["Ticker"] = cusip_tickers(m.fillna({"Issuer": ""}))
    m["Value $M"] = m["Value"] / 1e6
    m["Prev value $M"] = m["Prev value"] / 1e6
    m["Put/Call"] = m["PutCall"].replace("", np.nan)
    m = m.sort_values(["Value", "Prev value"], ascending=False)
    return dict(label=label, name=name, cik=cik, period=cur["reportDate"], filed=cur["filingDate"],
                prev_period=prev_period, total=total, df=m.reset_index(drop=True))


def sm_since_period(df, period):
    """% change from the quarter-end date to today, for the biggest holdings."""
    t = [x for x in df.loc[df["Value"] > 0, "Ticker"].dropna().unique()][:60]
    if not t or period is None:
        return pd.Series(dtype=float)
    px_ = rs_prices(tuple(t), "1y")
    if px_.empty:
        return pd.Series(dtype=float)
    base = px_[px_.index <= pd.Timestamp(period)].ffill().iloc[-1] if (px_.index <= pd.Timestamp(period)).any() \
        else px_.iloc[0]
    return (px_.ffill().iloc[-1] / base - 1) * 100


# ---- Congress ----
SM_AMOUNT_MIN = {"Any amount": 0, "$15K+": 15001, "$50K+": 50001, "$100K+": 100001, "$250K+": 250001,
                 "$1M+": 1000001}


def _amount_range(text):
    nums = [float(x.replace(",", "")) for x in re.findall(r"\$\s*([\d,]+)", str(text))]
    if not nums:
        return np.nan, np.nan
    return nums[0], nums[-1] if len(nums) > 1 else nums[0]


def _fmt_amount(lo, hi):
    f = lambda v: f"${v / 1e6:g}M" if v >= 1e6 else f"${v / 1e3:,.0f}K" if v >= 1e3 else f"${v:,.0f}"
    if pd.isna(lo):
        return ""
    return f(lo) if hi == lo or pd.isna(hi) else f"{f(lo)}–{f(hi)}"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def congress_members():
    """Current members: last name + state -> (full name, party, state, chamber). Free public dataset."""
    try:
        data = requests.get("https://unitedstates.github.io/congress-legislators/legislators-current.json",
                            headers=UA, timeout=30).json()
    except Exception:
        return []
    out = []
    for p in data:
        t = (p.get("terms") or [{}])[-1]
        n = p.get("name", {})
        out.append(dict(first=n.get("first", ""), nick=n.get("nickname", ""), last=n.get("last", ""),
                        full=n.get("official_full") or f"{n.get('first', '')} {n.get('last', '')}",
                        party=(t.get("party") or "")[:1], state=t.get("state", ""),
                        chamber="Senate" if t.get("type") == "sen" else "House"))
    return out


def _match_member(first, last, state, chamber):
    last_u = str(last).upper().split(",")[0].strip()
    first_u = str(first).upper().strip()
    cands = [m for m in congress_members() if m["last"].upper() == last_u and (not chamber or m["chamber"] == chamber)]
    if state:
        cands = [m for m in cands if m["state"] == state] or cands
    if len(cands) > 1 and first_u:
        cands = [m for m in cands if first_u[:3] in (m["first"].upper()[:3], m["nick"].upper()[:3])] or cands
    return cands[0] if cands else None


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def capitoltrades_trades(days):
    """Trades from Capitol Trades' public data feed, newest filings first."""
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    rows = []
    for page in range(1, 31):
        r = requests.get(f"https://bff.capitoltrades.com/trades?page={page}&pageSize=100&sortBy=-pubDate",
                         headers={**UA, "Origin": "https://www.capitoltrades.com",
                                  "Referer": "https://www.capitoltrades.com/"}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            break
        oldest = None
        for x in data:
            pol, iss, ast = x.get("politician") or {}, x.get("issuer") or {}, x.get("asset") or {}
            tick = str(iss.get("issuerTicker") or ast.get("assetTicker") or "").split(":")[0].replace("/", "-")
            pub = pd.to_datetime(x.get("pubDate"), errors="coerce", utc=True)
            pub = pub.tz_localize(None) if pd.notna(pub) else pub
            oldest = pub if oldest is None or (pd.notna(pub) and pub < oldest) else oldest
            lo, hi = x.get("sizeRangeLow"), x.get("sizeRangeHigh")
            lo = float(lo) if isinstance(lo, (int, float)) else np.nan
            hi = float(hi) if isinstance(hi, (int, float)) else lo
            if pd.isna(lo) and isinstance(x.get("value"), (int, float)):
                lo = hi = float(x["value"])
            tx = str(x.get("txType") or "").lower()
            rows.append(dict(
                Filed=pub, Traded=pd.to_datetime(x.get("txDate"), errors="coerce"),
                Politician=f"{pol.get('nickname') or pol.get('firstName') or ''} {pol.get('lastName') or ''}".strip(),
                Chamber=str(x.get("chamber") or pol.get("chamber") or "").title(),
                Party=str(pol.get("party") or "")[:1].upper(), State=str(pol.get("_stateId") or "").upper(),
                Ticker=tick or None, Asset=iss.get("issuerName") or ast.get("instrument") or "",
                Type="Purchase" if tx == "buy" else "Sale" if tx == "sell" else tx.title(),
                **{"Amount low": lo, "Amount high": hi}, Owner=str(x.get("owner") or "").title(),
                Link=x.get("filingURL") or "", Source="Capitol Trades"))
        if oldest is not None and pd.notna(oldest) and oldest < cutoff:
            break
    df = pd.DataFrame(rows)
    return df[df["Filed"] >= cutoff] if len(df) else df


def _senate_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA["User-Agent"]})
    home = s.get("https://efdsearch.senate.gov/search/home/", timeout=30)
    tok = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', home.text)
    s.post("https://efdsearch.senate.gov/search/home/", timeout=30,
           data={"prohibition_agreement": "1", "csrfmiddlewaretoken": tok.group(1) if tok else ""},
           headers={"Referer": "https://efdsearch.senate.gov/search/home/"})
    return s


def _html_rows(html):
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = [html_lib.unescape(re.sub(r"<[^>]+>", " ", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I)]
        if cells:
            out.append([re.sub(r"\s+", " ", c) for c in cells])
    return out


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def senate_trades(days):
    """Official Senate eFD periodic transaction reports (electronic ones)."""
    s = _senate_session()
    start = (date.today() - timedelta(days=days)).strftime("%m/%d/%Y 00:00:00")
    reports, offset = [], 0
    while offset < 1000:
        r = s.post("https://efdsearch.senate.gov/search/report/data/", timeout=30,
                   headers={"Referer": "https://efdsearch.senate.gov/search/",
                            "X-CSRFToken": s.cookies.get("csrftoken", "")},
                   data={"start": str(offset), "length": "100", "report_types": "[11]", "filer_types": "[]",
                         "submitted_start_date": start, "submitted_end_date": "", "candidate_state": "",
                         "senator_state": "", "office_id": "", "first_name": "", "last_name": "",
                         "csrfmiddlewaretoken": s.cookies.get("csrftoken", "")})
        data = r.json().get("data") or []
        reports += data
        if len(data) < 100:
            break
        offset += 100
    rows = []

    def one(rep):
        first, last, filed = rep[0], rep[1], rep[-1]
        link = re.search(r'href="([^"]+)"', rep[3] or "")
        if not link or "/ptr/" not in link.group(1):
            return []                                        # paper (scanned) filings can't be read
        url = "https://efdsearch.senate.gov" + link.group(1)
        html = s.get(url, timeout=30).text
        mem = _match_member(first, last, "", "Senate")
        out = []
        for c in _html_rows(html):
            if len(c) < 8 or not re.match(r"\d{2}/\d{2}/\d{4}", c[1]):
                continue
            lo, hi = _amount_range(c[7])
            typ = c[6]
            out.append(dict(Filed=pd.to_datetime(filed, errors="coerce"), Traded=pd.to_datetime(c[1], errors="coerce"),
                            Politician=mem["full"] if mem else f"{first} {last}".title(), Chamber="Senate",
                            Party=mem["party"] if mem else "", State=mem["state"] if mem else "",
                            Ticker=None if c[3] in ("--", "") else c[3].split()[0].replace(".", "-"),
                            Asset=c[4], Type="Purchase" if typ.startswith("Purchase") else
                            "Sale" if typ.startswith("Sale") else typ,
                            **{"Amount low": lo, "Amount high": hi}, Owner=c[2], Link=url, Source="Senate eFD"))
        return out
    with ThreadPoolExecutor(4) as ex:
        for part in ex.map(lambda rep: _safe(one, rep), reports):
            rows += part
    return pd.DataFrame(rows)


def _pdf_text(content):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        return " ".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(content)).pages).replace("\x00", "")
    except Exception:
        return ""


HOUSE_ROW = re.compile(
    r"\(([A-Z][A-Z0-9.\-/]{0,7})\)\s*\[(ST|OP|EF|ET|AB)\]\s*"
    r"(P|S \(partial\)|S|E)\s+(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}/\d{1,2}/\d{4})\s*"
    r"(\$[\d,]+\s*-\s*\$[\d,]+|Over \$[\d,]+|\$[\d,]+ \+|Spouse/DC Over \$[\d,]+)")


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def house_ptr(year, doc_id):
    r = requests.get(f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf",
                     headers=UA, timeout=40)
    text = _pdf_text(r.content)
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text)
    out, prev_end = [], 0
    for m in HOUSE_ROW.finditer(text):
        tick, kind, typ, tdate, _ndate, amt = m.groups()
        lo, hi = _amount_range(amt)
        # the asset name sits between the previous row and this ticker; drop headers and the notes of the last row
        pre = re.split(r"\$200\?|F S\s*:\s*\w+|S O\s*:|\bD\s*:", text[prev_end:m.start()])[-1]
        prev_end = m.end()
        om = re.search(r"(?:^|\s)(SP|JT|DC)\s+(.*)$", pre)
        owner, asset = (om.group(1), om.group(2)) if om else (None, pre[-120:])
        out.append(dict(Owner={"SP": "Spouse", "JT": "Joint", "DC": "Child"}.get(owner, "Self"),
                        Asset=asset.strip(" -:"), Ticker=tick.replace(".", "-").replace("/", "-"),
                        Kind="Option" if kind == "OP" else "Stock", Traded=tdate,
                        Type={"P": "Purchase", "E": "Exchange"}.get(typ, "Sale"), lo=lo, hi=hi))
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def house_filings(days):
    """List of House periodic transaction reports filed in the window (from the Clerk's yearly index)."""
    import zipfile
    import xml.etree.ElementTree as ET
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    rows = []
    for year in sorted({date.today().year, cutoff.year}):
        r = requests.get(f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip",
                         headers=UA, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = next(n for n in z.namelist() if n.lower().endswith(".xml"))
        for mem in ET.fromstring(z.read(name)).iter("Member"):
            g = lambda k: (mem.findtext(k) or "").strip()
            if g("FilingType") != "P":
                continue
            rows.append(dict(first=g("First"), last=g("Last"), dist=g("StateDst"), year=g("Year") or str(year),
                             filed=pd.to_datetime(g("FilingDate"), errors="coerce"), doc=g("DocID")))
    df = pd.DataFrame(rows)
    return df[df["filed"] >= cutoff].sort_values("filed", ascending=False) if len(df) else df


def house_trades(days, max_reports, bar=None):
    f = house_filings(days).head(max_reports)
    rows, no_pdf = [], False
    with ThreadPoolExecutor(6) as ex:
        results = ex.map(lambda r: (r, _safe(house_ptr, r["year"], r["doc"])), [r for _, r in f.iterrows()])
        for i, (r, trades) in enumerate(results):
            if bar is not None:
                bar.progress((i + 1) / max(len(f), 1), text=f"Reading House reports … {i + 1}/{len(f)}")
            if trades is None:
                no_pdf = True
                continue
            mem = _match_member(r["first"], r["last"], r["dist"][:2], "House")
            url = f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{r['year']}/{r['doc']}.pdf"
            for t in trades:
                rows.append(dict(Filed=r["filed"], Traded=pd.to_datetime(t["Traded"], errors="coerce"),
                                 Politician=mem["full"] if mem else f"{r['first']} {r['last']}",
                                 Chamber="House", Party=mem["party"] if mem else "", State=r["dist"][:2],
                                 Ticker=t["Ticker"], Asset=t["Asset"] + (" (option)" if t["Kind"] == "Option" else ""),
                                 Type=t["Type"], **{"Amount low": t["lo"], "Amount high": t["hi"]},
                                 Owner=t["Owner"], Link=url, Source="House Clerk"))
    return pd.DataFrame(rows), no_pdf


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return []


def congress_trades(days, max_house):
    """Capitol Trades first; if that's unreachable, the official Senate + House filings."""
    notes = []
    try:
        df = capitoltrades_trades(days)
        if len(df):
            return df, notes
    except Exception:
        notes.append("Capitol Trades unreachable — using the official Senate and House filings instead.")
    parts = []
    try:
        parts.append(senate_trades(days))
    except Exception:
        notes.append("Senate eFD site unreachable.")
    try:
        bar = st.progress(0.0, text="Reading House reports …")
        h, no_pdf = house_trades(days, max_house, bar)
        bar.empty()
        parts.append(h)
        if no_pdf:
            notes.append("House trades need the free **pypdf** package to read the PDF reports: run "
                         "`py -m pip install pypdf`, then restart the app.")
    except Exception:
        notes.append("House Clerk site unreachable.")
    parts = [p for p in parts if len(p)]
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), notes


def sm_table(df, cfg, key, height=None):
    """A clickable table; returns the clicked row's ticker (or None)."""
    data = df.data if hasattr(df, "data") else df          # a Styler wraps the DataFrame
    kw = dict(use_container_width=True, hide_index=True, column_config=cfg, on_select="rerun", key=key,
              height=height or min(35 * (len(data) + 1) + 3, 560))
    try:
        ev = st.dataframe(df, selection_mode="single-cell", **kw)
    except Exception:
        ev = st.dataframe(df, selection_mode="single-row", **kw)
    sel = []
    try:
        cells = ev.selection.get("cells", []) if hasattr(ev.selection, "get") else getattr(ev.selection, "cells", [])
        sel = [cells[0][0] if isinstance(cells[0], (list, tuple)) else cells[0].get("row")] if cells else \
            list(getattr(ev.selection, "rows", []) or [])
    except Exception:
        pass
    if sel and sel[0] is not None and 0 <= sel[0] < len(data) and "Ticker" in data:
        return data["Ticker"].iloc[sel[0]]
    return None


def sm_chart(tickers, clicked, key):
    tickers = [t for t in dict.fromkeys(tickers) if isinstance(t, str) and t]
    if not tickers:
        return
    if clicked in tickers:
        ss[key] = clicked
    if ss.get(key) not in tickers:
        ss[key] = tickers[0]
    ex_map = get_exchange_map()
    c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
    pick = c1.selectbox("Chart", tickers, key=key, help="Or click a row in the table above.")
    c2.button("🧭 Use these tickers in Relative Strength", key="smt_rs_" + key,
              help="Sends this list to Relative Strength → Stocks from watchlist → Stock Scanner results.",
              on_click=lambda: ss.update(last_results=tickers[:500], last_results_label="Smart Money"))
    tpls, active = load_templates()
    tpl = {**DEFAULT_TPL, **tpls.get(ss.get("tpl_name", active), tpls["Default"])}
    tradingview_chart(tv_symbol(pick, ex_map), tpl, [tv_symbol(t, ex_map) for t in tickers[:100]])


CHANGE_STYLE = {"NEW": "color:#22a355;font-weight:700", "ADD": "color:#22a355", "TRIM": "color:#e0a106",
                "SOLD": "color:#e5484d;font-weight:700", "Purchase": "color:#22a355;font-weight:700",
                "Sale": "color:#e5484d;font-weight:700", "D": "color:#3b82f6;font-weight:700",
                "R": "color:#e5484d;font-weight:700"}


def smart_money_page():
    st.sidebar.title("💰 Smart Money")
    view = st.sidebar.radio("Track", ["🏦 Famous investors (13F)", "🏛 Congress trades"], key="sm_view")
    st.title("Smart money")

    if view.startswith("🏦"):
        st.sidebar.multiselect("Investors", list(FAMOUS_13F), key="sm_funds", placeholder="Pick investors")
        st.sidebar.text_input("Add funds by CIK (optional)", key="sm_ciks", placeholder="e.g. 1067983, 1336528",
                              help="Any 13F filer: find the CIK with EDGAR company search on sec.gov.")
        st.sidebar.text_input("Your email for SEC requests (optional)", key="sm_email",
                              help="The SEC asks automated tools to identify themselves. Stays on your computer.")
        st.sidebar.caption("13F = the quarterly list of US stocks a fund manager with $100M+ holds. It's filed up to "
                           "45 days after each quarter ends, so it shows what they owned then, not today. Shorts and "
                           "most non-US holdings aren't included.")
        ua = f"StockScreener personal research ({ss['sm_email'].strip()})" if "@" in ss.get("sm_email", "") else None
        funds = [(l, FAMOUS_13F[l]) for l in ss["sm_funds"] if l in FAMOUS_13F]
        funds += [(f"CIK {c}", int(c)) for c in re.findall(r"\d{3,10}", ss.get("sm_ciks", ""))]
        if not funds:
            st.info("👈 Pick some investors in the sidebar.")
            return
        ports, errs = [], []
        bar = st.progress(0.0, text="Loading 13F filings from SEC EDGAR …")
        for i, (label, cik) in enumerate(funds):
            bar.progress(i / len(funds), text=f"Loading 13F · {label} …")
            try:
                p = fund_portfolio(label, cik, ua)
                if p:
                    ports.append(p)
                else:
                    errs.append(f"{label}: no 13F filings found")
            except Exception as e:
                errs.append(f"{label}: {str(e)[:80]}")
        bar.empty()
        if errs:
            st.caption("⚠️ " + " · ".join(errs))
        if not ports:
            st.error("Couldn't load any 13F filings. Check your internet connection (the data comes from sec.gov).")
            return
        mode = st.segmented_control("Show", ["Consensus", "One investor"], key="sm_mode",
                                    label_visibility="collapsed") or "Consensus"
        if mode == "One investor":
            labels = [p["label"] for p in ports]
            if ss.get("sm_one") not in labels:
                ss["sm_one"] = labels[0]
            p = next(x for x in ports if x["label"] == st.selectbox("Investor", labels, key="sm_one"))
            d = p["df"]
            cur = d[d["Value"] > 0]
            chg = d["Change"].value_counts()
            top10 = cur["Weight %"].head(10).sum()
            render_cards([("Portfolio", f"${p['total'] / 1e9:,.2f}B", p["name"], "n"),
                          ("Holdings", f"{len(cur):,}", f"top 10 = {top10:.0f}% of it", "b"),
                          ("New buys", f"{chg.get('NEW', 0)}", f"added to {chg.get('ADD', 0)}", "g"),
                          ("Sold out", f"{chg.get('SOLD', 0)}", f"trimmed {chg.get('TRIM', 0)}", "r"),
                          ("Quarter", f"{p['period']:%b %d, %Y}", f"filed {p['filed']:%b %d, %Y}", "n")])
            since = sm_since_period(d, p["period"])
            d = d.assign(**{"Since qtr end %": d["Ticker"].map(since)})
            cols = ["Ticker", "Issuer", "Put/Call", "Change", "Weight %", "Value", "Shares", "Shares chg %",
                    "Prev value", "Since qtr end %", "Class", "CUSIP"]
            d = d.assign(**{c: d[c].fillna("") for c in ["Put/Call", "Class", "Issuer"]})
            top = cur.head(15)
            fig = px.bar(top.iloc[::-1], x="Weight %", y=top.iloc[::-1]["Ticker"].fillna(top.iloc[::-1]["Issuer"]),
                         orientation="h", text=top.iloc[::-1]["Weight %"].map(lambda v: f"{v:.1f}%"))
            fig.update_layout(height=max(260, 24 * len(top) + 60), margin=dict(l=10, r=10, t=30, b=10),
                              title="Biggest positions (% of portfolio)", yaxis_title=None, xaxis_title=None)
            fig.update_traces(marker_color="#3b82f6")
            st.plotly_chart(fig, use_container_width=True)
            show = d[cols]
            cfg = {"Value": st.column_config.NumberColumn("Value $", format="compact"),
                   "Prev value": st.column_config.NumberColumn("Prev qtr $", format="compact"),
                   "Weight %": st.column_config.ProgressColumn("Weight", format="%.2f%%", min_value=0,
                                                               max_value=float(max(cur["Weight %"].max(), 1))),
                   "Shares": st.column_config.NumberColumn(format="compact"),
                   "Shares chg %": st.column_config.NumberColumn("Shares chg", format="%+.0f%%"),
                   "Since qtr end %": st.column_config.NumberColumn("Since qtr end", format="%+.1f%%",
                                                                    help="Price now vs. the quarter-end date"),
                   "Change": st.column_config.TextColumn(help="vs. the previous quarter's 13F")}
            clicked = sm_table(show.style.map(lambda v: CHANGE_STYLE.get(v, ""), subset=["Change"]), cfg,
                               f"smt_tbl_{p['cik']}", height=min(35 * (len(show) + 1) + 3, 600))
            st.download_button("⬇ CSV", show.to_csv(index=False).encode(), f"13F_{p['cik']}.csv", "text/csv")
            sm_chart(list(cur["Ticker"].dropna()), clicked, "sm_chart_fund")
        else:
            all_ = []
            for p in ports:
                d = p["df"].assign(Fund=p["label"].split(" · ")[0])
                all_.append(d[d["Ticker"].notna() & d["Put/Call"].isna()])      # shares only, not options
            a = pd.concat(all_, ignore_index=True)
            held = a[a["Value"] > 0]
            g = a.groupby("Ticker")
            cons = pd.DataFrame({
                "Holders": held.groupby("Ticker")["Fund"].nunique(),
                "New buys": a[a["Change"] == "NEW"].groupby("Ticker")["Fund"].nunique(),
                "Added": a[a["Change"] == "ADD"].groupby("Ticker")["Fund"].nunique(),
                "Trimmed": a[a["Change"] == "TRIM"].groupby("Ticker")["Fund"].nunique(),
                "Sold out": a[a["Change"] == "SOLD"].groupby("Ticker")["Fund"].nunique(),
                "Total value": held.groupby("Ticker")["Value"].sum(),
                "Avg weight %": held.groupby("Ticker")["Weight %"].mean(),
                "Name": g["Issuer"].first(),
                "Who": held.groupby("Ticker")["Fund"].agg(lambda s: ", ".join(sorted(set(s)))),
                "Bought by": a[a["Change"].isin(["NEW", "ADD"])].groupby("Ticker")["Fund"].agg(
                    lambda s: ", ".join(sorted(set(s)))),
            }).fillna({"Holders": 0, "New buys": 0, "Added": 0, "Trimmed": 0, "Sold out": 0, "Who": "", "Bought by": ""})
            for c in ["Holders", "New buys", "Added", "Trimmed", "Sold out"]:
                cons[c] = cons[c].astype(int)
            cons["Net buyers"] = cons["New buys"] + cons["Added"] - cons["Trimmed"] - cons["Sold out"]
            sort = st.segmented_control("Sort by", ["Most held", "Most bought", "Most sold", "Biggest $"],
                                        key="sm_sort", label_visibility="collapsed") or "Most held"
            by = {"Most held": ["Holders", "Total value"], "Most bought": ["Net buyers", "New buys"],
                  "Most sold": ["Net buyers", "Sold out"], "Biggest $": ["Total value", "Holders"]}[sort]
            cons = cons.sort_values(by, ascending=[True, False] if sort == "Most sold" else False).reset_index()
            periods = sorted({f"{p['period']:%b %Y}" for p in ports})
            render_cards([("Investors", f"{len(ports)}", "quarter " + " / ".join(periods), "n"),
                          ("Stocks held", f"{(cons['Holders'] > 0).sum():,}", "across all of them", "b"),
                          ("Held by 2+", f"{(cons['Holders'] >= 2).sum():,}", "shared ideas", "g"),
                          ("Fresh buys", f"{int((cons['New buys'] > 0).sum()):,}", "new positions last quarter", "g"),
                          ("Exits", f"{int((cons['Sold out'] > 0).sum()):,}", "sold completely", "r")])
            cols = ["Ticker", "Name", "Holders", "Net buyers", "New buys", "Added", "Trimmed", "Sold out",
                    "Total value", "Avg weight %", "Who", "Bought by"]
            show = cons[cols]
            cfg = {"Total value": st.column_config.NumberColumn("Total value $", format="compact"),
                   "Avg weight %": st.column_config.NumberColumn("Avg weight", format="%.1f%%"),
                   "Holders": st.column_config.ProgressColumn("Holders", format="%d", min_value=0,
                                                              max_value=len(ports)),
                   "Net buyers": st.column_config.NumberColumn(format="%+d",
                                                               help="New + added − trimmed − sold out"),
                   "Who": st.column_config.TextColumn("Held by", width="medium"),
                   "Bought by": st.column_config.TextColumn(width="medium"),
                   "Name": st.column_config.TextColumn(width="small")}
            sty = show.style.map(lambda v: "color:#22a355;font-weight:700" if isinstance(v, (int, float)) and v > 0
                                 else "color:#e5484d;font-weight:700" if isinstance(v, (int, float)) and v < 0 else "",
                                 subset=["Net buyers"])
            if show.empty:
                st.warning("Couldn't match the holdings to tickers right now (the OpenFIGI lookup may be busy — it "
                           "allows about 250 lookups a minute). Try again in a minute, or use **One investor**.")
            clicked = sm_table(sty, cfg, "smt_cons")
            st.caption("Click a row to chart it. Positions are matched by CUSIP; a few may lack a ticker "
                       "(bonds, funds, foreign listings) and are left out here — see them under **One investor**.")
            st.download_button("⬇ CSV", show.to_csv(index=False).encode(), "13F_consensus.csv", "text/csv")
            sm_chart(list(show["Ticker"]), clicked, "sm_chart_cons")
        return

    # ---------------- Congress ----------------
    st.sidebar.selectbox("Filed in the last", [14, 30, 60, 90, 180, 365], key="sm_days",
                         format_func=lambda d: f"{d} days")
    st.sidebar.selectbox("Chamber", ["Both", "House", "Senate"], key="sm_chamber")
    st.sidebar.selectbox("Trades", ["All", "Purchases", "Sales"], key="sm_type")
    st.sidebar.select_slider("Size at least", list(SM_AMOUNT_MIN), key="sm_amount")
    st.sidebar.text_input("Member name contains", key="sm_member", placeholder="e.g. Pelosi")
    st.sidebar.text_input("Tickers", key="sm_tick", placeholder="e.g. NVDA, MSFT")
    st.sidebar.toggle("Stocks with a ticker only", key="sm_only_tick")
    st.sidebar.slider("House reports to read (official source only)", 25, 400, step=25, key="sm_house_max",
                      help="Used only when Capitol Trades can't be reached: each House report is a PDF.")
    st.sidebar.caption("Members of Congress must report stock trades over \\$1,000 within 45 days (STOCK Act), "
                       "as a size range, not an exact amount. **Est. \\$** uses the middle of that range.")
    with st.spinner("Loading Congress trades …"):
        df, notes = congress_trades(int(ss["sm_days"]), int(ss["sm_house_max"]))
    for n in notes:
        st.caption("ℹ️ " + n)
    if df.empty:
        st.warning("No Congress trades loaded. The sources may be unreachable right now — try again later.")
        return
    if (df["Party"] == "").any() and congress_members():
        for i in df.index[df["Party"] == ""]:
            parts = str(df.at[i, "Politician"]).split()
            mem = _match_member(parts[0] if parts else "", parts[-1] if parts else "", df.at[i, "State"],
                                df.at[i, "Chamber"])
            if mem:
                df.at[i, "Party"] = mem["party"]
    df["Est. $"] = (df["Amount low"] + df["Amount high"].fillna(df["Amount low"])) / 2
    df["Amount"] = [_fmt_amount(lo, hi) for lo, hi in zip(df["Amount low"], df["Amount high"])]
    df["Days to report"] = (df["Filed"] - df["Traded"]).dt.days
    f = df
    if ss["sm_chamber"] != "Both":
        f = f[f["Chamber"] == ss["sm_chamber"]]
    if ss["sm_type"] != "All":
        f = f[f["Type"] == ss["sm_type"][:-1]]
    if SM_AMOUNT_MIN[ss["sm_amount"]]:
        f = f[f["Amount high"].fillna(f["Amount low"]) >= SM_AMOUNT_MIN[ss["sm_amount"]]]
    if ss.get("sm_member", "").strip():
        f = f[f["Politician"].str.contains(ss["sm_member"].strip(), case=False, na=False)]
    ticks = [t.replace(".", "-") for t in parse_tickers(ss.get("sm_tick", ""))] if ss.get("sm_tick", "").strip() else []
    if ticks:
        f = f[f["Ticker"].isin(ticks)]
    if ss["sm_only_tick"]:
        f = f[f["Ticker"].notna() & (f["Ticker"] != "")]
    buys, sells = f[f["Type"] == "Purchase"], f[f["Type"] == "Sale"]
    render_cards([("Trades", f"{len(f):,}", f"filed in the last {ss['sm_days']} days", "n"),
                  ("Members", f"{f['Politician'].nunique():,}", "who traded", "b"),
                  ("Purchases", f"{len(buys):,}", f"est. ${buys['Est. $'].sum() / 1e6:,.1f}M", "g"),
                  ("Sales", f"{len(sells):,}", f"est. ${sells['Est. $'].sum() / 1e6:,.1f}M", "r"),
                  ("Source", df["Source"].iloc[0] if df["Source"].nunique() == 1 else "Senate + House",
                   f"latest filing {df['Filed'].max():%b %d}" if df["Filed"].notna().any() else "", "n")])
    sub = st.segmented_control("View", ["Latest trades", "Most bought", "Most sold", "By member"], key="sm_cview",
                               label_visibility="collapsed") or "Latest trades"
    if sub == "Latest trades":
        show = f.sort_values(["Filed", "Traded"], ascending=False)[
            ["Filed", "Traded", "Politician", "Party", "Chamber", "State", "Ticker", "Asset", "Type", "Amount",
             "Owner", "Days to report", "Link"]].reset_index(drop=True)
        cfg = {"Filed": st.column_config.DateColumn(format="MMM D, YYYY"),
               "Traded": st.column_config.DateColumn(format="MMM D, YYYY"),
               "Link": st.column_config.LinkColumn("Filing", display_text="📄 open"),
               "Asset": st.column_config.TextColumn(width="medium"),
               "Days to report": st.column_config.NumberColumn("Lag", format="%d d",
                                                               help="Days between the trade and the filing")}
        sty = show.head(2000).style.map(lambda v: CHANGE_STYLE.get(v, ""), subset=["Type", "Party"])
        clicked = sm_table(sty, cfg, "smt_trades", height=min(35 * (len(show) + 1) + 3, 600))
        st.download_button("⬇ CSV", show.to_csv(index=False).encode(), "congress_trades.csv", "text/csv")
        sm_chart(list(show["Ticker"].dropna()), clicked, "sm_chart_cong")
    elif sub in ("Most bought", "Most sold"):
        t = f[f["Ticker"].notna() & (f["Ticker"] != "")]
        agg = t.groupby("Ticker").agg(
            Name=("Asset", "first"),
            Buyers=("Politician", lambda s: s[t.loc[s.index, "Type"] == "Purchase"].nunique()),
            Sellers=("Politician", lambda s: s[t.loc[s.index, "Type"] == "Sale"].nunique()),
            Buys=("Type", lambda s: (s == "Purchase").sum()), Sales=("Type", lambda s: (s == "Sale").sum()),
            **{"Est. bought $": ("Est. $", lambda s: s[t.loc[s.index, "Type"] == "Purchase"].sum()),
               "Est. sold $": ("Est. $", lambda s: s[t.loc[s.index, "Type"] == "Sale"].sum()),
               "Last trade": ("Traded", "max")},
            Who=("Politician", lambda s: ", ".join(sorted(set(s))[:6]))).reset_index()
        agg["Net buyers"] = agg["Buyers"] - agg["Sellers"]
        by = ["Net buyers", "Est. bought $"] if sub == "Most bought" else ["Net buyers", "Est. sold $"]
        agg = agg.sort_values(by, ascending=[sub != "Most bought", False])
        cfg = {"Est. bought $": st.column_config.NumberColumn(format="compact"),
               "Est. sold $": st.column_config.NumberColumn(format="compact"),
               "Last trade": st.column_config.DateColumn(format="MMM D"),
               "Net buyers": st.column_config.NumberColumn(format="%+d", help="Members buying − members selling"),
               "Who": st.column_config.TextColumn("Members", width="medium"),
               "Name": st.column_config.TextColumn(width="small")}
        clicked = sm_table(agg, cfg, "smt_agg")
        sm_chart(list(agg["Ticker"]), clicked, "sm_chart_agg")
    else:
        m = f.groupby("Politician").agg(
            Party=("Party", "first"), Chamber=("Chamber", "first"), State=("State", "first"),
            Trades=("Type", "size"), Buys=("Type", lambda s: (s == "Purchase").sum()),
            Sales=("Type", lambda s: (s == "Sale").sum()), **{"Est. volume $": ("Est. $", "sum"),
                                                              "Last filing": ("Filed", "max")},
            Tickers=("Ticker", lambda s: ", ".join(pd.Series(s.dropna()).value_counts().index[:8]))).reset_index()
        m = m.sort_values("Trades", ascending=False)
        cfg = {"Est. volume $": st.column_config.NumberColumn(format="compact"),
               "Last filing": st.column_config.DateColumn(format="MMM D, YYYY"),
               "Tickers": st.column_config.TextColumn("Most traded", width="large")}
        sm_table(m.style.map(lambda v: CHANGE_STYLE.get(v, ""), subset=["Party"]), cfg, "smt_members")
        st.caption("Tip: type a name under **Member name contains** to see just their trades.")




# ============================================================================
# Backtest: replay a scan over past days and trade every signal it gave
# ============================================================================
def build_pp(universe=None):
    """The calculation settings of every scan, exactly as set in the Stock Scanner tab."""
    pp = dict(max_range=ss["max_range"], min_pull=ss["pull_range"][0], max_pull=ss["pull_range"][1],
              min_bars=int(ss["min_bars"]), max_ema_dist=ss["max_ema_dist"], max_ema_gap=ss["max_ema_gap"],
              dry_up=ss["dry_up"], breakout_vol=ss["breakout_vol"],
              vr_days=int(ss["vr_days"]), vr_min_ratio=float(ss["vr_min_ratio"]), vr_dir=ss["vr_dir"],
              # only settings that change the calculations (a status choice or LiqLead values don't need a recompute)
              **{k: ss[k] for k in SCAN_EXTRA_DEFAULTS if k.startswith(("pb_", "ep_", "xb_", "gl_", "so_", "vcp_", "mac_", "htf_", "sh_", "rc_", "f5_"))
                 and k not in ("pb_status", "ep_status", "xb_status", "gl_status", "gl_ath", "so_status",
                               "vcp_status", "mac_status", "htf_status", "sh_status", "rc_status", "f5_status")})
    pp["xb_bench"] = MARKETS[universe]["bench"] if universe in MARKETS else "SPY"
    return pp


def rs_matrix(C):
    """IBD-style RS rating (1-99) for every stock on every day, ranked against the other stocks that day."""
    r = lambda n: C / C.shift(n) - 1
    raw = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)
    raw = raw.fillna(0.4 * r(63) + 0.6 * r(126))
    return (raw.rank(axis=1, pct=True) * 98 + 1).round()


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
BT_ANY, BT_ALL = "Signals from any of them", "Only stocks where all of them agree"
BT_AGREE = "✅ All agree"


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
BT_TRAILS = ["None", "10-day SMA", "20-day SMA", "50-day SMA"]
BT_SCALE_OUT = "Scale out: ⅓ · ⅓ · trail the rest"
BT_TARGETS = ["None", "R-multiple", "Percent", BT_SCALE_OUT]
BT_ENTRY_TRIGGER = "Trigger day, at the breakout price"
BT_ENTRY_330 = "Trigger day at 3:30 pm (above the trigger & up on the day)"
BT_ENTRIES = ["Next day's open", "Signal day's close", BT_ENTRY_TRIGGER, BT_ENTRY_330]
is_trig_entry = lambda e: e in (BT_ENTRY_TRIGGER, BT_ENTRY_330)       # setups armed the day before


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
BT_ATR_STOPS = {"ATR × 0.5": 0.5, "ATR × 0.8": 0.8, "ATR × 1": 1.0}
BT_STOPS = ["The setup's stop", "Low of the entry day", *BT_ATR_STOPS, "Fixed %"]
BT_UNIVERSES = ["S&P 500", "Nasdaq-listed", "All US stocks", *MARKETS, "Stock Scanner results", "My own tickers"]


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


PYR_STOPS = ["Raise to the previous buy (ladder)", "Raise to breakeven (average cost)", "Keep the trade's stop"]
PYR_RULES = ["Each day it still qualifies (SETUP or TRIGGER)", "Price ladder: every N × ATR higher",
             "Each day the trend is stacked (close ≥ EMA10 ≥ EMA21 ≥ EMA50)"]


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


STAGE_BENCHES = ["QQQE", "SPY", "QQQ", "RSP", "IWM"]
STAGE_SIZE_DEFAULT = {"1A": 50, "1B": 50, "2A": 100, "2B": 100, "2C": 75, "3A": 50, "3B": 25, "4A": 25, "4B": 0,
                      "4C": 0}


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


BT_LIST_UNIVERSES = {"S&P 500": "S&P 500 (~500 · 1–2 min)", "Nasdaq-listed": "Nasdaq-listed (~3,500)",
                     "All US stocks": "All US stocks (~6,000+)", **{m: m for m in MARKETS}}
BT_VOL = {"Any": 0, "100K+": 1e5, "300K+": 3e5, "500K+": 5e5, "1M+": 1e6, "2M+": 2e6, "5M+": 5e6}


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


MKT_PULLBACK = "SPY pullback only (below 21 EMA, above 50 EMA)"
MKT_CUSTOM = "Custom: pick the lines"
BT_MARKET_FILTERS = ["Off", "SPY in an uptrend", "SPY not in a downtrend", MKT_PULLBACK, MKT_CUSTOM]
MKT_LINES = ["10 EMA", "21 EMA", "50 EMA", "50 SMA", "200 SMA"]


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


# sectors the user left out (Stock Scanner sidebar → Sectors); used by every scan and backtest
DEFAULT_SECTOR_EXCL = ["Real Estate", "Utilities"]


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


BT_PRIORITY = ["Highest RS first", "Tightest stop first", "Alphabetical"]


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


OLD_BT_NAMES = {'🚩 Tight-flag breakout': '🚩 Tight Flag Pattern', '🔁 EMA crossback': '🔁 EMA Crossback', '🟢 Green line breakout': '✳️ Green Line Breakout', '🪤 Shakeout +3': '📈 Shakeout +3', '🚀 Episodic pivot': '🚀 Episodic Pivot', '📉 Parabolic short': '📉 Parabolic Short'}


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


def stock_theme(symbol, industry):
    """(theme name, its ETF) for a stock, or ("", None)."""
    name = THEME_OF_TICKER.get(str(symbol or "").upper())
    if name:
        return name, THEME_ETF.get(name)
    ind = str(industry or "")
    if ind and ind != "nan":
        etf = next((t for rx, t in ETF_INDUSTRY_RULES if ETF_LEVEL.get(t) == "Theme" and rx.search(ind)), None)
        if etf:
            return ETF_THEME_NAME.get(etf, ETF_NAME.get(etf, etf)), etf
    return "", None


def group_etfs(sector, industry, symbol=None):
    """{"Sector": ETF, "Industry": ETF, "Theme": ETF} for a stock (None where it has no match)."""
    ind = str(industry or "")
    ind = "" if ind == "nan" else ind
    hits = [t for rx, t in ETF_INDUSTRY_RULES if ind and rx.search(ind)]
    return {"Sector": ETF_SECTOR_OF.get(str(sector or "").strip().lower()),
            "Industry": next((t for t in hits if ETF_LEVEL.get(t) == "Industry"), None),
            "Theme": stock_theme(symbol, ind)[1]}


def group_check(sector, industry, levels, status_of, ok_set, symbol=None):
    """Does the stock pass the group filter? Every ticked level the stock HAS an ETF for must be in ok_set;
    a stock with no ETF at any ticked level fails. Returns (passes, has any group, text like '🔥 XLK · ❄️ SMH')."""
    g = group_etfs(sector, industry, symbol)
    used = [(lv, g[lv], status_of(g[lv])) for lv in levels if g[lv]]
    used = [(lv, t, s_) for lv, t, s_ in used if isinstance(s_, str)]
    if not used:
        return False, False, "no group"
    return (all(s_ in ok_set for _, t, s_ in used), True,
            " · ".join(f"{s_.split(' ')[0]} {t}" for _, t, s_ in used))


GRP_CORR_DAYS = 60


def group_used_etfs(sector, industry, levels, symbol=None):
    """The ETFs the group filter checks for a stock (one per ticked level it has)."""
    g = group_etfs(sector, industry, symbol)
    return [g[lv] for lv in levels if g[lv]]


def own_mover(rs, corr, rs_min, corr_min):
    """Skip the group check? A leader (RS ≥ rs_min) or a stock that doesn't move with its group ETFs (its highest
    60-day correlation < corr_min) is judged on its own chart. Returns ('' or a short label)."""
    if rs_min and pd.notna(rs) and rs >= rs_min:
        return f"⭐ leader (RS {rs:.0f})"
    if corr_min and pd.notna(corr) and corr < corr_min:
        return f"🚀 own mover (ρ {corr:.2f})"
    return ""


def group_etf(sector, industry, sector_only=False):
    """(industry-or-theme ETF or None, sector ETF or None) for a stock."""
    g = group_etfs(sector, "" if sector_only else industry)
    return g["Industry"] or g["Theme"], g["Sector"]


def etf_status_history(P, tickers=None):
    """Status of every ETF on every day (dates x ETFs), using only data up to that day:
    🔥 In play / 🌱 Emerging / ⚠️ Cooling / ❄️ Weak — the same rules as the Sectors in play tab.
    Also returns Week RS, Month RS and Trend (0-4) tables."""
    C = P["Close"]
    tk = [t for t in (tickers or ETF_TICKERS) if t in C.columns]
    C = C[tk]
    e21 = C.ewm(span=21, adjust=False).mean()
    s50, s200 = C.rolling(50).mean(), C.rolling(200, min_periods=150).mean()
    up = C > e21
    trend = up.astype(int) + (C > s50).astype(int) + (s50 > s200).astype(int) + (e21 > e21.shift(5)).astype(int)

    def pct_rank(r):                                   # 0-100 across the ETFs trading that day
        n = r.notna().sum(axis=1)
        return (r.rank(axis=1) - 1).div((n - 1).clip(lower=1), axis=0) * 100
    wk, mo = pct_rank(C / C.shift(5) - 1), pct_rank(C / C.shift(21) - 1)
    w, m = wk >= 50, mo >= 50
    names = list(ETF_STATUS)
    st_ = np.select([(up & w & m & (trend >= 3)).to_numpy(), (up & w).to_numpy(), m.to_numpy()], names[:3], names[3])
    status = pd.DataFrame(st_, index=C.index, columns=tk).where(C.notna() & (C.notna().cumsum() > 60))
    return status, wk, mo, trend


def market_trend_history(C, t="SPY"):
    """Uptrend / Choppy / Downtrend of an index ETF on every day (as on the Sectors in play tab)."""
    c = C[t]
    e21 = c.ewm(span=21, adjust=False).mean()
    s50, s200 = c.rolling(50).mean(), c.rolling(200, min_periods=150).mean()
    a21, a50, g200 = c > e21, c > s50, s50 > s200
    return pd.Series(np.where(a21 & a50 & g200, "Uptrend", np.where(~a21 & ~a50, "Downtrend", "Choppy")),
                     index=C.index).where(s50.notna())


def stock_group_status(sectors, industries, status_now, levels, ok_set, symbols=None):
    """Per stock: (passes the group filter, text of the ETFs checked)."""
    out_ok, out_t = [], []
    symbols = list(symbols) if symbols is not None else [None] * len(list(sectors))
    for sec, ind, sym in zip(sectors, industries, symbols):
        ok, _, txt = group_check(sec, ind, levels, status_now.get, ok_set, sym)
        out_ok.append(ok)
        out_t.append(txt if txt != "no group" else "")
    return out_ok, out_t


def etf_status_now():
    """{ETF: status} as of the latest close (prices shared with the Sectors in play tab, kept 30 min)."""
    P, _ = etf_panels(ETF_TICKERS + [t for t, _ in ETF_MARKET])
    if not P:
        return {}
    return etf_status_history(P)[0].iloc[-1].dropna().to_dict()


def etf_panels(tickers, force=False):
    """Daily prices of the ETFs, kept for 30 minutes."""
    key = tuple(sorted(tickers))
    c = ss.get("etf_cache")
    if not force and c and c["key"] == key and time.time() - c["t"] < 30 * 60:
        return c["P"], c["t"]
    box = st.empty()
    with box.container():
        msg, bar = st.empty(), st.progress(0.0)
        prices = download_all(list(tickers), bar, msg, period="2y")
        P = build_panels(prices, rows=600) if prices else None
    box.empty()
    ss["etf_cache"] = dict(key=key, P=P, t=time.time())
    return P, ss["etf_cache"]["t"]


def etf_trend(C, t):
    """(above 21 EMA, above 50 SMA, 50 SMA above 200 SMA, 21 EMA rising) for one ticker."""
    c = C[t].dropna()
    if len(c) < 60:
        return (False,) * 4
    e21 = c.ewm(span=21, adjust=False).mean()
    s50, s200 = c.rolling(50).mean(), c.rolling(200, min_periods=150).mean()
    last = c.iat[-1]
    return (bool(last > e21.iat[-1]), bool(last > s50.iat[-1]),
            bool(pd.notna(s200.iat[-1]) and s50.iat[-1] > s200.iat[-1]), bool(e21.iat[-1] > e21.iat[-6]))


def etf_table(P, rows, bench="SPY"):
    """One row per ETF: returns, RS vs the benchmark and the other ETFs, trend checks, stage, score, status."""
    C, H, L = P["Close"], P["High"], P["Low"]
    have = [r for r in rows if r[0] in C.columns and C[r[0]].notna().sum() > 30]
    tk = [r[0] for r in have]
    last = C.iloc[-1]
    ret = lambda k: (last / C.shift(k).iloc[-1] - 1) * 100
    r = {k: ret(k) for k in (1, 5, 21, 63, 126, 189, 252)}
    b = {k: (r[k].get(bench, np.nan) if bench in C else np.nan) for k in r}
    df = pd.DataFrame({"ETF": tk, "Name": [x[1] for x in have], "Sector": [x[2] for x in have],
                       "Type": [x[3] for x in have]}).set_index("ETF")
    for k, col in [(1, "1D %"), (5, "1W %"), (21, "1M %"), (63, "3M %"), (126, "6M %")]:
        df[col] = r[k].reindex(tk)
    df["vs SPY 1M"] = df["1M %"] - b[21]
    raw = 0.4 * r[63] + 0.2 * r[126] + 0.2 * r[189] + 0.2 * r[252]
    raw = raw.fillna(0.4 * r[63] + 0.6 * r[126]).reindex(tk)
    df["RS"] = (raw.rank(pct=True) * 98 + 1).round()
    hi52 = H[tk].rolling(252, min_periods=60).max().iloc[-1]
    df["Off high %"] = (last[tk] / hi52 - 1) * 100
    tr = {t: etf_trend(C, t) for t in tk}
    df["> 21 EMA"] = [tr[t][0] for t in tk]
    df["> 50 SMA"] = [tr[t][1] for t in tk]
    df["50 > 200"] = [tr[t][2] for t in tk]
    df["21 EMA rising"] = [tr[t][3] for t in tk]
    df["Trend"] = df[["> 21 EMA", "> 50 SMA", "50 > 200", "21 EMA rising"]].sum(axis=1).astype(int)
    e10, e20 = C[tk].ewm(span=10, adjust=False).mean().iloc[-1], C[tk].ewm(span=20, adjust=False).mean().iloc[-1]
    s50, s200 = C[tk].rolling(50).mean().iloc[-1], C[tk].rolling(200, min_periods=150).mean().iloc[-1]
    pc = C[tk].shift(1)
    trr = np.fmax(np.fmax((H[tk] - L[tk]).to_numpy(), (H[tk] - pc).abs().to_numpy()), (L[tk] - pc).abs().to_numpy())
    atr = pd.DataFrame(trr, index=C.index, columns=tk).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    df["Stage"] = [rs_classify_stage(last[t], e10[t], e20[t], s50[t], s200[t], atr[t]) for t in tk]
    hs, hw, hm, _ = etf_status_history(P, tk)
    df["Week RS"], df["Month RS"] = hw.iloc[-1].reindex(tk).round(), hm.iloc[-1].reindex(tk).round()
    df["Status"] = hs.iloc[-1].reindex(tk).fillna("❄️ Weak")
    df["Score"] = (0.35 * df["RS"] + 0.15 * df["Week RS"] + 0.2 * df["Month RS"] + 0.3 * df["Trend"] / 4 * 100).round()
    return df.sort_values("Score", ascending=False)


def market_cards(P):
    cards = []
    C = P["Close"]
    for t, name in ETF_MARKET:
        if t not in C or C[t].notna().sum() < 60:
            continue
        a21, a50, g200, rising = etf_trend(C, t)
        c = C[t].dropna()
        d1, m1 = (c.iat[-1] / c.iat[-2] - 1) * 100, (c.iat[-1] / c.iat[-22] - 1) * 100
        if a21 and a50 and g200:
            status, tone = "Uptrend", "g"
        elif not a21 and not a50:
            status, tone = "Downtrend", "r"
        else:
            status, tone = "Choppy", "y"
        ck = lambda b: "✓" if b else "✗"
        cards.append((f"{t} · {name}", status,
                      f"{d1:+.1f}% today · {m1:+.1f}% 1M · 21 EMA {ck(a21)} 50 SMA {ck(a50)} 50>200 {ck(g200)}", tone))
    return cards


def etf_page():
    st.sidebar.title("🗂 Sectors in play")
    kind_opts = ["All", "Sectors", "Industries", "Themes"]
    if ss.get("etf_kind") not in kind_opts:
        ss["etf_kind"] = {"11 sectors only": "Sectors", "Industries & themes only": "Industries"}.get(ss.get("etf_kind"), "All")
    kinds = st.sidebar.radio("ETFs", kind_opts, key="etf_kind", horizontal=True)
    rows = [r for r in ETF_LIST if kinds == "All" or r[3] == kinds[:-1].replace("Industrie", "Industry")]
    refresh = st.sidebar.button("🔄 Refresh prices", use_container_width=True)
    st.sidebar.caption("**How an ETF is rated**\n\n"
                       "**RS** 1–99: 3-, 6-, 9- and 12-month performance ranked against the other ETFs (IBD-style).\n\n"
                       "**Week RS / Month RS** 0–100: 1-week and 1-month performance vs SPY, ranked.\n\n"
                       "**Trend** 0–4: above the 21 EMA · above the 50 SMA · 50 SMA above the 200 SMA · 21 EMA rising.\n\n"
                       "**Score** = 35% RS + 15% Week RS + 20% Month RS + 30% Trend.\n\n"
                       "🔥 **In play** — above the 21 EMA, strong this week AND this month, trend 3–4 of 4 · "
                       "🌱 **Emerging** — above the 21 EMA and strong this week, not yet this month · "
                       "⚠️ **Cooling** — strong this month but not this week (or lost the 21 EMA) · "
                       "❄️ **Weak** — the rest.")

    st.title("Sectors in play")
    st.caption("Which sectors and industries are leading right now, from their ETFs. Trade stocks in the 🔥 groups; "
               "breakouts work far better when their group is moving too. To scan or backtest only these groups: "
               "**🗂 Groups** filter on the Stock Scanner, **Group filter** in the Backtest sidebar.")
    P, t_loaded = etf_panels(ETF_TICKERS + [t for t, _ in ETF_MARKET], force=refresh)
    if not P:
        st.error("Couldn't download ETF prices — check the internet connection and press Refresh.")
        return
    # ---- history slider: see the page as it looked on an earlier day (only data up to that day is used) ----
    P_all = P
    dates = P["Close"].index
    day_opts = list(dates[min(150, len(dates) - 1):])
    if ss.get("etf_asof") not in day_opts:
        ss["etf_asof"] = day_opts[-1]
    h1, h2 = st.columns([6, 1], vertical_alignment="bottom")
    h1.select_slider("As of", day_opts, key="etf_asof", format_func=lambda d: f"{d:%a %b %d, %Y}",
                     help="Slide back to see which groups were in play on an earlier day. Everything on the page "
                          "(market cards, statuses, ranking, map) uses only prices up to that day.")
    h2.button("Today", use_container_width=True, key="go_etf_today",
              on_click=lambda: ss.__setitem__("etf_asof", day_opts[-1]))
    asof = ss["etf_asof"]
    past = asof != dates[-1]
    if past:
        P = {k: v.loc[:asof] for k, v in P_all.items()}
    render_cards(market_cards(P))
    df = etf_table(P, ETF_LIST)                              # rated against ALL the ETFs (same as the filters)
    df = df[df.index.isin([r[0] for r in rows])]
    if df.empty:
        st.warning("No ETF prices came back.")
        return
    after_txt = ""
    if past:                                                # what happened next — how good the ratings were
        C_all = P_all["Close"]
        i0 = dates.get_loc(asof)
        i1 = min(i0 + 21, len(dates) - 1)
        nxt = (C_all.iloc[i1] / C_all.iloc[i0] - 1) * 100
        df.insert(df.columns.get_loc("Status") + 1, "Next %", nxt.reindex(df.index))
        spy_n = nxt.get("SPY", np.nan)
        grp = df.groupby("Status")["Next %"].mean()
        after_txt = (f" · **what happened next** ({i1 - i0} trading days, to {dates[i1]:%b %d}): SPY {spy_n:+.1f}%, "
                     + ", ".join(f"{k.split(' ')[0]} avg {grp[k]:+.1f}%" for k in ETF_STATUS if k in grp))
    n_in = (df["Status"] == "🔥 In play").sum()
    br21, br50 = df["> 21 EMA"].mean() * 100, df["> 50 SMA"].mean() * 100
    (st.warning if past else st.caption)(
        (f"🕰 **History view — as of {asof:%b %d, %Y}**" if past else f"Prices as of {asof:%b %d, %Y}")
        + f" · loaded {pd.Timestamp(t_loaded, unit='s', tz='UTC').tz_convert('America/New_York'):%H:%M} ET · "
        f"**{n_in}** of {len(df)} ETFs in play · breadth: {br21:.0f}% above the 21 EMA, "
        f"{br50:.0f}% above the 50 SMA" + after_txt)
    chips = "".join(f'<span class="chip" style="border-left-color:{ETF_STATUS[s]}">{t} <span style="font-weight:400;'
                    f'opacity:.75">{html_lib.escape(n)}</span></span>'
                    for t, n, s in zip(df.index, df["Name"], df["Status"]) if s in ("🔥 In play", "🌱 Emerging"))
    if chips:
        rs_html(f'<div style="margin:.2rem 0 .6rem 0"><span style="font-size:12px;opacity:.7;margin-right:6px">'
                f'🔥 In play &amp; 🌱 emerging:</span>{chips}</div>')

    view = (st.segmented_control("View", ["Ranking", "Rotation map", "Performance"], key="etf_view",
                                 label_visibility="collapsed") if hasattr(st, "segmented_control") else
            st.radio("View", ["Ranking", "Rotation map", "Performance"], key="etf_view", horizontal=True,
                     label_visibility="collapsed")) or "Ranking"
    pick = None
    if view == "Ranking":
        show = df.reset_index()[["ETF", "Name", "Status"] + (["Next %"] if past else []) + ["Score", "RS", "Week RS", "Month RS", "1D %", "1W %", "1M %",
                                 "3M %", "6M %", "vs SPY 1M", "Off high %", "Trend", "Stage", "Sector", "Type"]]
        pct = st.column_config.NumberColumn(format="%+.1f%%")
        cfg = {"Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
               "RS": st.column_config.NumberColumn(format="%d"), "Week RS": st.column_config.NumberColumn(format="%d"),
               "Month RS": st.column_config.NumberColumn(format="%d"),
               "1D %": pct, "1W %": pct, "1M %": pct, "3M %": pct, "6M %": pct,
               "vs SPY 1M": st.column_config.NumberColumn(format="%+.1f%%", help="1-month return minus SPY's"),
               "Next %": st.column_config.NumberColumn(format="%+.1f%%", help="What the ETF did AFTER the as-of day "
                                                       "(the next 21 trading days, or up to today)"),
               "Off high %": st.column_config.NumberColumn(format="%.1f%%", help="Distance from the 52-week high"),
               "Trend": st.column_config.NumberColumn(format="%d/4", help="Above the 21 EMA · above the 50 SMA · "
                                                                          "50 SMA above the 200 SMA · 21 EMA rising"),
               "Status": st.column_config.TextColumn(width="medium"),
               "Stage": st.column_config.TextColumn(help="Price cycle stage — 2A/2B = uptrend, 4A/4B = downtrend")}
        sty = show.style.map(lambda v: "color:#26a69a" if isinstance(v, float) and v > 0 else
                             "color:#ef5350" if isinstance(v, float) and v < 0 else "",
                             subset=["1D %", "1W %", "1M %", "3M %", "6M %", "vs SPY 1M"] + (["Next %"] if past else [])) \
            .map(lambda v: f"color:{ETF_STATUS.get(v, '')}; font-weight:600", subset=["Status"])
        ev = st.dataframe(sty, hide_index=True, use_container_width=True, column_config=cfg, key="etf_tbl",
                          on_select="rerun", selection_mode="single-row", height=min(35 * (len(show) + 1) + 3, 640))
        rows_sel = list(getattr(ev.selection, "rows", []) or [])
        pick = show["ETF"].iat[rows_sel[0]] if rows_sel else None
        st.download_button("⬇ ETF ranking CSV", show.to_csv(index=False).encode(), "sectors_in_play.csv", "text/csv",
                           key="dl_etf_csv")
        st.caption("Click a row to chart it below.")
    elif view == "Rotation map":
        d = df.reset_index()
        fig = rs_chart(d, "ETF", "Status", ["Name", "Score", "1W %", "1M %", "Trend"], height=640)
        for tr in fig.data:
            if tr.name in ETF_STATUS:
                tr.marker.color = ETF_STATUS[tr.name]
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Right = stronger than SPY this week, up = stronger this month (ranked among these ETFs). Groups "
                   "usually rotate clockwise: Improving → Strong → Weakening → Weak.")
    else:
        tf = st.radio("Period", ["1D %", "1W %", "1M %", "3M %", "6M %"], index=2, horizontal=True, key="etf_tf")
        d = df.sort_values(tf)
        fig = go.Figure(go.Bar(x=d[tf], y=[f"{t} · {n}" for t, n in zip(d.index, d["Name"])], orientation="h",
                               marker_color=[ETF_STATUS[s] for s in d["Status"]],
                               hovertemplate="%{y}<br>%{x:+.1f}%<extra></extra>"))
        if "SPY" in P["Close"]:
            k = {"1D %": 1, "1W %": 5, "1M %": 21, "3M %": 63, "6M %": 126}[tf]
            c = P["Close"]["SPY"].dropna()
            if len(c) > k:
                fig.add_vline(x=(c.iat[-1] / c.iat[-1 - k] - 1) * 100, line_dash="dot", line_color="#9ca3af",
                              annotation_text="SPY", annotation_position="top")
        fig.update_layout(height=max(360, 18 * len(d) + 80), margin=dict(l=10, r=10, t=30, b=10),
                          xaxis=dict(ticksuffix="%"), template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bar colour = status. The dotted line is SPY over the same period.")

    # ---- chart of one ETF ----
    opts = list(df.index)
    if pick and pick in opts:
        ss["etf_chart"] = pick
    if ss.get("etf_chart") not in opts:
        ss["etf_chart"] = opts[0]
    c1, c2 = st.columns([2, 1.2], vertical_alignment="bottom")
    sym = c1.selectbox("Chart", opts, key="etf_chart",
                       format_func=lambda t: f"{t} · {df.at[t, 'Name']} · {df.at[t, 'Status']}")
    ss.setdefault("etf_src", "Built-in")
    src = (c2.segmented_control("Chart type", ["Built-in", "TradingView"], key="etf_src",
                                label_visibility="collapsed") if hasattr(st, "segmented_control") else
           c2.radio("Chart type", ["Built-in", "TradingView"], key="etf_src", horizontal=True,
                    label_visibility="collapsed")) or "Built-in"
    if src == "TradingView":
        tpls, active = load_templates()
        tpl = {**DEFAULT_TPL, **tpls.get(ss.get("tpl_name", active), tpls["Default"])}
        ex_map = get_exchange_map()
        tradingview_chart(tv_symbol(sym, ex_map), tpl, [tv_symbol(x, ex_map) for x in opts[:100]])
    else:
        end = dates[min(dates.get_loc(asof) + 42, len(dates) - 1)]      # history view: also the next ~2 months
        one = pd.DataFrame({f: P_all[f][sym].loc[:end] for f in ["Open", "High", "Low", "Close", "Volume"]})
        fig = price_chart(sym, one, days=160 + (dates.get_loc(end) - dates.get_loc(asof)))
        if past:
            fig.add_vline(x=asof, line_dash="dot", line_color="#f59e0b")
            fig.add_annotation(x=asof, y=1, yref="paper", text=f"as of {asof:%b %d}", showarrow=False,
                               font=dict(color="#f59e0b", size=11), xanchor="left")
        st.plotly_chart(fig, use_container_width=True)
    r = df.loc[sym]
    st.caption(f"**{sym} · {r['Name']}** ({r['Sector']}) — {r['Status']} · score {r['Score']:.0f} · RS {r['RS']:.0f} · "
               f"trend {r['Trend']}/4 · stage {r['Stage']} · {r['Off high %']:.1f}% from its 52-week high")


# ---------------------------------------------------------------------------------------------------------------
# 💼 Positions: pilot + add to winners — the daily action list for your open positions
# ---------------------------------------------------------------------------------------------------------------
POS_FILE = Path(__file__).with_name("positions.json")
POS_COLS = ["Symbol", "Pilot date", "Buys", "Stop", "Note"]


def load_positions():
    try:
        rows = json.loads(POS_FILE.read_text(encoding="utf-8"))
        return pd.DataFrame(rows, columns=POS_COLS)
    except Exception:
        return pd.DataFrame(columns=POS_COLS)


def save_positions(df):
    rows = []
    for r in df.to_dict("records"):
        sym = str(r.get("Symbol") or "").strip().upper()
        if not sym:
            continue
        stop = r.get("Stop")
        rows.append({"Symbol": sym, "Pilot date": str(r.get("Pilot date") or "")[:10],
                     "Buys": str(r.get("Buys") or "").strip(),
                     "Stop": float(stop) if stop is not None and pd.notna(stop) and str(stop) != "" else None,
                     "Note": str(r.get("Note") or "")})
    try:
        POS_FILE.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return True
    except Exception:
        return False


def parse_buys(txt):
    """'27.48, 31.2 35' -> [27.48, 31.2, 35.0]"""
    out = []
    for x in re.split(r"[,\s;/]+", str(txt or "").strip()):
        try:
            v = float(x.replace("$", ""))
            if v > 0:
                out.append(v)
        except ValueError:
            pass
    return out


def pos_actions(pos, P, step, max_units, stop_mode, trail, acct, unit_pct, rule=None, profit_only=True, qual=None):
    """Today's action for each open position: SELL (stop hit / closed below the trailing SMA), ADD 1 unit (closed a
    rung above the last buy), NEAR, HOLD or FULL — with the stop to use after an add."""
    rows = []
    if P is None:
        return pd.DataFrame()
    C, H, L = P["Close"], P["High"], P["Low"]
    pc = C.shift(1)
    atr = pd.DataFrame(np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy()),
                       index=C.index, columns=C.columns).rolling(14, min_periods=10).mean()
    ma_n = {"10-day SMA": 10, "20-day SMA": 20, "50-day SMA": 50}.get(trail)
    rule = rule or PYR_RULES[1]
    if rule == PYR_RULES[2]:
        e10, e21, e50 = (C.ewm(span=k_, adjust=False).mean() for k_ in (10, 21, 50))
        stacked = (C >= e10) & (e10 >= e21) & (e21 >= e50)
    for r in pos.to_dict("records"):
        sym = str(r.get("Symbol") or "").strip().upper()
        buys = parse_buys(r.get("Buys"))
        if not sym:
            continue
        row = {"Symbol": sym, "Units": len(buys)}
        if sym not in C or C[sym].dropna().empty or not buys:
            row["Action"] = "⚠️ No price data" if buys else "⚠️ Enter the buy price(s)"
            rows.append(row)
            continue
        c_ = C[sym].dropna()
        last_day = c_.index[-1]
        close, low = float(c_.iat[-1]), float(L[sym].loc[last_day])
        try:
            pdt = pd.Timestamp(str(r.get("Pilot date") or "")[:10])
        except Exception:
            pdt = pd.NaT
        a_ser = atr[sym].dropna()
        known = a_ser[a_ser.index < pdt] if pd.notna(pdt) else a_ser.iloc[:0]
        a0 = float(known.iat[-1]) if len(known) else (float(a_ser.iat[-1]) if len(a_ser) else np.nan)
        rung = step * a0
        stop = r.get("Stop")
        stop = float(stop) if stop is not None and pd.notna(stop) and str(stop) != "" else np.nan
        units, lastb = len(buys), buys[-1]
        avg = units / sum(1 / b for b in buys)
        nxt = lastb + rung if np.isfinite(rung) else np.nan
        ma = float(C[sym].rolling(ma_n).mean().loc[last_day]) if ma_n else np.nan
        new_stop = np.nan
        if np.isfinite(stop) and low <= stop:
            act = "🔴 SELL — stop hit"
        elif ma_n and np.isfinite(ma) and close < ma and last_day > (pdt if pd.notna(pdt) else last_day - pd.Timedelta(days=1)):
            act = f"🔴 SELL — closed below the {trail}"
        elif units >= max_units:
            act = "🔵 FULL — hold, trail the stop"
        elif rule != PYR_RULES[1]:
            if rule == PYR_RULES[2]:
                q_ = bool(stacked[sym].loc[last_day])
                why = "trend stacked (close ≥ EMA10 ≥ 21 ≥ 50)"
            else:
                q_ = (qual or {}).get(sym)
                why = "still a SETUP/TRIGGER in the last scan"
            if q_ is None:
                act = "⚪ Hold — not in the last scan"
            elif q_ and (not profit_only or close > avg):
                act = f"🟢 ADD 1 unit — {why}"
            elif q_:
                act = "⚪ Hold — qualifies, but not in profit"
            else:
                act = "⚪ Hold — doesn't qualify today"
            if act.startswith("🟢"):
                if stop_mode == PYR_STOPS[1]:
                    new_stop = (units + 1) / (sum(1 / b for b in buys) + 1 / close)
                elif stop_mode == PYR_STOPS[0]:
                    new_stop = lastb
                else:
                    new_stop = stop
                new_stop = new_stop if (np.isfinite(new_stop) and new_stop < close) else stop
                if np.isfinite(stop) and np.isfinite(new_stop):
                    new_stop = max(new_stop, stop)
        elif np.isfinite(nxt) and close >= nxt:
            act = "🟢 ADD 1 unit"
            if stop_mode == PYR_STOPS[0]:
                new_stop = lastb
            elif stop_mode == PYR_STOPS[1]:
                new_stop = (units + 1) / (sum(1 / b for b in buys) + 1 / close)
            else:
                new_stop = stop
            if np.isfinite(stop):
                new_stop = max(new_stop, stop)
        elif np.isfinite(nxt) and close >= lastb + 0.75 * rung:
            act = "🟡 Near the next add"
        else:
            act = "⚪ Hold"
        row.update({"Action": act, "Close": close, "Date": last_day, "Avg cost": avg,
                    "P&L %": (close / avg - 1) * 100, "Stop": stop,
                    "To stop %": (close / stop - 1) * 100 if np.isfinite(stop) else np.nan,
                    "New stop": new_stop,
                    **({"Next add at": nxt if units < max_units else np.nan,
                        "To add %": (nxt / close - 1) * 100 if units < max_units and np.isfinite(nxt) else np.nan,
                        "Rung $": rung} if rule == PYR_RULES[1] else {}), "Add shares": np.floor(acct * unit_pct / 100 / close) if close > 0 else np.nan,
                    "Position $": sum(np.floor(acct * unit_pct / 100 / b) * close for b in buys)})
        rows.append(row)
    return pd.DataFrame(rows)


def positions_page():
    st.title("💼 Positions — pilot + add to winners")
    st.caption("Start every new trade with a small **pilot** (1 unit). Each time a position closes another rung "
               "(N × ATR) above your last buy, **add 1 unit** at the close and **raise the stop**. Losers cost only "
               "the pilot; winners grow. Test the same rules in 🧪 Backtest → **📈 Pilot + add to winners**.")
    st.sidebar.markdown("**Rules** (shared with the Backtest)")
    a1, a2 = st.sidebar.columns(2)
    a1.number_input("Account $", 1000, 100_000_000, step=10_000, key="bt_acct")
    a2.number_input("Unit, % of account", 0.25, 50.0, step=0.25, key="bt_pyr_unit")
    if ss.get("bt_pyr_rule") not in PYR_RULES:
        ss["bt_pyr_rule"] = PYR_RULES[0]
    st.sidebar.selectbox("Add 1 unit", PYR_RULES, key="bt_pyr_rule",
                         help="**Still qualifies** — the stock is a SETUP or TRIGGER in your last Stock Scanner run. "
                              "**Stacked trend** — close ≥ EMA10 ≥ EMA21 ≥ EMA50 today. **Price ladder** — the close "
                              "is N × ATR above your last buy.")
    c1, c2 = st.sidebar.columns(2)
    if ss["bt_pyr_rule"] == PYR_RULES[1]:
        c1.number_input("Add every × ATR", 0.25, 10.0, step=0.25, key="bt_pyr_step",
                        help="14-day ATR on the day before the pilot. Add when the close is this far above the last buy.")
    else:
        c1.checkbox("Only while in profit", key="bt_pyr_profit")
    c2.number_input("Max units", 1, 20, key="bt_pyr_max")
    st.sidebar.selectbox("After each add, the stop", PYR_STOPS, key="bt_pyr_stop")
    if ss.get("bt_trail") not in BT_TRAILS:
        ss["bt_trail"] = "20-day SMA"
    st.sidebar.selectbox("Also sell on a close below", BT_TRAILS, key="bt_trail")
    acct, unit = float(ss["bt_acct"]), float(ss["bt_pyr_unit"])
    st.sidebar.checkbox("📊 Size by market stage", key="bt_stage_size",
                        help="Same as in the Backtest: today's unit size is scaled by the market ETF's stage.")
    if ss["bt_stage_size"]:
        sh_ = bt_bench_stages(ss["bt_stage_bench"])
        if sh_ is not None and len(sh_):
            sg_now = sh_.iat[-1]
            m_now = bt_stage_pct().get(sg_now, 100)
            st.sidebar.markdown(f"Market **{ss['bt_stage_bench']}** is **{sg_now}** "
                                f"({RS_STAGES.get(sg_now, ('', ''))[0]}) → units at **{m_now}%** size.")
            unit = unit * m_now / 100
            if m_now == 0:
                st.warning(f"📊 {ss['bt_stage_bench']} is in stage **{sg_now}** — your stage sizing says **no new "
                           "trades** today (0% size). Existing positions: follow their stops.")
        else:
            st.sidebar.caption(f"Couldn't load {ss['bt_stage_bench']} — stage sizing off.")
    st.sidebar.caption(f"1 unit = **\\${acct * unit / 100:,.0f}** · a full position = {int(ss['bt_pyr_max'])} units = "
                       f"**\\${acct * unit / 100 * int(ss['bt_pyr_max']):,.0f}**.")

    if "pos_df" not in ss:
        ss["pos_df"] = load_positions()
        ss["pos_ver"] = 0
    st.markdown("**Your open positions** — one row per stock. *Buys* = every price you bought at, pilot first "
                "(e.g. `27.48, 31.20`). *Stop* = where your stop is now.")
    ed = st.data_editor(ss["pos_df"], num_rows="dynamic", use_container_width=True, hide_index=True,
                        key=f"pos_ed_{ss.get('pos_ver', 0)}",
                        column_config={"Symbol": st.column_config.TextColumn(required=True, width="small"),
                                       "Pilot date": st.column_config.TextColumn(help="YYYY-MM-DD — sets the ATR "
                                                                                 "used for the rungs"),
                                       "Buys": st.column_config.TextColumn(help="Every buy price, pilot first, "
                                                                           "separated by commas", width="medium"),
                                       "Stop": st.column_config.NumberColumn(format="%.2f"),
                                       "Note": st.column_config.TextColumn(width="medium")})
    ed = ed.copy()
    ed["Symbol"] = ed["Symbol"].astype(str).str.strip().str.upper().replace({"NAN": "", "NONE": ""})
    if not ed.fillna("").astype(str).equals(ss["pos_df"].fillna("").astype(str)):
        ss["pos_df"] = ed
        if not save_positions(ed):
            st.warning("Couldn't save positions.json next to the app — changes last only for this session.")
    pos = ss["pos_df"][ss["pos_df"]["Symbol"].astype(str).str.len() > 0]

    tick = sorted(set(pos["Symbol"]))
    refresh = st.button("🔄 Refresh prices", help="Prices are kept for the session; press after the close for "
                                                  "today's actions.")
    pkey = (tuple(tick), date.today().isoformat())
    if tick and (refresh or ss.get("pos_pkey") != pkey):
        with st.spinner(f"Downloading prices for {len(tick)} stocks …"):
            m_, b_ = st.empty(), st.empty()
            prices = download_all(tick, b_, m_, period="1y")
            m_.empty(); b_.empty()
            ss["pos_panels"] = build_panels(prices, rows=100000) if prices else None
            ss["pos_pkey"] = pkey
    if not tick:
        st.info("**No positions yet.** Type your open trades in the table above (click the empty row: Symbol, "
                "Pilot date, Buys, Stop) — or add new pilots from the list below. Then this page shows what to do "
                "with each one today: SELL, ADD 1 unit, or HOLD.")
    else:
        qual = None
        if ss["bt_pyr_rule"] == PYR_RULES[0] and ss.get("metrics") is not None and ss.get("scan_mode") in MODE_SPEC:
            qcol = F.get(MODE_SPEC[ss["scan_mode"]][0], {}).get("col")
            mt = ss["metrics"]
            if qcol and qcol in mt:
                qual = {x: str(mt.at[x, qcol]) in ("SETUP", "TRIGGER", "BREAKOUT", "RETEST", "NEAR", "RECLAIMED")
                        for x in pos["Symbol"] if x in mt.index}
            st.caption(f"'Still qualifies' uses your last scan: **{ss.get('last_results_label') or ss['scan_mode']}**.")
        elif ss["bt_pyr_rule"] == PYR_RULES[0]:
            st.caption("'Still qualifies' needs a Stock Scanner run in this session — run your scan first.")
        act = pos_actions(pos, ss.get("pos_panels"), float(ss["bt_pyr_step"]), int(ss["bt_pyr_max"]),
                          ss["bt_pyr_stop"], ss["bt_trail"], acct, unit, rule=ss["bt_pyr_rule"],
                          profit_only=bool(ss["bt_pyr_profit"]), qual=qual)
        if len(act):
            n_ = lambda k: int(act["Action"].astype(str).str.startswith(k).sum())
            render_cards([("Positions", f"{len(act)}", f"{int(act['Units'].sum())} units held", "n"),
                          ("Add", f"{n_('🟢')}", "closed a rung above the last buy", "g"),
                          ("Sell", f"{n_('🔴')}", "stop hit / below the trailing SMA", "r"),
                          ("Near an add", f"{n_('🟡')}", "within ¼ rung", "y")])
            order = {"🔴": 0, "🟢": 1, "🟡": 2, "⚪": 3, "🔵": 4, "⚠": 5}
            act = act.assign(_o=act["Action"].astype(str).str[:1].map(order).fillna(9)).sort_values(["_o", "Symbol"]) \
                .drop(columns="_o")
            asof = act["Date"].dropna().max() if "Date" in act else None
            st.markdown(f"**Today's actions**" + (f" — prices as of {asof:%a %b %d, %Y}" if pd.notna(asof) else ""))
            st.dataframe(act.drop(columns=["Date"], errors="ignore"), hide_index=True, use_container_width=True,
                         column_config={"Close": st.column_config.NumberColumn(format="%.2f"),
                                        "Avg cost": st.column_config.NumberColumn(format="%.2f"),
                                        "P&L %": st.column_config.NumberColumn(format="%+.1f%%"),
                                        "Stop": st.column_config.NumberColumn(format="%.2f"),
                                        "To stop %": st.column_config.NumberColumn(format="%+.1f%%",
                                                                                   help="How far the close is above the stop"),
                                        "New stop": st.column_config.NumberColumn(format="%.2f",
                                                                                  help="Raise the stop here after the add"),
                                        "Next add at": st.column_config.NumberColumn(format="%.2f",
                                                                                     help="Close at or above this → add 1 unit"),
                                        "To add %": st.column_config.NumberColumn(format="%+.1f%%"),
                                        "Rung $": st.column_config.NumberColumn(format="%.2f", help="N × ATR"),
                                        "Add shares": st.column_config.NumberColumn(format="%d",
                                                                                    help="Shares for 1 unit at today's close"),
                                        "Position $": st.column_config.NumberColumn(format="dollar",
                                                                                    help="Value now (unit $ ÷ each buy price × close)")})
            adds = act[act["Action"].astype(str).str.startswith("🟢")]
            sells = act[act["Action"].astype(str).str.startswith("🔴")]
            b1, b2 = st.columns(2)
            if b1.button(f"✅ Record {len(adds)} add(s) at today's close", disabled=adds.empty,
                         help="Appends today's close to Buys and moves the Stop to 'New stop'."):
                df = ss["pos_df"].copy()
                for r in adds.to_dict("records"):
                    m_ = df["Symbol"] == r["Symbol"]
                    df.loc[m_, "Buys"] = df.loc[m_, "Buys"].astype(str) + f", {r['Close']:.2f}"
                    if pd.notna(r["New stop"]):
                        df.loc[m_, "Stop"] = round(float(r["New stop"]), 2)
                ss["pos_df"] = df; ss["pos_ver"] = ss.get("pos_ver", 0) + 1
                save_positions(df)
                st.rerun()
            if b2.button(f"🗑 Remove {len(sells)} sold position(s)", disabled=sells.empty):
                df = ss["pos_df"][~ss["pos_df"]["Symbol"].isin(sells["Symbol"])]
                ss["pos_df"] = df.reset_index(drop=True); ss["pos_ver"] = ss.get("pos_ver", 0) + 1
                save_positions(ss["pos_df"])
                st.rerun()
            st.caption("SELL = the day's low reached your stop, or the close was below the trailing SMA. ADD = today's add "
                       "rule is met (one add per day). Stops are only ever raised, never to or above the price.")

    # ---- new pilots from the last Stock Scanner run ----
    st.markdown("**New pilots** — today's TRIGGERs from your last Stock Scanner run")
    metrics, syms, mode = ss.get("metrics"), ss.get("last_results") or [], ss.get("scan_mode")
    spec = MODE_SPEC.get(mode) if mode else None
    col = F.get(spec[0], {}).get("col") if spec else None
    go_scan = lambda: ss.update(page=PAGES[0])
    if metrics is None or not ss.get("last_results_label"):
        st.info("**No scan run yet in this session.** Go to 📈 Stock Scanner, pick a setup scan (e.g. 🔂 50-Day "
                "Reclaim, 🚩 Tight Flag, ⛳ High Tight Flag, 🌀 VCP …) and press **▶ SCAN**. Come back here and its "
                "TRIGGERs are listed as pilot candidates. (Scan results are kept only while the app is open.)")
        st.button("📈 Go to the Stock Scanner", on_click=go_scan)
        return
    TRIG = ("TRIGGER", "BREAKOUT")
    SETUP_SCANS = ("pattern", "xback", "glb", "so3", "vcp", "mac", "htf", "reclaim")     # long setups with a trigger
    if not spec or spec[0] not in SETUP_SCANS or not col or col not in metrics:
        st.info(f"The last scan was **{ss.get('last_results_label') or mode}**, which has no TRIGGER signals. Run a "
                "setup scan (🔂 50-Day Reclaim, 🚩 Tight Flag, ⛳ High Tight Flag, 🌀 VCP, 📏 MA Consolidation, "
                "✳️ Green Line, 📈 Shakeout +3 …) to get pilot candidates.")
        st.button("📈 Go to the Stock Scanner", on_click=go_scan)
        return
    held = set(pos["Symbol"])
    cand = [x for x in syms if x in metrics.index and str(metrics.at[x, col]) in TRIG and x not in held]
    if not cand:
        st.info(f"No new TRIGGERs today in your last **{ss.get('last_results_label') or mode}** scan "
                f"({len(syms)} stocks listed; stocks you already hold are left out). SETUPs become pilots only when "
                "they trigger — check again after tomorrow's close, or try another scan.")
        return
    rows = []
    for x in cand:
        lv = dict((nm, v) for v, nm, _ in setup_levels(x, metrics, spec[0])[1])
        px_ = _num(metrics.loc[x], "Price")
        stp = next((v for nm, v in lv.items() if nm.startswith("Stop")), np.nan)
        if not (pd.notna(stp) and 0 < stp < px_):
            atrp = _num(metrics.loc[x], "ATR %")
            stp = px_ * (1 - atrp / 100) if pd.notna(atrp) else np.nan
        sh = np.floor(acct * unit / 100 / px_) if px_ else np.nan
        rows.append({"Symbol": x, "Price": px_, "Stop": stp, "Stop %": (1 - stp / px_) * 100 if pd.notna(stp) else np.nan,
                     "Shares (1 unit)": sh, "Risk $": sh * (px_ - stp) if pd.notna(stp) else np.nan,
                     "RS": _num(metrics.loc[x], "RS")})
    cdf = pd.DataFrame(rows)
    st.dataframe(cdf, hide_index=True, use_container_width=True,
                 column_config={"Price": st.column_config.NumberColumn(format="%.2f"),
                                "Stop": st.column_config.NumberColumn(format="%.2f",
                                                                      help="The setup's stop (else 1 ATR under the price)"),
                                "Stop %": st.column_config.NumberColumn(format="%.1f%%"),
                                "Shares (1 unit)": st.column_config.NumberColumn(format="%d"),
                                "Risk $": st.column_config.NumberColumn(format="dollar", help="Lost if the stop is hit"),
                                "RS": st.column_config.NumberColumn(format="%d")})
    pick = st.multiselect("Add as pilots (1 unit at today's price)", cand, key=f"pos_pick_{ss.get('pos_ver', 0)}")
    if st.button("➕ Add pilots", disabled=not pick):
        new = cdf[cdf["Symbol"].isin(pick)]
        day = pd.Timestamp.today().strftime("%Y-%m-%d")
        df = pd.concat([ss["pos_df"], pd.DataFrame([{"Symbol": r["Symbol"], "Pilot date": day,
                                                      "Buys": f"{r['Price']:.2f}",
                                                      "Stop": round(float(r["Stop"]), 2) if pd.notna(r["Stop"]) else None,
                                                      "Note": spec[2]} for r in new.to_dict("records")])],
                       ignore_index=True)
        ss["pos_df"] = df; ss["pos_ver"] = ss.get("pos_ver", 0) + 1
        save_positions(df)
        st.rerun()


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


def stock_stages_now(panels, syms):
    """Each stock's 1A–4C stage from its latest close (the same rules as the Relative Strength page)."""
    syms = [s for s in syms if s in panels["Close"].columns]
    if not syms:
        return pd.Series(dtype=object)
    C, H, L = (panels[k][syms] for k in ("Close", "High", "Low"))
    last = C.ffill().iloc[-1]
    e10 = C.ewm(span=10, adjust=False).mean().iloc[-1]
    e20 = C.ewm(span=20, adjust=False).mean().iloc[-1]
    s50 = C.rolling(50).mean().iloc[-1]
    s200 = C.rolling(200, min_periods=150).mean().iloc[-1]
    pc = C.shift(1)
    tr = np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy())
    atr = pd.DataFrame(tr, index=C.index, columns=syms).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    return pd.Series([rs_classify_stage(last[t], e10[t], e20[t], s50[t], s200[t], atr[t]) for t in syms], index=syms)


def industry_ranks(m, min_stocks):
    """Industry strength = median RS (1–99) of its stocks; ranked across industries with ≥ min_stocks.
    Returns a table: Industry, Sector, Stocks, Median RS, Rank (1 = best), Top % (0 = best)."""
    d = m.dropna(subset=["Industry", "RS"])
    d = d[d["Industry"].astype(str).str.strip().ne("") & d["Industry"].astype(str).ne("nan")]
    if d.empty:
        return pd.DataFrame(columns=["Industry", "Sector", "Stocks", "Median RS", "Rank", "Top %"])
    g = d.groupby("Industry").agg(Stocks=("RS", "size"), **{"Median RS": ("RS", "median")})
    g["Sector"] = d.groupby("Industry")["Sector"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    g = g[g["Stocks"] >= min_stocks].sort_values("Median RS", ascending=False)
    g["Rank"] = np.arange(1, len(g) + 1)
    g["Top %"] = (g["Rank"] - 1) / max(len(g), 1) * 100            # 0 = the strongest industry
    return g.reset_index()[["Industry", "Sector", "Stocks", "Median RS", "Rank", "Top %"]]


def peer_corr(C, m, syms, days=GRP_CORR_DAYS):
    """{symbol: 60-day correlation of its daily returns with the average of its industry peers (itself left out)}."""
    R = C.pct_change().iloc[-days:]
    out = {}
    ind = m["Industry"].astype(str)
    for name, members in ind.groupby(ind).groups.items():
        mem = [s for s in members if s in R.columns]
        if len(mem) < 3:
            continue
        tot, cnt = R[mem].sum(axis=1, min_count=1), R[mem].notna().sum(axis=1)
        for s in mem:
            if s not in syms or R[s].notna().sum() < 40:
                continue
            peer = (tot - R[s].fillna(0)) / (cnt - R[s].notna().astype(int)).replace(0, np.nan)
            c_ = R[s].corr(peer)
            if pd.notna(c_):
                out[s] = c_
    return out


def first_close_above_50(panels, syms, days=30):
    """Stocks whose close today is above the 50-day SMA for the first time in `days` sessions: every close in between was
    at or below it (so it dipped under and has just reclaimed it). Returns {symbol: sessions since the last close above}."""
    C = panels["Close"]
    C = C[[x for x in syms if x in C.columns]]
    s50 = C.rolling(50).mean()
    above = (C > s50) & s50.notna()
    n = len(C)
    if n < 51 + days:
        return {}
    ok = above.iloc[-1] & (above.iloc[-1 - days:-1].sum() == 0) & s50.iloc[-1 - days].notna()
    return {t: days for t in C.columns[ok.to_numpy()]}


def funnel_compute(m, o, stages=None, corr=None, etf_now=None, first50=None):
    """Run the stocks in m through the funnel. o = settings (fn_* values without the prefix).
    Returns (steps [(name, kept symbols, dropped DataFrame with Reason)], industry table, pilots, watchlist)."""
    m = m.copy()
    stages = stages if stages is not None else pd.Series(dtype=object)
    corr = corr or {}
    m["Stage"] = stages.reindex(m.index).fillna("?")
    dv_col = "Avg $ Vol 50d M" if "Avg $ Vol 50d M" in m else "$ Vol M"
    info = lambda idx: m.loc[idx, [c for c in ["RS", "Stage", "Price", "ADR % 14d", "Industry"] if c in m]]
    steps = []

    # 1 · liquid stocks
    px_, dv = m["Price"], m[dv_col]
    reason = pd.Series("", index=m.index)
    reason[dv.fillna(0) < o["mindv"]] = f"Avg $ vol under ${o['mindv']:g}M"
    reason[px_.fillna(0) < o["minpx"]] = f"Price under ${o['minpx']:g}"
    if o.get("adr", 0) > 0 and "ADR % 14d" in m:
        adr = m["ADR % 14d"]
        low = ((adr.fillna(0) < o["adr"]) & (reason == "")).to_numpy()
        txt = np.array([f"ADR {v:.1f}% under {o['adr']:g}%" if pd.notna(v) else "no ADR" for v in adr], dtype=object)
        reason = pd.Series(np.where(low, txt, reason.to_numpy(dtype=object)), index=reason.index)
    keep = list(m.index[reason == ""])
    steps.append((FN_STEPS[0], keep, info(reason.index[reason != ""]).assign(Reason=reason[reason != ""])))

    # 2 · leading groups — industry strength is ranked over the whole universe, not just the liquid stocks
    ind = industry_ranks(m, int(o["ind_min"]))
    top = set(ind.loc[ind["Top %"] < float(o["ind_top"]), "Industry"])
    rank_of = dict(zip(ind["Industry"], ind["Rank"]))
    m["Ind rank"] = m["Industry"].map(rank_of)
    m["Group"] = ""
    ok_set = set(ETF_OK.get(o.get("etf", "Off"), ()))
    cur, drop, why = [], [], {}
    for s in keep:
        r = m.loc[s]
        i_ok = r["Industry"] in top
        txt = f"#{int(rank_of[r['Industry']])} of {len(ind)}" if r["Industry"] in rank_of else "industry too small"
        note, g_ok = "", True
        if ok_set and etf_now:
            lv = ["Sector", "Industry", "Theme"]
            g_ok, has, gtxt = group_check(r.get("Sector"), r.get("Industry"), lv, etf_now.get, ok_set, s)
            if has:
                note = gtxt
                txt += f" · {gtxt}"
        if i_ok and g_ok:
            m.at[s, "Group"] = note
            cur.append(s)
            continue
        own = own_mover(r.get("RS", np.nan), corr.get(s, np.nan), int(o["lead_rs"]), float(o["corr"]))
        if own:
            m.at[s, "Group"] = own + (f" · {note}" if note else "")
            cur.append(s)
        else:
            drop.append(s)
            why[s] = ("Industry not in the top " + f"{o['ind_top']:g}% ({txt})") if not i_ok else f"Group ETF not in play ({txt})"
    steps.append((FN_STEPS[1], cur, info(drop).assign(Reason=[why[s] for s in drop])))

    # 3 · leading stocks: RS and stage
    keep, cur, drop, why = cur, [], [], {}
    for s in keep:
        rs, sg = m.at[s, "RS"], m.at[s, "Stage"]
        if not (pd.notna(rs) and rs >= o["rs"]):
            why[s] = f"RS {rs:.0f} < {o['rs']:g}" if pd.notna(rs) else "no RS"
        elif sg not in o["stages"]:
            why[s] = f"Stage {sg} ({RS_STAGES.get(sg, ('?',))[0]})"
        else:
            cur.append(s)
            continue
        drop.append(s)
    steps.append((FN_STEPS[2], cur, info(drop).assign(Reason=[why[s] for s in drop])))

    # 4 · a setup today in one of the chosen scans; 5 · it TRIGGERED today
    keep, cur, drop, trig = cur, [], [], []
    setup_txt, first_trig, scans_of, why4 = {}, {}, {}, {}
    f50_mode, first50 = o.get("f50", FN_F50[0]), first50 or {}
    f50_txt = f"First close above the 50-day in {int(o.get('f50_days', 30))} days"
    for s in keep:
        parts = []
        for k in o["scans"]:
            col = F[k]["col"]
            if col not in m:
                continue
            v = str(m.at[s, col])
            if v in FN_SCANS[k][0] + FN_SCANS[k][1]:
                parts.append(f"{F[k]['label']} {v}")
                scans_of.setdefault(s, []).append(k)
                if v in FN_SCANS[k][0] and s not in first_trig:
                    first_trig[s] = k
        hit50 = s in first50 and f50_mode != FN_F50[0]
        if f50_mode == FN_F50[2] and not hit50:                 # "only these": everything else drops out here
            drop.append(s)
            why4[s] = f"No {f50_txt.lower()}"
            continue
        if hit50:
            parts.append(f50_txt)
            first_trig.setdefault(s, "f50")
        if parts:
            setup_txt[s] = " · ".join(parts)
            cur.append(s)
            (trig if s in first_trig else []).append(s)
        else:
            drop.append(s)
            why4[s] = "No setup in the chosen scans"
    steps.append((FN_STEPS[3], cur, info(drop).assign(Reason=[why4[s] for s in drop])))
    watch = [s for s in cur if s not in first_trig]
    steps.append((FN_STEPS[4], trig, info(watch).assign(Reason=[f"Not triggered yet — {setup_txt[s]}" for s in watch])))

    base = m.loc[:, [c for c in ["Price", "ADR % 14d", "RS", "Stage", "Industry", "Ind rank", "Group", "ATR %"] if c in m]]
    pil = base.loc[trig].assign(Setup=[setup_txt[s] for s in trig], _scan=[first_trig[s] for s in trig])
    wl = base.loc[watch].assign(Setup=[setup_txt[s] for s in watch], _scans=[scans_of[s] for s in watch])
    return steps, ind, pil, wl


def funnel_levels(pil, metrics, unit_usd):
    """Stop, shares and risk for each pilot (the stop of the scan that triggered, else 1 ATR under the price)."""
    rows = []
    for s, r in pil.iterrows():
        px_ = r["Price"]
        if r["_scan"] == "f50":                              # stop = the lowest low of the last 10 sessions
            try:
                stp = float(ss["panels"]["Low"][s].iloc[-10:].min())
            except Exception:
                stp = np.nan
        else:
            lv = setup_levels(s, metrics, r["_scan"])[1]
            stp = next((v for v, nm, _ in lv if nm.startswith("Stop")), np.nan)
        if not (pd.notna(stp) and 0 < stp < px_):
            atrp = r.get("ATR %", np.nan)
            stp = px_ * (1 - atrp / 100) if pd.notna(atrp) else np.nan
        sh = np.floor(unit_usd / px_) if px_ and unit_usd > 0 else 0
        rows.append({"Stop": stp, "Stop %": (1 - stp / px_) * 100 if pd.notna(stp) else np.nan,
                     "Shares": sh, "Cost $": sh * px_, "Risk $": sh * (px_ - stp) if pd.notna(stp) else np.nan})
    return pil.join(pd.DataFrame(rows, index=pil.index)) if rows else pil.assign(
        Stop=np.nan, **{"Stop %": np.nan, "Shares": np.nan, "Cost $": np.nan, "Risk $": np.nan})


def funnel_watch_levels(wl, metrics):
    """For each watchlist stock: the trigger price (of the setup closest to firing), the % the price must rise to reach
    it, the stop, and the risk from the trigger to the stop — the numbers for a buy-stop order."""
    rows = []
    for s_, r in wl.iterrows():
        px_, best = r["Price"], None
        for k in r["_scans"]:
            lv = setup_levels(s_, metrics, k)[1]
            trg = next((v for v, nm, _ in lv if nm.startswith(("Trigger", "Entry"))), np.nan)
            stp = next((v for v, nm, _ in lv if nm.startswith("Stop")), np.nan)
            if pd.notna(trg) and trg > 0 and (best is None or abs(trg / px_ - 1) < abs(best[0] / px_ - 1)):
                best = (trg, stp)
        trg, stp = best if best else (np.nan, np.nan)
        rows.append({"Trigger": trg, "To trigger %": (trg / px_ - 1) * 100 if pd.notna(trg) else np.nan,
                     "Stop": stp, "Risk %": (1 - stp / trg) * 100 if pd.notna(trg) and pd.notna(stp) and trg else np.nan})
    out = wl.join(pd.DataFrame(rows, index=wl.index))
    return out.assign(_d=out["To trigger %"].abs()).sort_values("_d", na_position="last").drop(columns="_d")


def funnel_page():
    st.title("🔻 Funnel — from the whole market down to today's pilot buys")
    st.caption("Top-down, like the pros: trade only when the **market** allows, only in **leading groups**, only the "
               "**leading stocks** in them, only on a **setup**, and only when it **triggers**. Every stock that "
               "drops out shows why.")
    lab2key = {F[k]["label"]: k for k in FN_SCANS}
    sb = st.sidebar
    sb.markdown("**1 · Liquidity**")
    a1, a2 = sb.columns(2)
    a1.number_input("Min price \\$", 0.0, 10000.0, step=1.0, key="fn_minpx")
    a2.number_input("Min avg \\$ vol (\\$M)", 0.0, 100000.0, step=5.0, key="fn_mindv",
                    help="50-day average dollar volume, in millions")
    sb.number_input("Min ADR % (14-day)", 0.0, 50.0, step=0.5, key="fn_adr",
                    help="Average daily range: how far the stock moves in a day. Under 3% is usually too slow to be "
                         "worth a pilot. 0 = off.")
    sb.markdown("**Market gate**")
    sb.checkbox("📊 Size by market stage", key="fn_mkt",
                help="Pilot size = unit × the market ETF's stage %. The stage % are the same as in 🧪 Backtest → "
                     "Size by market stage (0% = no new pilots).")
    if ss.get("bt_stage_bench") not in STAGE_BENCHES:
        ss["bt_stage_bench"] = "QQQE"
    if ss["fn_mkt"]:
        sb.selectbox("Market ETF", STAGE_BENCHES, key="bt_stage_bench")
    sb.markdown("**2 · Leading groups**")
    b1, b2 = sb.columns(2)
    b1.number_input("Industry in top %", 1, 100, step=5, key="fn_ind_top",
                    help="Industries ranked by the median RS of their stocks. 30 = the strongest 30% of industries.")
    b2.number_input("Min stocks / industry", 1, 50, key="fn_ind_min",
                    help="Industries with fewer stocks in your scan aren't ranked (too few to judge)")
    sb.selectbox("Group ETF must also be", ["Off", *ETF_OK], key="fn_etf",
                 help="Also check the stock's sector / industry / theme ETFs, as on 🗂 Sectors in play "
                      "(downloads the ETFs once, kept 30 min).")
    c1, c2 = sb.columns(2)
    c1.number_input("…unless RS ≥", 0, 99, key="fn_lead_rs",
                    help="Leaders pass even in a weak group (0 = off)")
    c2.number_input("…or ρ with peers <", 0.0, 1.0, step=0.05, format="%.2f", key="fn_corr",
                    help="Own movers (like TSLA) pass: 60-day correlation with the average of their industry peers "
                         "is below this (0 = off)")
    sb.markdown("**3 · Leading stocks**")
    sb.number_input("Stock RS ≥", 0, 99, key="fn_rs")
    ss["fn_stages"] = [x for x in ss["fn_stages"] if x in RS_STAGE_ORDER]
    sb.multiselect("Stock stage", RS_STAGE_ORDER[:-1], key="fn_stages",
                   help="The stock's own stage today (the app's 1A–4C rules, as on 🧭 Relative Strength)")
    sb.markdown("**4 · Setups**")
    sb.selectbox("First close above the 50-day", FN_F50, key="fn_f50",
                 help="A stock that closed above its 50-day SMA today for the first time in N sessions — every close in "
                      "between was at or below it (it dipped under and has just reclaimed it). Stop = the lowest low "
                      "of the last 10 sessions. **Also count** adds these to the setups below; **Only these** shows "
                      "just them.")
    if ss["fn_f50"] != FN_F50[0]:
        sb.number_input("…for the first time in N days", 5, 120, step=5, key="fn_f50_days")
    ss["fn_scans"] = [x for x in ss["fn_scans"] if x in lab2key]
    sb.multiselect("Setup scans", list(lab2key), key="fn_scans")
    sb.markdown("**5 · Pilot size** (shared with 💼 Positions)")
    d1, d2 = sb.columns(2)
    d1.number_input("Account $", 1000, 100_000_000, step=10_000, key="bt_acct")
    d2.number_input("Unit, % of account", 0.25, 50.0, step=0.25, key="bt_pyr_unit")

    metrics = ss.get("metrics")
    go_scan = lambda: ss.update(page=PAGES[0])
    if metrics is None or "panels" not in ss:
        st.info("**No scan in this session yet.** The funnel works on the stocks your last Stock Scanner run "
                "downloaded — go to 📈 Stock Scanner, pick any universe (e.g. S&P 1500 or All US) and press **▶ SCAN**, "
                "then come back here.")
        st.button("📈 Go to the Stock Scanner", on_click=go_scan)
        return
    m = metrics
    if ss.get("scan_set"):
        m = m[m.index.isin(ss["scan_set"])]
    o = {k[3:]: ss[k] for k in FN_DEFAULTS}
    o["scans"] = [lab2key[x] for x in ss["fn_scans"]]
    if not o["scans"] and ss["fn_f50"] != FN_F50[2]:
        st.warning("Pick at least one setup scan in the sidebar (step 4).")
        return

    # ---- market gate ----
    mult, sg_now = 1.0, None
    if ss["fn_mkt"]:
        sh_ = bt_bench_stages(ss["bt_stage_bench"])
        if sh_ is not None and len(sh_):
            sg_now = sh_.iat[-1]
            mult = bt_stage_pct().get(sg_now, 100) / 100
    unit_usd = float(ss["bt_acct"]) * float(ss["bt_pyr_unit"]) / 100 * mult

    with st.spinner("Running the funnel …"):
        stages = stock_stages_now(ss["panels"], list(m.index))
        corr = peer_corr(ss["panels"]["Close"], m, set(m.index)) if float(ss["fn_corr"]) > 0 else {}
        etf_now = etf_status_now() if ss["fn_etf"] != "Off" else None
        f50 = first_close_above_50(ss["panels"], list(m.index), int(ss["fn_f50_days"])) \
            if ss["fn_f50"] != FN_F50[0] else {}
        steps, ind, pil, wl = funnel_compute(m, o, stages, corr, etf_now, f50)
        pil = funnel_levels(pil, metrics, unit_usd)

    asof = ss["panels"]["Close"].index[-1]
    st.caption(f"Using the **{len(m):,} stocks** from your last scan (prices to **{asof:%a %b %d, %Y}**). The scan's own "
               "filters aren't applied here — the funnel has its own, in the sidebar.")

    # market card + counts
    if sg_now:
        nm_, col_ = RS_STAGES.get(sg_now, ("?", "#777"))
        mk = ("Market", f"{sg_now} · size {mult * 100:.0f}%", f"{ss['bt_stage_bench']} · {nm_}",
              "g" if mult >= 1 else ("r" if mult == 0 else "y"))
    elif ss["fn_mkt"]:
        mk = ("Market", "?", f"couldn't load {ss['bt_stage_bench']} — full size", "n")
    else:
        mk = ("Market", "off", "stage sizing off — full size", "n")
    render_cards([mk, ("Liquid stocks", f"{len(steps[0][1]):,}", f"of {len(m):,}", "n"),
                  ("Leading industries", f"{int((ind['Top %'] < float(ss['fn_ind_top'])).sum())}",
                   f"of {len(ind)} ranked", "n"),
                  ("Setups", f"{len(steps[3][1])}", "SETUP or TRIGGER today", "y"),
                  ("Pilot buys", f"{len(pil)}", f"1 unit = ${unit_usd:,.0f}", "g" if len(pil) else "n")])
    if sg_now and mult == 0:
        st.error(f"📊 The market ({ss['bt_stage_bench']}) is in stage **{sg_now}** — your stage sizing says **no new "
                 "pilots** today. The list below is a watchlist only.")

    fig = go.Figure(go.Funnel(y=[n for n, _, _ in steps], x=[len(k) for _, k, _ in steps],
                              textinfo="value+percent initial",
                              marker=dict(color=["#94a3b8", "#60a5fa", "#a78bfa", "#fbbf24", "#22c55e"])))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ---- pilots ----
    st.subheader(f"🟢 Pilot buys today ({len(pil)})")
    if pil.empty:
        st.info("No stock made it all the way through today. That's normal — most days only a handful do. "
                "The **watchlist** below shows the setups that could trigger next.")
    else:
        show = pil.drop(columns=["_scan", "ATR %"], errors="ignore").reset_index().rename(columns={"index": "Symbol"})
        show = show[["Symbol", "Setup", "Price", "ADR % 14d", "Stop", "Stop %", "Shares", "Cost $", "Risk $", "RS", "Stage",
                     "Industry", "Ind rank", "Group"]]
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"Price": st.column_config.NumberColumn(format="%.2f"),
                                    "ADR % 14d": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Stop": st.column_config.NumberColumn(format="%.2f",
                                                                          help="The setup's stop (else 1 ATR under)"),
                                    "Stop %": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Shares": st.column_config.NumberColumn(format="%d", help="1 unit at the close"),
                                    "Cost $": st.column_config.NumberColumn(format="dollar"),
                                    "Risk $": st.column_config.NumberColumn(format="dollar"),
                                    "RS": st.column_config.NumberColumn(format="%d"),
                                    "Ind rank": st.column_config.NumberColumn(format="%d",
                                                                              help="1 = strongest industry"),
                                    "Group": st.column_config.TextColumn(
                                        help="⭐ leader / 🚀 own mover = let through although its group isn't leading")})
        st.caption(f"Total: **\\${show['Cost $'].sum():,.0f}** invested, **\\${show['Risk $'].sum():,.0f}** at risk "
                   f"if every stop is hit. Take them all — the funnel is the selection; adds to the winners come from "
                   "💼 Positions.")
        if "pos_df" not in ss:
            ss["pos_df"] = load_positions()
            ss["pos_ver"] = 0
        held = set(ss["pos_df"]["Symbol"].astype(str).str.upper())
        new = show[~show["Symbol"].isin(held)]
        lbl = f"➕ Send {len(new)} pilot(s) to 💼 Positions" + (f" ({len(show) - len(new)} already held)"
                                                                if len(new) < len(show) else "")
        if st.button(lbl, disabled=new.empty or (sg_now is not None and mult == 0), type="primary"):
            day = pd.Timestamp.today().strftime("%Y-%m-%d")
            add = pd.DataFrame([{"Symbol": r["Symbol"], "Pilot date": day, "Buys": f"{r['Price']:.2f}",
                                 "Stop": round(float(r["Stop"]), 2) if pd.notna(r["Stop"]) else None,
                                 "Note": "Funnel: " + r["Setup"]} for r in new.to_dict("records")])
            ss["pos_df"] = pd.concat([ss["pos_df"], add], ignore_index=True)
            ss["pos_ver"] = ss.get("pos_ver", 0) + 1
            save_positions(ss["pos_df"])
            st.success(f"Added {len(new)} pilot(s) to 💼 Positions.")

    # ---- watchlist ----
    st.subheader(f"🟡 Watchlist — setups not triggered yet ({len(wl)})")
    if len(wl):
        st.caption("Sorted by how close each is to its trigger. **Trigger** = the price for a buy-stop order (a close above it "
                   "on volume makes it a TRIGGER); if it stays under, it isn't a buy today.")
        wlv = funnel_watch_levels(wl, metrics)
        st.dataframe(wlv.drop(columns=["ATR %", "_scans"], errors="ignore").reset_index().rename(columns={"index": "Symbol"})
                     [["Symbol", "Setup", "Price", "ADR % 14d", "Trigger", "To trigger %", "Stop", "Risk %", "RS", "Stage", "Industry",
                       "Ind rank", "Group"]],
                     hide_index=True, use_container_width=True,
                     column_config={"Price": st.column_config.NumberColumn(format="%.2f"),
                                    "ADR % 14d": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Trigger": st.column_config.NumberColumn(format="%.2f"),
                                    "To trigger %": st.column_config.NumberColumn(
                                        format="%+.1f%%", help="How far the price must rise to reach the trigger "
                                                               "(negative = already above it, waiting for the close / volume)"),
                                    "Stop": st.column_config.NumberColumn(format="%.2f"),
                                    "Risk %": st.column_config.NumberColumn(format="%.1f%%",
                                                                            help="From the trigger down to the stop"),
                                    "RS": st.column_config.NumberColumn(format="%d"),
                                    "Ind rank": st.column_config.NumberColumn(format="%d")})

    # ---- why stocks dropped out ----
    st.subheader("Why stocks dropped out")
    for i, (name, kept, dropped) in enumerate(steps):
        prev = len(m) if i == 0 else len(steps[i - 1][1])
        with st.expander(f"{name}: {len(kept):,} kept · {len(dropped):,} dropped (of {prev:,})"):
            if dropped.empty:
                st.caption("Nobody dropped out here.")
            else:
                st.dataframe(dropped.reset_index().rename(columns={"index": "Symbol"}).sort_values(
                    "RS", ascending=False, na_position="last"), hide_index=True, use_container_width=True,
                    column_config={"Price": st.column_config.NumberColumn(format="%.2f"),
                                   "RS": st.column_config.NumberColumn(format="%d")})
    with st.expander(f"🏭 Industry ranking ({len(ind)} industries with ≥ {ss['fn_ind_min']} stocks)"):
        st.dataframe(ind.assign(Leading=ind["Top %"] < float(ss["fn_ind_top"])).drop(columns=["Top %"]),
                     hide_index=True, use_container_width=True,
                     column_config={"Median RS": st.column_config.NumberColumn(format="%.0f")})


PAGES = ["📈 Stock Scanner", "🗂 Sectors in play", "🧭 Relative Strength", "🔗 Similar stocks", "💰 Smart Money",
         "🧪 Backtest", "💼 Positions", "🔻 Funnel"]
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
PRESET_FILE = Path(__file__).with_name("scan_presets.json")
PRESET_PREFIXES = ("pb_", "ep_", "sp_", "xb_", "gl_", "so_", "vcp_", "mac_", "htf_", "sh_", "rc_", "f5_", "ll_", "rss_", "rs_", "vr_", "earn_", "v_f_", "v_fmin_", "v_fmax_", "v_fcat_")


def load_scan_presets():
    try:
        return json.loads(PRESET_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _jsonable(v):
    if isinstance(v, (str, bool, int, float)) or v is None:
        return True
    return isinstance(v, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in v)


def save_scan_preset():
    name = (ss.get("preset_new") or "").strip()
    if not name:
        ss["preset_msg"] = "⚠️ Type a name first."
        return
    keys = ["scan_mode", "v_visible"] + list(PATTERN_PRESETS["Default"]) + \
        [k for k in ss.keys() if isinstance(k, str) and k.startswith(PRESET_PREFIXES)]
    data = {k: (list(ss[k]) if isinstance(ss[k], tuple) else ss[k]) for k in keys if k in ss and _jsonable(ss[k])}
    presets = load_scan_presets()
    presets[name] = data
    try:
        PRESET_FILE.write_text(json.dumps(presets, indent=1), encoding="utf-8")
        ss["preset_pick"], ss["preset_new"] = name, ""
        ss["preset_msg"] = f"✅ Saved “{name}”."
    except Exception as e:
        ss["preset_msg"] = f"Couldn't save: {e}"


def apply_scan_preset():
    data = load_scan_presets().get(ss.get("preset_pick"), {})
    for k, v in data.items():
        if k == "scan_mode":
            v = OLD_SCAN_NAMES.get(v, v)
            if v not in SCAN_MODES:
                continue
        ss[k] = tuple(v) if k in ("pull_range", "gl_months", "so_under", "htf_depth") else v
    ss["v_tab"] = "Setup"
    ss["preset_msg"] = f"Loaded “{ss.get('preset_pick')}” — press SCAN if the stock list changed."


def delete_scan_preset():
    presets = load_scan_presets()
    presets.pop(ss.get("preset_pick"), None)
    try:
        PRESET_FILE.write_text(json.dumps(presets, indent=1), encoding="utf-8")
        ss["preset_msg"] = "🗑 Deleted."
    except Exception as e:
        ss["preset_msg"] = f"Couldn't delete: {e}"


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
def results_area():
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


results_area()
