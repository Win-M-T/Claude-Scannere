"""Universe providers: index members, listed-stock rosters, foreign markets, tickers."""

import html as html_lib
import io
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import requests

from scanner.constants import MARKETS, SPDJI_BASE, SP_INDEX_NAMES, UA, _TAG
from scanner.rs import rs_prices

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
