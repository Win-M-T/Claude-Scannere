"""Smart Money page: 13F funds and congressional trades."""

import html as html_lib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests

from scanner.charts import load_templates, tradingview_chart
from scanner.constants import CHANGE_STYLE, CUSIP_FILE, DEFAULT_TPL, FAMOUS_13F, HOUSE_ROW, SM_AMOUNT_MIN, SM_NAME_DROP, UA
from scanner.data.universe import get_exchange_map, get_screener_meta, market_of, parse_tickers, tv_symbol
from scanner.rs import rs_prices

ss = st.session_state

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
