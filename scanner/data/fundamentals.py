"""Fundamentals, earnings calendar and simple filter predicates."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests

from scanner.constants import CAL_COLS, EARN_TIME, PERF_PRESETS, UA

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


def perf_filter(col, label):
    return dict(label=label, col=col, kind="num", unit="%",
                presets=[(a, b, fn(col, *args)) for a, b, fn, args in PERF_PRESETS])


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
