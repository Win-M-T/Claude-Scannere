"""Price download helpers (yfinance) and panel building."""

import pandas as pd
import streamlit as st
import yfinance as yf


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
