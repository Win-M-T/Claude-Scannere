"""'Similar stocks' peer-correlation page."""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from scanner.charts import load_templates, tradingview_chart
from scanner.constants import DEFAULT_TPL, PEER_BULL, PEER_CSS, PEER_ETFS
from scanner.data.universe import get_exchange_map, get_screener_meta, tv_symbol
from scanner.rs import rs_stock_table, watchlist_panels, watchlist_tickers

ss = st.session_state

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
