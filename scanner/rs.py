"""Relative-strength engine, scan-mode helpers and the RS page."""

import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from scanner.constants import (F, HTF_PRESETS, MARKET_FALLBACK, MARKETS, MODE_SPEC, PATTERN_PRESETS, RS_CSS,
                              RS_PALETTE, RS_SECTOR_ETFS, RS_STAGES, RS_STAGE_ORDER, RS_TIMEFRAMES,
                              RS_VIEWS, WATCH_SOURCES)
from scanner.constants import (M_EARN, M_EP, M_F5, M_GL, M_HTF, M_LL, M_MAC, M_PB, M_RC, M_RS, M_SH,
                              M_SO, M_SP, M_TF, M_VCP, M_VOL, M_XB)
from scanner.data.prices import _fetch, _split, build_panels
from scanner.data.universe import get_all_us, get_market_meta, get_nasdaq_listed, get_screener_meta, get_sp500, parse_tickers

ss = st.session_state

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


def load_pattern_preset(name):
    for k, v in PATTERN_PRESETS[name].items():
        ss[k] = v


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
