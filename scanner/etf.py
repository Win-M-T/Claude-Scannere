"""'Sectors in play' ETF/theme page."""

import html as html_lib
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scanner.charts import load_templates, price_chart, tradingview_chart
from scanner.constants import DEFAULT_TPL, ETF_INDUSTRY_RULES, ETF_LEVEL, ETF_LIST, ETF_MARKET, ETF_NAME, ETF_SECTOR_OF, ETF_STATUS, ETF_THEME_NAME, ETF_TICKERS, THEME_ETF, THEME_OF_TICKER
from scanner.data.prices import build_panels, download_all
from scanner.data.universe import get_exchange_map, tv_symbol
from scanner.rs import rs_chart, rs_classify_stage, rs_html
from scanner.smart_money import render_cards

ss = st.session_state

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
