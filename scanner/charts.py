"""TradingView widget embedding + local Plotly price chart."""

import json
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from scanner.constants import DEFAULT_TPL, TPL_FILE, TV_EXTRAS, TV_INTERVALS, TV_MA_IDS, TV_STYLES

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
