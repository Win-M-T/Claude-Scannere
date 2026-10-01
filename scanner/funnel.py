"""Minervini-style 'Funnel' page."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scanner.backtest import bt_bench_stages, bt_stage_pct
from scanner.constants import ETF_OK, F, FN_DEFAULTS, FN_F50, FN_SCANS, FN_STEPS, GRP_CORR_DAYS, PAGES, RS_STAGES, RS_STAGE_ORDER, STAGE_BENCHES
from scanner.etf import etf_status_now, group_check, own_mover
from scanner.positions import load_positions, save_positions
from scanner.rs import rs_classify_stage
from scanner.smart_money import render_cards, setup_levels

ss = st.session_state

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
