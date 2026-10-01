"""Positions tracker page."""

import json
import re
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from scanner.backtest import bt_bench_stages, bt_stage_pct
from scanner.constants import BT_TRAILS, F, MODE_SPEC, PAGES, POS_COLS, POS_FILE, PYR_RULES, PYR_STOPS, RS_STAGES
from scanner.data.prices import build_panels, download_all
from scanner.smart_money import _num, render_cards, setup_levels

ss = st.session_state

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
