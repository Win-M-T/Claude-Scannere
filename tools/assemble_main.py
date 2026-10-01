"""Assemble the final scanner/main.py from main_new.py (raw chunks of the monolith).

- keeps docstring, third-party imports and st.set_page_config at module level
- injects cross-module imports + a `ss = st.session_state` alias
- inserts the two global CSS st.markdown blocks (moved out of constants.py by wire_imports)
- wraps the Stock Scanner page flow (sidebar/settings block + scan-execution block +
  final results_area() call) into stock_scanner_page() so page routing can st.stop()
"""
import pathlib, re

ROOT = pathlib.Path("scanner")
lines = (ROOT / "main_new.py").read_text(encoding="utf-8").splitlines(keepends=True)

IMPORT_RE = re.compile(r"^(import |from )")


def is_import(l):
    return bool(IMPORT_RE.match(l))


# ---- locate the structural landmarks in main_new.py ----
i_cfg = next(i for i, l in enumerate(lines) if l.startswith("st.set_page_config"))
i_ss = next(i for i, l in enumerate(lines) if l.strip() == "ss = st.session_state")
i_route = next(i for i, l in enumerate(lines) if l.startswith("ss.setdefault(\"page\""))
i_side = next(i for i, l in enumerate(lines) if l.rstrip() == "with st.sidebar:")
i_results = max(i for i, l in enumerate(lines) if l.strip() == "results_area()")

head = lines[:i_cfg + 1]                      # docstring + stdlib/3rd-party imports + set_page_config
body = lines[i_cfg + 1:i_ss]                  # leftover chunk (usually empty/blank)
routing = lines[i_route:i_side]               # page-routing block (ends with st.stop())
scanflow = lines[i_side:]                     # sidebar + scan execution + trailing results_area()

# remove the bare trailing call from scanflow (it becomes the last line inside the function)
while scanflow and scanflow[-1].strip() in ("", "results_area()"):
    if scanflow[-1].strip() == "results_area()":
        scanflow.pop()
        break
    scanflow.pop()

# ---- extract the two global CSS st.markdown blocks from the monolith ----
mono = pathlib.Path("scanner_app.py").read_text(encoding="utf-8").splitlines(keepends=True)
import ast
mtree = ast.parse("".join(mono))
css_ranges = []
for n in mtree.body:
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) \
       and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "markdown":
        s = "".join(mono[n.lineno - 1:n.end_lineno])
        if "<style>" in s:
            css_ranges.append((n.lineno, n.end_lineno))
assert len(css_ranges) == 2, css_ranges          # main CSS + shared page furniture
css_texts = ["".join(mono[a - 1:b]) for a, b in css_ranges]

# drop them from constants.py if they are still there (fresh split before wiring)
cpath = ROOT / "constants.py"
csrc = cpath.read_text(encoding="utf-8")
clines = csrc.splitlines(keepends=True)
ctree = ast.parse(csrc)
cblocks = [(n.lineno, n.end_lineno) for n in ctree.body
           if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
           and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "markdown"]
if cblocks:
    assert len(cblocks) == 2, cblocks
    for a, b in sorted(cblocks, reverse=True):
        del clines[a - 1:b]
    cpath.write_text("".join(clines), encoding="utf-8")

# ---- cross-module imports needed by main ----
MAIN_IMPORTS = """from scanner.constants import (CAL_COLS, CORE_KEYS, DEFAULT_CHOICE, DEFAULT_SECTOR_EXCL,
                           DEFAULT_VISIBLE, EARN_DAYS, EARN_WINDOWS, F, FUND_COLS, HTF_PRESETS, LL_G1, LL_G2,
                           MAC_LINES, MARKETS, M_EARN, M_TF, OLD_SCAN_NAMES, PAGES, PATTERN_PRESETS,
                           PYR_RULES, PYR_STOPS, RC_STOPS, RC_TRENDS, RS_STAGE_ORDER, RS_STAGES,
                           RS_TIMEFRAMES, SCAN_MODES, SCAN_PARTS, SH_MAS, SH_SETUPS, SM_DEFAULT_FUNDS,
                           SO_ENTRIES, SO_RECLAIM, SP_INDEXES, SP_WINDOWS, STAGE_SIZE_DEFAULT,
                           VOLREC_OPTS, XB_20, XB_EMAS)
from scanner.data.fundamentals import (_nearest_per_stock, all_time_highs, earnings_calendar,
                                       earnings_reaction, yahoo_earnings)
from scanner.data.prices import build_panels, download_all, volume_history
from scanner.patterns.common import build_pp, compute_metrics
from scanner.data.universe import (get_all_us, get_market_meta, get_nasdaq_listed, get_screener_meta,
                                   get_sp500, market_fx, parse_tickers, sp_candidates, sp_changes_all,
                                   sp_member_dates, sp_pick_index)
from scanner.etf import etf_page
from scanner.funnel import FN_DEFAULTS, funnel_page
from scanner.peers import peers_page
from scanner.positions import positions_page
from scanner.presets import apply_scan_preset, delete_scan_preset, load_scan_presets, save_scan_preset
from scanner.results import results_area
from scanner.backtest import (backtest_page, drop_excluded, is_excluded_sector, sector_lookup,
                           sector_picker)
from scanner.rs import (V, base_universe, earn_window_changed, ep_status_changed, f5_status_changed,
                        gl_status_changed, htf_load_preset, htf_status_changed, is_earn_mode,
                        is_ep_mode, is_f5_mode, is_gl_mode, is_htf_mode, is_ll_mode, is_mac_mode,
                        is_pb_mode, is_rc_mode, is_record_mode, is_rs_mode, is_sh_mode, is_so_mode,
                        is_sp_mode, is_vcp_mode, is_xb_mode, ll_status_changed, load_pattern_preset,
                        mac_status_changed, pb_status_changed, prefilter, pwidget, rc_status_changed,
                        rs_groups, rs_page, rs_stock_table, rss_status_changed, scan_mode_changed,
                        sh_status_changed, so_status_changed, sp_type_changed, vcp_status_changed,
                        vr_window_changed, xb_status_changed)
from scanner.sketching import (bear_flag_sketch, failed_breakout_sketch, green_line_sketch, htf_sketch,
                               mac_sketch, reclaim_sketch, shakeout_sketch, tight_flag_sketch,
                               vcp_sketch, xb_sketch)
from scanner.smart_money import smart_money_page
"""

HEADER_TAIL = "\n\nss = st.session_state\n"

# ---- session-state bridge: main writes plain locals (universe, metrics, sp_changes …)
# that other modules read through ss — mirror the important ones into session_state.
BRIDGE = """

def _sync_session_state():
    for _k in ("universe", "metrics", "pp", "sp_changes", "tick_list", "full_hist"):
        try:
            ss[_k] = globals()[_k]
        except KeyError:
            pass


"""

# ---- indent the Stock Scanner page flow into a function ----
indented = ["def stock_scanner_page():\n"]
for l in scanflow:
    indented.append(("    " + l) if l.strip() else "\n")
indented.append("    results_area(metrics, universe, sp_changes)\n")
indented.append("    _sync_session_state()\n")

out = "".join(head) + MAIN_IMPORTS + HEADER_TAIL
out += "".join(css_texts)
out += "\n\n" + "".join(routing) + "\n\n" + BRIDGE
out += "".join(indented)

(ROOT / "main.py").write_text(out, encoding="utf-8")
print(f"scanner/main.py written: {out.count(chr(10))} lines")
