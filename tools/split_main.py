"""Split scanner/main.py into modules using byte-exact AST line ranges."""
import ast, pathlib

SRC = pathlib.Path("scanner/main.py")
lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
tree = ast.parse("".join(lines))

defs, assigns = {}, {}
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defs[n.name] = n
    elif isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name):
                assigns[t.id] = n


def seg(node):
    start = node.lineno - 1
    for d in getattr(node, "decorator_list", []):
        start = min(start, d.lineno - 1)
    i = start - 1
    while i >= 0 and lines[i].strip().startswith("#"):
        start = i; i -= 1
    return "".join(lines[start:node.end_lineno])


def rng(a, b):
    return "".join(lines[a - 1:b])


MODULES = {}          # mod -> list of names / ("R", a, b) tuples
ORDER = []

def add(mod, *names):
    MODULES.setdefault(mod, []).extend(names)

def addr(mod, a, b):
    MODULES.setdefault(mod, []).append(("R", a, b))

# ---------------- constants.py ----------------
add("constants", "UA")
addr("constants", 42, 168)                      # st.markdown(<style>) block
add("constants", "SPDJI_BASE", "_TAG", "SP_INDEX_NAMES", "MARKETS", "MARKET_FALLBACK",
    "VOL_WINDOWS", "VOL_REC_NAME", "VOL_REC_RANK", "XB_20", "XB_EMAS",
    "SO_RECLAIM", "SO_ENTRIES", "MAC_LINES", "HTF_PRESETS", "SH_SETUPS", "SH_MAS",
    "RC_TRENDS", "RC_STOPS", "EARN_TIME", "CAL_COLS", "FUND_COLS", "PERF_PRESETS",
    "SP_INDEXES", "sp_pick_index", "SP_WINDOWS",
    "DEFAULT_VISIBLE", "DEFAULT_CHOICE", "COLUMN_SETS",
    "TV_INTERVALS", "TV_STYLES", "TV_EXTRAS", "TV_MA_IDS", "TPL_FILE", "DEFAULT_TPL",
    "TPL_FIELDS",
    "RS_SECTOR_ETFS", "RS_STAGES", "RS_STAGE_ORDER", "RS_PALETTE",
    "RS_TIMEFRAMES", "RS_VIEWS", "RS_CSS",
    "PATTERN_PRESETS", "M_SH", "M_RC", "M_F5", "SCAN_MODES", "OLD_SCAN_NAMES",
    "SCAN_EXTRA_DEFAULTS", "VOLREC_OPTS", "MODE_SPEC", "WATCH_SOURCES",
    "PEER_ETFS", "PEER_BULL", "PEER_CSS",
    "FAMOUS_13F", "SM_DEFAULT_FUNDS", "CUSIP_FILE", "SM_NAME_DROP", "SM_AMOUNT_MIN",
    "HOUSE_ROW", "CHANGE_STYLE",
    "BT_STRATEGIES", "BT_COMBINED", "BT_AGREE", "BT_TRAILS", "BT_SCALE_OUT",
    "BT_TARGETS", "BT_ENTRY_TRIGGER", "BT_ENTRY_330", "BT_ENTRIES", "is_trig_entry",
    "BT_ATR_STOPS", "BT_STOPS", "BT_UNIVERSES", "PYR_STOPS", "PYR_RULES",
    "STAGE_BENCHES", "STAGE_SIZE_DEFAULT", "BT_LIST_UNIVERSES", "BT_VOL",
    "MKT_PULLBACK", "MKT_CUSTOM", "BT_MARKET_FILTERS", "MKT_LINES",
    "DEFAULT_SECTOR_EXCL", "BT_PRIORITY", "OLD_BT_NAMES",
    "ETF_MARKET", "ETF_LIST", "ETF_STATUS", "ETF_TICKERS", "ETF_NAME", "ETF_OK",
    "GROUP_CHOICES", "ETF_SECTOR_OF", "ETF_INDUSTRY_RULES", "GROUP_LEVELS", "ETF_LEVEL",
    "THEMES", "THEME_OF_TICKER", "THEME_ETF", "ETF_THEME_NAME", "GRP_CORR_DAYS",
    "POS_FILE", "POS_COLS",
    "FN_SCANS", "FN_F50", "FN_DEFAULTS", "FN_STEPS",
    "PAGES", "RSP_DEFAULTS", "PRESET_FILE", "PRESET_PREFIXES", "CORE_KEYS")

# F dict + filter definitions (mixed stmts: keep as raw ranges)
addr("constants", 2658, 2756)                   # F={} .. last F[...] before LL_G1 comment
addr("constants", 2763, 2866)                   # F["price"]..F[..] after LL_G2 block
addr("constants", 2867, 2877)                   # EARN_WINDOWS..F["short"] block
addr("constants", 2758, 2762)                   # LL_G1 / LL_G2 dicts
addr("constants", 3519, 3525)                   # M_RS..M_F5 name assignments
addr("constants", 4627, 4648)                   # shared "page furniture" st.markdown(<style>)
addr("constants", 5774, 5774)                   # BT_ANY, BT_ALL

# ---------------- data/universe.py ----------------
add("universe", "get_screener_meta", "_sp500_wiki_html", "get_sp500", "get_sp_changes",
    "sp_candidates", "sp_member_dates", "_txt", "_slug_title", "spdji_release_list",
    "_norm_index", "_parse_date",
    "spdji_release_changes", "spdji_changes", "_merge_changes", "sp_changes_all",
    "_nasdaqtrader", "get_nasdaq_listed", "get_all_us", "get_exchange_map",
    "market_of", "market_fx", "get_market_meta", "tv_symbol", "tv_watchlist_text",
    "parse_tickers")

# ---------------- data/prices.py ----------------
add("prices", "_fetch", "_split", "download_all", "build_panels", "volume_history")

# ---------------- data/fundamentals.py ----------------
add("fundamentals", "fetch_fundamentals", "fetch_mcap", "_money", "earnings_day",
    "_nearest_per_stock", "earnings_calendar", "yahoo_earnings", "earnings_reaction",
    "between", "is_true", "perf_filter", "all_time_highs")

# ---------------- patterns/* ----------------
add("patterns_common", "_ago", "_bars_since_high", "compute_metrics", "build_pp", "rs_matrix")
add("patterns_volume", "volume_records", "parabolic_scan", "ep_scan")
add("patterns_crossback", "ema_crossback", "_xb_path")
add("patterns_greenline", "green_line")
add("patterns_shakeout", "so_rs_gate", "shakeout_plus3")
add("patterns_trend", "trend_template", "vcp_scan")
add("patterns_mac", "mac_frames", "mac_scan")
add("patterns_htf", "_win_view", "htf_frames", "htf_scan")
add("patterns_short", "_since", "_ffill_at", "short_frames", "short_scan")
add("patterns_reclaim", "rc_frames", "rc_scan")
add("patterns_misc", "f5_scan")

# ---------------- sketching.py ----------------
add("sketching", "xb_sketch", "_sk_path", "_sk_ema", "_sk_render", "tight_flag_sketch",
    "green_line_sketch", "shakeout_sketch", "htf_sketch", "reclaim_sketch",
    "bear_flag_sketch", "failed_breakout_sketch", "mac_sketch", "vcp_sketch")

# ---------------- charts.py ----------------
add("charts", "load_templates", "save_templates", "parse_mas", "tradingview_chart",
    "tv_open_url", "price_chart")

# ---------------- rs.py (incl. scan-mode helpers + widgets) ----------------
add("rs", "rs_add_rank", "rs_pct_rank", "rs_classify_stage", "rs_prices", "rs_stock_table",
    "rs_groups", "rs_chart", "rs_rank_color", "rs_chg_html", "rs_html", "rs_score_circle",
    "rs_stage_badge", "render_rs_view", "base_universe", "prefilter",
    "V", "pwidget", "load_pattern_preset",
    "is_record_mode", "is_earn_mode", "is_pb_mode", "is_ep_mode", "is_rs_mode",
    "rss_status_changed", "rs_quad_changed", "is_xb_mode", "xb_status_changed",
    "is_gl_mode", "gl_status_changed", "is_so_mode", "so_status_changed",
    "is_htf_mode", "htf_status_changed", "htf_load_preset", "is_rc_mode",
    "rc_status_changed", "is_f5_mode", "f5_status_changed", "is_sh_mode",
    "sh_status_changed", "is_mac_mode", "mac_status_changed", "is_vcp_mode",
    "vcp_status_changed", "is_ll_mode", "ll_status_changed", "is_sp_mode",
    "scan_mode_changed", "pb_status_changed", "ep_status_changed", "sp_type_changed",
    "earn_window_changed", "vr_window_changed",
    "watchlist_tickers", "watchlist_panels", "rs_page")

# ---------------- peers.py ----------------
add("peers", "peer_pool", "peer_intraday", "peer_table", "_pct_pill", "peers_page")

# ---------------- smart_money.py ----------------
add("smart_money", "market_mood", "render_cards", "_num", "setup_levels", "detail_card",
    "sec_headers", "sec_13f_filings", "_xml_text", "sec_13f_holdings", "_norm_name",
    "cusip_tickers", "fund_portfolio", "sm_since_period", "_amount_range", "_fmt_amount",
    "congress_members", "_match_member", "capitoltrades_trades", "_senate_session",
    "_html_rows", "senate_trades", "_pdf_text", "house_ptr", "house_filings",
    "house_trades", "_safe", "congress_trades", "sm_table", "sm_chart", "smart_money_page")

# ---------------- backtest.py ----------------
add("backtest", "_bt_tight_all", "_bt_tight", "_bt_xback", "_bt_glb", "_bt_ep", "_bt_so3",
    "_bt_vcp", "_bt_mac_all", "_bt_mac", "_bt_htf_all", "_bt_htf", "_bt_rc_all", "_bt_rc",
    "_bt_f50_all", "_bt_f50", "_bt_bf_all", "_bt_bf", "_bt_fb_all", "_bt_fb", "_bt_para",
    "bt_agree_signals", "intraday_330", "bt_at_330",
    "bt_signals", "bt_trades", "bt_pause_filter", "bt_industry_pause", "bt_by_strategy",
    "bt_size", "stage_history", "bt_bench_stages", "bt_stage_pct", "bt_stage_sizing",
    "_profit_factor", "bt_stats", "bt_universe_meta", "sector_lookup", "market_lines",
    "market_ok_history", "market_filter_label", "bt_liquidity_filter", "bt_funnel_filter",
    "bt_group_market_filter", "bt_sector_info", "month_list_text", "month_picker",
    "sector_picker", "is_excluded_sector", "drop_excluded", "bt_universe_list",
    "bt_portfolio", "bt_account_view", "bt_detail_sections", "bt_levels",
    "bt_apply_funnel_preset", "backtest_page")

# ---------------- etf.py ----------------
add("etf", "stock_theme", "group_etfs", "group_check", "group_used_etfs", "own_mover",
    "group_etf", "etf_status_history", "market_trend_history", "stock_group_status",
    "etf_status_now", "etf_panels", "etf_trend", "etf_table", "market_cards", "etf_page")

# ---------------- positions.py ----------------
add("positions", "load_positions", "save_positions", "parse_buys", "pos_actions",
    "positions_page")

# ---------------- funnel.py ----------------
add("funnel", "stock_stages_now", "industry_ranks", "peer_corr", "first_close_above_50",
    "funnel_compute", "funnel_levels", "funnel_watch_levels", "funnel_page")

# ---------------- presets.py ----------------
add("presets", "load_scan_presets", "_jsonable", "save_scan_preset", "apply_scan_preset",
    "delete_scan_preset")

# ---------------- results.py ----------------
addr("results", 10460, 10461)   # MODE_FILTER_KEYS, _fragment
add("results", "ALSO_SCANS", "also_in", "results_area")

OUT = {
    "constants": "scanner/constants.py",
    "universe": "scanner/data/universe.py",
    "prices": "scanner/data/prices.py",
    "fundamentals": "scanner/data/fundamentals.py",
    "patterns_common": "scanner/patterns/common.py",
    "patterns_volume": "scanner/patterns/volume.py",
    "patterns_crossback": "scanner/patterns/crossback.py",
    "patterns_greenline": "scanner/patterns/greenline.py",
    "patterns_shakeout": "scanner/patterns/shakeout.py",
    "patterns_trend": "scanner/patterns/trend.py",
    "patterns_mac": "scanner/patterns/mac.py",
    "patterns_htf": "scanner/patterns/htf.py",
    "patterns_short": "scanner/patterns/short.py",
    "patterns_reclaim": "scanner/patterns/reclaim.py",
    "patterns_misc": "scanner/patterns/misc.py",
    "sketching": "scanner/sketching.py",
    "charts": "scanner/charts.py",
    "rs": "scanner/rs.py",
    "peers": "scanner/peers.py",
    "smart_money": "scanner/smart_money.py",
    "backtest": "scanner/backtest.py",
    "etf": "scanner/etf.py",
    "positions": "scanner/positions.py",
    "funnel": "scanner/funnel.py",
    "presets": "scanner/presets.py",
    "results": "scanner/results.py",
}

MAIN_RANGES = [
    (1, 15),       # docstring
    (17, 34),      # imports
    (36, 36),      # st.set_page_config
    (3516, 3518),  # ss = st.session_state ; seed PATTERN_PRESETS Default
    (3570, 3590),  # session-state normalization loops/ifs
    (3608, 3611),  # for key in F ... visible defaults
    (9226, 9268),  # page routing
    (9326, 10163), # big With(...) sidebar+settings block  -- main UI flow
    (10165, 10457),# scan execution
    (11440, 11441),# final results_area() call
]

consumed = set()
for mod, items in MODULES.items():
    chunks = []
    for it in items:
        if isinstance(it, tuple):
            _, a, b = it
            chunks.append(rng(a, b)); consumed.update(range(a, b + 1))
        else:
            node = defs.get(it) or assigns.get(it)
            if node is None:
                raise SystemExit(f"NOT FOUND: {it}")
            chunks.append(seg(node))
            consumed.update(range(node.lineno, node.end_lineno + 1))
    p = pathlib.Path(OUT[mod])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    print(f"{OUT[mod]:38s} {len(chunks):3d} blocks")

main_chunks = []
for a, b in MAIN_RANGES:
    main_chunks.append(rng(a, b)); consumed.update(range(a, b + 1))
pathlib.Path("scanner/main_new.py").write_text("\n\n".join(main_chunks) + "\n", encoding="utf-8")

# report leftovers: any top-level node not fully consumed
left = []
for n in tree.body:
    span = set(range(n.lineno, n.end_lineno + 1))
    if not span <= consumed:
        nm = getattr(n, "name", "")
        if isinstance(n, ast.Assign):
            nm = ",".join(t.id for t in n.targets if isinstance(t, ast.Name))
        left.append((n.lineno, type(n).__name__, nm))
print("LEFTOVER:", left if left else "none")
