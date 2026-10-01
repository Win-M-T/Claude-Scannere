"""Wire the split modules together:
- add stdlib/3rd-party + cross-module imports to each scanner module
- move the global CSS st.markdown into main_new.py
- emit final scanner/main.py, delete the old monolith copy
"""
import ast, pathlib, re

ROOT = pathlib.Path("scanner")

THIRD_PARTY = {
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "px": "import plotly.express as px",
    "go": "import plotly.graph_objects as go",
    "st": "import streamlit as st",
    "components": "import streamlit.components.v1 as components",
    "yf": "import yfinance as yf",
    "make_subplots": "from plotly.subplots import make_subplots",
    "requests": "import requests",
}
STDLIB = {
    "html_lib": "import html as html_lib",
    "io": "import io",
    "json": "import json",
    "re": "import re",
    "time": "import time",
    "Path": "from pathlib import Path",
    "ThreadPoolExecutor": "from concurrent.futures import ThreadPoolExecutor",
    "date": "from datetime import date",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
}
# merge datetime imports if several are needed
def header_lines(nm_set):
    stdl = [v for k, v in STDLIB.items() if k in nm_set]
    # collapse datetime bits
    dt = sorted(k for k in ("date", "datetime", "timedelta") if k in nm_set)
    if len(dt) > 1 or (dt and dt[0] != "date"):
        stdl = [s for s in stdl if not s.startswith("from datetime")]
        stdl.append("from datetime import " + ", ".join(dt))
    tp = [v for k, v in THIRD_PARTY.items() if k in nm_set]
    return stdl, tp


def free_names(tree):
    loads, stores, args = set(), set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (loads if isinstance(n.ctx, ast.Load) else stores).add(n.id)
        elif isinstance(n, ast.arg):
            args.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.add(n.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                stores.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                stores.add(a.asname or a.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            stores.add(n.name)
        elif isinstance(n, ast.Global):
            stores.update(n.names)
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                for t in ast.walk(g.target):
                    if isinstance(t, ast.Name):
                        stores.add(t.id)
    return loads - stores - args


# ---- module registry: name -> dotted module ----
files = sorted(str(p) for p in ROOT.rglob("*.py"))
registry = {}
per_file_free = {}
for f in files:
    if f.endswith("main_new.py"):
        continue
    tree = ast.parse(pathlib.Path(f).read_text())
    mod = f[:-3].replace("/", ".").replace("\\", ".")
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
    for nm in names:
        registry.setdefault(nm, mod)
    per_file_free[f] = free_names(tree)

# ---- extract & remove the two global CSS st.markdown blocks from constants.py ----
csrc = pathlib.Path("scanner/constants.py").read_text()
clines = csrc.splitlines(keepends=True)
ctree = ast.parse(csrc)
css_blocks = []
for n in ctree.body:
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) \
       and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "markdown":
        css_blocks.append((n.lineno, n.end_lineno, "".join(clines[n.lineno - 1:n.end_lineno])))
assert len(css_blocks) == 2, [b[:2] for b in css_blocks]
# drop them (descending order so ranges stay valid)
for a, b, _ in sorted(css_blocks, reverse=True):
    del clines[a - 1:b]
pathlib.Path("scanner/constants.py").write_text("".join(clines))
CSS_MAIN, CSS_FURNITURE = css_blocks[0][2], css_blocks[1][2]

# ---- fix known cross-module deps that AST can't see as free names ----
u = pathlib.Path("scanner/data/universe.py")
usrc = u.read_text()
if "MARKET_FALLBACK" in usrc and "from scanner.constants import" in usrc:
    usrc = usrc.replace("from scanner.constants import",
                        "from scanner.constants import MARKET_FALLBACK,", 1)
elif "MARKET_FALLBACK" in usrc:
    usrc = "from scanner.constants import MARKET_FALLBACK\n" + usrc
u.write_text(usrc)

# ---- rewrite every module with proper imports ----
DOCSTRINGS = {
    "scanner/constants.py": "Constants: user-agent, page CSS data, option lists, presets, filter definitions.",
    "scanner/data/universe.py": "Universe providers: index members, listed-stock rosters, foreign markets, tickers.",
    "scanner/data/prices.py": "Price download helpers (yfinance) and panel building.",
    "scanner/data/fundamentals.py": "Fundamentals, earnings calendar and simple filter predicates.",
    "scanner/patterns/common.py": "Shared pattern-scan helpers: metrics, parameter pack, RS matrix.",
    "scanner/patterns/volume.py": "Record-volume, parabolic and episodic-pivot scans.",
    "scanner/patterns/crossback.py": "EMA crossback scan.",
    "scanner/patterns/greenline.py": "Green-line breakout scan.",
    "scanner/patterns/shakeout.py": "Shakeout +3 scan.",
    "scanner/patterns/trend.py": "Trend template and VCP (Minervini) scans.",
    "scanner/patterns/mac.py": "MA-consolidation frames/scan.",
    "scanner/patterns/htf.py": "High-tight-flag frames/scan.",
    "scanner/patterns/short.py": "Swing-short setups.",
    "scanner/patterns/reclaim.py": "50-day reclaim setups.",
    "scanner/patterns/misc.py": "First-close-above-50 scan.",
    "scanner/sketching.py": "Small SVG rule-sketch generators for the settings panes.",
    "scanner/charts.py": "TradingView widget embedding + local Plotly price chart.",
    "scanner/rs.py": "Relative-strength engine, scan-mode helpers and the RS page.",
    "scanner/peers.py": "'Similar stocks' peer-correlation page.",
    "scanner/smart_money.py": "Smart Money page: 13F funds and congressional trades.",
    "scanner/backtest.py": "Backtest engine and page.",
    "scanner/etf.py": "'Sectors in play' ETF/theme page.",
    "scanner/positions.py": "Positions tracker page.",
    "scanner/funnel.py": "Minervini-style 'Funnel' page.",
    "scanner/presets.py": "Saved scan presets (load/save/apply/delete).",
    "scanner/results.py": "Results table, filter pills and detail charts of the Stock Scanner page.",
}

for f in files:
    if f.endswith("main_new.py"):
        continue
    src = pathlib.Path(f).read_text()
    tree = ast.parse(src)
    used = per_file_free[f] | {"st"}           # every module touches streamlit
    stdl, tp = header_lines(used)
    mods = {}
    for nm in used:
        m = registry.get(nm)
        if m and m != f[:-3].replace("/", ".").replace("\\", "."):
            mods.setdefault(m, []).append(nm)
    out = ['"""%s"""' % DOCSTRINGS.get(f, ""), "",
           *stdl, *([""] if stdl and tp else []), *tp,
           *([""] if tp else [])]
    for m in sorted(mods):
        out.append("from %s import %s" % (m, ", ".join(sorted(set(mods[m])))))
    out.append("")
    out.append(src.rstrip("\n") + "\n")
    pathlib.Path(f).write_text("\n".join(out))
    print(f"{f:38s} stdlib={len(stdl)} 3p={len(tp)} crossmod={len(mods)}")
