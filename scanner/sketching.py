"""Small SVG rule-sketch generators for the settings panes."""

import numpy as np
import streamlit as st

from scanner.constants import MAC_LINES, XB_20
from scanner.patterns.crossback import _xb_path

def xb_sketch(rally, rally_days, ema, tol, T, pull_max, tight, tight_max, vol, candle):
    """Minimal line sketch of an EMA-crossback TRIGGER, drawn with the current settings."""
    use20 = ema == XB_20
    T = int(T)
    R = _xb_path(rally, int(np.clip(round(rally_days * 0.6), 6, 70)), use20, tol, T, tight, tight_max, vol, candle)
    rb, rhi, rtag, rtrig = R[0], R[2], R[3], R[4]
    pull_pct = (rb["H"][rhi] - min(rb["L"][rtag:rtrig])) / rb["H"][rhi] * 100
    B, lo_i, hi_i, tag_i, trig_i, entry, stop, atr = _xb_path(
        float(np.clip(rally, 12, 40)), int(np.clip(round(rally_days * 0.3), 6, 16)), use20, tol, T, tight, tight_max,
        vol, candle)
    C, L, H = B["C"], B["L"], B["H"]
    E = B["E20"] if use20 else B["E10"]
    n = len(C)
    start = max(0, lo_i - 4)
    W, Hh, top, pb, right = 300, 130, 16, 112, 262
    split = 8 + (right - 8) * 0.38

    def xs(i):
        if i <= hi_i:
            return 8 + (i - start) / max(1, hi_i - start) * (split - 8)
        return split + (i - hi_i) / max(1, n - 1 - hi_i) * (right - split)
    # price line: closes, but the tag day uses its low so the touch shows
    P = list(C)
    P[tag_i] = L[tag_i]
    lo_y, hi_y = min(min(P[start:]), stop, min(E[start:])) - atr, max(max(P[start:]), entry) + atr * 1.8
    ys = lambda v: top + (hi_y - v) / (hi_y - lo_y) * (pb - top)
    g, r, blue, ec = "#26a69a", "#ef5350", "#3b82f6", ("#a855f7" if use20 else "#f59e0b")
    path = lambda arr: " ".join(f"{xs(i):.1f},{ys(arr[i]):.1f}" for i in range(start, n))
    o = [f'<svg viewBox="0 0 {W} {Hh}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         'style="font:9px sans-serif;fill:currentColor;display:block">',
         f'<polyline points="{path(E)}" fill="none" stroke="{ec}" stroke-width="1.4" opacity=".9"/>',
         f'<polyline points="{path(P)}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>']
    xe = xs(trig_i - 5)
    for y, colr, lbl in [(entry, g, "entry"), (stop, r, "stop")]:
        o.append(f'<line x1="{xe:.1f}" x2="{right + 4}" y1="{ys(y):.1f}" y2="{ys(y):.1f}" stroke="{colr}" '
                 f'stroke-width="1" stroke-dasharray="3 2"/>')
    ye, yst = ys(entry) + 3, ys(stop) + 3
    if yst - ye < 10:
        ye, yst = (ye + yst) / 2 - 5, (ye + yst) / 2 + 5
    o.append(f'<text x="{W - 2}" y="{ye:.1f}" text-anchor="end" fill="{g}">entry</text>'
             f'<text x="{W - 2}" y="{yst:.1f}" text-anchor="end" fill="{r}">stop</text>')
    o.append(f'<text x="{right + 6}" y="{ys(E[-1]) + 3:.1f}" fill="{ec}">{"SMA20" if use20 else "EMA10"}</text>'
             if abs(ys(E[-1]) - ys(stop)) > 9 and abs(ys(E[-1]) - ys(entry)) > 9 else "")
    # marks
    o.append(f'<circle cx="{xs(tag_i):.1f}" cy="{ys(P[tag_i]):.1f}" r="3.2" fill="{blue}"/>')
    o.append(f'<text x="{xs(tag_i):.1f}" y="{ys(P[tag_i]) + 13:.1f}" text-anchor="middle" fill="{blue}">tag ±{tol:g}%</text>')
    o.append(f'<circle cx="{xs(trig_i):.1f}" cy="{ys(C[trig_i]):.1f}" r="3.2" fill="{g}"/>')
    o.append(f'<text x="{min(xs(trig_i), right - 16):.1f}" y="{ys(C[trig_i]) - 7:.1f}" text-anchor="middle" fill="{g}" '
             f'style="font-weight:700">TRIGGER</text>')
    o.append(f'<text x="{xs(hi_i):.1f}" y="{ys(H[hi_i]) - 6:.1f}" text-anchor="middle" opacity=".7">+{rally:g}% in ≤{rally_days:g}d</text>')
    o.append(f'<text x="{xs(lo_i):.1f}" y="{ys(P[lo_i]) + 12:.1f}" text-anchor="middle" opacity=".55">low</text>')
    # the "tagged within N days" span
    bx1, bx2 = xs(trig_i - T), xs(trig_i - 1)
    o.append(f'<path d="M{bx1:.1f},{pb + 4} H{max(bx2, bx1 + 2):.1f}" stroke="currentColor" opacity=".4"/>'
             f'<text x="{(bx1 + bx2) / 2:.1f}" y="{pb + 14}" text-anchor="middle" opacity=".6">≤ {T}d</text>')
    o.append("</svg>")
    note = ""
    if pull_pct > pull_max:
        note = (f"⚠️ After a +{rally:g}% rally, a dip to the {'20' if use20 else '10'}-day EMA is usually about "
                f"{pull_pct:.0f}% — more than your {pull_max:g}% max pullback, so few stocks will pass.")
    return "".join(o), note


def _sk_path(knots, amp=0.005):
    """Straight segments between (x, price) knots with a small zig-zag so it reads like a price line."""
    out = []
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        for x in range(int(x0), int(x1)):
            y = y0 + (y1 - y0) * (x - x0) / max(1, x1 - x0)
            out.append((x, y * (1 + (amp if x % 2 else -amp) * (0 if x == x0 else 1))))
    out.append(knots[-1])
    return out


def _sk_ema(path, span):
    e, out = None, []
    for x, y in path:
        e = y if e is None else e + (y - e) * 2 / (span + 1)
        out.append((x, e))
    return out


def _sk_render(price, x_from=0, ema=None, hl=(), dots=(), texts=(), band=None, brackets=(), vols=None, hot_col=None):
    """Tiny minimal line chart (SVG). Everything is given in (bar, price) units."""
    W, top, pb, right = 300, 16, 106, 262
    vol_h = 16 if vols else 0
    br_h = 14 if brackets else 0
    Hh = pb + br_h + vol_h + 6
    pts = [p for p in price if p[0] >= x_from]
    ys_all = [y for _, y in pts] + [h[0] for h in hl] + ([y for x, y in ema[0] if x >= x_from] if ema else [])
    if band:
        ys_all += [band[2], band[3]]
    lo, hi = min(ys_all), max(ys_all)
    pad = (hi - lo) * 0.12 or 1
    lo, hi = lo - pad * 0.6, hi + pad * 1.4
    x_to = max(x for x, _ in pts)
    xs = lambda x: 8 + (x - x_from) / max(1, x_to - x_from) * (right - 8)
    ys = lambda v: top + (hi - v) / (hi - lo) * (pb - top)
    poly = lambda arr: " ".join(f"{xs(x):.1f},{ys(y):.1f}" for x, y in arr if x >= x_from)
    g, r = "#22c55e", "#ef5350"
    o = [f'<svg viewBox="0 0 {W} {Hh}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         'style="font:9px sans-serif;fill:currentColor;display:block">']
    if band:
        x0, x1, y0, y1, colr, lbl = band
        o.append(f'<rect x="{xs(x0):.1f}" y="{ys(max(y0, y1)):.1f}" width="{xs(x1) - xs(x0):.1f}" '
                 f'height="{abs(ys(y0) - ys(y1)):.1f}" fill="{colr}" opacity=".16"/>')
        if lbl:
            o.append(f'<text x="{xs(x0) + 2:.1f}" y="{ys(min(y0, y1)) + 9:.1f}" fill="{colr}" opacity=".9">{lbl}</text>')
    if ema:
        pts_e, colr, lbl = ema
        o.append(f'<polyline points="{poly(pts_e)}" fill="none" stroke="{colr}" stroke-width="1.3" opacity=".9"/>')
        if lbl:
            o.append(f'<text x="{right + 6}" y="{ys(pts_e[-1][1]) + 3:.1f}" fill="{colr}">{lbl}</text>')
    labels_r = []
    for y, x0, x1, colr, dash, lbl in hl:
        o.append(f'<line x1="{xs(x0):.1f}" x2="{right + 4 if x1 is None else xs(x1):.1f}" y1="{ys(y):.1f}" '
                 f'y2="{ys(y):.1f}" stroke="{colr}" stroke-width="1.1" stroke-dasharray="{dash}"/>')
        if lbl:
            labels_r.append([ys(y) + 3, colr, lbl])
    labels_r.sort()
    for i in range(1, len(labels_r)):                      # keep right-hand labels from overlapping
        labels_r[i][0] = max(labels_r[i][0], labels_r[i - 1][0] + 10)
    for yv, colr, lbl in labels_r:
        o.append(f'<text x="{W - 2}" y="{yv:.1f}" text-anchor="end" fill="{colr}">{lbl}</text>')
    o.append(f'<polyline points="{poly(price)}" fill="none" stroke="currentColor" stroke-width="1.6" '
             f'stroke-linejoin="round"/>')
    for x, y, colr, lbl, dy, bold in dots:
        o.append(f'<circle cx="{xs(x):.1f}" cy="{ys(y):.1f}" r="3.2" fill="{colr}"/>')
        if lbl:
            fw = "700" if bold else "400"
            o.append(f'<text x="{min(max(xs(x), 20), right - 14):.1f}" y="{ys(y) + dy:.1f}" text-anchor="middle" '
                     f'fill="{colr}" style="font-weight:{fw}">{lbl}</text>')
    for x, y, txt, anchor, colr, op in texts:
        o.append(f'<text x="{xs(x):.1f}" y="{ys(y):.1f}" text-anchor="{anchor}" fill="{colr}" opacity="{op}">{txt}</text>')
    by = pb + 5
    for x0, x1, lbl in brackets:
        o.append(f'<path d="M{xs(x0):.1f},{by - 2} V{by} H{xs(x1):.1f} V{by - 2}" fill="none" stroke="currentColor" '
                 f'opacity=".45"/><text x="{(xs(x0) + xs(x1)) / 2:.1f}" y="{by + 9}" text-anchor="middle" '
                 f'opacity=".65">{lbl}</text>')
    if vols:
        vb = Hh - 3
        vmax = max(v for _, v, _ in vols)
        bw = max(1.2, (right - 8) / max(1, x_to - x_from) * 0.6)
        for x, v, hot in vols:
            if x >= x_from:
                h = v / vmax * vol_h
                o.append(f'<rect x="{xs(x) - bw / 2:.1f}" y="{vb - h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                         f'fill="{(hot_col or g) if hot else "currentColor"}" opacity="{.9 if hot else .25}"/>')
    o.append("</svg>")
    return "".join(o)


def tight_flag_sketch(max_range, pull_range, min_bars, max_ema_dist, max_ema_gap, dry_up, breakout_vol):
    """Minimal line sketch of a Tight-flag TRIGGER with the current settings."""
    pull = float(np.clip((pull_range[0] + pull_range[1]) / 2, 1.5, 30))
    days = int(max(min_bars, 4)) + 1
    hi_x = 26
    flag_lo = 100 * (1 - pull / 100)
    flag_top = flag_lo * (1 + min(max_range, pull) * 0.55 / 100)
    knots = [(0, 72), (8, 73), (hi_x, 100), (hi_x + 2, flag_lo)]
    x = hi_x + 2
    for k in range(days - 2):
        x += 1
        knots.append((x, flag_top if k % 2 == 0 else flag_lo * 1.006))
    path = _sk_path(knots, 0.004)
    trig = max(y for xx, y in path if xx >= x - 6)
    brk = x + 1
    path.append((brk, trig * 1.025))
    path.append((brk + 1, trig * 1.035))
    stop = min(y for xx, y in path if brk - 7 <= xx < brk) * 0.995
    e21 = [(x_, sum(y for _, y in path[max(0, i - 19):i + 1]) / len(path[max(0, i - 19):i + 1]))
           for i, (x_, _) in enumerate(path)]                              # 20-day SMA
    vols = [(xx, (dry_up if hi_x < xx < brk else 1.0), False) for xx, _ in path[:-2]] + \
           [(brk, breakout_vol, True), (brk + 1, 1.1, False)]
    rng7 = (trig - min(y for xx, y in path if brk - 7 <= xx < brk)) / trig * 100
    svg = _sk_render(
        path, x_from=15, ema=(e21, "#a855f7", ""),
        hl=[(trig, brk - 7, None, "#22c55e", "3 2", "trigger"), (stop, brk - 7, None, "#ef5350", "3 2", "stop")],
        dots=[(hi_x, 100, "currentColor", "20-day high", -7, False),
              (brk, trig * 1.025, "#22c55e", "TRIGGER", -8, True)],
        texts=[(hi_x - 1, flag_lo * 0.985, f"pullback {pull_range[0]:g}–{pull_range[1]:g}%", "end",
                "currentColor", .7), (15, e21[15][1] * 0.975, "SMA20", "start", "#a855f7", .9)],
        brackets=[(hi_x, brk - 1, f"≥ {min_bars}d since high · range ≤ {max_range:g}%")],
        vols=vols)
    notes = [f"Flag hugs the 20-day SMA (≤ {max_ema_dist:g}% above) · volume dries up (≤ {dry_up:g}× avg), "
             f"then the breakout comes on ≥ {breakout_vol:g}× volume."]
    if pull_range[0] > max_range and min_bars < 7:
        notes.append(f"⚠️ A pullback of at least {pull_range[0]:g}% can't fit in a 7-day range of {max_range:g}% "
                     f"when the high is less than 7 days back — few stocks will pass.")
    return svg, " ".join(notes)


def green_line_sketch(months, vol_x, near, retest_days, tol, stop_pct):
    """Minimal line sketch of a Green line BREAKOUT (and RETEST) with the current settings."""
    months = f"{months[0]:g}–{months[1]:g}" if isinstance(months, (tuple, list)) else f"≥ {months:g}"
    line = 100.0
    knots = [(0, 68), (12, line), (20, 80), (28, 91), (35, 82), (45, 96.5), (50, 97.2), (53, 104),
             (57, line * (1 + tol / 200)), (62, 110)]
    path = _sk_path(knots, 0.006)
    stop = line * (1 - stop_pct / 100)
    vols = [(x, 1.0, False) for x, _ in path]
    vols[53] = (53, vol_x, True)
    svg = _sk_render(
        path, x_from=0,
        hl=[(line, 12, None, "#22c55e", "none", "green line"), (stop, 50, None, "#ef5350", "3 2", "stop")],
        band=(38, 53, line * (1 - near / 100), line, "#3b82f6", ""),
        dots=[(12, line, "#22c55e", "all-time high", -7, False),
              (53, 104, "#22c55e", "BREAKOUT", -8, True),
              (57, line * (1 + tol / 200), "#3b82f6", "", 0, False)],
        texts=[(37, line * (1 - near / 100) * 1.004, f"NEAR ≤ {near:g}%", "end", "#3b82f6", .95),
               (59, line * 0.935, f"RETEST ±{tol:g}%", "middle", "#3b82f6", .95)],
        brackets=[(12, 52, f"line set {months} months ago"), (53, 58, f"≤ {retest_days:g}d")],
        vols=vols)
    note = (f"BREAKOUT = first close above the line on ≥ {vol_x:g}× average volume · RETEST = back to the line "
            f"within {retest_days:g} days and holding · stop {stop_pct:g}% under the line.")
    return svg, note


def shakeout_sketch(rule, pct, base_days, recent, under, mid, depth, trend, vol_x, stop_pct, entry="", fast=0,
                    shvol=0.0):
    """Minimal line sketch of a Shakeout +3 TRIGGER with the current settings."""
    if isinstance(under, (tuple, list)):
        under_lbl = f"{under[0]:g}–{under[1]:g}"
        under = (under[0] + under[1]) / 2
    else:
        under_lbl = f"{under:g}"
    first = 100.0
    shake = first * (1 - under / 100)
    level = first + (6 if first > 60 else 3) if rule.startswith("Livermore") else first * (1 + pct / 100)
    mid_top = first * (1 + mid / 100)
    base_hi = max(shake / (1 - depth * 0.85 / 100), mid_top * 1.03, level * 1.03)
    true_depth = (base_hi - shake) / base_hi * 100
    start = base_hi / (1 + max(trend, 5) / 100)
    knots = [(0, start), (14, base_hi), (24, first), (30, mid_top), (38, shake), (41, first * 1.01),
             (45, level * 1.012), (47, level * 1.03)]
    path = _sk_path(knots, 0.005)
    stop = shake * (1 - stop_pct / 100)
    lvl_lbl = "+3 = +$6" if rule.startswith("Livermore") else f"+3 = +{pct:g}%"
    reclaim = entry.startswith("Reclaim")
    tx = 41 if reclaim else 45
    vols = [(x, 1.0, False) for x, _ in path]
    vols[tx] = (tx, max(vol_x, 1.0) * 1.3, True)
    if shvol > 0:
        vols[38] = (38, max(shvol, 1.0) * 1.3, False)
    hl = [(first, 24, None, "#3b82f6", "3 2", "first low" + (" = buy" if reclaim else "")),
          (stop, 34, None, "#ef5350", "3 2", "stop")]
    if not reclaim:
        hl.insert(1, (level, 24, None, "#22c55e", "3 2", lvl_lbl))
    svg = _sk_render(
        path, x_from=0,
        hl=hl,
        dots=[(24, first, "#3b82f6", "", 0, False), (38, shake, "#ef5350", "", 0, False),
              (tx, first * 1.01 if reclaim else level * 1.012, "#22c55e", "TRIGGER", -8, True)],
        texts=[(30, mid_top * 1.012, f"+{mid:g}%", "middle", "currentColor", .7),
               (36.5, shake * 0.975, f"shakeout −{under_lbl}%", "end", "#ef5350", .95),
               (14, base_hi * 1.015, f"base high (+{trend:g}% uptrend)", "middle", "currentColor", .7)],
        brackets=[(14, 38, f"base ~{base_days:g}d · depth ≤ {depth:g}%")] + ([(38, tx, f"≤ {fast:g}d")] if fast > 0 else []),
        vols=vols)
    note = (f"W-bottom: a first low, a bounce, then a shakeout {under_lbl}% under the first low"
            f"{f' on ≥ {shvol:g}× volume' if shvol > 0 else ''} that quickly reclaims it. TRIGGER = "
            + ("the first close back above the first low" if reclaim else "close above the +3 level")
            + f"{f' on ≥ {vol_x:g}× volume' if vol_x > 1 else ''}"
            + (f", within {fast:g} days of the shakeout low" if fast > 0 else "")
            + f" · stop {stop_pct:g}% under the shakeout low.")
    if true_depth > depth + 0.5:
        note += (f" ⚠️ With a +{mid:g}% bounce and a {under:.0f}% undercut the base is about {true_depth:.0f}% deep — "
                 f"deeper than your {depth:g}% max, so few stocks will pass.")
    return svg, note


def htf_sketch(pole, depth, fmin, fmax, vol_x, dry):
    """Minimal line sketch of a High Tight Flag: a steep pole, a shallow quiet flag, a breakout."""
    base, top = 50.0, 50.0 * (1 + pole / 100)
    dep = (depth[0] + depth[1]) / 2
    lo = top * (1 - dep / 100)
    fl = int(np.clip((fmin + fmax) / 2, 6, 25))
    knots = [(0, base * 0.97), (8, base), (38, top)]
    x, i = 38, 0
    while x < 38 + fl:                                     # a few gentle swings inside the flag
        x = min(38 + fl, x + 3)
        knots.append((x, lo if i == 1 else top * (1 - dep / 100 * (0.3 if i % 2 == 0 else 0.7))))
        i += 1
    knots += [(x + 2, top * 1.03), (x + 3, top * 1.06)]
    path = _sk_path(knots, 0.004)
    vols = [(xx, 1.6 if 8 < xx <= 38 else (min(dry, 1.0) * 0.8 if 38 < xx <= x else 1.0), False) for xx, _ in path]
    vols[x + 2] = (x + 2, max(vol_x, 1.0) * 1.4, True)
    svg = _sk_render(path, x_from=0,
                     hl=[(top, 38, None, "#22c55e", "3 2", "flag high"), (lo, 38, None, "#ef5350", "3 2", "flag low / stop")],
                     dots=[(x + 2, top * 1.03, "#22c55e", "TRIGGER", -8, True)],
                     texts=[(22, (base + top) / 2, f"pole +{pole:g}%", "end", "currentColor", .8),
                            (38 + fl / 2, lo * 0.92, f"−{depth[0]:g}–{depth[1]:g}%", "middle", "#ef5350", .95)],
                     brackets=[(8, 38, "≤ 8 weeks"), (38, x, f"{fmin}–{fmax}d flag")], vols=vols)
    note = (f"A stock up ≥ {pole:g}% in about 8 weeks rests in a flag {depth[0]:g}–{depth[1]:g}% under its high for "
            f"{fmin}–{fmax} days on drying volume · TRIGGER = close above the flag high"
            f"{f' on ≥ {vol_x:g}× volume' if vol_x > 0 else ''} · stop = the flag low.")
    return svg, note


def reclaim_sketch(trend, dip_max, min_dip, max_days, hl, stop_hl=True, prior=True):
    """Minimal line sketch of a 50-day reclaim: under the SMA, a pop above it that fails, a shallow dip whose
    pullback makes a higher low, then a close back above."""
    rising = trend.startswith("Pullback")
    falling = trend.startswith("Turnaround")
    sma = (lambda x: 100 + 0.12 * (x - 40)) if rising else (lambda x: 100 - 0.12 * (x - 40)) if falling else \
        (lambda x: 100.0)
    dd = min(max(dip_max, 2.0), 12.0) / 100
    lo1, lo2 = sma(35) * (1 - dd * 0.75), sma(39) * (1 - dd * 0.42)
    lo0 = min(sma(18) * 0.915, lo1 * 0.965)                 # the low before the pop — under the dip low
    knots = [(0, sma(0) * 0.9), (10, sma(10) * 0.95), (18, lo0), (26, sma(26) * 0.97),
             (30, sma(30) * 1.012), (32, sma(32) * 1.005), (35, lo1), (37, sma(37) * (1 - dd * 0.15)),
             (39, lo2) if hl else (39, sma(39) * (1 - dd * 0.3)), (42, sma(42) * 0.99), (43, sma(43) * 1.012),
             (46, sma(46) * 1.04), (48, sma(48) * 1.055)]
    path = _sk_path(knots, 0.002)
    line = [(x, sma(x)) for x, _ in path]
    stop_y = (lo2 if hl and stop_hl else lo1) * 0.992
    vols = [(x, 1.0, False) for x, _ in path]
    vols[43] = (43, 1.4, True)
    dots = [(30, sma(30) * 1.012, "#f59e0b", "pop", -8, False), (35, lo1, "#94a3b8", "dip low", 14, False),
            (43, sma(43) * 1.012, "#22c55e", "TRIGGER", -9, True)]
    if hl:
        dots.append((39, lo2, "#3b82f6", "higher low", 14, False))
    lines = [(stop_y, 38 if hl and stop_hl else 34, None, "#ef5350", "3 2", "stop")]
    if prior:
        dots.append((18, lo0, "#a855f7", "low before pop", 14, False))
        lines.append((lo0, 18, None, "#a855f7", "2 3", ""))
    svg = _sk_render(path, x_from=10, ema=(line, "#cddc39", ""),
                     hl=lines,
                     dots=dots,
                     texts=[(11, sma(11) * 1.02, "SMA50" + (" rising" if rising else " falling" if falling else ""),
                             "start", "#cddc39", .95)],
                     brackets=[(30, 43, f"≤ {max_days}d · dip ≤ {dip_max:g}% under")], vols=vols)
    note = (f"Under the 50-day SMA, a pop closes above it and fails; the dip stays within {dip_max:g}% of the line"
            + (" and above the low before the pop" if prior else "")
            + (", bounces and pulls back to a higher low above its lowest low" if hl else "")
            + f" (at least {min_dip} closes under) · TRIGGER = the first close back above, within {max_days} days "
            f"of the pop · stop = under the {'higher low' if hl and stop_hl else 'dip low'}.")
    return svg, note


def bear_flag_sketch(ma, tol, bounce, dry, days):
    """Minimal line sketch of a bear flag: a downtrend, a light-volume bounce into the falling SMA, then a break."""
    use50 = ma == "50-day SMA"
    span = 50 if use50 else 20
    x0 = 55 if use50 else 25                                 # start drawing once the SMA exists
    lo_x, x_t = x0 + 45, x0 + 53                             # the low, then the bounce tags the SMA
    start, low = 150.0, 100.0
    sma_of = lambda path, i: sum(y for _, y in path[max(0, i - span + 1):i + 1]) / len(path[max(0, i - span + 1):i + 1])
    top = low * 1.08
    for _ in range(4):                                       # the bounce top sits right at the SMA (within tol)
        knots = [(0, start + (start - low) * x0 / 45), (x0, start), (lo_x, low), (x_t, top)]
        path0 = _sk_path(knots, 0.0)
        top = sma_of(path0, x_t) * (1 - tol / 200)
    knots = [(0, start + (start - low) * x0 / 45), (x0, start), (lo_x, low), (x_t, top),
             (x_t + 2, top * 0.975), (x_t + 3, top * 0.955), (x_t + 6, top * 0.91), (x_t + 10, top * 0.86),
             (x_t + 16, top * 0.8)]
    path = _sk_path(knots, 0.003)
    line = [(x_, sma_of(path, i)) for i, (x_, _) in enumerate(path)]
    sell = top * 0.962
    vols = [(xx, 1.2 if xx <= lo_x else (min(dry, 1.0) * 0.6 if xx <= x_t else 1.0), False) for xx, _ in path]
    vols[x_t + 4] = (x_t + 4, 1.6, True)
    x0 = lo_x - 22
    svg = _sk_render(path, x_from=x0, ema=(line, "#a855f7", ""),
                     hl=[(top * 1.012, x_t - 1, None, "#ef5350", "3 2", "stop = bounce high"),
                         (sell, x_t, None, "#f97316", "3 2", "sell below")],
                     dots=[(x_t, top, "#a855f7", "", -8, False),
                           (x_t + 4, top * 0.935, "#ef5350", "TRIGGER", 13, True)],
                     texts=[(x0 + 2, line[x0 + 2][1] * 1.015, f"falling {'SMA50' if use50 else 'SMA20'}", "start",
                             "#a855f7", .9),
                            (x_t - 1, top * 1.035, "tags the SMA", "end", "#a855f7", .9)],
                     brackets=[(lo_x, x_t, f"bounce ≥ {bounce:g}%, light volume")], vols=vols,
                     hot_col="#ef5350")
    return svg


def failed_breakout_sketch(n, days, vol_x):
    """Minimal line sketch of a failed breakout: a base, a close above its high, then a close back under it."""
    knots = [(0, 92.0), (6, 100.0), (12, 94.0), (18, 99.5), (24, 95.0), (30, 99.0), (33, 103.5), (34, 104.5),
             (35, 102.0), (36, 98.0), (38, 95.0), (42, 90.0), (48, 84.0)]
    path = _sk_path(knots, 0.003)
    vols = [(xx, 0.9, False) for xx, _ in path]
    vols[33] = (33, 1.4, False)
    vols[36] = (36, max(vol_x, 1.0) * 1.5, True)
    svg = _sk_render(path, x_from=0,
                     hl=[(100.0, 6, None, "#f97316", "3 2", f"{n}-day high = sell below"),
                         (104.8, 33, None, "#ef5350", "3 2", "stop (failed high)")],
                     dots=[(33, 103.5, "#22c55e", "", -7, False),
                           (36, 98.0, "#ef5350", "TRIGGER", 12, True)],
                     texts=[(31.5, 104.2, "breakout", "end", "#22c55e", .95)],
                     brackets=[(0, 32, f"base ({n}-day high)"), (33, 33 + days, f"≤ {days}d")], vols=vols,
                     hot_col="#ef5350")
    return svg


def mac_sketch(ma, days, tol, rng, near, vol_x, hold):
    """Minimal line sketch: an uptrend, then a tight sideways box resting on the rising 20/50-day SMA."""
    use50 = ma == "50-day SMA"
    span = 50 if use50 else 20
    d = int(np.clip(days, 6, 30))
    run = 70 if use50 else 45
    box_lo = 100.0 * (1 - min(rng, 20) / 100 * 0.75)
    knots = [(0, 55.0 if use50 else 75.0), (run, 100.0)]
    x, k = run, 0
    while x < run + d:                                   # a few gentle swings inside the box
        x = min(run + d, x + 3)
        knots.append((x, box_lo * 1.01 if k % 2 == 0 else 99.3))
        k += 1
    knots += [(x + 2, 103.0), (x + 3, 104.2)]
    path = _sk_path(knots, 0.002)
    line = [(x_, sum(y for _, y in path[max(0, i - span + 1):i + 1]) / len(path[max(0, i - span + 1):i + 1]))
            for i, (x_, _) in enumerate(path)]
    vols = [(xx, 0.7 if run < xx <= x else 1.0, False) for xx, _ in path]
    vols[x + 2] = (x + 2, max(vol_x, 1.0) * 1.3, True)
    x0 = run - 25
    svg = _sk_render(path, x_from=x0, ema=(line, "#a855f7", ""),
                     hl=[(100.0, run, None, "#22c55e", "3 2", "box high"), (box_lo, run, None, "#ef5350", "3 2", "stop")],
                     dots=[(x + 2, 103.0, "#22c55e", "TRIGGER", -8, True)],
                     texts=[(x0 + 1, line[x0 + 1][1] * 0.975, "SMA50" if use50 else "SMA20", "start", "#a855f7", .9)],
                     brackets=[(run, x, f"{d}d box · range ≤ {rng:g}%")], vols=vols)
    note = (f"Every {'low' if hold == 'Every low' else 'close'} of the last {days:g} days held above the rising "
            f"{ma if ma != MAC_LINES[0] else '20- or 50-day SMA'} (up to {tol:g}% under it allowed), range ≤ {rng:g}%, "
            f"close ≤ {near:g}% above the line · TRIGGER = close above the box high"
            f"{f' on ≥ {vol_x:g}× volume' if vol_x > 0 else ''} · stop = the box low.")
    return svg, note


def vcp_sketch(swing, min_c, first_max, last_max, dry, vol_x, near, base):
    """Minimal line sketch of a VCP with the current settings: shrinking pullbacks, then a breakout."""
    nc = max(2, min(int(min_c) + 1, 4))
    d = [first_max * 0.8]
    while len(d) < nc:
        d.append(d[-1] * 0.5)
    d[-1] = min(d[-1], last_max * 0.8)
    knots, x, top = [(0, 70.0), (12, 100.0)], 12, 100.0
    highs = []
    for i, dep in enumerate(d):
        lo = top * (1 - dep / 100)
        x += 9 - i
        knots.append((x, lo))
        nxt = top * (1 - dep / 100 * 0.25) if i < len(d) - 1 else top * (1 - d[-1] / 100 * 0.15)
        x += 8 - i
        knots.append((x, nxt))
        highs.append((x - (8 - i), top, dep))
        top = nxt
    pivot = top
    knots += [(x + 3, pivot * 1.03), (x + 5, pivot * 1.05)]
    path = _sk_path(knots, 0.004)
    vols = [(xx, 1.0 - 0.6 * min(1, max(0, xx - 12) / max(1, x - 12)), False) for xx, _ in path]
    vols[x + 3] = (x + 3, max(vol_x, 1.0) * 1.2, True)
    texts = [(hx + 3, top_ * (1 - dep / 100) * 0.975, f"−{dep:.0f}%", "middle", "#ef5350", .95)
             for hx, top_, dep in highs]
    svg = _sk_render(path, x_from=0,
                     hl=[(pivot, x - 6, None, "#22c55e", "3 2", "pivot"),
                         (pivot * (1 - d[-1] / 100), x - 6, None, "#ef5350", "3 2", "stop")],
                     dots=[(x + 3, pivot * 1.03, "#22c55e", "TRIGGER", -8, True)],
                     texts=texts + [(12, 101.5, "base high", "middle", "currentColor", .7)],
                     brackets=[(12, x + 3, f"base ≤ {base:g}d · {len(d)} contractions, each smaller")], vols=vols)
    note = (f"Each pullback smaller than the last (first ≤ {first_max:g}%, last ≤ {last_max:g}%), swings of "
            f"≥ {swing:g}% · volume dries up (≤ {dry:g}× avg) · TRIGGER = close above the pivot on ≥ {vol_x:g}× "
            f"volume · SETUP = within {near:g}% under it · stop = the last contraction's low.")
    return svg, note
