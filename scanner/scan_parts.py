"""The pattern detectors run during a scan, as (prefix, label, callable) triples."""

from scanner.patterns.crossback import ema_crossback
from scanner.patterns.greenline import green_line
from scanner.patterns.htf import htf_scan
from scanner.patterns.mac import mac_scan
from scanner.patterns.misc import f5_scan
from scanner.patterns.reclaim import rc_scan
from scanner.patterns.shakeout import shakeout_plus3, so_rs_gate
from scanner.patterns.short import short_scan
from scanner.patterns.trend import vcp_scan
from scanner.patterns.volume import ep_scan, parabolic_scan, volume_records

SCAN_PARTS = [
    ("vr_", "record volume", lambda P, m, pp, vh: volume_records(P["Close"], P["Volume"], pp, vh)),
    ("pb_", "parabolic runs", lambda P, m, pp, vh: parabolic_scan(P["Open"], P["High"], P["Low"], P["Close"],
                                                                   P["Close"].ewm(span=10, adjust=False).mean(), pp)),
    ("ep_", "episodic pivots", lambda P, m, pp, vh: ep_scan(P["Open"], P["High"], P["Low"], P["Close"], P["Volume"], pp)),
    ("gl_", "green line breakouts", lambda P, m, pp, vh: green_line(P["High"], P["Low"], P["Close"], P["Volume"], pp)),
    ("so_", "shakeout +3 setups", lambda P, m, pp, vh: so_rs_gate(
        shakeout_plus3(P["High"], P["Low"], P["Close"], P["Volume"], pp), m["RS"], pp)),
    ("mac_", "MA consolidations", lambda P, m, pp, vh: mac_scan(P, m["RS"], pp)),
    ("htf_", "high tight flags", lambda P, m, pp, vh: htf_scan(P, pp)),
    ("sh_", "swing shorts", lambda P, m, pp, vh: short_scan(P, pp)),
    ("rc_", "50-day reclaims", lambda P, m, pp, vh: rc_scan(P, pp)),
    ("f5_", "first closes above the 50-day", lambda P, m, pp, vh: f5_scan(P, pp)),
    ("vcp_", "VCP patterns", lambda P, m, pp, vh: vcp_scan(P["High"], P["Low"], P["Close"], P["Volume"], m["RS"], pp)),
    ("xb_", "EMA crossbacks", lambda P, m, pp, vh: ema_crossback(P["Open"], P["High"], P["Low"], P["Close"], P["Volume"],
                                                                 m["RS"], pp)),
]
