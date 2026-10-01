"""Saved scan presets (load/save/apply/delete)."""

import json

import streamlit as st

from scanner.constants import OLD_SCAN_NAMES, PATTERN_PRESETS, PRESET_FILE, PRESET_PREFIXES, SCAN_MODES

ss = st.session_state

def load_scan_presets():
    try:
        return json.loads(PRESET_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _jsonable(v):
    if isinstance(v, (str, bool, int, float)) or v is None:
        return True
    return isinstance(v, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in v)


def save_scan_preset():
    name = (ss.get("preset_new") or "").strip()
    if not name:
        ss["preset_msg"] = "⚠️ Type a name first."
        return
    keys = ["scan_mode", "v_visible"] + list(PATTERN_PRESETS["Default"]) + \
        [k for k in ss.keys() if isinstance(k, str) and k.startswith(PRESET_PREFIXES)]
    data = {k: (list(ss[k]) if isinstance(ss[k], tuple) else ss[k]) for k in keys if k in ss and _jsonable(ss[k])}
    presets = load_scan_presets()
    presets[name] = data
    try:
        PRESET_FILE.write_text(json.dumps(presets, indent=1), encoding="utf-8")
        ss["preset_pick"], ss["preset_new"] = name, ""
        ss["preset_msg"] = f"✅ Saved “{name}”."
    except Exception as e:
        ss["preset_msg"] = f"Couldn't save: {e}"


def apply_scan_preset():
    data = load_scan_presets().get(ss.get("preset_pick"), {})
    for k, v in data.items():
        if k == "scan_mode":
            v = OLD_SCAN_NAMES.get(v, v)
            if v not in SCAN_MODES:
                continue
        ss[k] = tuple(v) if k in ("pull_range", "gl_months", "so_under", "htf_depth") else v
    ss["v_tab"] = "Setup"
    ss["preset_msg"] = f"Loaded “{ss.get('preset_pick')}” — press SCAN if the stock list changed."


def delete_scan_preset():
    presets = load_scan_presets()
    presets.pop(ss.get("preset_pick"), None)
    try:
        PRESET_FILE.write_text(json.dumps(presets, indent=1), encoding="utf-8")
        ss["preset_msg"] = "🗑 Deleted."
    except Exception as e:
        ss["preset_msg"] = f"Couldn't delete: {e}"
