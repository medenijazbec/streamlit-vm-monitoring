# ui/dashboard.py
"""
dashboard.py
------------

Streamlit UI for sentra.

What's new in this version:
- Docker container list is collapsible (expander).
- Containers render with custom HTML/CSS:
  - compact 1-line rows
  - smaller monospace font
  - nowrap + ellipsis (no wrap / no double lines)
  - scrollable if you have a lot of containers
  - responsive flex columns
- Columns are: Name | RAM | CPU% | Lifetime | Command
- Sorting still works (Name / RAM / CPU% / Lifetime).
- RAM/CPU show live values if SENTRA_DOCKER_STATS=1. Otherwise N/A, but UI renders instantly.
- Top status block:
  - "GPU TEMP" moved under the LOAD column.
  - Columns made wider (bigger min-width per col).
"""

import os
import sys
import time
import platform
import subprocess
from datetime import datetime, timedelta
import html as html_lib  # for escaping container names/cmds in HTML

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import psutil
from streamlit_autorefresh import st_autorefresh

# Ensure project root is on sys.path (so imports work with `streamlit run ui/dashboard.py`)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import config
import api.datastore as datastore
import collector.logger as logger


# ---------- Page setup ----------
st.set_page_config(page_title="sentra dashboard", layout="wide")
# Intentionally no st.title() to save vertical space
st.markdown(
    """
    <style>
    .stApp, .block-container, .main {
        transition: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Height (px) of the top status block iframe.
STATUS_BLOCK_HEIGHT = int(os.getenv("SENTRA_STATUS_BLOCK_HEIGHT", "600"))
GPU_LOG_FILENAME = "gpu_log.csv"
GPU_LOG_HEADER = "timestamp,gpu_index,temp,util,power,vram"


def get_gpu_log_path() -> str:
    path = config.get_db_path()
    dirpath = os.path.dirname(path)
    if not dirpath:
        dirpath = "."
    os.makedirs(dirpath, exist_ok=True)
    return os.path.join(dirpath, GPU_LOG_FILENAME)


def ensure_gpu_log_path() -> str:
    path = get_gpu_log_path()
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(GPU_LOG_HEADER + "\n")
        return path

    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        first = ""

    if first != GPU_LOG_HEADER:
        try:
            df_old = pd.read_csv(path)
            df_old["power"] = pd.NA
            df_old["vram"] = pd.NA
            df_old.to_csv(path, index=False)
        except pd.errors.EmptyDataError:
            with open(path, "w", encoding="utf-8") as f:
                f.write(GPU_LOG_HEADER + "\n")
        except Exception as err:
            st.warning(f"Could not migrate GPU log header: {err}")
        finally:
            # Ensure header exists even if migration failed
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(GPU_LOG_HEADER + "\n")
            else:
                with open(path, "r+", encoding="utf-8") as f:
                    content = f.read()
                    if not content.startswith(GPU_LOG_HEADER):
                        f.seek(0)
                        f.write(GPU_LOG_HEADER + "\n")
                        f.truncate()

    return path


def append_gpu_log(path: str, records: list[tuple[int, float | None, float | None, float | None, float | None]]) -> None:
    now = datetime.now().isoformat()
    with open(path, "a", encoding="utf-8") as f:
        for idx, temp, util, power, vram in records:
            line = ",".join(
                [
                    now,
                    str(idx),
                    "" if temp is None else str(temp),
                    "" if util is None else str(util),
                    "" if power is None else str(power),
                    "" if vram is None else str(vram),
                ]
            )
            f.write(line + "\n")


def read_gpu_history(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except (pd.errors.EmptyDataError, ValueError):
        df = pd.DataFrame()
    return df


def get_cpu_temp() -> float | None:
    try:
        out = subprocess.check_output(
            ["sensors"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if "Package id 0:" in line and "°C" in line:
                parts = line.split("+", 1)
                if len(parts) < 2:
                    continue
                temp_str = parts[1].split("°")[0]
                return float(temp_str)
    except Exception:
        pass
    return None


# ---------- Sidebar controls ----------
refresh_interval = st.sidebar.slider(
    "Refresh interval (seconds)",
    min_value=1,
    max_value=10,
    value=config.SAMPLE_INTERVAL,
)
st_autorefresh(
    interval=int(float(refresh_interval) * 1000),
    limit=None,
    key="sentra_autorefresh",
)

st.sidebar.markdown("---")

col_purge_a, col_purge_b = st.sidebar.columns(2)
if col_purge_a.button("Purge ALL"):
    datastore.purge_before(time.time() + 10**9)
if col_purge_b.button("Purge >24h"):
    datastore.purge_before(time.time() - config.ONE_DAY)

st.sidebar.markdown("---")
st.sidebar.write(f"DB path: `{config.get_db_path()}`")


# ---------- Stateful counters for throughput ----------
if "prev_disk" not in st.session_state:
    st.session_state.prev_disk = None
if "prev_net" not in st.session_state:
    st.session_state.prev_net = None
if "last_sample_ts" not in st.session_state:
    st.session_state.last_sample_ts = time.time()

# persistent fan max map for display scaling
if "fan_max" not in st.session_state:
    st.session_state.fan_max = {}  # label -> max rpm (int, default 4000)

# docker container table sort state
if "container_sort_key" not in st.session_state:
    # default sort: memory consumption high -> low
    st.session_state.container_sort_key = "mem_bytes"
if "container_sort_reverse" not in st.session_state:
    st.session_state.container_sort_reverse = True


# ---------- Collect metrics & log ----------
now_ts = time.time()
interval_s = now_ts - st.session_state.last_sample_ts
if interval_s <= 0:
    interval_s = float(refresh_interval)

snapshot, gpus, containers, new_disk, new_net = logger.collect_and_store(
    prev_disk=st.session_state.prev_disk,
    prev_net=st.session_state.prev_net,
    interval_s=interval_s,
)

st.session_state.prev_disk = new_disk
st.session_state.prev_net = new_net
st.session_state.last_sample_ts = now_ts

# initialize fan defaults if we saw new fans
for fan_label in snapshot["fans"].keys():
    st.session_state.fan_max.setdefault(fan_label, 4000)


# ---------- Helpers ----------
def format_seconds(sec: float) -> str:
    d = timedelta(seconds=int(sec))
    return str(d)


def human_bytes(num: float | None) -> str:
    """Return nice human-readable byte string like '18.2G', '163M', etc."""
    if num is None:
        return "N/A"
    kb = 1024.0
    mb = kb * 1024.0
    gb = mb * 1024.0
    if num >= gb:
        return f"{num/gb:.1f}G"
    if num >= mb:
        return f"{num/mb:.0f}M"
    if num >= kb:
        return f"{num/kb:.0f}K"
    return f"{int(num)}"


def fmt_pct(v) -> str:
    """Format a percentage or return 'N/A' if None."""
    if isinstance(v, (int, float)):
        return f"{v:.1f}%"
    return "N/A"


def fmt_ram_col(mem_bytes, mem_pct) -> str:
    """Combine mem bytes + pct into one cell."""
    return f"{human_bytes(mem_bytes)} ({fmt_pct(mem_pct)})"


def pct_bar_html(
    pct: float,
    warn_threshold: float,
    display_text: str | None = None,
    width_chars: int = 20,
) -> str:
    """
    Build an ASCII-style bar (██████      ) + inline text.
    - Fixed width so layout doesn't jump
    - Colored red when above warn_threshold, else light blue
    - All content lives inside one inline-block span
    """
    if pct is None:
        return "<span class='sentra-barbox'>[no data]</span>"

    pct = max(0.0, min(100.0, pct))

    filled = int(round((pct / 100.0) * width_chars))
    empty = width_chars - filled
    bar_txt = "█" * filled + " " * empty

    color = "#ff0033" if pct >= warn_threshold else "#00aaff"
    if display_text is None:
        display_text = f"{pct:4.1f}%"

    return (
        "<span class='sentra-barbox' style='color:"
        + color
        + ";'><span class='sentra-bar-inner'>"
        + bar_txt
        + "</span> "
        + display_text
        + "</span>"
    )


def fan_bar_html(label: str, rpm: float, assumed_max_map: dict) -> str:
    """
    Renders a bar for a fan.
    - Default assume max 4000 RPM.
    - If overridden in Fan Settings, show "1234 RPM / 10000 RPM".
    """
    max_rpm = float(assumed_max_map.get(label, 4000))
    if max_rpm <= 0:
        max_rpm = 1.0

    pct = min((rpm / max_rpm) * 100.0, 100.0)

    if int(max_rpm) == 4000:
        text = f"{rpm:.0f} RPM"
    else:
        text = f"{rpm:.0f} RPM / {max_rpm:.0f} RPM"

    return pct_bar_html(
        pct=pct,
        warn_threshold=200.0,  # practically never red
        display_text=text,
        width_chars=20,
    )


def build_top_status_block_html(snap, gpu_list) -> str:
    """
    Build the full responsive header row as pure HTML+CSS.
    Sections: CPU/HOST | LOAD + GPU TEMP | MEM | SWAP
    GPU TEMP now appears under the LOAD column.
    """

    header_color = "#8dd3ff"  # light blue header text

    # ---------- CPU + HOST ----------
    cpu_total = snap["cpu"]["total_util"]
    cpu_user = snap["cpu"]["breakdown"]["user"]
    cpu_sys = snap["cpu"]["breakdown"]["system"]
    cpu_idle = snap["cpu"]["breakdown"]["idle"]
    cpu_iowait = snap["cpu"]["breakdown"].get("iowait", 0.0)
    cpu_temp_val = snap["cpu"]["temp"]
    cpu_temp_str = f"{cpu_temp_val:.1f}°C" if cpu_temp_val is not None else "N/A"

    cpu_user_bar = pct_bar_html(
        cpu_user, warn_threshold=90.0, display_text=f"{cpu_user:.1f}%"
    )
    cpu_sys_bar = pct_bar_html(
        cpu_sys, warn_threshold=90.0, display_text=f"{cpu_sys:.1f}%"
    )
    cpu_idle_bar = pct_bar_html(
        cpu_idle, warn_threshold=90.0, display_text=f"{cpu_idle:.1f}%"
    )

    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    os_name = f"{platform.system()} {platform.release()}"
    uptime_str = format_seconds(snap["meta"]["uptime_sec"])

    cpu_col_html = (
        "<div class='sentra-col'>"
        f"<div class='sentra-head'>CPU {cpu_total:.1f}%</div>"
        f"<div>user:&nbsp;&nbsp;{cpu_user_bar}</div>"
        f"<div>system:{cpu_sys_bar}</div>"
        f"<div>idle:&nbsp;&nbsp;{cpu_idle_bar}</div>"
        f"<div>temp:&nbsp;{cpu_temp_str}</div>"
        f"<div>iowait:&nbsp;{cpu_iowait:.1f}%</div>"
        "<div class='sentra-space'></div>"
        "<div class='sentra-head'>HOST</div>"
        f"<div>cores:&nbsp;{logical_cores} ({physical_cores} phys)</div>"
        f"<div>RAM:&nbsp;&nbsp;{total_ram_gb:.2f} GB</div>"
        f"<div>OS:&nbsp;&nbsp;{os_name}</div>"
        f"<div>Uptime:{uptime_str}</div>"
        "</div>"
    )

    # ---------- LOAD ----------
    load1 = snap["cpu"]["load"]["1m"]
    load5 = snap["cpu"]["load"]["5m"]
    load15 = snap["cpu"]["load"]["15m"]

    load1_str = f"{load1:.2f}" if load1 is not None else "N/A"
    load5_str = f"{load5:.2f}" if load5 is not None else "N/A"
    load15_str = f"{load15:.2f}" if load15 is not None else "N/A"

    # ---------- GPU TEMP (moved under LOAD) ----------
    gpu_lines = []
    any_gpu_ok = False
    for g in gpu_list:
        if "error" in g:
            continue
        any_gpu_ok = True

        temp_c = g["temp"]
        pct_temp = (
            0.0
            if temp_c is None
            else max(0.0, min(float(temp_c), 100.0))
        )
        temp_bar = pct_bar_html(
            pct_temp,
            warn_threshold=80.0,  # turn red if >=80C
            display_text=f"{temp_c:.1f}°C",
        )
        gpu_lines.append(f"<div>GPU{g['index']}: {temp_bar}</div>")

    if not any_gpu_ok:
        gpu_lines.append("<div>no-gpu</div>")

    load_gpu_col_html = (
        "<div class='sentra-col'>"
        "<div class='sentra-head'>LOAD</div>"
        f"<div>1 min:&nbsp;{load1_str}</div>"
        f"<div>5 min:&nbsp;{load5_str}</div>"
        f"<div>15 min:{load15_str}</div>"
        "<div class='sentra-space'></div>"
        "<div class='sentra-head'>GPU TEMP</div>"
        + "".join(gpu_lines)
        + "</div>"
    )

    # ---------- MEM ----------
    # We want the "high" number like landscape-sysinfo usually prints:
    # usage ~ (total - MemFree) / total.
    mem_total_b = snap["mem"]["total_bytes"]
    mem_free_raw_b = snap["mem"].get("free_bytes_raw")  # strict MemFree
    mem_available_b = snap["mem"]["free_bytes"]         # reclaimable-friendly "available"
    mem_psutil_pct = snap["mem"]["used_percent"]        # psutil (total-available)/total*100
    mem_used_bytes_reported = snap["mem"]["used_bytes"] # psutil's .used

    mem_used_pct_display = mem_psutil_pct
    mem_used_b_display = mem_used_bytes_reported
    mem_free_b_display = mem_available_b

    if (
        mem_total_b is not None
        and mem_free_raw_b is not None
        and mem_total_b > 0
    ):
        real_used = mem_total_b - mem_free_raw_b
        mem_used_pct_display = (real_used / mem_total_b) * 100.0
        mem_used_b_display = real_used
        mem_free_b_display = mem_free_raw_b

    mem_bar = pct_bar_html(
        mem_used_pct_display,
        warn_threshold=90.0,
        display_text=f"{mem_used_pct_display:.1f}%",
    )

    mem_col_html = (
        "<div class='sentra-col'>"
        f"<div class='sentra-head'>MEM {mem_used_pct_display:.1f}%</div>"
        f"<div>{mem_bar}</div>"
        f"<div>total:&nbsp;{human_bytes(mem_total_b)}</div>"
        f"<div>used:&nbsp;&nbsp;{human_bytes(mem_used_b_display)}</div>"
        f"<div>free:&nbsp;&nbsp;{human_bytes(mem_free_b_display)}</div>"
        "</div>"
    )

    # ---------- SWAP ----------
    swap_used_pct = snap["mem"]["swap_used_percent"]
    swap_total_b = snap["mem"]["swap_total_bytes"]
    swap_used_b = snap["mem"]["swap_used_bytes"]
    swap_free_b = snap["mem"]["swap_free_bytes"]

    swap_bar = pct_bar_html(
        swap_used_pct,
        warn_threshold=50.0,
        display_text=f"{swap_used_pct:.1f}%",
    )

    swap_col_html = (
        "<div class='sentra-col'>"
        f"<div class='sentra-head'>SWAP {swap_used_pct:.1f}%</div>"
        f"<div>{swap_bar}</div>"
        f"<div>total:&nbsp;{human_bytes(swap_total_b)}</div>"
        f"<div>used:&nbsp;&nbsp;{human_bytes(swap_used_b)}</div>"
        f"<div>free:&nbsp;&nbsp;{human_bytes(swap_free_b)}</div>"
        "</div>"
    )

    # ---------- Final block with inline CSS ----------
    html = (
        "<style>"
        ".sentra-status{"
        "  background-color:#1a1a1a;"
        "  color:#eee;"
        "  padding:8px 10px;"
        "  border-radius:4px;"
        "  font-family:monospace;"
        "  font-size:12px;"
        "  line-height:1.4;"
        "  -webkit-user-select:none;"
        "  -moz-user-select:none;"
        "  -ms-user-select:none;"
        "  user-select:none;"
        "  width:100%;"
        "  max-width:100%;"
        "  box-sizing:border-box;"
        "}"
        ".sentra-row{"
        "  display:flex;"
        "  flex-wrap:wrap;"
        "  width:100%;"
        "  max-width:100%;"
        "  background-color:#000;"
        "  border:1px solid #333;"
        "  border-radius:4px;"
        "  box-sizing:border-box;"
        "}"
        ".sentra-col{"
        "  flex:1 1 240px;"
        "  min-width:240px;"
        "  box-sizing:border-box;"
        "  padding:8px 12px;"
        "  border-right:1px solid #333;"
        "  white-space:normal;"
        "  overflow:hidden;"
        "}"
        ".sentra-col:last-child{"
        "  border-right:none;"
        "}"
        ".sentra-head{"
        f" color:{header_color};"
        "  font-weight:bold;"
        "  margin-bottom:4px;"
        "}"
        ".sentra-barbox{"
        "  display:inline-block;"
        "  border:1px solid #444;"
        "  background-color:#000;"
        "  padding:0 4px;"
        "  font-family:monospace;"
        "  font-size:12px;"
        "  line-height:1.2;"
        "  white-space:pre;"
        "}"
        ".sentra-bar-inner{"
        "  white-space:pre;"
        "}"
        ".sentra-space{"
        "  margin-top:6px;"
        "  height:6px;"
        "}"
        "</style>"
        "<div class='sentra-status'>"
        "<div class='sentra-row'>"
        + cpu_col_html
        + load_gpu_col_html
        + mem_col_html
        + swap_col_html
        + "</div></div>"
    )

    return html


def build_containers_table_html(rows_sorted: list[dict]) -> str:
    """
    Build a compact, scrollable, nowrap, single-line-per-container flex table.

    Visible columns (in order):
    - name
    - ram
    - cpu%
    - life
    - command
    """

    style = """
    <style>
    .sentra-docker-box{
        background-color:#1a1a1a;
        border:1px solid #333;
        border-radius:4px;
        font-family:monospace;
        font-size:11px;
        color:#eee;
        width:100%;
        box-sizing:border-box;
        padding:0;
    }
    .sentra-docker-headrow{
        background-color:#000;
        color:#8dd3ff;
        font-weight:bold;
        border-bottom:1px solid #333;
    }
    .sentra-docker-inner{
        max-height:360px;
        overflow-y:auto;
        overflow-x:auto;
        white-space:nowrap;
    }
    .sentra-docker-row{
        display:flex;
        flex-direction:row;
        align-items:center;
        border-bottom:1px solid #333;
        padding:2px 6px;
        box-sizing:border-box;
    }
    .sentra-docker-row:last-child{
        border-bottom:none;
    }

    /* column sizing */
    .c-name{
        flex:0 0 180px;
        max-width:180px;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
        padding-right:8px;
    }
    .c-ram{
        flex:0 0 110px;
        max-width:110px;
        text-align:right;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
        padding-right:8px;
    }
    .c-cpu{
        flex:0 0 60px;
        max-width:60px;
        text-align:right;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
        padding-right:8px;
    }
    .c-life{
        flex:0 0 70px;
        max-width:70px;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
        padding-right:8px;
    }
    .c-cmd{
        flex:1 1 auto;
        min-width:200px;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
    }
    </style>
    """

    # header row
    header_html = (
        "<div class='sentra-docker-row sentra-docker-headrow'>"
        "<div class='c-name'>name</div>"
        "<div class='c-ram'>ram</div>"
        "<div class='c-cpu'>cpu%</div>"
        "<div class='c-life'>life</div>"
        "<div class='c-cmd'>command</div>"
        "</div>"
    )

    # body rows
    body_parts = []
    for r in rows_sorted:
        # escape all text to avoid HTML injection
        name_html = html_lib.escape(str(r.get("name", "")))
        ram_html = html_lib.escape(str(r.get("ram", "")))
        cpu_html = html_lib.escape(fmt_pct(r.get("cpu_pct_raw", None)))
        life_html = html_lib.escape(str(r.get("lifetime", "")))
        cmd_html = html_lib.escape(str(r.get("command", "")))

        row_html = (
            "<div class='sentra-docker-row'>"
            f"<div class='c-name'>{name_html}</div>"
            f"<div class='c-ram'>{ram_html}</div>"
            f"<div class='c-cpu'>{cpu_html}</div>"
            f"<div class='c-life'>{life_html}</div>"
            f"<div class='c-cmd'>{cmd_html}</div>"
            "</div>"
        )
        body_parts.append(row_html)

    body_html = "".join(body_parts)

    full_html = (
        style
        + "<div class='sentra-docker-box'>"
        + "<div class='sentra-docker-inner'>"
        + header_html
        + body_html
        + "</div></div>"
    )
    return full_html


# ---------- Tabs ----------
tab_overview, tab_fansettings, tab_legacy = st.tabs(
    ["Overview", "Fan Settings", "Legacy GPU Monitor"]
)


# =========================================
# OVERVIEW TAB
# =========================================
with tab_overview:
    # Top status row (render via HTML iframe so CSS works)
    status_html = build_top_status_block_html(snapshot, gpus)
    components.html(status_html, height=STATUS_BLOCK_HEIGHT, scrolling=False)

    # CPU Cores Utilization (live)
    st.subheader("CPU Cores Utilization (live)")
    per_core = snapshot["cpu"]["per_core"]
    core_df = pd.DataFrame({
        "core": list(range(len(per_core))),
        "util": per_core,
    })
    cpu_bar_chart = (
        alt.Chart(core_df)
        .mark_bar()
        .encode(
            x=alt.X("core:O", title="Core"),
            y=alt.Y(
                "util:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="Util (%)",
            ),
            tooltip=["core", "util"],
        )
        .properties(height=200)
    )
    st.altair_chart(cpu_bar_chart, use_container_width=True)

    # ----------------------------------------------------------------------
    # Docker Containers (live)
    # ----------------------------------------------------------------------
    with st.expander("Docker Containers (live)", expanded=True):
        if containers and isinstance(containers, list) and "error" in containers[0]:
            st.warning(f"Docker stats error: {containers[0]['error']}")
        elif not containers:
            st.write("No running containers or Docker not available.")
        else:
            # sorting controls row
            # Columns: Name | RAM | CPU% | Lifetime | Command
            sort_cols = st.columns([1, 1, 1, 1, 3])

            def _update_sort(key: str, default_desc: bool):
                if st.session_state.container_sort_key == key:
                    st.session_state.container_sort_reverse = (
                        not st.session_state.container_sort_reverse
                    )
                else:
                    st.session_state.container_sort_key = key
                    st.session_state.container_sort_reverse = default_desc

            if sort_cols[0].button("Name"):
                _update_sort("name", False)
            if sort_cols[1].button("RAM"):
                _update_sort("mem_bytes", True)
            if sort_cols[2].button("CPU%"):
                _update_sort("cpu_pct_sortable", True)
            if sort_cols[3].button("Lifetime"):
                _update_sort("lifetime_sortable", True)
            sort_cols[4].write("Command")

            # build rows for table/sorting
            table_rows = []
            for cinfo in containers:
                name = cinfo.get("name", "")
                cpu_pct_val = cinfo.get("cpu_pct", None)
                mem_pct_val = cinfo.get("mem_pct", None)
                mem_bytes_val = cinfo.get("mem_bytes", None)
                lifetime = cinfo.get("lifetime", "N/A")
                command = cinfo.get("command", "")

                # For stable sorting: None -> 0
                sort_mem_bytes = mem_bytes_val if isinstance(mem_bytes_val, (int, float)) else 0

                # Lifetime sortable: convert "16h 29m", "2m", "3d 4h" -> rough minutes
                lifetime_minutes = 0
                try:
                    parts = lifetime.split()
                    for p in parts:
                        if p.endswith("d"):
                            lifetime_minutes += int(p[:-1]) * 24 * 60
                        elif p.endswith("h"):
                            lifetime_minutes += int(p[:-1]) * 60
                        elif p.endswith("m"):
                            lifetime_minutes += int(p[:-1])
                except Exception:
                    pass

                # CPU sortable: None -> 0.0
                cpu_sortable = float(cpu_pct_val) if isinstance(cpu_pct_val, (int, float)) else 0.0

                table_rows.append({
                    "name": name,
                    "cpu_pct_raw": cpu_pct_val,
                    "ram": fmt_ram_col(mem_bytes_val, mem_pct_val),
                    "lifetime": lifetime,
                    "command": command,
                    # hidden sort helpers
                    "mem_bytes": sort_mem_bytes,
                    "cpu_pct_sortable": cpu_sortable,
                    "lifetime_sortable": lifetime_minutes,
                })

            df_cont = pd.DataFrame(table_rows)

            sort_key = st.session_state.container_sort_key
            sort_reverse = st.session_state.container_sort_reverse

            df_cont_sorted = df_cont.sort_values(
                by=sort_key,
                ascending=not sort_reverse,
                kind="mergesort",  # stable
            ).reset_index(drop=True)

            rows_sorted = df_cont_sorted.to_dict(orient="records")
            table_html = build_containers_table_html(rows_sorted)

            # render custom table
            components.html(table_html, height=420, scrolling=False)

    # Disk IO / Usage (live)
    st.subheader("Disk IO / Usage (live)")

    disk_rows = []
    for dev, vals in snapshot["disk"]["throughput"].items():
        disk_rows.append({
            "device": dev,
            "read_MB/s": None if vals["read_bps"] is None else (vals["read_bps"] / (1024.0 * 1024.0)),
            "write_MB/s": None if vals["write_bps"] is None else (vals["write_bps"] / (1024.0 * 1024.0)),
        })
    disk_df = pd.DataFrame(disk_rows)

    disk_usage_percent = snapshot["disk"]["root_usage_percent"]
    disk_cols = st.columns(2)
    disk_cols[0].metric(
        "Root FS Usage",
        f"{disk_usage_percent:.1f}%" if disk_usage_percent is not None else "N/A",
    )

    if not disk_df.empty:
        disk_df_display = disk_df.copy()

        def _fmt_io(v):
            if v is None:
                return ""
            return f"{v:.2f}"

        disk_df_display["read_MB/s"] = disk_df_display["read_MB/s"].map(_fmt_io)
        disk_df_display["write_MB/s"] = disk_df_display["write_MB/s"].map(_fmt_io)

        disk_cols[1].table(disk_df_display)
    else:
        disk_cols[1].write("No disk throughput data.")

    # Network Throughput (live)
    st.subheader("Network Throughput (live)")

    net_rows = []
    for iface, vals in snapshot["net"]["throughput"].items():
        net_rows.append({
            "iface": iface,
            "rx_Mb/s": None if vals["rx_bps"] is None else ((vals["rx_bps"] * 8.0) / 1e6),
            "tx_Mb/s": None if vals["tx_bps"] is None else ((vals["tx_bps"] * 8.0) / 1e6),
        })
    net_df = pd.DataFrame(net_rows)

    if not net_df.empty:
        net_df_display = net_df.copy()

        def _fmt_net(v):
            if v is None:
                return ""
            return f"{v:.2f}"

        net_df_display["rx_Mb/s"] = net_df_display["rx_Mb/s"].map(_fmt_net)
        net_df_display["tx_Mb/s"] = net_df_display["tx_Mb/s"].map(_fmt_net)

        st.table(net_df_display)
    else:
        st.write("No network throughput data.")

    # Hardware Fans (live)
    st.subheader("Hardware Fans (live)")
    fan_map = snapshot["fans"]
    if fan_map:
        for label, rpm in fan_map.items():
            fan_html = fan_bar_html(label, rpm, st.session_state.fan_max)
            components.html(
                "<div style='font-family:monospace;font-size:12px;color:#eee;"
                "-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;"
                "background-color:transparent;'>"
                f"{label}: {fan_html}</div>",
                height=40,
                scrolling=False,
            )
    else:
        st.write("No fan RPM data.")

    # NVIDIA GPU(s) (live)
    st.subheader("NVIDIA GPU(s) (live)")
    if gpus and not (len(gpus) == 1 and "error" in gpus[0]):
        for g in gpus:
            cols = st.columns(5)
            cols[0].metric("GPU", f"{g['index']} ({g['name']})")
            temp_val = g.get("temp")
            cols[1].metric(
                "Temp",
                f"{temp_val}°C" if isinstance(temp_val, (int, float)) else "N/A",
            )
            power_val = g.get("power_w")
            cols[2].metric(
                "Power",
                f"{power_val:.2f}W" if isinstance(power_val, (int, float)) else "N/A",
            )
            vram_used = g.get("vram_used_mb")
            vram_total = g.get("vram_total_mb")
            if isinstance(vram_used, (int, float)) and isinstance(vram_total, (int, float)):
                vram_text = f"{vram_used} / {vram_total} MB"
            else:
                vram_text = "N/A"
            cols[3].metric(
                "VRAM",
                vram_text
            )
            util_val = g.get("util")
            cols[4].metric(
                "Util",
                f"{util_val}%" if isinstance(util_val, (int, float)) else "N/A"
            )
            if g.get("fan_percent") is not None:
                st.write(f"Fan: {g['fan_percent']}%")
    else:
        if gpus and "error" in gpus[0]:
            st.warning(f"GPU stats error: {gpus[0]['error']}")
        else:
            st.write("No NVIDIA GPU detected / NVML not available.")

    # Historic Trends
    st.subheader("Historic Trends (last 60 min)")

    host_hist_df = datastore.get_cpu_mem_history(minutes=60)
    if not host_hist_df.empty:
        cpu_usage_chart = (
            alt.Chart(host_hist_df)
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y(
                    "total_util:Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="CPU Util (%)"
                ),
            )
            .properties(height=200, title="CPU Usage (%)")
        )
        st.altair_chart(cpu_usage_chart, use_container_width=True)

        if "cpu_temp" in host_hist_df.columns:
            cpu_temp_chart = (
                alt.Chart(host_hist_df)
                .mark_line()
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y(
                        "cpu_temp:Q",
                        scale=alt.Scale(domain=[0, 100]),
                        title="CPU Temp (°C)"
                    ),
                )
                .properties(height=200, title="CPU Temp (°C)")
            )
            st.altair_chart(cpu_temp_chart, use_container_width=True)

        mem_melt = host_hist_df.melt(
            id_vars=["timestamp"],
            value_vars=["used_percent", "swap_used_percent"],
            var_name="metric",
            value_name="percent",
        )
        mem_chart = (
            alt.Chart(mem_melt)
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y(
                    "percent:Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="Usage (%)"
                ),
                color=alt.Color("metric:N", legend=alt.Legend(title="Metric")),
            )
            .properties(height=200, title="Memory / Swap Usage (%)")
        )
        st.altair_chart(mem_chart, use_container_width=True)
    else:
        st.write("No CPU/memory history yet.")

    gpu_hist_df = datastore.get_gpu_history(minutes=60)
    if not gpu_hist_df.empty:
        gpu_temp_chart = (
            alt.Chart(gpu_hist_df)
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y(
                    "temp:Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="Temp (°C)"
                ),
                color=alt.Color("gpu_index:N", legend=alt.Legend(title="GPU")),
            )
            .properties(height=200, title="GPU Temperature (°C)")
        )
        st.altair_chart(gpu_temp_chart, use_container_width=True)

        gpu_util_chart = (
            alt.Chart(gpu_hist_df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y=alt.Y(
                    "util:Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="GPU Util (%)"
                ),
                color=alt.Color("gpu_index:N", legend=alt.Legend(title="GPU")),
            )
            .properties(height=200, title="GPU Utilization (%)")
        )
        st.altair_chart(gpu_util_chart, use_container_width=True)

        gpu_power_chart = (
            alt.Chart(gpu_hist_df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y=alt.Y(
                    "power_w:Q",
                    scale=alt.Scale(domain=[0, 300]),
                    title="GPU Power (W)"
                ),
                color=alt.Color("gpu_index:N", legend=alt.Legend(title="GPU")),
            )
            .properties(height=200, title="GPU Power (W)")
        )
        st.altair_chart(gpu_power_chart, use_container_width=True)

        gpu_vram_chart = (
            alt.Chart(gpu_hist_df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y=alt.Y(
                    "vram_used_mb:Q",
                    title="VRAM Used (MB)"
                ),
                color=alt.Color("gpu_index:N", legend=alt.Legend(title="GPU")),
            )
            .properties(height=200, title="GPU VRAM Used (MB)")
        )
        st.altair_chart(gpu_vram_chart, use_container_width=True)
    else:
        st.write("No GPU history yet.")


# =========================================
# FAN SETTINGS TAB
# =========================================
with tab_fansettings:
    st.header("Fan Settings")
    st.write(
        "By default, sentra assumes each fan's max speed is 4000 RPM. "
        "If you set a custom max here, Overview will show `current / max` "
        "and fan utilization bars will scale to that max."
    )

    for label, current_rpm in snapshot["fans"].items():
        cur_max = int(st.session_state.fan_max.get(label, 4000))
        widget_key = f"fanmax_{abs(hash(label))}"
        new_max = st.number_input(
            f"Max RPM for {label}",
            min_value=1000,
            max_value=100000,
            step=100,
            value=cur_max,
            key=widget_key,
        )
        st.session_state.fan_max[label] = int(new_max)

    if not snapshot["fans"]:
        st.write("No fans detected on this host.")

with tab_legacy:
    st.header("Legacy GPU Monitor")
    log_path = ensure_gpu_log_path()
    gpu_records: list[tuple[int, float | None, float | None, float | None, float | None]] = []
    placeholder = st.empty()

    with placeholder.container():
        st.subheader("🖥️ System Metrics")
        cpu_perc = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory()
        st.metric("CPU Usage", f"{cpu_perc}%")
        st.metric("RAM Usage", f"{ram.percent}%")

        cpu_temp = get_cpu_temp()
        if cpu_temp is not None:
            st.metric("CPU Temp", f"{cpu_temp}°C")
        else:
            st.write("CPU Temp: N/A (check `sensors` output)")

        st.subheader("NVIDIA GPU(s)")
        if gpus and not (len(gpus) == 1 and "error" in gpus[0]):
            for g in gpus:
                idx = g["index"]
                st.markdown(f"### GPU {idx}: {g['name']}")
                cols = st.columns(4)
                cols[0].metric("Temp", f"{g['temp']}°C" if isinstance(g.get("temp"), (int, float)) else "N/A")
                cols[1].metric(
                    "Power",
                    f"{g['power_w']:.2f}W" if isinstance(g.get("power_w"), (int, float)) else "N/A",
                )
                vram_used = g.get("vram_used_mb")
                vram_total = g.get("vram_total_mb")
                if isinstance(vram_used, (int, float)) and isinstance(vram_total, (int, float)):
                    cols[2].metric("VRAM Used", f"{vram_used} / {vram_total} MB")
                else:
                    cols[2].metric("VRAM Used", "N/A")
                util_val = g.get("util")
                cols[3].metric(
                    "Utilization",
                    f"{util_val}%" if isinstance(util_val, (int, float)) else "N/A",
                )
                st.markdown(f"### GPU {idx}: {g['name']}")
                gpu_records.append(
                    (
                        idx,
                        g.get("temp"),
                        g.get("util"),
                        g.get("power_w"),
                        g.get("vram_used_mb"),
                    )
                )
        else:
            if gpus and "error" in gpus[0]:
                st.warning(f"Error loading NVIDIA stats: {gpus[0]['error']}")
            else:
                st.write("No NVIDIA GPU detected / NVML not available.")

    if gpu_records:
        append_gpu_log(log_path, gpu_records)

    df = read_gpu_history(log_path)
    if not df.empty:
        st.subheader("GPU Temperatures (°C)")
        temp_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("temp:Q", scale=alt.Scale(domain=[0, 100]), title="Temp (°C)"),
                color=alt.Color("gpu_index:N", legend=alt.Legend(title="GPU")),
            )
            .properties(height=200)
        )
        st.altair_chart(temp_chart, use_container_width=True)

        st.subheader("GPU Utilization (%)")
        util_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y=alt.Y("util:Q", scale=alt.Scale(domain=[0, 100]), title="Util (%)"),
                color=alt.Color("gpu_index:N", legend=None),
            )
            .properties(height=200)
        )
        st.altair_chart(util_chart, use_container_width=True)

        st.subheader("GPU Power (W)")
        power_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y=alt.Y("power:Q", scale=alt.Scale(domain=[0, 600]), title="Power (W)"),
                color=alt.Color("gpu_index:N", legend=None),
            )
            .properties(height=200)
        )
        st.altair_chart(power_chart, use_container_width=True)

        st.subheader("GPU VRAM Used (MB)")
        vram_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y=alt.Y("vram:Q", scale=alt.Scale(domain=[0, 12000]), title="VRAM Used (MB)"),
                color=alt.Color("gpu_index:N", legend=None),
            )
            .properties(height=200)
        )
        st.altair_chart(vram_chart, use_container_width=True)
    else:
        st.write("No GPU history to plot yet.")
