# collector/system_collector.py
"""
system_collector.py
-------------------

Collects CPU, memory, disk, network, fan and basic host stats using psutil
and sensors_collector. This module does NOT touch GPUs (that's gpu_collector).

The main entrypoint is `collect_system_snapshot(prev_disk, prev_net, interval_s)`:
- prev_disk, prev_net: previous psutil counters so we can compute deltas/sec
- interval_s: seconds between samples (float)

It returns:
(snapshot_dict, new_disk_counters, new_net_counters)
"""

import os
import time
import psutil
from typing import Any, Dict, Optional, Tuple

from . import sensors_collector


def _safe_disk_usage(path: str) -> Optional[psutil._common.sdiskusage]:
    try:
        return psutil.disk_usage(path)
    except Exception:
        # On Windows dev or weird containers, "/" may not exist
        try:
            return psutil.disk_usage(os.getcwd())
        except Exception:
            return None


def _get_swap_memory():
    """
    psutil.swap_memory() can raise on super restricted /proc environments.
    We'll gracefully fall back to zeros.
    """
    try:
        return psutil.swap_memory()
    except Exception:
        class _FakeSwap:
            percent = 0.0
            total = 0
            used = 0
            free = 0
        return _FakeSwap()


def collect_system_snapshot(
    prev_disk: Optional[Dict[str, psutil._common.sdiskio]],
    prev_net: Optional[Dict[str, psutil._common.snetio]],
    interval_s: float,
) -> Tuple[Dict[str, Any], Dict[str, psutil._common.sdiskio], Dict[str, psutil._common.snetio]]:
    """
    Collect one snapshot of host metrics.

    Returns:
        snapshot: dict
        disk_now: current psutil.disk_io_counters(perdisk=True)
        net_now: current psutil.net_io_counters(pernic=True)
    """

    now = time.time()

    # CPU total %, per-core % (non-blocking after first call to psutil)
    cpu_total = psutil.cpu_percent(interval=0.0)
    per_core = psutil.cpu_percent(interval=0.0, percpu=True)

    # CPU times breakdown
    cpu_times = psutil.cpu_times_percent(interval=0.0)
    cpu_user_pct = getattr(cpu_times, "user", 0.0)
    cpu_sys_pct = getattr(cpu_times, "system", 0.0)
    cpu_idle_pct = getattr(cpu_times, "idle", 0.0)
    cpu_iowait_pct = getattr(cpu_times, "iowait", 0.0)

    # load average (may not exist on Windows -> catch OSError)
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = None

    # temps / fans
    cpu_temp, fans_rpm = sensors_collector.get_temperatures_and_fans()

    # memory
    vm = psutil.virtual_memory()
    swap = _get_swap_memory()

    # uptime
    boot_ts = psutil.boot_time()
    uptime_sec = now - boot_ts

    # disk usage
    disk_root = _safe_disk_usage("/")

    # disk IO throughput
    try:
        disk_now = psutil.disk_io_counters(perdisk=True)
    except Exception:
        disk_now = {}

    disk_stats: Dict[str, Dict[str, Optional[float]]] = {}
    if prev_disk and interval_s > 0:
        for dev, nowc in disk_now.items():
            if dev in prev_disk:
                then = prev_disk[dev]
                read_bps = (nowc.read_bytes - then.read_bytes) / interval_s
                write_bps = (nowc.write_bytes - then.write_bytes) / interval_s
                disk_stats[dev] = {
                    "read_bps": read_bps,
                    "write_bps": write_bps,
                }
    else:
        # first run -> can't compute deltas yet
        for dev in getattr(disk_now, "keys", lambda: [])():
            disk_stats[dev] = {
                "read_bps": None,
                "write_bps": None,
            }

    # network throughput
    try:
        net_now = psutil.net_io_counters(pernic=True)
    except Exception:
        net_now = {}

    net_stats: Dict[str, Dict[str, Optional[float]]] = {}
    # FIXED: `&&` -> `and`
    if prev_net and interval_s > 0:
        for nic, nowc in net_now.items():
            if nic in prev_net:
                then = prev_net[nic]
                rx_bps = (nowc.bytes_recv - then.bytes_recv) / interval_s
                tx_bps = (nowc.bytes_sent - then.bytes_sent) / interval_s
                net_stats[nic] = {
                    "rx_bps": rx_bps,
                    "tx_bps": tx_bps,
                }
    else:
        for nic in getattr(net_now, "keys", lambda: [])():
            net_stats[nic] = {
                "rx_bps": None,
                "tx_bps": None,
            }

    # Build snapshot dict
    snapshot: Dict[str, Any] = {
        "ts": now,
        "cpu": {
            "total_util": cpu_total,
            "per_core": per_core,
            "iowait": cpu_iowait_pct,  # keep top-level for DB insert
            "temp": cpu_temp,
            "load": {"1m": load1, "5m": load5, "15m": load15},
            "breakdown": {
                "user": cpu_user_pct,
                "system": cpu_sys_pct,
                "idle": cpu_idle_pct,
                "iowait": cpu_iowait_pct,
            },
        },
        "mem": {
            # vm.percent is psutil's "(total-available)/total*100" view
            "used_percent": vm.percent,
            # vm.used and vm.free are "total - free" style accounting
            # We'll also ship raw free so the UI can calculate its own % that
            # matches what `landscape-sysinfo` prints (often ~"Memory usage: 91%")
            "used_bytes": vm.used,
            "total_bytes": vm.total,
            "free_bytes": getattr(vm, "available", None),   # reclaimable-friendly "free"
            "free_bytes_raw": getattr(vm, "free", None),    # strict MemFree
            "swap_used_percent": getattr(swap, "percent", 0.0),
            "swap_total_bytes": getattr(swap, "total", 0),
            "swap_used_bytes": getattr(swap, "used", 0),
            "swap_free_bytes": getattr(swap, "free", 0),
        },
        "disk": {
            "root_usage_percent": disk_root.percent if disk_root else None,
            "throughput": disk_stats,   # { "sda": {"read_bps":..,"write_bps":..}, ... }
        },
        "net": {
            "throughput": net_stats,    # { "eth0":{"rx_bps":..,"tx_bps":..}, ... }
        },
        "fans": fans_rpm,               # { "chip:label": rpm, ... }
        "meta": {
            "uptime_sec": uptime_sec,
        },
    }

    return snapshot, disk_now, net_now
