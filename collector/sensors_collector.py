# collector/sensors_collector.py
"""
sensors_collector.py
--------------------

Helpers for reading temperatures and fan RPMs from the host.

Strategy:
1. Try "sensors -j" (lm-sensors) for CPU temp + fan RPM.
2. Fallback: /sys (or /host_sys in Docker) hwmon fan*_input files for RPM.
3. Fallback for CPU temp: /sys/class/thermal/.../temp

All functions are designed to fail gracefully on systems that don't expose
these metrics (Windows dev boxes, minimal containers, etc.).
"""

import os
import json
import glob
import subprocess
from typing import Dict, Optional, Tuple


def _get_sys_prefix() -> str:
    """
    In Docker, we expect the host's /sys to be mounted read-only at /host_sys.
    Outside Docker, that path won't exist, so just use /sys.
    """
    host_sys = os.getenv("SENTRA_HOST_SYS", "/host_sys")
    if os.path.isdir(host_sys):
        return host_sys
    return "/sys"


def _read_file_float(path: str) -> Optional[float]:
    try:
        with open(path, "r") as f:
            raw = f.read().strip()
        if raw == "" or raw.lower() == "na":
            return None
        return float(raw)
    except Exception:
        return None


def read_sensors_json() -> dict:
    """
    Try to call `sensors -j` which returns JSON.
    If lm-sensors isn't installed or accessible, returns {}.
    """
    try:
        out = subprocess.check_output(
            ["sensors", "-j"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return json.loads(out)
    except Exception:
        return {}


def extract_cpu_temp_from_sensors(sensors_json: dict) -> Optional[float]:
    """
    Attempt to identify a CPU temperature reading from sensors -j output.
    We heuristically search for typical CPU temp labels like 'Package id 0',
    'Tctl', 'Tdie', etc.
    We'll return the first plausible value we find.
    """
    # Common substrings that often indicate CPU package temp.
    HINTS = ["package id", "tctl", "tdie", "cpu", "k10temp", "coretemp"]

    candidates = []

    for chip_name, chip_data in sensors_json.items():
        if not isinstance(chip_data, dict):
            continue
        for label, metrics in chip_data.items():
            if not isinstance(metrics, dict):
                continue
            # label might be something like "Package id 0", "Tctl", etc.
            label_lower = str(label).lower()
            for key, val in metrics.items():
                if not isinstance(val, (int, float)):
                    continue
                key_lower = str(key).lower()
                # Typical convention in sensors -j:
                #   "temp1_input": 54.0
                if "temp" in key_lower and "input" in key_lower:
                    if any(h in label_lower for h in HINTS):
                        candidates.append(float(val))

    if candidates:
        return float(candidates[0])

    return None


def extract_fans_from_sensors(sensors_json: dict) -> Dict[str, float]:
    """
    Parse fan RPM readings out of sensors -j output.
    We'll build a dict label->rpm.
    """
    fans: Dict[str, float] = {}

    for chip_name, chip_data in sensors_json.items():
        if not isinstance(chip_data, dict):
            continue
        for label, metrics in chip_data.items():
            if not isinstance(metrics, dict):
                continue
            for key, val in metrics.items():
                # Example: "fan1_input": 1234.0
                if "fan" in key.lower() and "input" in key.lower():
                    if isinstance(val, (int, float)):
                        fan_label = f"{chip_name}:{label}"
                        fans[fan_label] = float(val)

    return fans


def read_hwmon_fans(sys_prefix: str) -> Dict[str, float]:
    """
    Fallback for fan RPM readings via /sys/class/hwmon/hwmon*/fan*_input
    We'll label fans as "<chipname>:fanX".
    """
    fans: Dict[str, float] = {}

    hwmons = glob.glob(os.path.join(sys_prefix, "class", "hwmon", "hwmon*"))
    for hm in hwmons:
        # Try to read the chip name (/sys/class/hwmon/hwmonX/name)
        name_path = os.path.join(hm, "name")
        try:
            with open(name_path, "r") as f:
                chip_name = f.read().strip()
        except Exception:
            chip_name = os.path.basename(hm)

        for fan_file in glob.glob(os.path.join(hm, "fan*_input")):
            rpm = _read_file_float(fan_file)
            if rpm is None:
                continue
            base = os.path.basename(fan_file)  # e.g. fan1_input
            fan_label = f"{chip_name}:{base.replace('_input','')}"
            fans[fan_label] = rpm

    return fans


def read_thermal_cpu_temp(sys_prefix: str) -> Optional[float]:
    """
    Fallback for CPU temperature via thermal zones.

    We'll scan /sys/class/thermal/thermal_zone*/type for a name that looks
    like CPU (contains 'cpu', 'x86_pkg_temp', etc.) and read the matching temp.
    Values are typically millidegrees C.
    """
    zones = glob.glob(os.path.join(sys_prefix, "class", "thermal", "thermal_zone*"))
    for zone in zones:
        type_path = os.path.join(zone, "type")
        temp_path = os.path.join(zone, "temp")
        try:
            with open(type_path, "r") as f:
                zone_type = f.read().strip().lower()
        except Exception:
            zone_type = ""

        if any(hint in zone_type for hint in ["cpu", "x86_pkg_temp", "package", "soc"]):
            millic = _read_file_float(temp_path)
            if millic is not None:
                # convert millideg C -> deg C
                return millic / 1000.0

    # As a last resort, pick first thermal_zone temp at all:
    for zone in zones:
        temp_path = os.path.join(zone, "temp")
        millic = _read_file_float(temp_path)
        if millic is not None:
            return millic / 1000.0

    return None


def get_temperatures_and_fans() -> Tuple[Optional[float], Dict[str, float]]:
    """
    High-level helper used by system_collector:
    Returns:
        cpu_temp (float or None),
        fan_map {label: rpm}
    """
    sys_prefix = _get_sys_prefix()
    sensors_json = read_sensors_json()

    # temps
    cpu_temp = extract_cpu_temp_from_sensors(sensors_json)
    if cpu_temp is None:
        cpu_temp = read_thermal_cpu_temp(sys_prefix)

    # fans
    fans = extract_fans_from_sensors(sensors_json)
    hwmon_fans = read_hwmon_fans(sys_prefix)
    # merge hwmon fallback fans if they aren't already present
    for k, v in hwmon_fans.items():
        fans.setdefault(k, v)

    return cpu_temp, fans
