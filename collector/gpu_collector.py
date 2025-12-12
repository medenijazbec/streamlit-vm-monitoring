# collector/gpu_collector.py
"""
gpu_collector.py
----------------

NVIDIA GPU collection using pynvml.
If no NVIDIA GPUs or NVML access fails, we return a list containing a dict
with an "error" key so the UI can handle gracefully.

Returned per-GPU dict fields:
- index
- name
- temp (°C)
- util (%)
- power_w (float)
- vram_used_mb (int)
- vram_total_mb (int)
- fan_percent (% or None)
"""

from typing import List, Dict, Any, TypeVar, Callable
import glob
import os
import ctypes
from ctypes.util import find_library
import pynvml

T = TypeVar("T")

SOFT_NVML_ERRORS = {
    getattr(pynvml, "NVML_ERROR_NOT_SUPPORTED", None),
    getattr(pynvml, "NVML_ERROR_NO_PERMISSION", None),
}
_UNKNOWN_ERR = getattr(pynvml, "NVML_ERROR_UNKNOWN", None)
if _UNKNOWN_ERR is not None:
    SOFT_NVML_ERRORS.add(_UNKNOWN_ERR)
SOFT_NVML_ERRORS.discard(None)

NVML_ENV_HINTS = (
    "SENTRA_NVML_LIBRARY",
    "SENTRA_NVML_LIBRARY_PATH",
    "NVML_LIBRARY_PATH",
)


def _gather_nvml_candidates() -> list[str]:
    """
    Build an ordered list of potential NVML library locations.
    We look at env hints first, then common distro paths.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        if not path:
            return
        normalized = os.path.abspath(path) if os.path.isabs(path) else path
        if normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)

    # Allow users to point at the exact library or directory via env.
    for env_var in NVML_ENV_HINTS:
        raw = os.environ.get(env_var, "")
        if not raw:
            continue
        for segment in raw.split(os.pathsep):
            segment = segment.strip()
            if not segment:
                continue
            if os.path.isdir(segment):
                pattern = "nvml.dll" if os.name == "nt" else "libnvidia-ml.so*"
                for match in sorted(glob.glob(os.path.join(segment, pattern)), reverse=True):
                    if os.path.isfile(match):
                        _add(match)
            else:
                _add(segment)

    lib_hint = find_library("nvidia-ml")
    if lib_hint and os.path.isabs(lib_hint):
        _add(lib_hint)

    # Common locations for NVIDIA drivers (covers bare metal, WSL, CUDA containers, etc.)
    search_roots = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/wsl/lib",
        "/usr/lib",
        "/usr/lib64",
        "/lib/x86_64-linux-gnu",
        "/lib64",
        "/usr/local/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/nvidia/lib64",
        "/run/host/usr/lib/x86_64-linux-gnu",
        "/run/host/usr/lib64",
    ]
    pattern = "nvml.dll" if os.name == "nt" else "libnvidia-ml.so*"
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for match in sorted(glob.glob(os.path.join(root, pattern)), reverse=True):
            if os.path.isfile(match):
                _add(match)

    return candidates


def _try_load_custom_nvml() -> bool:
    """
    Attempt to manually load libnvidia-ml.so (or nvml.dll) from known paths.
    On success we assign pynvml.nvmlLib so subsequent nvmlInit() works.
    """
    for candidate in _gather_nvml_candidates():
        try:
            pynvml.nvmlLib = ctypes.CDLL(candidate)
            return True
        except OSError:
            continue
    return False


def _safe_nvml_call(func: Callable[..., T], *args, default: T | None = None) -> T | None:
    """
    Execute NVML calls and downgrade common unsupported errors to None so that
    partially supported GPUs (P400, some datacenter cards) still show stats.
    """
    try:
        return func(*args)
    except pynvml.NVMLError as err:
        if getattr(err, "value", None) in SOFT_NVML_ERRORS:
            return default
        raise


def collect_gpu_snapshot() -> List[Dict[str, Any]]:
    gpus = []
    initialized = False
    try:
        try:
            pynvml.nvmlInit()
            initialized = True
        except pynvml.NVMLError_LibraryNotFound:
            if not _try_load_custom_nvml():
                raise RuntimeError(
                    "NVML shared library not found. Install NVIDIA drivers in this "
                    "environment, start the container with GPU pass-through (--gpus all), "
                    "or set SENTRA_NVML_LIBRARY_PATH to the libnvidia-ml.so location."
                )
            pynvml.nvmlInit()
            initialized = True
        gpu_count = _safe_nvml_call(pynvml.nvmlDeviceGetCount, default=0)
        if gpu_count is None:
            gpu_count = 0
        gpu_count = int(gpu_count)
        for i in range(gpu_count):
            h = _safe_nvml_call(
                pynvml.nvmlDeviceGetHandleByIndex,
                i,
                default=None,
            )
            if h is None:
                gpus.append({
                    "index": i,
                    "name": f"GPU {i}",
                    "temp": None,
                    "util": None,
                    "power_w": None,
                    "vram_used_mb": None,
                    "vram_total_mb": None,
                    "fan_percent": None,
                })
                continue

            raw_name = _safe_nvml_call(
                pynvml.nvmlDeviceGetName,
                h,
                default=b"Unknown",
            )
            if isinstance(raw_name, bytes):
                name = raw_name.decode(errors="ignore").strip()
            else:
                name = str(raw_name).strip() if raw_name else ""
            if not name:
                name = f"GPU {i}"
            temp = _safe_nvml_call(
                pynvml.nvmlDeviceGetTemperature,
                h,
                pynvml.NVML_TEMPERATURE_GPU,
            )
            mem = _safe_nvml_call(pynvml.nvmlDeviceGetMemoryInfo, h)
            util = _safe_nvml_call(pynvml.nvmlDeviceGetUtilizationRates, h)
            raw_power = _safe_nvml_call(pynvml.nvmlDeviceGetPowerUsage, h)
            power_w = None if raw_power is None else raw_power / 1000.0  # mW -> W
            fan_percent = _safe_nvml_call(pynvml.nvmlDeviceGetFanSpeed, h)

            gpus.append({
                "index": i,
                "name": name.decode() if isinstance(name, bytes) else name,
                "temp": temp,
                "util": None if util is None else util.gpu,
                "power_w": power_w,
                "vram_used_mb": None if mem is None else int(mem.used / 1e6),
                "vram_total_mb": None if mem is None else int(mem.total / 1e6),
                "fan_percent": fan_percent,
            })
    except Exception as e:
        gpus = [{"error": str(e)}]
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    return gpus
