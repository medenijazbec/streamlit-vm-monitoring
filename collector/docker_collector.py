# collector/docker_collector.py
"""
docker_collector.py
-------------------

Collect live Docker container info by talking directly to the Docker Engine
API over /var/run/docker.sock (via docker-py).

Why this exists:
- We want per-container CPU% / RAM without freezing the dashboard.
- Calling .stats() serially for 100+ containers can block for many seconds.
  Streamlit won't render until that's done -> you saw a blank page.
- Now we:
    * always collect cheap metadata immediately
    * IF SENTRA_DOCKER_STATS=1, we TRY to collect stats in parallel
      using a ThreadPoolExecutor
    * we enforce a global timeout (default 2s), so we never stall forever
    * containers whose stats didn't finish in time just show N/A

Env vars:
    SENTRA_DOCKER_STATS              "1" -> attempt stats, else "0"
    SENTRA_DOCKER_STATS_THREADS      max worker threads (default "8")
    SENTRA_DOCKER_STATS_TIMEOUT_SEC  global timeout in seconds (default "2.0")

Returned per-container dict fields:
- id        (12-char ID)
- name      (container name)
- cpu_pct   (float % CPU or None if unavailable/timeout)
- mem_pct   (float % mem or None if unavailable/timeout)
- mem_bytes (int bytes used or None if unavailable/timeout)
- lifetime  ("1h 20m", "3d 4h", etc.)
- command   (string from entrypoint/cmd; may include [stats_error: ...])
- error     (only if something really broke at container level)

If we cannot talk to Docker at all, we return:
    [{ "error": "…message…" }]
Dashboard shows that instead of silently going blank.
"""

from __future__ import annotations

import os
import time
import datetime
import concurrent.futures
from typing import List, Dict, Any, Tuple

# Try docker SDK
try:
    import docker
    from docker import errors as docker_errors  # noqa: F401
except Exception:
    docker = None
    docker_errors = None  # type: ignore


def _parse_started_at(ts: str) -> datetime.datetime | None:
    """
    Docker gives StartedAt like "2025-11-02T14:09:45.123456789Z".
    Trim 'Z', clamp nanos -> micros.
    """
    if not ts:
        return None

    ts = ts.strip().rstrip("Z").rstrip("z")

    if "." in ts:
        main, frac = ts.split(".", 1)
        frac = frac[:6]
        ts_usable = f"{main}.{frac}"
    else:
        ts_usable = ts

    try:
        return datetime.datetime.fromisoformat(ts_usable)
    except ValueError:
        try:
            base = ts.split(".")[0]
            return datetime.datetime.strptime(base, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None


def _format_lifetime(delta: datetime.timedelta) -> str:
    """
    Convert timedelta -> "Xd Yh", "Yh Zm", "Xm".
    """
    total_sec = int(delta.total_seconds())
    days = total_sec // 86400
    hours = (total_sec % 86400) // 3600
    minutes = (total_sec % 3600) // 60

    if days > 0:
        if hours > 0:
            return f"{days}d {hours}h"
        return f"{days}d"
    if hours > 0:
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h"
    return f"{minutes}m"


def _calc_cpu_pct(stats: Dict[str, Any]) -> float:
    """
    Docker stats CPU% formula:
    CPU% = (cpu_delta / system_delta) * online_cpus * 100
    """
    try:
        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})

        cpu_usage = cpu_stats.get("cpu_usage", {})
        precpu_usage = precpu_stats.get("cpu_usage", {})

        cpu_delta = (
            cpu_usage.get("total_usage", 0)
            - precpu_usage.get("total_usage", 0)
        )
        system_delta = (
            cpu_stats.get("system_cpu_usage", 0)
            - precpu_stats.get("system_cpu_usage", 0)
        )

        online_cpus = cpu_stats.get("online_cpus")
        if not online_cpus:
            per_cpu = cpu_usage.get("percpu_usage") or []
            online_cpus = len(per_cpu) if per_cpu else 1

        if cpu_delta > 0 and system_delta > 0 and online_cpus > 0:
            return (cpu_delta / system_delta) * online_cpus * 100.0
    except Exception:
        pass

    return 0.0


def _calc_mem(stats: Dict[str, Any]) -> Tuple[int | None, float | None]:
    """
    Return (mem_bytes, mem_pct).
    Guard div0.
    """
    try:
        mem_stats = stats.get("memory_stats", {})
        used = int(mem_stats.get("usage", 0))
        limit_val = mem_stats.get("limit", 0)
        limit = float(limit_val) if limit_val else 1.0
        pct = (used / limit) * 100.0
        return used, pct
    except Exception:
        return None, None


def _build_command_str(attrs: Dict[str, Any]) -> str:
    """
    Roughly like `docker ps --no-trunc --format {{.Command}}`.
    Join Entrypoint + Cmd. Fallback to Path + Args.
    """
    try:
        cfg = attrs.get("Config", {})
        entrypoint = cfg.get("Entrypoint") or []
        cmd = cfg.get("Cmd") or []
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        if isinstance(cmd, str):
            cmd = [cmd]
        pieces = list(entrypoint) + list(cmd)
        if not pieces:
            path = attrs.get("Path", "")
            args = attrs.get("Args", [])
            if isinstance(args, str):
                args = [args]
            pieces = [path] + list(args)
        return " ".join(pieces).strip()
    except Exception:
        return ""


def _grab_stats(api_client, cid: str) -> Dict[str, Any]:
    """
    Worker fn run in thread pool.
    Returns either docker stats dict or {"__error__": "..."}.
    """
    try:
        # stream=False -> single snapshot
        return api_client.stats(cid, stream=False)
    except Exception as e:
        return {"__error__": str(e)}


def _collect_stats_parallel(
    api_client,
    container_ids: List[str],
    max_workers: int,
    global_timeout: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Fire off stats() calls in parallel threads and wait up to `global_timeout`
    seconds total. Whatever finishes in time is returned in stats_map.
    Others just won't appear in the map.

    We *don't* block forever. This is the key fix for the "blank page" hang.
    """
    stats_map: Dict[str, Dict[str, Any]] = {}

    if not container_ids:
        return stats_map

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures_map = {
            pool.submit(_grab_stats, api_client, cid): cid
            for cid in container_ids
        }

        start = time.time()
        try:
            # as_completed with timeout stops waiting after global_timeout,
            # leaving slow containers as missing stats.
            for fut in concurrent.futures.as_completed(
                futures_map.keys(),
                timeout=global_timeout,
            ):
                cid = futures_map[fut]
                try:
                    res = fut.result(timeout=0)
                except Exception as e:
                    res = {"__error__": str(e)}

                stats_map[cid] = res

                # tiny safety valve: if we somehow loop too long, bail
                if time.time() - start > global_timeout:
                    break
        except concurrent.futures.TimeoutError:
            # hit the global timeout -> just return whatever we have
            pass

        # actively cancel anything still running so we don't leak threads
        pool.shutdown(cancel_futures=True)

    return stats_map


def collect_docker_containers() -> List[Dict[str, Any]]:
    """
    Returns a list of:
      {
        "id", "name",
        "cpu_pct", "mem_pct", "mem_bytes",
        "lifetime", "command",
        maybe "error"
      }

    Behavior:
    - If docker SDK isn't importable at all, return [] (dashboard: "No running containers...").
    - If we fail to talk to daemon (socket missing / perms), return [{"error": "..."}].
    - Per-container stats are OPTIONAL (SENTRA_DOCKER_STATS=1). When enabled,
      we fetch them in parallel with a strict timeout so Streamlit doesn't hang.
    """
    if docker is None:
        # Docker SDK not installed -> act like "no containers"
        return []

    # read env knobs
    do_stats = os.getenv("SENTRA_DOCKER_STATS", "0") == "1"
    try:
        max_workers = int(os.getenv("SENTRA_DOCKER_STATS_THREADS", "8"))
    except Exception:
        max_workers = 8
    try:
        global_timeout = float(os.getenv("SENTRA_DOCKER_STATS_TIMEOUT_SEC", "2.0"))
    except Exception:
        global_timeout = 2.0

    # connect to daemon
    try:
        client = docker.from_env()
    except Exception as e:
        return [{"error": f"docker.from_env() failed: {e}"}]

    # list running containers
    try:
        running = client.containers.list()  # only running
    except Exception as e:
        try:
            client.close()
        except Exception:
            pass
        return [{
            "error": (
                "cannot list containers via docker.sock: " + str(e)
            )
        }]

    # pre-collect static/container metadata (cheap)
    now_utc = datetime.datetime.utcnow()
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for c in running:
        try:
            cid_full = getattr(c, "id", "") or ""
            cid_short = cid_full[:12] if cid_full else ""

            name = getattr(c, "name", "") or cid_short

            attrs = getattr(c, "attrs", {}) or {}
            started_raw = attrs.get("State", {}).get("StartedAt")
            lifetime_str = "N/A"
            if started_raw:
                started_dt = _parse_started_at(started_raw)
                if started_dt is not None:
                    delta = now_utc - started_dt
                    lifetime_str = _format_lifetime(delta)

            command_str = _build_command_str(attrs)

            meta_by_id[cid_full] = {
                "short_id": cid_short,
                "name": name,
                "lifetime": lifetime_str,
                "command": command_str,
            }

        except Exception as per_container_err:
            # we still want *something* for this container
            meta_by_id[getattr(c, "id", "unknown")] = {
                "short_id": getattr(c, "id", "unknown")[:12],
                "name": getattr(c, "name", "unknown"),
                "lifetime": "N/A",
                "command": f"[meta_error: {per_container_err}]",
            }

    # optionally collect live stats in parallel
    stats_map: Dict[str, Dict[str, Any]] = {}
    if do_stats and meta_by_id:
        ids_all = list(meta_by_id.keys())  # full-length container IDs
        stats_map = _collect_stats_parallel(
            client.api,
            ids_all,
            max_workers=max_workers,
            global_timeout=global_timeout,
        )

    # combine meta + stats into final output
    containers_out: List[Dict[str, Any]] = []
    for full_id, meta in meta_by_id.items():
        base_cmd = meta["command"]

        stat_blob = stats_map.get(full_id)
        if stat_blob is None:
            # either stats disabled or not finished in time
            cpu_pct = None
            mem_bytes = None
            mem_pct = None
        else:
            if "__error__" in stat_blob:
                # we'll surface the error inline in the command column
                base_cmd = (base_cmd + f" [stats_error:{stat_blob['__error__']}]").strip()
                cpu_pct = None
                mem_bytes = None
                mem_pct = None
            else:
                cpu_pct = _calc_cpu_pct(stat_blob)
                mem_bytes, mem_pct = _calc_mem(stat_blob)

        containers_out.append({
            "id": meta["short_id"],
            "name": meta["name"],
            "cpu_pct": cpu_pct,
            "mem_pct": mem_pct,
            "mem_bytes": mem_bytes,
            "lifetime": meta["lifetime"],
            "command": base_cmd,
        })

    try:
        client.close()
    except Exception:
        pass

    return containers_out
