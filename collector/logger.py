# collector/logger.py
"""
logger.py
---------

High-level "one shot" sampler that:
1. Collects system snapshot (CPU/RAM/Disk/Net/Fans/etc.)
2. Collects GPU snapshot(s)
3. Collects Docker container stats
4. Inserts host+gpu snapshot data into the SQLite datastore

This is called from the Streamlit dashboard every refresh cycle.
We pass in the previous disk/net counters (for throughput deltas),
and get the updated counters back.
"""

from typing import Tuple, Dict, Any, List, Optional
import psutil

import collector.system_collector as system_collector
import collector.gpu_collector as gpu_collector
import collector.docker_collector as docker_collector
import api.datastore as datastore


def collect_and_store(
    prev_disk: Optional[Dict[str, psutil._common.sdiskio]],
    prev_net: Optional[Dict[str, psutil._common.snetio]],
    interval_s: float,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, psutil._common.sdiskio],
    Dict[str, psutil._common.snetio],
]:
    """
    Collect a fresh snapshot (system + gpu + docker), write host+gpu to SQLite,
    and return everything for immediate UI display.

    Returns:
        snapshot (dict)
        gpus (list of dicts)
        containers (list of dicts)
        new_disk_counters (psutil.disk_io_counters(perdisk=True))
        new_net_counters (psutil.net_io_counters(pernic=True))
    """
    snapshot, new_disk, new_net = system_collector.collect_system_snapshot(
        prev_disk=prev_disk,
        prev_net=prev_net,
        interval_s=interval_s,
    )
    gpus = gpu_collector.collect_gpu_snapshot()

    # collect docker containers (does NOT go into DB)
    containers = docker_collector.collect_docker_containers()

    datastore.insert_snapshot(snapshot, gpus)

    return snapshot, gpus, containers, new_disk, new_net
