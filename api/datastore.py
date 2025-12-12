# api/datastore.py
"""
datastore.py
------------

Thin wrapper around SQLite for sentra.

We store short-timescale samples of:
- cpu_samples
- mem_samples
- gpu_samples
- disk_samples
- net_samples
- fan_samples

We expose:
    init_db_if_needed()
    insert_snapshot(snapshot, gpus)
    get_cpu_mem_history(minutes=60)
    get_gpu_history(minutes=60)
    purge_before(cutoff_ts)
"""

import time
import json
import sqlite3
from typing import Any, Dict, List

import pandas as pd

from config import config

_initialized = False


def _conn() -> sqlite3.Connection:
    db_path = config.get_db_path()
    # check_same_thread=False lets Streamlit call from different threads if needed
    return sqlite3.connect(db_path, check_same_thread=False)


def init_db_if_needed() -> None:
    """Create tables if they don't already exist."""
    global _initialized
    if _initialized:
        return

    conn = _conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS cpu_samples (
            ts INTEGER,
            total_util REAL,
            iowait REAL,
            per_core TEXT,
            cpu_temp REAL,
            load1 REAL,
            load5 REAL,
            load15 REAL,
            uptime_sec REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS mem_samples (
            ts INTEGER,
            used_percent REAL,
            used_bytes INTEGER,
            total_bytes INTEGER,
            swap_used_percent REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS gpu_samples (
            ts INTEGER,
            gpu_index INTEGER,
            temp REAL,
            util REAL,
            power_w REAL,
            vram_used_mb INTEGER,
            vram_total_mb INTEGER,
            fan_percent REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS disk_samples (
            ts INTEGER,
            device TEXT,
            read_bps REAL,
            write_bps REAL,
            usage_percent REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS net_samples (
            ts INTEGER,
            iface TEXT,
            rx_bps REAL,
            tx_bps REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fan_samples (
            ts INTEGER,
            label TEXT,
            rpm REAL
        )
    """)

    # Some light indices for faster time slicing
    c.execute("CREATE INDEX IF NOT EXISTS idx_cpu_ts ON cpu_samples(ts)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_mem_ts ON mem_samples(ts)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_gpu_ts ON gpu_samples(ts)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_disk_ts ON disk_samples(ts)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_net_ts ON net_samples(ts)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fan_ts ON fan_samples(ts)")

    conn.commit()
    conn.close()

    _initialized = True


def insert_snapshot(snapshot: Dict[str, Any], gpus: List[Dict[str, Any]]) -> None:
    """
    Take snapshot dict from system_collector + list of GPU dicts from gpu_collector,
    and append them to SQLite.
    """
    init_db_if_needed()

    conn = _conn()
    c = conn.cursor()

    ts = int(snapshot["ts"])

    # CPU
    cpu = snapshot["cpu"]
    meta = snapshot["meta"]
    c.execute(
        """
        INSERT INTO cpu_samples
        (ts,total_util,iowait,per_core,cpu_temp,load1,load5,load15,uptime_sec)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            ts,
            cpu["total_util"],
            cpu["iowait"],
            json.dumps(cpu["per_core"]),
            cpu["temp"],
            cpu["load"]["1m"],
            cpu["load"]["5m"],
            cpu["load"]["15m"],
            meta["uptime_sec"],
        ),
    )

    # MEM
    mem = snapshot["mem"]
    c.execute(
        """
        INSERT INTO mem_samples
        (ts,used_percent,used_bytes,total_bytes,swap_used_percent)
        VALUES (?,?,?,?,?)
        """,
        (
            ts,
            mem["used_percent"],
            mem["used_bytes"],
            mem["total_bytes"],
            mem["swap_used_percent"],
        ),
    )

    # GPU
    for g in gpus:
        if "error" in g:
            continue
        c.execute(
            """
            INSERT INTO gpu_samples
            (ts,gpu_index,temp,util,power_w,vram_used_mb,vram_total_mb,fan_percent)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                ts,
                g["index"],
                g["temp"],
                g["util"],
                g["power_w"],
                g["vram_used_mb"],
                g["vram_total_mb"],
                g["fan_percent"],
            ),
        )

    # DISK throughput
    disk_data = snapshot["disk"]
    usage_percent = disk_data["root_usage_percent"]
    for dev, vals in disk_data["throughput"].items():
        c.execute(
            """
            INSERT INTO disk_samples
            (ts,device,read_bps,write_bps,usage_percent)
            VALUES (?,?,?,?,?)
            """,
            (
                ts,
                dev,
                vals["read_bps"],
                vals["write_bps"],
                usage_percent,
            ),
        )

    # NET throughput
    for iface, vals in snapshot["net"]["throughput"].items():
        c.execute(
            """
            INSERT INTO net_samples
            (ts,iface,rx_bps,tx_bps)
            VALUES (?,?,?,?)
            """,
            (
                ts,
                iface,
                vals["rx_bps"],
                vals["tx_bps"],
            ),
        )

    # FANS
    for label, rpm in snapshot["fans"].items():
        c.execute(
            """
            INSERT INTO fan_samples
            (ts,label,rpm)
            VALUES (?,?,?)
            """,
            (
                ts,
                label,
                rpm,
            ),
        )

    conn.commit()
    conn.close()


def get_cpu_mem_history(minutes: int = 60) -> pd.DataFrame:
    """
    Return CPU+memory+swap history for the last `minutes` minutes as one DataFrame.
    Columns:
        ts, total_util, cpu_temp, used_percent, swap_used_percent, timestamp
    """
    init_db_if_needed()

    since = int(time.time() - minutes * 60)
    conn = _conn()

    # CPU history
    df_cpu = pd.read_sql_query(
        """
        SELECT ts, total_util, cpu_temp
        FROM cpu_samples
        WHERE ts >= ?
        ORDER BY ts ASC
        """,
        conn,
        params=(since,),
    )

    # MEM history
    df_mem = pd.read_sql_query(
        """
        SELECT ts, used_percent, swap_used_percent
        FROM mem_samples
        WHERE ts >= ?
        ORDER BY ts ASC
        """,
        conn,
        params=(since,),
    )

    conn.close()

    if df_cpu.empty and df_mem.empty:
        return pd.DataFrame()

    # Outer merge on ts to avoid losing rows
    df = pd.merge(df_cpu, df_mem, on="ts", how="outer").sort_values("ts")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
    return df


def get_gpu_history(minutes: int = 60) -> pd.DataFrame:
    """
    Return GPU telemetry for the last `minutes` minutes as a pandas DataFrame
    with a timestamp column for plotting.
    """
    init_db_if_needed()

    since = int(time.time() - minutes * 60)
    conn = _conn()
    df = pd.read_sql_query(
        """
        SELECT ts, gpu_index, temp, util, power_w, vram_used_mb, vram_total_mb, fan_percent
        FROM gpu_samples
        WHERE ts >= ?
        ORDER BY ts ASC
        """,
        conn,
        params=(since,),
    )
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
    return df


def purge_before(cutoff_ts: float) -> None:
    """
    Delete all rows older than cutoff_ts from every table.
    Then vacuum the DB to reclaim space.
    """
    init_db_if_needed()

    conn = _conn()
    c = conn.cursor()
    for tbl in [
        "cpu_samples",
        "mem_samples",
        "gpu_samples",
        "disk_samples",
        "net_samples",
        "fan_samples",
    ]:
        c.execute(f"DELETE FROM {tbl} WHERE ts < ?", (int(cutoff_ts),))

    conn.commit()
    c.execute("VACUUM")
    conn.close()
