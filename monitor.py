import streamlit as st
import psutil
import subprocess
import pynvml
import pandas as pd
import os
import time
from datetime import datetime
import altair as alt

st.set_page_config(page_title="Server Monitor", layout="wide")
st.title("Server Monitor (TrueNAS VM Edition)")

refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 1, 10)

# 1) SYSTEM INFO (unchanged)
with st.expander("🖥️ System Info", expanded=True):
    st.write(f"**CPU:** {psutil.cpu_count(logical=True)} cores "
             f"({psutil.cpu_count(logical=False)} physical)")
    st.write(f"**RAM:** {round(psutil.virtual_memory().total / 1e9, 2)} GB")
    st.write(f"**OS:** Ubuntu 24.04")

# 2) MIGRATE OR CREATE gpu_log.csv
log_path = "gpu_log.csv"
desired_header = "timestamp,gpu_index,temp,util,power,vram"
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        first = f.readline().strip()
    if first != desired_header:
        try:
            df_old = pd.read_csv(log_path)  # reads timestamp,gpu_index,temp,util
            df_old["power"] = pd.NA
            df_old["vram"]  = pd.NA
            df_old.to_csv(log_path, index=False)
        except Exception as err:
            st.warning(f"Could not migrate existing log file: {err}")
else:
    with open(log_path, "w") as f:
        f.write(desired_header + "\n")


placeholder = st.empty()

while True:
    with placeholder.container():
        # — your original system metrics —
        cpu_perc = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory()
        st.metric("CPU Usage", f"{cpu_perc}%")
        st.metric("RAM Usage", f"{ram.percent}%")

        def get_cpu_temp():
            try:
                # redirect stderr so we never see the "No sensors found!" messages
                out = subprocess.check_output(
                    ["sensors"],
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                for line in out.splitlines():
                    if "Package id 0:" in line and "°C" in line:
                        return float(line.split("+")[1].split("°")[0])
            except Exception:
                # any error (including command-not-found) just yields no temp
                pass
            return None

        cpu_temp = get_cpu_temp()
        if cpu_temp is not None:
            st.metric("CPU Temp", f"{cpu_temp}°C")
        else:
            st.write("CPU Temp: N/A (Check 'sensors' output)")

        # — your original GPU text display & record gathering —
        gpu_records = []
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            st.subheader("NVIDIA GPU(s)")
            for i in range(gpu_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name  = pynvml.nvmlDeviceGetName(h)
                temp  = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                mem   = pynvml.nvmlDeviceGetMemoryInfo(h)
                util  = pynvml.nvmlDeviceGetUtilizationRates(h)
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
                vram  = int(mem.used / 1e6)

                # —— EXACTLY YOUR TEXT BLOCK ——
                st.markdown(f"### GPU {i}: {name}")
                cols = st.columns(4)
                cols[0].metric("Temp", f"{temp}°C")
                cols[1].metric("Power", f"{power:.2f}W")
                cols[2].metric("VRAM Used", f"{vram} / {int(mem.total/1e6)} MB")
                cols[3].metric("Utilization", f"{util.gpu}%")

                # record for CSV
                gpu_records.append((i, temp, util.gpu, power, vram))

            pynvml.nvmlShutdown()
        except Exception as e:
            st.write(f"Error loading NVIDIA stats: {e}")

        # — append this iteration to CSV —
        if gpu_records:
            now = datetime.now().isoformat()
            with open(log_path, "a") as f:
                for idx, t, u, p, v in gpu_records:
                    f.write(f"{now},{idx},{t},{u},{p},{v}\n")

        # — NOW: historical plots with fixed domains —
        df = pd.read_csv(log_path, parse_dates=["timestamp"])
        if not df.empty:
            # Temperature: 0–100°C
            st.subheader("GPU Temperatures (°C)")
            temp_chart = (
                alt.Chart(df)
                   .mark_line()
                   .encode(
                       x=alt.X("timestamp:T", title="Time"),
                       y=alt.Y("temp:Q", scale=alt.Scale(domain=[0,100]), title="Temp (°C)"),
                       color=alt.Color("gpu_index:N", legend=alt.Legend(title="GPU"))
                   )
                   .properties(height=200)
            )
            st.altair_chart(temp_chart, use_container_width=True)

            # Utilization: 0–100%
            st.subheader("GPU Utilization (%)")
            util_chart = (
                alt.Chart(df)
                   .mark_line()
                   .encode(
                       x="timestamp:T",
                       y=alt.Y("util:Q", scale=alt.Scale(domain=[0,100]), title="Util (%)"),
                       color=alt.Color("gpu_index:N", legend=None)
                   )
                   .properties(height=200)
            )
            st.altair_chart(util_chart, use_container_width=True)

            # Power: 0–600W
            st.subheader("GPU Power (W)")
            power_chart = (
                alt.Chart(df)
                   .mark_line()
                   .encode(
                       x="timestamp:T",
                       y=alt.Y("power:Q", scale=alt.Scale(domain=[0,600]), title="Power (W)"),
                       color=alt.Color("gpu_index:N", legend=None)
                   )
                   .properties(height=200)
            )
            st.altair_chart(power_chart, use_container_width=True)

            # VRAM Used: 0–12000MB
            st.subheader("GPU VRAM Used (MB)")
            vram_chart = (
                alt.Chart(df)
                   .mark_line()
                   .encode(
                       x="timestamp:T",
                       y=alt.Y("vram:Q", scale=alt.Scale(domain=[0,12000]), title="VRAM Used (MB)"),
                       color=alt.Color("gpu_index:N", legend=None)
                   )
                   .properties(height=200)
            )
            st.altair_chart(vram_chart, use_container_width=True)
        else:
            st.write("No GPU history to plot yet.")

    time.sleep(refresh_interval)
    st.rerun()
