import streamlit as st
import psutil
import subprocess
import pynvml
import time

st.set_page_config(page_title="Server Monitor", layout="wide")
st.title("Server Monitor (TrueNAS VM Edition)")

refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 1, 10)

# System info (static, outside the refresh loop)
with st.expander("🖥️ System Info", expanded=True):
    st.write(f"**CPU:** {psutil.cpu_count(logical=True)} cores ({psutil.cpu_count(logical=False)} physical)")
    st.write(f"**RAM:** {round(psutil.virtual_memory().total / 1e9, 2)} GB")
    st.write(f"**OS:** Ubuntu 24.04")

placeholder = st.empty()

while True:
    with placeholder.container():
        cpu_perc = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory()
        st.metric("CPU Usage", f"{cpu_perc}%")
        st.metric("RAM Usage", f"{ram.percent}%")

        def get_cpu_temp():
            try:
                sensors = subprocess.check_output("sensors", text=True)
                lines = sensors.splitlines()
                for line in lines:
                    if "Package id 0:" in line and "°C" in line:
                        temp = line.split("+")[1].split("°")[0]
                        return float(temp)
                return None
            except Exception:
                return None

        cpu_temp = get_cpu_temp()
        if cpu_temp:
            st.metric("CPU Temp", f"{cpu_temp}°C")
        else:
            st.write("CPU Temp: N/A (Check 'sensors' output)")

        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            st.subheader("NVIDIA GPU(s)")
            for i in range(gpu_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
                st.markdown(f"### GPU {i}: {name}")
                cols = st.columns(4)
                cols[0].metric("Temp", f"{temp}°C")
                cols[1].metric("Power", f"{power}W")
                cols[2].metric("VRAM Used", f"{int(mem.used/1e6)} / {int(mem.total/1e6)} MB")
                cols[3].metric("Utilization", f"{util.gpu}%")
            pynvml.nvmlShutdown()
        except Exception as e:
            st.write(f"Error loading NVIDIA stats: {e}")

    time.sleep(refresh_interval)
    st.rerun()
