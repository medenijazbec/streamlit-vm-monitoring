# docker/Dockerfile
#
# Lightweight image to run sentra.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build --no-cache -t sentra:latest -f docker/Dockerfile .
#
# Run (manual example):
#   docker run -d --restart unless-stopped --name sentra \
#     -p 8501:8501 \
#     -e STREAMLIT_SERVER_HEADLESS=true \
#     -e STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
#     -e SENTRA_DB_PATH=/data/sentra.db \
#     -e SENTRA_HOST_SYS=/host_sys \
#     -e SENTRA_SAMPLE_INTERVAL=2 \
#     -e SENTRA_DOCKER_STATS=0 \
#     -v sentra_data:/data \
#     -v /sys:/host_sys:ro \
#     -v /var/run/docker.sock:/var/run/docker.sock:ro \
#     sentra:latest
#
# Notes:
#   - We DO NOT exec the docker CLI inside the container.
#     We use the Python docker SDK to talk to the host daemon via docker.sock.
#   - We default SENTRA_DOCKER_STATS=0 so we DON'T spam stats() for 100+
#     containers and freeze the UI. Flip to 1 if you really want per-container
#     CPU/MEM, knowing it can be heavy.
#   - We don't need to mount host /proc anymore for the dashboard to work.
#
# After it's running, open http://<host-ip>:8501

FROM python:3.12-slim

# Install minimal OS deps for sensors, pci info, procps (ps, uptime, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        lm-sensors \
        pciutils \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Helpful runtime env defaults
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# Tell docker SDK inside the container to talk to the host's socket.
ENV DOCKER_HOST=unix:///var/run/docker.sock

# Where SQLite DB will live (we mount /data as a volume)
ENV SENTRA_DB_PATH=/data/sentra.db
ENV SENTRA_HOST_SYS=/host_sys

# How often Streamlit refreshes (seconds)
ENV SENTRA_SAMPLE_INTERVAL=2

# Disable expensive per-container docker stats by default
ENV SENTRA_DOCKER_STATS=0

WORKDIR /app

# Install Python deps first for better build caching
COPY requirements.txt /app/requirements.txt
# Include docker SDK so we can query daemon
RUN pip install --no-cache-dir -r /app/requirements.txt docker

# Copy application code
COPY api /app/api
COPY collector /app/collector
COPY config /app/config
COPY ui /app/ui
COPY docker/entrypoint.sh /entrypoint.sh

# Ensure entrypoint is executable
RUN chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
