#!/bin/sh
set -e

# It's okay if sensors fails (some hosts don't expose hwmon/fan data).
# We don't treat that as fatal.
sensors >/dev/null 2>&1 || true

exec streamlit run ui/dashboard.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --browser.gatherUsageStats=false
