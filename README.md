docker run -d --restart unless-stopped --name sentra \
  -p 8501:8501 \
  -e SENTRA_DOCKER_STATS=0 \
  -e SENTRA_SAMPLE_INTERVAL=2 \
  -e SENTRA_DB_PATH=/data/sentra.db \
  -e SENTRA_HOST_SYS=/host_sys \
  -v sentra_data:/data \
  -v /sys:/host_sys:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  sentra:latest
