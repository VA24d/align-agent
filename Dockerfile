# ALIGN: analyze agent trajectories for stale edits; run mini-swe-agent with the write-through hook.
# Pre-built: docker run --rm ghcr.io/va24d/align-agent:0.3.0 --help
# Build from the repository root:  docker build -t align-agent .
FROM python:3.12-slim
LABEL org.opencontainers.image.source=https://github.com/VA24d/align-agent
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/align-agent
COPY pyproject.toml README.md LICENSE ./
COPY align_agent ./align_agent
RUN pip install --no-cache-dir '.[run]' && git config --global user.email agent@align-agent && git config --global user.name align-agent
COPY examples ./examples
ENV MSWEA_SILENT_STARTUP=1
WORKDIR /work
ENTRYPOINT ["align-agent"]
CMD ["--help"]
