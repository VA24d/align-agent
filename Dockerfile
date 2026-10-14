# stalecheck: analyze agent trajectories for stale edits; run mini-swe-agent with the write-through hook.
# Build from the repository root:  docker build -t stalecheck .
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/stalecheck
COPY pyproject.toml README.md LICENSE ./
COPY stalecheck ./stalecheck
RUN pip install --no-cache-dir '.[run]' && git config --global user.email agent@stalecheck && git config --global user.name stalecheck
COPY examples ./examples
ENV MSWEA_SILENT_STARTUP=1
WORKDIR /work
ENTRYPOINT ["stalecheck"]
CMD ["--help"]
