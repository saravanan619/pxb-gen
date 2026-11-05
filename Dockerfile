FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pxb-selfplay /usr/local/bin/pxb-selfplay
RUN chmod +x /usr/local/bin/pxb-selfplay
CMD ["pxb-selfplay", "--games", "20000", "--depth", "5", "--max-nodes", "15000", "--threads", "12", "--seed", "101", "--output", "/app/out.txt"]
