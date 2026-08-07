# GPU worker nodes for the main deployment

A common layout: the main host runs whisper-stt-web-app on CPU (e.g. an
ARM64 VPS with no NVIDIA GPU), while one or more GPU hosts run the same app
as a **worker** and the main deployment dispatches jobs to them. This
document covers running a worker on each GPU host and wiring the main
deployment to dispatch to them.

GPU workers are plain instances of this same app built with the GPU
Dockerfile (`docker-compose.worker.yml`) — dispatch reuses the worker's
normal `/api/transcribe` + `/api/transcribe/status` job flow, so the result
shape, retention, and the `/j/{job_id}` page are identical whether a job
ran on a worker GPU or the main host.

## Architecture

```
browser → main site (Cloudflare) → main app (8561, CPU disabled by default)
                                       │  nodes: config/nodes.yaml
                                       ▼
                   reverse SSH tunnels bound on the main host's
                   docker bridge gateway (10.99.0.1 in this estate)
                                       │
                       worker A (8564) ┘   worker B (8565)
                       GPU               GPU
```

- Nodes are listed in `config/nodes.yaml` on the main deployment (copy
  `config/nodes.example.yaml`); `WHISPER_WORKER_URLS`/`WHISPER_WORKER_NAMES`
  env vars are the legacy fallback. Node labels are public — keep them
  neutral ("GPU Node A", "GPU Node B").
- **No CPU fallback by default**: `WHISPER_ALLOW_LOCAL_CPU` defaults to
  false, so when no node can take a job it fails cleanly with "No
  transcription node available — try again later" instead of running on the
  main host's CPU. Set it true only for plain CPU-only single-host setups.
- The main host dispatches only models that fit the workers' VRAM in float32
  (`WHISPER_GPU_MODELS`, default `tiny,base,small`). The worker GPUs here
  are Maxwell/Pascal (CC 5.0/6.1): CTranslate2 runs float32 there (no
  fp16/int8 kernels), so `medium` and larger are excluded and fail cleanly
  when CPU is disabled.
- The user can also choose a specific node per job (node picker on the
  page, or a `worker` param — a name from the nodes list, a 1-based index,
  or the built-in "CPU" node). A chosen node is used exclusively: if it
  fails, the job fails with a message naming the node, rather than silently
  running somewhere else.

## Deploy a worker on a GPU host (run once per host)

Prerequisites: docker with the NVIDIA container toolkit (nvidia-ctk/CDI —
verify with `docker info`), the NVIDIA closed-source driver matching the GPU
(`nvidia-580xx-dkms` on Arch for Maxwell/Pascal — the `nvidia-open` modules
do NOT support them), and the repo checked out.

```bash
cd <repo checkout>
docker compose -f docker-compose.worker.yml up -d --build
curl -sf http://localhost:8564/health     # device should read "cuda"
```

The worker binds localhost only and auto-restarts (`restart: unless-stopped`).
Multiple workers use distinct local ports:

```bash
HOST_PORT=8565 docker compose -f docker-compose.worker.yml up -d --build
```

The worker image compiles CTranslate2 from source for the GPU's compute
capability (`WORKER_CUDA_ARCH` build arg, default 5.0; set 6.1 for Pascal)
because the stock pip wheel ships only sm_86/90/100 kernels. This build
takes a while — reuse the built image across restarts.

## Reverse tunnel to the main host

The workers bind localhost on the GPU hosts, so the main host cannot reach
them directly. Each GPU host runs a systemd reverse-tunnel service (estate
pattern) forwarding its worker port to the main host's **docker bridge
gateway** (10.99.0.1 here — the interface the app container reaches via
`host.docker.internal`).

On the main host, `sshd_config` needs `GatewayPorts clientspecified`
(drop-in in `/etc/ssh/sshd_config.d/`) so the tunnel can bind the
non-loopback gateway address, and the firewall must allow the bridge ports:

```bash
# main host (one-time)
echo 'GatewayPorts clientspecified' | sudo tee /etc/ssh/sshd_config.d/99-gatewayports.conf
sudo systemctl reload sshd
BR=$(ip -o -4 addr show | awk '/10\.99\.0\.1/{print $2; exit}')
sudo ufw allow in on $BR to any port 8564 proto tcp
sudo ufw allow in on $BR to any port 8565 proto tcp
```

When the GPU host already has a reverse tunnel service (common on laptops
that also tunnel SSH), add the worker forward to its ExecStart. Example —
appending `-R 10.99.0.1:8564:localhost:8564` to an existing unit running as
user `tunnel` with a key authorized on the main host:

```ini
[Service]
User=tunnel
ExecStart=/usr/bin/ssh -NT -R 10.99.0.1:8564:localhost:8564 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new \
  -i /home/tunnel/.ssh/tunnel_key -R 2222:localhost:22 \
  <main-host-user>@<main-host>
Restart=always
RestartSec=10
```

(`systemctl daemon-reload && systemctl restart reverse-tunnel.service` —
note that restarting the tunnel drops your own SSH path if you connect
through it, so reconnect after.) For a fresh host, create a dedicated
`reverse-tunnel-worker.service` with the same shape and just the worker
forward. Worker B forwards `-R 10.99.0.1:8565:localhost:8564` (its local
port is 8565). Then verify from the main host:

```bash
curl -sf http://10.99.0.1:8564/health   # and 8565 — must answer
```

## Configure the main deployment

Copy `config/nodes.example.yaml` to `config/nodes.yaml` with the real URLs
(e.g. `http://host.docker.internal:8564`, `:8565`) and rebuild:

```bash
docker compose up -d --build
```

## Verification

- `curl http://10.99.0.1:8564/health` on the main host → `"device":"cuda"`.
- Transcribe a file through the live site and check the app logs:
  `Remote transcription complete via ...` when the worker path is used, or
  the dispatch-failure log + a clean "No transcription node available"
  failure when no worker is reachable (CPU is never used).
- Stop a worker (`sudo systemctl stop reverse-tunnel-worker.service` and/or
  stop the container) → the next job fails cleanly instead of running on
  the main host's CPU.
