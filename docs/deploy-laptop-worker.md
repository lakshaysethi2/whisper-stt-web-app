# Laptop GPU workers for audio.lak.nz

The VPS (Oracle A1, ARM64) has no NVIDIA GPU, so transcription has always run
on CPU. The estate's two Arch laptops each have a real CUDA GPU — laptop1
(NVIDIA GeForce 940MX), laptop2 (NVIDIA MX330) — and this document covers
running a whisper **worker** on each and wiring the main deployment to
dispatch jobs to them, falling back to VPS CPU when they are unreachable.

## Architecture

```
browser → audio.lak.nz (Cloudflare) → VPS whisper-web (8561, CPU fallback)
                                          │  WHISPER_WORKER_URLS=
                                          │  http://host.docker.internal:8564,http://host.docker.internal:8565
                                          ▼
                      reverse SSH tunnels bound on VPS whisper-net gateway 10.99.0.1
                                          │
                    laptop1 worker (8564) ┘   laptop2 worker (8565)
                    GPU: 940MX               GPU: MX330
```

- The worker is **this same app** built with the GPU Dockerfile — no separate
  worker code. Dispatch reuses the worker's normal `/api/transcribe` +
  `/api/transcribe/status` job flow, so the result shape, retention, and the
  `/j/{job_id}` page are identical whether a job ran on a laptop GPU or the
  VPS CPU.
- The VPS dispatches only models that fit ~2 GB VRAM in float32
  (`WHISPER_GPU_MODELS`, default `tiny,base,small,crisperwhisper-small`).
  The laptop GPUs are Maxwell/Pascal (CC 5.0/6.1): CTranslate2 runs float32
  there (no fp16/int8 kernels), so `medium` and larger would OOM — they are
  excluded and transcribe locally as before. Large models requested by the
  user therefore always fall back to VPS CPU.
- The user can also choose a specific node per job (node picker on the page,
  or a `worker` param — a name from `WHISPER_WORKER_NAMES` or a 1-based
  index). A chosen node is used exclusively: if it fails, the job fails with
  a message naming the node, rather than silently running somewhere else.
- Fallback is automatic and total: if every worker is unreachable, fails the
  health check, rejects the upload, fails the job, or stops responding while
  polling, the VPS transcribes the job locally on CPU. The user never sees an
  error from the laptop path.

## Deploy on a laptop (run once per laptop)

Prerequisites: docker with the NVIDIA container toolkit (`nvidia-ctk` +
nvidia runtime or CDI — verify with `docker info`), the NVIDIA closed-source
driver for Maxwell/Pascal (`nvidia-580xx-dkms` on Arch — the `nvidia-open`
modules do NOT support Maxwell/Pascal), and the repo checked out.

```bash
cd <repo checkout>
docker compose -f docker-compose.worker.yml up -d --build
curl -sf http://localhost:8564/health     # device should read "cuda"
```

laptop2 uses `HOST_PORT=8565` so both laptops can run the same port locally
while the VPS exposes distinct ports:

```bash
HOST_PORT=8565 docker compose -f docker-compose.worker.yml up -d --build
```

### Reverse tunnel to the VPS

The workers bind localhost on the laptops, so the VPS cannot reach them
directly. Each laptop runs a systemd `reverse-tunnel.service` (estate pattern;
laptop1 already had one for SSH port 2222) that forwards its worker port to
the VPS **whisper-net gateway** (10.99.0.1, the docker bridge the main app's
container is on — the container reaches it via `host.docker.internal`).

On the VPS, `sshd_config` has `GatewayPorts clientspecified` (drop-in in
`/etc/ssh/sshd_config.d/`) so the tunnel can bind the non-loopback gateway
address, and UFW allows the bridge port:

```bash
# VPS (one-time)
echo 'GatewayPorts clientspecified' | sudo tee /etc/ssh/sshd_config.d/99-gatewayports.conf
sudo systemctl reload sshd
BR=$(ip -o -4 addr show | awk '/10\.99\.0\.1/{print $2; exit}')
sudo ufw allow in on $BR to any port 8564 proto tcp
sudo ufw allow in on $BR to any port 8565 proto tcp
```

Laptop unit (`/etc/systemd/system/reverse-tunnel-worker.service`, user
`tunnel` with a key authorized on the VPS):

```ini
[Unit]
Description=Reverse SSH tunnel (whisper worker) to VPS
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0
[Service]
User=tunnel
ExecStart=/usr/bin/ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -i /home/tunnel/.ssh/tunnel_key -R 10.99.0.1:8564:localhost:8564 ubuntu@192.9.187.157
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
```

laptop2 tunnels `-R 10.99.0.1:8565:localhost:8564` (worker port 8565 locally,
remote port 8564 forwarded to the VPS's 8565). Then:

```bash
sudo systemctl enable --now reverse-tunnel-worker.service
# from the VPS:
curl -sf http://127.0.0.1:8564/health   # and 8565 — must answer
```

## Configure the main deployment (VPS)

In the VPS `.env`:

```
WHISPER_WORKER_URLS=http://host.docker.internal:8564,http://host.docker.internal:8565
WHISPER_HOST_GATEWAY=10.99.0.1
```

Then `docker compose up -d --build` (the compose file passes the variables
through and maps `host.docker.internal` to the whisper-net gateway).

## Verification

- `curl http://127.0.0.1:8564/health` on the VPS → `"device":"cuda"`.
- Transcribe a file through https://audio.lak.nz/ and check the app logs:
  `Remote transcription complete via http://host.docker.internal:8564 ...`
  when the laptop path is used, or the fallback warning when it isn't.
- Stop a worker (`sudo systemctl stop reverse-tunnel-worker.service` and/or
  stop the container) → next job logs the fallback warning and still completes.
