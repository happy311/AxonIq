# Deploying AxonIQ (merged single-server) to AWS

This build runs the whole app — chat/agent, auth, RAG, **and** the MRI
segmentation engine — as one Docker container. No separate Kaggle server,
no tunnel. CPU-only inference, 3-checkpoint ensemble (one per orientation).

## 0. Before you start

- You need your 3 trained checkpoints (`fold*_axial.pt`, `fold*_coronal.pt`,
  `fold*_sagittal.pt` — any filenames, see `checkpoints/README.md`).
- You need an `ANTHROPIC_API_KEY` (or `HF_TOKEN` as fallback).

## 1. Push this repo to GitHub

```bash
cd AxonIq-patched
git init
git add .
git commit -m "Merged single-server MRI engine, CPU-only, 3-checkpoint ensemble"
git branch -M main
git remote add origin https://github.com/<you>/axoniq.git
git push -u origin main
```

`checkpoints/*.pt` is **not** committed (see `.gitignore`) — checkpoint
files are large binaries that don't belong in git. You'll copy them onto
the server directly in step 5.

## 2. Launch an EC2 instance

The MRI engine (ANTs registration + skull-stripping + 3-model CPU inference,
on top of the existing chat app) is CPU- and RAM-hungry. Recommendations:

| Instance | vCPU | RAM | Notes |
|---|---|---|---|
| `t3.xlarge` | 4 | 16 GB | Minimum comfortable size |
| `m6i.xlarge` / `m6i.2xlarge` | 4–8 | 16–32 GB | Better sustained CPU (t3 is burstable/throttled) |

- AMI: **Ubuntu 22.04 LTS**
- Storage: **at least 40 GB** gp3 EBS (Docker image + template/atlas caches +
  Python deps add up)
- Security group: allow inbound **22** (SSH, your IP only) and **80**/**443**
  (or **8000** if you're not putting a reverse proxy in front yet)

## 3. Install Docker on the instance

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
newgrp docker
```

## 4. Clone the repo onto the instance

```bash
git clone https://github.com/<you>/axoniq.git
cd axoniq
```

## 5. Add your 3 checkpoints

```bash
mkdir -p checkpoints
# from your local machine:
scp -i your-key.pem fold0_axial.pt fold0_coronal.pt fold0_sagittal.pt \
    ubuntu@<EC2_PUBLIC_IP>:~/axoniq/checkpoints/
```

## 6. Configure environment

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Required (the app **refuses to start** without them in production):
- `JWT_SECRET` — ≥ 32 random chars, must not contain `CHANGE`/`example`/`placeholder`:
  `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`
- `DATA_ENCRYPTION_KEY` — Fernet key for messages + MRI findings at rest:
  `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
  **Store a copy in a different place than the backups** (Secrets Manager, §A.3). Losing it = losing the data.
- `ANTHROPIC_API_KEY`, `SMTP_EMAIL`, `SMTP_PASSWORD`
- `DOMAIN`, `ACME_EMAIL`, `PUBLIC_BASE_URL` (HTTPS URL used in e-mailed reset links)
- `MRI_CHECKPOINT_SHA256` — pin each checkpoint: `sha256sum checkpoints/*.pt > checkpoints/SHA256SUMS`
  (the app refuses to start if a file's hash differs; checkpoints must also be loadable with
  `torch.load(weights_only=True)` — re-save any that contain numpy scalars as plain Python types).

**There is no `ADMIN_USERNAME` any more.** Register the account in the UI, then grant admin once, by hand:

```bash
docker compose exec api python make_admin.py <username>
```

## 7. Build and run (docker compose — the supported path)

```bash
docker compose up -d --build
docker compose ps
```

What you get: Traefik is the only thing publishing ports (80→443 redirect, Let's Encrypt, HSTS); the API
is on an internal network only, runs as UID 10001 with a read-only root filesystem, `cap_drop: ALL`,
`no-new-privileges`, CPU/RAM limits, a `tmpfs` /tmp for uploads, `/data` on a named volume and backups on a
separate volume. Uvicorn runs with `--proxy-headers --forwarded-allow-ips=172.28.0.0/24` (the proxy
subnet), so the real client IP is trusted only when it comes from Traefik.

If you must use plain `docker run`, copy the same flags: `--read-only --tmpfs /tmp:size=4g --cap-drop ALL
--security-opt no-new-privileges -p 127.0.0.1:8000:8000 -e FORWARDED_ALLOW_IPS=<proxy ip>`.

## 8. Watch it come up

```bash
docker compose logs -f api
```

In order: `[AxonIQ] SQLite at: ... (encryption at rest: True)` → `[janitor] startup: ...` →
`[LLM] Active model: ...` → `[MRI Checkpoints] loaded ...` (×3) → `[MRI Engine] Ready`.
Templates/atlases/pretrained nets are baked into the image at build time, so there are **no runtime downloads**.

`GET /health` is public and returns only `{"status":"ok"}`. Detailed LLM/MRI status:
`GET /admin/health` (admin login required; audited).

## 9. HTTPS

Handled by the `proxy` service in `docker-compose.yml` (Traefik + Let's Encrypt, 80→443 redirect, HSTS,
TLS ≥ 1.2). Point your DNS A record at the instance's Elastic IP *before* first start so the ACME
challenge succeeds. Note `/var/run/docker.sock` is mounted into Traefik read-only: that is
root-equivalent if Traefik is compromised — for stricter setups put a docker-socket-proxy in front of it,
or use an ALB + ACM certificate (see §A.1) and drop the proxy service.

## 10. Updating later

```bash
cd ~/axoniq
git pull
docker compose build
docker stop axoniq && docker rm axoniq
docker run -d --name axoniq --restart unless-stopped -p 8000:8000 \
  --env-file .env -v /data:/data \
  -v $(pwd)/checkpoints:/app/checkpoints:ro axoniq
```

## Things worth knowing

- **Single container, single worker, always.** The MRI engine keeps its
  loaded models and in-flight job state in that one process's memory — see
  `ARCHITECTURE.md`'s "v21 Changes" section. Don't scale this horizontally
  (multiple containers behind a load balancer) without first moving job
  state into shared storage (e.g. the existing `mri_jobs` SQLite table would
  need to carry the actual report, not just the case_id).
- **CPU inference is slow.** Expect single-digit minutes per MRI case
  (registration + skull-strip dominate, not the 3-model ensemble itself).
  `MRI_INFERENCE_TIMEOUT` (default 1500s/25min) is the patient-facing poll
  budget — raise it if your instance is slower than that.
- **First request after a restart may be slow** if the template/atlas cache
  in `/tmp` was cleared (container restart wipes `/tmp` unless you mount it).
  If you want to avoid re-downloading these on every restart, mount a volume
  at `MRI_WORK_DIR` too, e.g. add `-v /data/mri_cache:/tmp/axoniq_mri` to the
  `docker run` command and it'll persist across restarts.

## GPU deployment (optional)

The engine auto-detects a GPU (`MRI_DEVICE=auto`, the default in
`Dockerfile.gpu`) and uses it if torch can see one — nothing in
`api/mri_engine/` needs to change, only the base image and torch build.

**Registration and skull-stripping (ANTs) still run on CPU regardless** —
those libraries don't use CUDA. Only the 3-model ensemble's forward pass
speeds up. In practice this means a GPU cuts total per-case time noticeably
but not by an order of magnitude — ANTs preprocessing is still the floor.

### 1. Use a GPU instance

`g4dn.xlarge` (1x T4, 4 vCPU, 16 GB RAM) is the cheapest sensible starting
point. Bigger checkpoints or wanting `MRI_MAX_CONCURRENT_JOBS > 1` would push
you toward `g5.xlarge` or larger.

### 2. Install the NVIDIA driver + Container Toolkit on the host

```bash
# NVIDIA driver (Ubuntu 22.04, AWS-recommended path):
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

# after reboot, confirm the host sees the GPU:
nvidia-smi

# NVIDIA Container Toolkit (lets Docker pass the GPU into containers):
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Build the GPU image and run with `--gpus all`

```bash
docker build -f Dockerfile.gpu -t axoniq-gpu .

docker run -d \
  --name axoniq \
  --restart unless-stopped \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -v /data:/data \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  axoniq-gpu
```

### 4. Verify it's actually using the GPU

```bash
docker logs -f axoniq
# look for: [MRI Engine] Ready — 3 models loaded (device: cuda).

curl http://localhost:8000/health | python3 -m json.tool
# mri_engine.device should read "cuda"
```

If `device` still says `"cpu"` after this, the most common causes are: the
NVIDIA driver isn't installed/loaded on the host (`nvidia-smi` fails), the
Container Toolkit step above was skipped, or `--gpus all` was left off the
`docker run` command.

### Switching an existing CPU deployment to GPU

You don't need to start over — stop the container, rebuild with
`Dockerfile.gpu`/`requirements-gpu.txt`, and re-run with `--gpus all`. Your
`/data` volume (DB, chroma) and `checkpoints/` folder carry over unchanged.


---

# Appendix A — AWS security baseline

## A.1 Network: security group + IMDSv2
- **Inbound:** `443/tcp` from `0.0.0.0/0` (and `80/tcp` only for the HTTP→HTTPS redirect / ACME challenge).
  `22/tcp` **only from your own IP** (`x.x.x.x/32`) — or better, no SSH at all: use SSM Session Manager and
  remove the rule. **Never** open 8000 (the API isn't published; compose binds nothing but 80/443).
- **Outbound:** 443 (Anthropic, SMTP submission port 587, Let's Encrypt, package mirrors) — restrict if you can.
- **Require IMDSv2** (blocks SSRF-style credential theft from the metadata service):
  ```bash
  aws ec2 modify-instance-metadata-options --instance-id i-XXXX \
      --http-tokens required --http-put-response-hop-limit 1 --http-endpoint enabled
  ```
  (hop limit 1 also keeps containers from reaching the metadata service; launch new instances with
  `--metadata-options HttpTokens=required,HttpPutResponseHopLimit=1`.)
- Optional: put an ALB + ACM + AWS WAF in front, keep the instance in a private subnet, and set
  `FORWARDED_ALLOW_IPS` to the ALB subnet CIDR.

## A.2 Storage: EBS encryption
- Create the volume (or the whole account default) **encrypted with a KMS CMK**:
  `aws ec2 enable-ebs-encryption-by-default --region <r>`; for the existing root volume, snapshot → copy
  with `--encrypted` → replace the volume.
- Keep `/data` on a dedicated encrypted EBS volume, mounted with `nodev,nosuid`. The application also
  encrypts message content, session titles and MRI findings itself (`DATA_ENCRYPTION_KEY`), so a stolen
  snapshot or DB file alone does not expose them. *Not* column-encrypted (rely on EBS/KMS): usernames,
  e-mail addresses, timestamps, symptom/feature metadata, tier logs.

## A.3 Secrets in AWS Secrets Manager (not in `.env` on disk)
Store `JWT_SECRET`, `DATA_ENCRYPTION_KEY`, `ANTHROPIC_API_KEY`, `SMTP_PASSWORD` as one secret, give the
instance role `secretsmanager:GetSecretValue` on that secret **only**, and render `.env` at boot:
```bash
aws secretsmanager get-secret-value --secret-id axoniq/prod --query SecretString --output text \
  | python3 -c 'import sys,json; [print(f"{k}={v}") for k,v in json.load(sys.stdin).items()]' \
  > /run/axoniq.env && chmod 600 /run/axoniq.env      # /run is tmpfs → never hits the disk
docker compose --env-file /run/axoniq.env up -d
```
(Or ECS/EKS with native secret injection.) Rotate `JWT_SECRET` by changing it and restarting — all sessions end.
`DATA_ENCRYPTION_KEY` supports comma-separated keys for rotation: put the **new key first**, keep the old one
after it, restart; then run a re-encrypt pass before dropping the old key.

## A.4 Automated, encrypted backups
- The app takes a consistent SQLite backup daily into `/backups` (separate volume, mode 0600, Fernet-encrypted
  when `DATA_ENCRYPTION_KEY` is set, 7 kept). That is a *local* safety net only.
- Off-box: a nightly cron/systemd timer that copies `/backups` to an S3 bucket with **SSE-KMS**, versioning,
  Block Public Access, and a lifecycle rule; plus an **AWS Backup** plan (daily, 35-day retention, KMS-encrypted vault)
  for the `/data` EBS volume:
  ```bash
  aws s3 sync /var/lib/docker/volumes/axoniq_backups/_data s3://axoniq-backups-<acct>/db/ \
      --sse aws:kms --sse-kms-key-id alias/axoniq-backups
  ```
- **Test a restore quarterly.** Remember retention/erasure: backups still contain data a user later deleted
  until they age out — keep the backup window short and document it in your privacy notice.
- Keep `DATA_ENCRYPTION_KEY` backed up separately from the data (Secrets Manager + an offline copy).

## A.5 Patch & monitor
`unattended-upgrades` on the host, rebuild the image regularly (`docker compose build --pull`), CloudWatch
alarms on CPU/memory/disk, and ship container logs to CloudWatch (app logs are PHI-redacted and carry a
per-request `rid=` for correlation).
