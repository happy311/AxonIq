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
nano .env
```

Set at minimum:
- `JWT_SECRET` — generate with `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`
- `ANTHROPIC_API_KEY` — your Claude API key
- `ADMIN_USERNAME` — your admin account username (create the account via the
  app first, then this grants it admin on next restart)

Leave the `MRI_*` vars at their defaults (`MRI_DEVICE=cpu`,
`MRI_MAX_CONCURRENT_JOBS=1`) unless you've sized the instance for more.

## 7. Build and run

```bash
sudo mkdir -p /data/db /data/chroma /data/backups
sudo chown -R $USER:$USER /data

docker build -t axoniq .

docker run -d \
  --name axoniq \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -v /data:/data \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  axoniq
```

`-v $(pwd)/checkpoints:/app/checkpoints:ro` mounts your checkpoints folder
into the container read-only — you can swap in new checkpoints later
without rebuilding the image (`docker restart axoniq` to pick them up).

## 8. Watch it come up

```bash
docker logs -f axoniq
```

You're looking for, in order:
1. `[AxonIQ] SQLite at: ...`
2. `[LLM] Active model: ...`
3. `[MRI Checkpoints] loaded ... (orientation=axial, ...)`  (×3)
4. `[MRI Engine] Ready — 3 models loaded, CPU-only.`

Step 3–4 can take a few minutes the *first* time (downloading the MNI
template + Harvard-Oxford/AAL atlases). The rest of the app (chat, auth) is
already serving traffic while that happens — MRI uploads just get a "still
starting up" message until it's ready.

Check status anytime:
```bash
curl http://localhost:8000/health | python3 -m json.tool
```
Look at the `mri_engine` block — `"ready": true` and `"n_models_in_ensemble": 3`
means you're good.

## 9. Put a reverse proxy + HTTPS in front (recommended)

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

Nginx site config (`/etc/nginx/sites-available/axoniq`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 1800s;   # MRI jobs can run for minutes
    }
}
```
```bash
sudo ln -s /etc/nginx/sites-available/axoniq /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d your-domain.com
```

## 10. Updating later

```bash
cd ~/axoniq
git pull
docker build -t axoniq .
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
