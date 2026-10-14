# syntax=docker/dockerfile:1.7
# ══ Stage 1: builder — compilers, pip, asset downloads. Nothing from here ships except /opt/* ══
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch (CPU build) comes from the PyTorch index and is pinned to an exact version.
COPY requirements-torch-cpu.txt .
RUN pip install -r requirements-torch-cpu.txt
# Everything else: exact pins (+ sha256 hashes when requirements.txt was generated with
# `pip-compile --generate-hashes` — pip then enforces them automatically).
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

# Bake templates / atlases / pretrained nets NOW (no runtime downloads → read-only rootfs works).
ENV MRI_WORK_DIR=/opt/mri_assets/work \
    MRI_TEMPLATE_CACHE=/opt/mri_assets/templateflow \
    TEMPLATEFLOW_HOME=/opt/mri_assets/templateflow \
    MRI_ATLAS_DATASET_DIR=/opt/mri_assets/atlases \
    HOME=/opt/mri_assets/home
COPY api/ /src/api/
COPY scripts/bake_assets.py /src/scripts/bake_assets.py
RUN mkdir -p /opt/mri_assets/home && cd /src && python scripts/bake_assets.py \
    && rm -rf /opt/mri_assets/work && find /opt/mri_assets -type d -exec chmod 755 {} + \
    && find /opt/mri_assets -type f -exec chmod 644 {} +

# ══ Stage 2: runtime — no compilers, no git, non-root, read-only-rootfs friendly ═══════════
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd  --system --uid 10001 --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/mri_assets /opt/mri_assets
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
# Code is root-owned and read-only to the app user.
COPY --chown=root:root api/ ./api/
COPY --chown=root:root frontend/ ./frontend/
COPY --chown=root:root make_admin.py start.py ./
RUN mkdir -p /app/checkpoints /data/db /data/chroma /data/mri_store /backups /tmp/axoniq_mri \
    && chown -R app:app /data /backups \
    && chmod 755 /app/checkpoints

ENV PORT=8000 \
    ENV=production \
    PYTHONPATH=/app \
    NEUROCHECK_DB=/data/db/neurocheck.db \
    CHROMA_DB_PATH=/data/chroma \
    BACKUP_DIR=/backups \
    HOME=/opt/mri_assets/home \
    TEMPLATEFLOW_HOME=/opt/mri_assets/templateflow \
    MRI_TEMPLATE_CACHE=/opt/mri_assets/templateflow \
    MRI_ATLAS_DATASET_DIR=/opt/mri_assets/atlases \
    MRI_WORK_DIR=/tmp/axoniq_mri \
    MRI_DEVICE=cpu \
    MRI_MAX_CONCURRENT_JOBS=1 \
    MRI_CHECKPOINT_DIR=/app/checkpoints \
    HF_HUB_OFFLINE=0
# FORWARDED_ALLOW_IPS = the reverse proxy's IP/CIDR (compose pins the subnet). Never "*".
ENV FORWARDED_ALLOW_IPS=127.0.0.1

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

# SINGLE worker on purpose: the MRI engine's models and job table live in this process's memory.
# --proxy-headers + --forwarded-allow-ips: client IP is taken from X-Forwarded-For ONLY when the
# TCP peer is the trusted proxy (so the rate limiter can't be bypassed or poisoned by spoofing).
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 1800 --proxy-headers --forwarded-allow-ips=\"${FORWARDED_ALLOW_IPS}\""]
