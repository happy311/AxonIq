FROM python:3.11-slim

# gcc/g++: some MRI-engine deps build native extensions on install.
# libgomp1: OpenMP runtime required by ANTs/antspynet at import time.
# wget: used by a couple of one-time atlas/template fallback downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc g++ git libgomp1 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# MRI engine deps (torch-cpu, antspyx, antspynet, segmentation-models-pytorch,
# templateflow, nilearn) are large — this install can take several minutes
# and needs real disk/RAM headroom. See README_DEPLOY_AWS.md for instance sizing.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/db /data/chroma /data/backups /app/checkpoints

ENV PORT=8000
ENV HOST=0.0.0.0
ENV NEUROCHECK_DB=/data/db/neurocheck.db
ENV CHROMA_DB_PATH=/data/chroma
ENV PYTHONPATH=/app

# MRI engine defaults — CPU only, one job at a time. Override at `docker run`
# time with -e if your instance can handle more concurrent jobs.
ENV MRI_DEVICE=cpu
ENV MRI_MAX_CONCURRENT_JOBS=1
ENV MRI_CHECKPOINT_DIR=/app/checkpoints

EXPOSE 8000

# NOTE: intentionally no --workers flag. The MRI engine loads its 3 models
# and tracks job state in this process's memory — running more than one
# worker process would mean job submissions and polls could land on
# different processes that never see each other's data. Keep this at a
# single worker; scale by vertical instance size, not worker count.
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app", "--timeout-keep-alive", "1800"]
