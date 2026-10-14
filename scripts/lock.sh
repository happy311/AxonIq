#!/usr/bin/env sh
# Produce a HASH-LOCKED requirements file (every artifact sha256-pinned; pip enforces it automatically).
# Run on a machine with ~15 GB free disk and the same Python as the Dockerfile (3.11):
#   python3.11 -m venv /tmp/lockenv && . /tmp/lockenv/bin/activate && pip install pip-tools
set -eu
cp requirements.txt requirements.in
pip-compile --generate-hashes --allow-unsafe --no-header --strip-extras \
            --constraint constraints.txt --output-file requirements.lock requirements.in
echo "Now:  mv requirements.lock requirements.txt   (and keep constraints.txt for reference)"
echo "Audit: pip-audit -r requirements.txt --require-hashes"
