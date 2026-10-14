"""
AxonIQ — application-level encryption for sensitive columns (Fernet / AES-128-CBC + HMAC).

  messages.content, mri_results.findings_json  →  "enc:v1:<token>"

Rows written before encryption was enabled (no prefix) are still readable, so
turning the key on is a no-downtime change; run `python -m api.tools.encrypt_existing`
(or scripts/encrypt_existing.py) to re-encrypt old rows.

KEY HANDLING: the key lives in env/secret manager, never in the DB or volume.
LOSING THE KEY MEANS LOSING THE DATA — back it up separately from the DB backups.
Rotation: DATA_ENCRYPTION_KEY accepts several comma-separated keys; the first
encrypts, all decrypt (MultiFernet).
"""
from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

_PREFIX = "enc:v1:"
_fernet: MultiFernet | None = None
_loaded = False


def _load() -> MultiFernet | None:
    global _fernet, _loaded
    if _loaded:
        return _fernet
    raw = os.environ.get("DATA_ENCRYPTION_KEY", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    _fernet = MultiFernet([Fernet(k.encode()) for k in keys]) if keys else None
    _loaded = True
    return _fernet


def reload_key() -> None:
    """For tests / key rotation without restart."""
    global _loaded
    _loaded = False


def enabled() -> bool:
    return _load() is not None


def require_key_if_configured_strict(strict: bool) -> None:
    """Called at startup: refuse to run in production without an encryption key."""
    if strict and not enabled():
        raise RuntimeError(
            "DATA_ENCRYPTION_KEY is not set. Patient messages and MRI findings would be stored "
            "in plaintext. Generate a key with: python -c \"from cryptography.fernet import "
            "Fernet; print(Fernet.generate_key().decode())\" — or set REQUIRE_DATA_ENCRYPTION=0 "
            "ONLY if the data volume is encrypted at the infrastructure level."
        )


def encrypt_text(plain: str) -> str:
    f = _load()
    if f is None or plain is None:
        return plain
    return _PREFIX + f.encrypt(plain.encode("utf-8")).decode("ascii")


def decrypt_text(stored: str) -> str:
    if stored is None or not isinstance(stored, str) or not stored.startswith(_PREFIX):
        return stored                      # legacy plaintext row
    f = _load()
    if f is None:
        raise RuntimeError("Encrypted data found but DATA_ENCRYPTION_KEY is not set.")
    try:
        return f.decrypt(stored[len(_PREFIX):].encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise RuntimeError("Could not decrypt stored data — wrong DATA_ENCRYPTION_KEY?") from e


def encrypt_bytes(data: bytes) -> bytes:
    f = _load()
    return f.encrypt(data) if f else data


def is_encrypted(stored: str) -> bool:
    return isinstance(stored, str) and stored.startswith(_PREFIX)


# ── Streaming file encryption (MRI scans can be hundreds of MB — never held in RAM whole) ─────
_FILE_MAGIC_ENC = b"AXQ1"      # chunked Fernet
_FILE_MAGIC_RAW = b"AXQ0"      # no key configured (dev / infra-level encryption)
_CHUNK = 4 * 1024 * 1024


def encrypt_file(src_path: str, dst_path: str) -> None:
    """Encrypt src → dst in 4 MB Fernet-token frames: MAGIC | (len:4 BE | token)*."""
    f = _load()
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        dst.write(_FILE_MAGIC_ENC if f else _FILE_MAGIC_RAW)
        while True:
            chunk = src.read(_CHUNK)
            if not chunk:
                break
            if f:
                tok = f.encrypt(chunk)
                dst.write(len(tok).to_bytes(4, "big"))
                dst.write(tok)
            else:
                dst.write(chunk)


def decrypt_file(src_path: str, dst_path: str) -> None:
    """Inverse of encrypt_file. Raises RuntimeError on a wrong key / corrupt or foreign file."""
    f = _load()
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        magic = src.read(4)
        if magic == _FILE_MAGIC_RAW:
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    return
                dst.write(chunk)
        if magic != _FILE_MAGIC_ENC:
            raise RuntimeError("Stored scan has an unknown format.")
        if f is None:
            raise RuntimeError("Encrypted scan found but DATA_ENCRYPTION_KEY is not set.")
        while True:
            hdr = src.read(4)
            if not hdr:
                return
            n = int.from_bytes(hdr, "big")
            tok = src.read(n)
            if len(tok) != n:
                raise RuntimeError("Stored scan is truncated.")
            try:
                dst.write(f.decrypt(tok))
            except InvalidToken as e:
                raise RuntimeError("Could not decrypt stored scan — wrong DATA_ENCRYPTION_KEY?") from e
