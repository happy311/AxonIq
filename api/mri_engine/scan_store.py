"""Encrypted at-rest copies of validated uploads, so an approved doctor can download them later."""
import os
import uuid
from pathlib import Path

from api.core import crypto
from api.core.config import MRI_STORE_DIR


def store_scan(src_path: str, user_id: int, session_uuid: str, size: int, suffix: str) -> int:
    """Encrypt src_path into the private store (random filename, 0600) and record it. Returns scan id."""
    from api import database as db
    MRI_STORE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    dst = str(MRI_STORE_DIR / f"{uuid.uuid4().hex}.axq")
    try:
        crypto.encrypt_file(src_path, dst)
        os.chmod(dst, 0o600)
        return db.add_scan(user_id, session_uuid, dst, size, suffix)
    except BaseException:
        Path(dst).unlink(missing_ok=True)
        raise
