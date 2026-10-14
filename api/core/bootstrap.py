"""
AxonIQ — administrator account from environment variables.

    ADMIN_USERNAME / ADMIN_PASSWORD  (+ optional ADMIN_EMAIL)

At startup — before the server accepts any request, so nobody can pre-register that name — the
account is created (or its password re-synced if the env value changed). Safety rules:

  * the password must pass the normal password policy (>= 12 chars etc.), else startup aborts;
  * if the username/e-mail already belongs to an account NOT created by this mechanism, startup
    aborts instead of promoting it (that account could have been registered by anyone);
  * if ADMIN_USERNAME is later changed/removed, the old env-managed account loses admin;
  * the account is flagged env_managed and never appears as a patient or doctor.

The plain password lives only in the environment; the database stores the bcrypt hash.
"""
from loguru import logger

from api.core import config
from api.core.passwords import hash_password, verify_password, validate_password_policy


def ensure_env_admin() -> None:
    from api import database as db

    name = config.ADMIN_USERNAME
    if not name:
        n = _demote_all()
        logger.info("[bootstrap] ADMIN_USERNAME not set — env-based admin disabled ({} demoted)", n)
        return
    if not config.ADMIN_PASSWORD:
        raise RuntimeError("ADMIN_USERNAME is set but ADMIN_PASSWORD is empty.")
    email = config.ADMIN_EMAIL or f"{name.lower()}@admin.invalid"
    try:
        validate_password_policy(config.ADMIN_PASSWORD, username=name, email=email)
    except Exception as e:                                   # PasswordPolicyError / ValueError
        raise RuntimeError(f"ADMIN_PASSWORD rejected: {e}") from None

    demoted = db.demote_env_admins_except(name)
    if demoted:
        logger.warning("[bootstrap] {} previous env-managed admin(s) demoted", demoted)

    existing = db.get_user_by_username(name)
    by_email = db.get_user_by_email(email)
    if existing is None:
        if by_email is not None:
            raise RuntimeError("ADMIN_EMAIL already belongs to another account. Choose a different ADMIN_EMAIL.")
        uid = db.create_env_admin(name, email, hash_password(config.ADMIN_PASSWORD))
        logger.info("[bootstrap] administrator account created (id={})", uid)
        return
    if not existing.get("env_managed"):
        raise RuntimeError(
            f"ADMIN_USERNAME '{name}' already exists as a normal account. Refusing to promote it "
            "(anyone could have registered that name). Pick an unused ADMIN_USERNAME, or grant admin "
            "explicitly with `python make_admin.py <username>`.")
    if by_email is not None and by_email["id"] != existing["id"]:
        raise RuntimeError("ADMIN_EMAIL already belongs to another account.")
    db.set_role_admin_from_env(existing["id"])
    if not verify_password(config.ADMIN_PASSWORD, existing["password_hash"]):
        db.update_user_password(existing["id"], hash_password(config.ADMIN_PASSWORD))   # bumps token_version
        logger.info("[bootstrap] administrator password re-synced from environment")


def _demote_all() -> int:
    from api import database as db
    return db.demote_env_admins_except("\x00none\x00")
