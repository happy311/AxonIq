"""
NeuroCheck — Email Utility
Sends OTP emails for password reset via SMTP.

Supported providers (set via environment variables):

Option A — Brevo (recommended, free 300 emails/day):
  SMTP_EMAIL    = your@email.com
  SMTP_PASSWORD = your Brevo SMTP key
  SMTP_HOST     = smtp-relay.brevo.com
  SMTP_PORT     = 587

Option B — Gmail (requires App Password):
  SMTP_EMAIL    = yourname@gmail.com
  SMTP_PASSWORD = 16-char app password
  SMTP_HOST     = smtp.gmail.com (default)
  SMTP_PORT     = 587 (default)

Option C — Any SMTP provider:
  Set SMTP_HOST, SMTP_PORT, SMTP_EMAIL, SMTP_PASSWORD accordingly

If SMTP is not configured the OTP is NEVER returned to the client; delivery simply fails (logged).
"""
from __future__ import annotations
import hashlib
import hmac
import html
import os
import re
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger

from api.core.config import JWT_SECRET, OTP_EXPIRY_MINUTES

SMTP_EMAIL    = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_HOST     = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "587"))

_EMAIL_RE = re.compile(r"^[^@\s'\"<>]+@[^@\s'\"<>]+\.[^@\s'\"<>]+$")


def generate_otp(length: int = 6) -> str:
    """Cryptographically secure numeric OTP (secrets, never `random`)."""
    return "".join(str(secrets.randbelow(10)) for _ in range(length))


def new_salt() -> str:
    return secrets.token_hex(16)


def hash_otp(otp: str, salt: str, email: str) -> str:
    """HMAC-SHA256 keyed with JWT_SECRET over salt|email|otp. A DB leak alone cannot be
    brute-forced offline (the 10^6 space is useless without the server key)."""
    msg = f"{salt}|{email.lower()}|{otp}".encode()
    return hmac.new(JWT_SECRET.encode(), msg, hashlib.sha256).hexdigest()


def verify_otp(plain_otp: str, salt: str, email: str, stored_mac: str) -> bool:
    """Constant-time comparison."""
    return hmac.compare_digest(hash_otp(plain_otp, salt, email), stored_mac)


def otp_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()


def _smtp_send(msg: MIMEMultipart, to_email: str) -> bool:
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        logger.warning("[Email] SMTP not configured; cannot send.")
        return False
    if not _EMAIL_RE.match(to_email) or "\n" in to_email or "\r" in to_email:
        logger.error("[Email] refusing to send to malformed address")
        return False
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        logger.info("[Email] message sent")
        return True
    except smtplib.SMTPException as e:
        logger.error("[Email] SMTP error: {}", type(e).__name__)
    except Exception as e:
        logger.error("[Email] unexpected error: {}", type(e).__name__)
    return False


def send_admin_reset_email(to_email: str, username: str, link: str, minutes: int) -> bool:
    """Admin-initiated reset: the user receives a single-use link (token in the URL fragment)."""
    body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:24px;">
      <h2>Password reset required</h2>
      <p>Hello <strong>{html.escape(username)}</strong>, an administrator has reset your
      NeuroCheck account. Use the one-time link below to choose a new password
      (valid for {minutes} minutes):</p>
      <p><a href="{html.escape(link, quote=True)}">Set a new password</a></p>
      <p style="color:#666;font-size:12px;">If you did not expect this, contact support.</p>
    </div>"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "NeuroCheck — set a new password"
    msg["From"] = f"NeuroCheck AI <{SMTP_EMAIL}>"
    msg["To"] = to_email
    msg.attach(MIMEText(body, "html"))
    return _smtp_send(msg, to_email)


def send_otp_email(to_email: str, username: str, otp: str) -> bool:
    """
    Send OTP email via Gmail SMTP.
    Returns True on success, False on failure.
    """
    subject = "NeuroCheck — Password Reset Code"
    html_body = f"""
    <div style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:480px;margin:0 auto;padding:32px;">
      <div style="background:#0B1F3A;border-radius:12px;padding:24px;text-align:center;margin-bottom:24px;">
        <div style="background:#0A7B74;width:48px;height:48px;border-radius:10px;display:inline-flex;
                    align-items:center;justify-content:center;font-size:22px;font-weight:700;
                    color:#fff;font-family:serif;margin-bottom:12px;">NC</div>
        <h1 style="color:#fff;font-size:20px;margin:0;font-family:'Georgia',serif;">NeuroCheck AI</h1>
        <p style="color:#8FA8BF;font-size:13px;margin:4px 0 0;">MS Clinical Assistant</p>
      </div>

      <h2 style="color:#0B1F3A;font-size:18px;margin-bottom:8px;">Password Reset Request</h2>
      <p style="color:#6B6056;font-size:14px;line-height:1.7;">
        Hello <strong>{html.escape(username)}</strong>, we received a request to reset your NeuroCheck password.
        Use the code below to proceed:
      </p>

      <div style="background:#F5F1EB;border:1px solid #DDD8CF;border-radius:12px;
                  padding:24px;text-align:center;margin:24px 0;">
        <div style="font-size:36px;font-weight:700;letter-spacing:12px;color:#0B1F3A;
                    font-family:'Courier New',monospace;">{otp}</div>
        <p style="color:#6B6056;font-size:12px;margin:10px 0 0;">
          This code expires in <strong>{OTP_EXPIRY_MINUTES} minutes</strong>
        </p>
      </div>

      <p style="color:#6B6056;font-size:13px;line-height:1.7;">
        If you didn't request this, you can safely ignore this email.
        Your password will not be changed.
      </p>

      <hr style="border:none;border-top:1px solid #DDD8CF;margin:24px 0;">
      <p style="color:#B0A898;font-size:11px;text-align:center;">
        NeuroCheck AI — MS Clinical Decision Support<br>
        Built by Dr. Avasarala (UKY) &amp; Dr. Kadambari (NITW)
      </p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"NeuroCheck AI <{SMTP_EMAIL}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))
    return _smtp_send(msg, to_email)
