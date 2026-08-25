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

If SMTP not configured: OTP is returned in API response (dev mode).
"""
from __future__ import annotations
import os
import smtplib
import random
import string
import hashlib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger

SMTP_EMAIL    = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_HOST     = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "587"))
OTP_EXPIRY_MINUTES = 15


def generate_otp(length: int = 6) -> str:
    """Generate a 6-digit numeric OTP."""
    return "".join(random.choices(string.digits, k=length))


def hash_otp(otp: str) -> str:
    """Hash the OTP before storing (SHA-256)."""
    return hashlib.sha256(otp.encode()).hexdigest()


def verify_otp(plain_otp: str, stored_hash: str) -> bool:
    """Check if plain OTP matches stored hash."""
    return hash_otp(plain_otp) == stored_hash


def otp_expiry() -> str:
    """Return UTC expiry timestamp (15 minutes from now)."""
    return (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()


def send_otp_email(to_email: str, username: str, otp: str) -> bool:
    """
    Send OTP email via Gmail SMTP.
    Returns True on success, False on failure.
    """
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        logger.warning("[Email] SMTP_EMAIL or SMTP_PASSWORD not set. Cannot send email.")
        return False

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
        Hello <strong>{username}</strong>, we received a request to reset your NeuroCheck password.
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

    try:
        logger.info("[Email] Connecting to {}:{}", SMTP_HOST, SMTP_PORT)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.set_debuglevel(0)
            server.ehlo()
            server.starttls()
            server.ehlo()
            logger.info("[Email] Logging in as {}", SMTP_EMAIL)
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        logger.info("[Email] OTP sent successfully to {}", to_email)
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error("[Email] AUTH FAILED — wrong email/password or App Password not set. Error: {}", e)
        return False
    except smtplib.SMTPConnectError as e:
        logger.error("[Email] CONNECTION FAILED to {}:{} — {}", SMTP_HOST, SMTP_PORT, e)
        return False
    except smtplib.SMTPException as e:
        logger.error("[Email] SMTP error: {}", e)
        return False
    except Exception as e:
        logger.error("[Email] Unexpected error sending to {}: {}", to_email, e)
        return False
