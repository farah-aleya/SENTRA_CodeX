"""SENTRA accounts: registration, login and signed session cookies.

Two roles:
  resident   registers with a house number; sees only guard-confirmed alerts for their
             house (and street-wide alerts), and controls Away Mode for that house
  guard      registers with the guard invite code; sees every case the agent sends for
             review and approves or dismisses it

Passwords are stored as PBKDF2-SHA256 hashes (200,000 rounds, random salt).
Sessions are HMAC-signed tokens in an HttpOnly cookie. The signing secret comes from
the SENTRA_SECRET environment variable, or config/secret.key (created once, git-ignored).
"""
import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_FILE = ROOT / "config" / "secret.key"
SESSION_HOURS = 12
ROLES = ("resident", "guard")


def _secret() -> bytes:
    env = os.environ.get("SENTRA_SECRET")
    if env:
        return env.encode()
    if not SECRET_FILE.exists():
        SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECRET_FILE.write_text(secrets.token_hex(32))
    return SECRET_FILE.read_text().strip().encode()


def guard_code() -> str:
    """Invite code a guard must enter to register. Set SENTRA_GUARD_CODE in production."""
    return os.environ.get("SENTRA_GUARD_CODE", "codex-guard")


def hash_password(pw: str, salt: bytes = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    return salt.hex(), hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000).hex()


def check_password(pw: str, salt_hex: str, hash_hex: str) -> bool:
    return hmac.compare_digest(hash_password(pw, bytes.fromhex(salt_hex))[1], hash_hex)


def make_token(user_id: int) -> str:
    exp = int(time.time()) + SESSION_HOURS * 3600
    body = f"{user_id}.{exp}"
    sig = hmac.new(_secret(), body.encode(), hashlib.sha256).digest()
    return body + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")


def read_token(token: str):
    """Return the user id if the token is valid and not expired, else None."""
    try:
        uid, exp, sig = token.split(".")
        body = f"{uid}.{exp}"
        good = base64.urlsafe_b64encode(hmac.new(_secret(), body.encode(), hashlib.sha256).digest()).decode().rstrip("=")
        if hmac.compare_digest(good, sig) and int(exp) > time.time():
            return int(uid)
    except (ValueError, AttributeError):
        pass
    return None


def validate_registration(username, password, role, house, code):
    """Return an error message, or None if the form is valid."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username or ""):
        return "Username: 3 to 32 letters, numbers, dot, dash or underscore."
    if len(password or "") < 8:
        return "Password must be at least 8 characters."
    if role not in ROLES:
        return "Choose resident or guard."
    if role == "resident" and not re.fullmatch(r"[A-Za-z0-9-]{1,10}", str(house or "")):
        return "Enter your house number."
    if role == "guard" and not hmac.compare_digest(str(code or ""), guard_code()):
        return "Wrong guard invite code."
    return None
