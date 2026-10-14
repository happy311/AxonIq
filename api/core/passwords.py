"""
AxonIQ — password policy + hashing.

Hashing: bcrypt over a SHA-256 pre-hash (versioned prefix) so that passwords up
to 128 chars work (bcrypt itself only uses the first 72 bytes and bcrypt>=5
raises on longer input). Legacy hashes (plain bcrypt) still verify and are
upgraded on next successful login (see needs_rehash()).
"""
import base64
import hashlib
import re

import bcrypt as _bcrypt

MIN_LEN = 10
MAX_LEN = 128
_PREFIX = "bcsha256$"

# Most frequently breached passwords/bases (lower-case). Matching is done on the
# lower-cased password AND on a "normalised" form with digits/symbols stripped,
# so "Password123!" and "p@ssw0rd2024" are both caught.
_COMMON = frozenset("""
password passw0rd p@ssword p@ssw0rd pass word123 123456 1234567 12345678 123456789 1234567890
12345678910 0123456789 1q2w3e4r 1q2w3e4r5t 1qaz2wsx qwerty qwerty123 qwertyuiop qwertyuiop123
qwertyui asdfgh asdfghjkl zxcvbnm abc123 abcd1234 abcdefgh abcdefghij iloveyou iloveyou1
letmein welcome welcome1 welcome123 admin admin123 administrator root toor changeme
changeme123 default guest login master monkey dragon football baseball basketball soccer
hockey batman superman spiderman starwars pokemon naruto princess sunshine shadow michael
jennifer jessica ashley daniel charlie thomas hunter ranger buster tigger trustno1 whatever
freedom computer internet secret secret123 hello hello123 hellohello service access
mustang harley cheese summer winter spring autumn flower purple orange yellow silver golden
liverpool chelsea arsenal manchester barcelona realmadrid cricket india india123 password1
password12 password123 password1234 password12345 passw0rd123 pa55word pa55w0rd test test123
testing testing123 temp temp123 user user123 demo demo123 sample qazwsx qazwsxedc zaq12wsx
asdf1234 asdfasdf qwer1234 qweasdzxc 1234qwer 11111111 111111111 1111111111 00000000
000000000 0000000000 aaaaaaaa aaaaaaaaaa 12341234 123123123 123321 654321 666666 7777777
88888888 987654321 9876543210 neurocheck axoniq neurocheck123 axoniq123 multiplesclerosis
doctor doctor123 hospital hospital123 patient patient123 medical medical123 clinic nurse
""".split())

_SEQUENCES = ("0123456789", "abcdefghijklmnopqrstuvwxyz", "qwertyuiopasdfghjklzxcvbnm")


def validate_password_policy(password: str, *, username: str = "", email: str = "") -> None:
    """Raise ValueError (message safe to show the user) if the password is unacceptable."""
    if not isinstance(password, str):
        raise ValueError("Invalid password.")
    if len(password) < MIN_LEN:
        raise ValueError(f"Password must be at least {MIN_LEN} characters.")
    if len(password) > MAX_LEN:
        raise ValueError(f"Password must be at most {MAX_LEN} characters.")
    low = password.lower()
    leet = low.translate(str.maketrans("@0$1!3457", "aosiieast"))
    forms = {low, re.sub(r"[^a-z]", "", low), re.sub(r"[^a-z]", "", leet),
             re.sub(r"\d+\W*$", "", low), re.sub(r"[^a-z]+$", "", leet)}
    if forms & _COMMON:
        raise ValueError("That password is too common. Choose something less guessable.")
    if len(set(low)) <= 3:
        raise ValueError("Password is too repetitive.")
    if any(low in seq or low in seq[::-1] for seq in _SEQUENCES):
        raise ValueError("Password is a simple keyboard/alphabet sequence.")
    uname = (username or "").strip().lower()
    if len(uname) >= 3 and uname in low:
        raise ValueError("Password must not contain your username.")
    local = (email or "").split("@")[0].strip().lower()
    if len(local) >= 4 and local in low:
        raise ValueError("Password must not contain your email name.")


def _prehash(plain: str) -> bytes:
    return base64.b64encode(hashlib.sha256(plain.encode("utf-8")).digest())  # 44 bytes


def hash_password(plain: str) -> str:
    return _PREFIX + _bcrypt.hashpw(_prehash(plain), _bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        if hashed.startswith(_PREFIX):
            return _bcrypt.checkpw(_prehash(plain), hashed[len(_PREFIX):].encode("ascii"))
        # legacy: plain bcrypt, only ever used the first 72 bytes
        return _bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("ascii"))
    except Exception:
        return False


def needs_rehash(hashed: str) -> bool:
    return not hashed.startswith(_PREFIX)


# Pre-computed at import so the unknown-user path costs the same as a real check.
DUMMY_HASH = hash_password("dummy-password-for-timing-equalisation")
