"""
Run this once to grant admin access to your account:
    python make_admin.py YOUR_USERNAME

On HuggingFace — add this as a startup command or run it via the Space terminal.
"""
import sys, os
os.environ.setdefault("NEUROCHECK_DB", "neurocheck.db")

from api.database import init_db, set_admin, migrate_add_admin_column, get_user_by_username

username = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ADMIN_USERNAME", "")
if not username:
    print("Usage: python make_admin.py YOUR_USERNAME")
    sys.exit(1)

init_db()
migrate_add_admin_column()

user = get_user_by_username(username)
if not user:
    print(f"User '{username}' not found. Register first at /chat-ui then run this.")
    sys.exit(1)

ok = set_admin(username, True)
print(f"{'✅' if ok else '❌'} Admin access {'granted to' if ok else 'failed for'} '{username}'")
