#!/usr/bin/env python3
"""
Grant (or revoke) admin rights — the ONLY way to create an admin.

    python make_admin.py USERNAME            # grant (asks you to retype the username)
    python make_admin.py USERNAME --revoke
    python make_admin.py USERNAME --yes      # non-interactive (scripts)

Run it inside the container / on the server with the same environment as the app
(JWT_SECRET, DATA_DIR/NEUROCHECK_DB, DATA_ENCRYPTION_KEY …), e.g.:
    docker compose exec api python make_admin.py alice

The account must already exist. There is deliberately NO environment-variable or
startup-time auto-grant: anyone can register any free username, so a name-based
bootstrap is a privilege-escalation hole. Every grant/revoke is written to the
append-only admin_audit table.
"""
import argparse
import getpass
import sys

from api.database import init_all, set_admin, get_user_by_username, audit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("username")
    ap.add_argument("--revoke", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    init_all()
    user = get_user_by_username(args.username)
    if not user:
        print(f"User '{args.username}' not found. Register first, then run this.")
        return 1
    verb = "REVOKE admin from" if args.revoke else "GRANT admin to"
    if not args.yes:
        again = input(f"About to {verb} '{user['username']}' (id={user['id']}, {user['email']}). "
                      f"Retype the username to confirm: ").strip()
        if again.lower() != user["username"].lower():
            print("Confirmation mismatch — aborted.")
            return 1
    ok = set_admin(user["username"], not args.revoke)
    audit(None, f"cli:{getpass.getuser()}", "CLI_REVOKE_ADMIN" if args.revoke else "CLI_GRANT_ADMIN",
          f"user:{user['id']}", "local-cli")
    print(f"{'OK' if ok else 'FAILED'}: {verb} '{user['username']}'")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
