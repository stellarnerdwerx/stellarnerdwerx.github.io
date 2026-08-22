"""One-time (or as-needed) CLI to create/update a login for the movie site.

Usage:
  python manage_users.py add <username> "<Display Name>"

Prompts for a password with hidden input and stores its hash in ratings.db.
Run locally to test, then run again on the PythonAnywhere console to set up
the real login there -- passwords never need to touch git or chat history.
"""

import getpass
import sys

from ratings_db import create_user, init_db


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "add":
        print('Usage: python manage_users.py add <username> "<Display Name>"')
        sys.exit(1)

    _, _, username, display_name = sys.argv
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if not password:
        print("Password can't be empty.")
        sys.exit(1)
    if password != confirm:
        print("Passwords didn't match.")
        sys.exit(1)

    init_db()
    create_user(username, display_name, password)
    print(f"Saved login for {display_name} ({username}).")


if __name__ == "__main__":
    main()
