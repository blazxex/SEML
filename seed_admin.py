"""Run once to create the first admin user in Supabase."""
import sys
from dotenv import load_dotenv
load_dotenv()

from dashboard.auth import create_user

username = sys.argv[1] if len(sys.argv) > 1 else "admin"
password = sys.argv[2] if len(sys.argv) > 2 else "changeme"

result = create_user(username, password)
if result:
    print(f"Created user: {username}")
else:
    print(f"User '{username}' already exists.")
