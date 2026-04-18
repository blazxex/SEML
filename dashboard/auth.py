import bcrypt
from dashboard.supabase_client import get_client


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_user(username: str, password: str) -> dict | None:
    db = get_client()
    existing = db.table("users").select("id").eq("username", username).execute()
    if existing.data:
        return None  # already exists
    hashed = hash_password(password)
    result = db.table("users").insert({"username": username, "password_hash": hashed}).execute()
    return result.data[0] if result.data else None


def authenticate(username: str, password: str) -> bool:
    db = get_client()
    result = db.table("users").select("password_hash").eq("username", username).execute()
    if not result.data:
        return False
    return check_password(password, result.data[0]["password_hash"])
