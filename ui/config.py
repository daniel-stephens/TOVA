# config.py
import os

# Prefer explicit DATABASE_URL; otherwise derive from standard Postgres envs.
_db_url = os.getenv("DATABASE_URL")
if not _db_url:
    db_user = os.getenv("POSTGRES_USER", "user")
    db_pass = os.getenv("POSTGRES_PASSWORD", "password")
    db_name = os.getenv("POSTGRES_DB", "mydb")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    _db_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

SQLALCHEMY_DATABASE_URI = _db_url
SQLALCHEMY_TRACK_MODIFICATIONS = False
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_SAMESITE = "Lax"