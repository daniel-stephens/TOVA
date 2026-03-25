"""Django settings for the TOVA web UI."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from django.contrib.messages import constants as msg_constants
from django.urls import reverse_lazy

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY") or "change-me-in-production"

DEBUG = os.getenv("DJANGO_DEBUG", "false").lower() == "true"

ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",") if h.strip()
]


def _database_from_env():
    raw = (os.getenv("DATABASE_URL") or "").strip()
    if not raw:
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    url = raw
    if url.startswith("postgresql+psycopg2://"):
        url = "postgresql://" + url[len("postgresql+psycopg2://") :]
    if url.startswith("postgresql://"):
        parsed = urlparse(url)
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": (parsed.path or "").lstrip("/") or "",
            "USER": parsed.username or "",
            "PASSWORD": parsed.password or "",
            "HOST": parsed.hostname or "",
            "PORT": str(parsed.port or 5432),
        }
    return {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }


DATABASES = {"default": _database_from_env()}

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "web.apps.WebConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "web.middleware.TovaUserMiddleware",
]

ROOT_URLCONF = "tova_site.urls"

# Repo checkout: templates live next to `ui/` (TOVA/templates/). Docker: COPY templates /app/templates → BASE_DIR/templates.
_template_dirs: list[Path] = []
for _tpl in (BASE_DIR / "templates", BASE_DIR.parent / "templates"):
    if _tpl.is_dir() and _tpl.resolve() not in {p.resolve() for p in _template_dirs}:
        _template_dirs.append(_tpl)

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": _template_dirs,
        "APP_DIRS": True,
        "OPTIONS": {
            "builtins": ["web.templatetags.web_urls"],
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "web.context_processors.template_extras",
            ],
        },
    },
]

WSGI_APPLICATION = "wsgi.application"

AUTH_USER_MODEL = "web.User"

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
]

LOGIN_URL = reverse_lazy("web:login")
LOGIN_REDIRECT_URL = reverse_lazy("web:home")
LOGOUT_REDIRECT_URL = reverse_lazy("web:login")

APPEND_SLASH = False

SESSION_ENGINE = "django.contrib.sessions.backends.db"
SESSION_COOKIE_NAME = "sessionid"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = os.getenv("USE_HTTPS", "false").lower() == "true"
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_SAVE_EVERY_REQUEST = False

STATIC_URL = "/static/"
_REPO_STATIC = BASE_DIR.parent / "static"
STATICFILES_DIRS = []
if _REPO_STATIC.is_dir():
    STATICFILES_DIRS.append(_REPO_STATIC)
elif (BASE_DIR / "static").is_dir():
    STATICFILES_DIRS.append(BASE_DIR / "static")

# Collected for production (Gunicorn + WhiteNoise); see docker/entrypoint-ui.sh
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

CSRF_TRUSTED_ORIGINS = [
    o.strip()
    for o in os.getenv("CSRF_TRUSTED_ORIGINS", "").split(",")
    if o.strip()
]

MESSAGE_TAGS = {
    msg_constants.DEBUG: "secondary",
    msg_constants.INFO: "info",
    msg_constants.SUCCESS: "success",
    msg_constants.WARNING: "warning",
    msg_constants.ERROR: "danger",
}
