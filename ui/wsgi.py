"""Django WSGI entrypoint for the TOVA UI. Production: ``gunicorn wsgi:application``."""

import os
import sys
from pathlib import Path

# Match manage.py: make `tova` importable (Docker /app/src or checkout TOVA/src).
_base = Path(__file__).resolve().parent
for _src in (_base / "src", _base.parent / "src"):
    if _src.is_dir():
        s = str(_src)
        if s not in sys.path:
            sys.path.insert(0, s)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tova_site.settings")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
