#!/usr/bin/env python3
"""Django management entrypoint for the TOVA UI package."""

import os
import sys
from pathlib import Path

# Monorepo: ui/…/manage.py → ../src. Docker: /app/manage.py → /app/src.
_base = Path(__file__).resolve().parent
for _src in (_base / "src", _base.parent / "src"):
    if _src.is_dir():
        s = str(_src)
        if s not in sys.path:
            sys.path.insert(0, s)


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tova_site.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Is it installed and available on your PYTHONPATH?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
