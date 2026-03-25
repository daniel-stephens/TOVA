"""Django WSGI entrypoint for the TOVA UI. Production: ``gunicorn wsgi:application``."""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tova_site.settings")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
