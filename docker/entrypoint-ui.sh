#!/bin/sh
set -e
cd /app
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-tova_site.settings}"
python manage.py migrate --noinput
python manage.py collectstatic --noinput
exec gunicorn -w 2 -b 0.0.0.0:8080 wsgi:application
