from __future__ import annotations

import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction


class Command(BaseCommand):
    help = "Create or update the env-configured superuser by email."

    def handle(self, *args, **options):
        email = (os.getenv("DJANGO_SUPERUSER_EMAIL") or "").strip().lower()
        password = (os.getenv("DJANGO_SUPERUSER_PASSWORD") or "").strip()
        if not email or not password:
            raise CommandError("Set DJANGO_SUPERUSER_EMAIL and DJANGO_SUPERUSER_PASSWORD.")

        User = get_user_model()
        with transaction.atomic():
            user = User.objects.filter(email=email).first()
            created = user is None
            if created:
                user = User(email=email, name=email, auth_source="local")

            user.is_active = True
            user.is_staff = True
            user.is_superuser = True
            if hasattr(user, "is_admin"):
                user.is_admin = True
            user.set_password(password)
            user.save()

        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} superuser for {email}."))
