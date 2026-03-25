"""TOVA UI database models (greenfield Django ORM)."""

from __future__ import annotations

import uuid

from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("email required")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_admin", True)
        return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    """Email-based user (no username)."""

    username = None
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField("email address", unique=True)
    name = models.CharField(max_length=255, blank=True)
    auth_source = models.CharField(max_length=50, blank=True)
    is_admin = models.BooleanField(default=False)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list[str] = []

    objects = UserManager()

    class Meta:
        db_table = "web_users"
        ordering = ["-date_joined"]


class UserConfig(models.Model):
    user = models.OneToOneField(User, models.CASCADE, primary_key=True, related_name="config_row")
    config = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "web_user_configs"


class AuditLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    actor = models.ForeignKey(User, models.SET_NULL, null=True, blank=True, related_name="audit_actions")
    action = models.CharField(max_length=64)
    target_type = models.CharField(max_length=32, blank=True, null=True)
    target_id = models.CharField(max_length=255, blank=True, null=True)
    details = models.CharField(max_length=1024, blank=True, null=True)

    class Meta:
        db_table = "web_audit_logs"
        ordering = ["-created_at"]


class ChatMessage(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, models.CASCADE, null=True, blank=True, related_name="chat_messages")
    model_id = models.CharField(max_length=255, db_index=True)
    role = models.CharField(max_length=16)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "web_chat_messages"
        indexes = [
            models.Index(fields=["user", "model_id"], name="ix_web_chat_user_model"),
        ]
        ordering = ["created_at"]
