"""Django admin registration for TOVA web models."""

from __future__ import annotations

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _

from web.models import AuditLog, ChatMessage, User, UserConfig

# Branding (templates/admin/base_site.html uses these)
admin.site.site_header = _("TOVA")
admin.site.site_title = _("TOVA Admin")
admin.site.index_title = _("TOVA site administration")


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    ordering = ("email",)
    list_display = ("email", "name", "is_staff", "is_superuser", "is_admin", "is_active", "date_joined")
    list_filter = ("is_staff", "is_superuser", "is_active", "is_admin", "auth_source")
    search_fields = ("email", "name")
    readonly_fields = ("date_joined", "last_login")
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        (_("Personal info"), {"fields": ("name", "auth_source")}),
        (
            _("Permissions"),
            {"fields": ("is_active", "is_staff", "is_superuser", "is_admin", "groups", "user_permissions")},
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "password1", "password2"),
            },
        ),
    )


@admin.register(UserConfig)
class UserConfigAdmin(admin.ModelAdmin):
    list_display = ("user", "updated_at")
    search_fields = ("user__email",)
    readonly_fields = ("updated_at",)


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "actor", "action", "target_type", "target_id")
    list_filter = ("action", "target_type")
    search_fields = ("details", "target_id", "actor__email")
    readonly_fields = ("created_at", "actor", "action", "target_type", "target_id", "details")
    date_hierarchy = "created_at"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return request.user.is_active and request.user.is_staff

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user", "model_id", "role")
    list_filter = ("role", "model_id")
    search_fields = ("content", "user__email", "model_id")
    readonly_fields = ("created_at", "user", "model_id", "role", "content")
    date_hierarchy = "created_at"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return request.user.is_active and request.user.is_staff

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser
