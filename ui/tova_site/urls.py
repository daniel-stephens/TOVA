"""Root URL configuration for the TOVA web UI."""

from django.contrib import admin
from django.urls import include, path

from web import admin_views

urlpatterns = [
    # Custom Django-admin pages (TOVA console)
    path(
        "admin/tova/",
        admin.site.admin_view(admin_views.tova_console_root),
        name="tova_admin_root",
    ),
    path(
        "admin/tova/corpora/",
        admin.site.admin_view(admin_views.tova_corpora),
        name="tova_admin_corpora",
    ),
    path(
        "admin/tova/corpora/<str:corpus_id>/delete/",
        admin.site.admin_view(admin_views.tova_delete_corpus),
        name="tova_admin_corpora_delete",
    ),
    path(
        "admin/tova/dashboards/",
        admin.site.admin_view(admin_views.tova_dashboards),
        name="tova_admin_dashboards",
    ),
    path(
        "admin/tova/dashboards/<str:model_id>/delete/",
        admin.site.admin_view(admin_views.tova_delete_dashboard),
        name="tova_admin_dashboards_delete",
    ),
    # Default Django admin
    path("admin/", admin.site.urls),
    path("", include("web.urls")),
]
