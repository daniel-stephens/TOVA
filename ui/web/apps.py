from django.apps import AppConfig


class WebConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web"
    verbose_name = "TOVA web UI"

    def ready(self):
        # Register Okta OAuth client when env is set
        from web import oauth_okta

        oauth_okta.ensure_registered()
