"""URL routes for the TOVA web UI."""

from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from web import views
from web.endpoints.chat import (
    chat,
    chat_stream,
    chat_llm_options,
    chat_openai_key_status,
    get_chat_messages,
)

app_name = "web"

_x = csrf_exempt

urlpatterns = [
    path("signup", views.signup, name="signup"),
    path("login", views.login, name="login"),
    path("login/okta", views.login_okta, name="login_okta"),
    path("auth/okta/callback", views.auth_okta_callback, name="auth_okta_callback"),
    path("logout", views.logout, name="logout"),
    path("staff", views.admin_page, name="admin_page"),
    path(
        "staff/users/<str:user_id>/toggle-admin",
        _x(views.admin_toggle_user),
        name="admin_toggle_user",
    ),
    path("staff/users", _x(views.admin_create_user), name="admin_create_user"),
    path(
        "staff/users/<str:user_id>/delete",
        _x(views.admin_delete_user),
        name="admin_delete_user",
    ),
    path("staff/corpora", views.admin_list_corpora, name="admin_list_corpora"),
    path(
        "staff/corpora/<str:corpus_id>/delete",
        _x(views.admin_delete_corpus),
        name="admin_delete_corpus",
    ),
    path("staff/models", views.admin_list_models, name="admin_list_models"),
    path(
        "staff/models/<str:model_id>/delete",
        _x(views.admin_delete_model),
        name="admin_delete_model",
    ),
    path("staff/stats", views.admin_stats, name="admin_stats"),
    path("staff/audit", views.admin_audit, name="admin_audit"),
    path("", views.home, name="home"),
    path("check-backend", views.check_backend, name="check_backend"),
    path("terms", views.terms, name="terms"),
    path("privacy", views.privacy, name="privacy"),
    path("llm/ui-config", views.llm_ui_config, name="llm_ui_config"),
    path("get-llm-config", views.get_llm_config, name="get_llm_config"),
    path("save-llm-config", _x(views.save_llm_config), name="save_llm_config"),
    path("api/user-config", _x(views.api_user_config), name="api_user_config"),
    path(
        "api/user-config/overrides",
        views.api_get_user_config_overrides,
        name="api_get_user_config_overrides",
    ),
    path("api/user-config/reset", _x(views.api_reset_user_config), name="api_reset_user_config"),
    path("load-data-page/", views.load_data_page, name="load_data_page"),
    path("load-corpus-page/", views.load_corpus_page, name="load_corpus_page"),
    path("data/create/corpus/", _x(views.create_corpus), name="create_corpus"),
    path("data/corpus/add_model", _x(views.add_model_to_corpus), name="add_model_to_corpus"),
    path("delete-corpus/", _x(views.delete_corpus), name="delete_corpus"),
    path("data/create/dataset/", _x(views.create_dataset), name="create_dataset"),
    path("training/", views.training_page_get, name="training_page_get"),
    path("get-training-session", views.get_training_session, name="get_training_session"),
    path(
        "train/corpus/<str:corpus_id>/tfidf/",
        _x(views.train_corpus_tfidf_route),
        name="train_corpus_tfidf",
    ),
    path(
        "corpus/<str:corpus_id>/tfidf/",
        views.get_tfidf_data,
        name="get_tfidf_data",
    ),
    path("training/start", _x(views.training_start), name="training_start"),
    path("status/jobs/<str:job_id>", views.get_status, name="get_status_jobs"),
    path("status/", views.get_status, name="get_status_root"),
    path("model", views.loadModel, name="loadModel"),
    path("getUniqueCorpusNames", views.get_unique_corpus_names, name="get_unique_corpus_names"),
    path("getAllCorpora", views.getAllCorpora, name="getAllCorpora"),
    path("getCorpus/<str:corpus_id>", views.get_corpus, name="get_corpus"),
    path("model-registry", views.get_model_registry, name="get_model_registry"),
    path("trained-models", views.trained_models, name="trained_models"),
    path("get-trained-models", views.get_trained_models, name="get_trained_models"),
    path("get-models-names", views.get_model_names, name="get_model_names"),
    path("delete-model", _x(views.delete_model), name="delete_model"),
    path("dashboard", _x(views.dashboard), name="dashboard"),
    path("get-dashboard-data", _x(views.proxy_dashboard_data), name="get_dashboard_data"),
    path("text-info", _x(views.text_info), name="text_info"),
    path("infer-text", _x(views.infer_text), name="infer_text"),
    path("save-settings", _x(views.save_settings), name="save_settings"),
    path("api/chat-openai-key-status", chat_openai_key_status, name="chat_openai_key_status"),
    path("api/chat-llm-options", chat_llm_options, name="chat_llm_options"),
    path("api/chat/messages", get_chat_messages, name="get_chat_messages"),
    path("api/chat", _x(chat), name="chat"),
    path("api/chat/stream", _x(chat_stream), name="chat_stream"),
    path(
        "api/models/<str:model_id>/topics/<int:topic_id>/rename",
        _x(views.rename_topic),
        name="rename_topic",
    ),
    path(
        "api/models/<str:model_id>/topics/renames",
        views.get_topic_renames,
        name="get_topic_renames",
    ),
    path("__django_health__/", views.django_health, name="django_health"),
]
