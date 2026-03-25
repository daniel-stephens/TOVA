"""Optional Okta OAuth (Authlib Django client)."""

from __future__ import annotations

import os

try:
    from authlib.integrations.django_client import OAuth

    oauth = OAuth()
except ImportError:
    oauth = None  # type: ignore[assignment]

_registered = False


def ensure_registered() -> None:
    global _registered
    if _registered or oauth is None:
        return
    cid = os.getenv("OKTA_CLIENT_ID")
    iss = os.getenv("OKTA_ISSUER")
    if not cid or not iss:
        return
    oauth.register(
        name="okta",
        client_id=cid,
        client_secret=os.getenv("OKTA_CLIENT_SECRET"),
        server_metadata_url=f"{iss}/.well-known/openid-configuration",
        client_kwargs={"scope": "openid profile email"},
    )
    _registered = True


def get_client():
    if oauth is None:
        return None
    ensure_registered()
    try:
        return oauth.create_client("okta")
    except Exception:
        return None
