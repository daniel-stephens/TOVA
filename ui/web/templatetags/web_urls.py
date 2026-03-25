"""Named URL helper: ``{% url_for 'view_name' %}`` (and ``static`` with ``filename=``)."""

from __future__ import annotations

from django import template
from django.templatetags.static import static
from django.urls import NoReverseMatch, reverse

register = template.Library()


@register.simple_tag
def url_for(endpoint: str, **kwargs) -> str:
    if endpoint == "static":
        filename = kwargs.get("filename") or ""
        return static(filename)
    try:
        return reverse(f"web:{endpoint}", kwargs=kwargs)
    except NoReverseMatch:
        try:
            return reverse(endpoint, kwargs=kwargs)
        except NoReverseMatch:
            # Avoid hard-failing template rendering when a route is renamed.
            # Broken links are easier to diagnose than a full 500 error.
            return "#"
