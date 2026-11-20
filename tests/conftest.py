"""Shared test setup: Open WebUI stand-ins are registered before any plugin is imported."""

from tests import owui_stubs

owui_stubs.install()
