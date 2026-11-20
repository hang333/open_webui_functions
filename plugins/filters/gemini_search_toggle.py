"""
title: Google Search
id: gemini_search_toggle
description: Chat toggle for Gemini's native Google Search grounding (Gemini Manifold pipes).
author: hang333
author_url: https://github.com/hang333
funding_url: https://github.com/hang333/open_webui_functions
license: MIT
version: 1.0.0
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel

# Globe icon (stroke uses the text color, like Open WebUI's own icons).
ICON: Final = (
    'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAy'
    'NCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBz'
    'dHJva2UtbGluZWpvaW49InJvdW5kIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxwYXRoIGQ9Ik0xMiAyYTE0LjUgMTQu'
    'NSAwIDAgMCAwIDIwIDE0LjUgMTQuNSAwIDAgMCAwLTIwIi8+PHBhdGggZD0iTTIgMTJoMjAiLz48L3N2Zz4='
)


class Filter:
    """
    While the toggle is on, Gemini Manifold pipes add Google Search grounding to the request: the toggle sets
    the `google_search_tool` feature, the same flag the companion filter derives from Open WebUI's web search
    toggle. Open WebUI's own web search therefore does not have to be enabled. Models the pipe's model table
    marks as unable to ground, and other providers, ignore the flag.
    """

    class Valves(BaseModel):
        pass

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True  # Shown as an on/off switch in the chat input.
        self.icon = ICON

    def inlet(self, body: dict[str, Any], __metadata__: dict[str, Any] | None = None) -> dict[str, Any]:
        # Open WebUI snapshots `body['features']` into `__metadata__['features']` before the filters run, and the
        # pipe reads the latter. Both must hold the same dict for the flag to reach the pipe (as in the companion).
        features = body.get('features')
        if not isinstance(features, dict):
            features = {}
            body['features'] = features
        if isinstance(__metadata__, dict):
            __metadata__['features'] = features
        features['google_search_tool'] = True
        return body
