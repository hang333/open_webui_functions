"""
title: Reasoning Effort
id: reasoning_effort_toggle
description: Chat toggle that sets the reasoning effort. Install once per level (e.g. xhigh, low).
author: hang333
author_url: https://github.com/hang333
funding_url: https://github.com/hang333/open_webui_functions
license: MIT
version: 1.0.0
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, Field

# Levels from least to most thinking. When several of these toggles are on, the most thinking wins.
LEVELS: Final[tuple[str, ...]] = ('minimal', 'low', 'medium', 'high', 'xhigh')
# Key in the request's __metadata__ that remembers which level a toggle of this kind already set.
_METADATA_KEY: Final = 'reasoning_effort_toggle'
# Gauge icon (stroke uses the text color, like Open WebUI's own icons).
ICON: Final = (
    'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAy'
    'NCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBz'
    'dHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJtMTIgMTQgNC00Ii8+PHBhdGggZD0iTTMuMzQgMTlhMTAgMTAgMCAxIDEgMTcu'
    'MzIgMCIvPjwvc3ZnPg=='
)


class Filter:
    """
    While the toggle is on, the request's `reasoning_effort` is set to `LEVEL`. It overrides the chat's own
    parameter (Open WebUI applies model and chat parameters before inlet filters run).

    `reasoning_effort` is the OpenAI parameter, so the toggle works for any model that understands it:
    Gemini Manifold pipes map it to each model's nearest supported thinking level, OpenAI-compatible
    connections pass it on. Models without reasoning support may ignore or reject it.
    """

    class Valves(BaseModel):
        LEVEL: Literal['minimal', 'low', 'medium', 'high', 'xhigh'] = Field(
            default='xhigh',
            description='The reasoning effort sent while this toggle is on.',
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True  # Shown as an on/off switch in the chat input.
        self.icon = ICON

    def inlet(self, body: dict[str, Any], __metadata__: dict[str, Any] | None = None) -> dict[str, Any]:
        level = self.valves.LEVEL
        state = __metadata__ if isinstance(__metadata__, dict) else {}
        previous = state.get(_METADATA_KEY)
        # Another toggle of this kind may have run first (filters run by priority, then id): keep the higher level.
        if previous not in LEVELS or LEVELS.index(level) > LEVELS.index(previous):
            body['reasoning_effort'] = level
            state[_METADATA_KEY] = level
        return body
