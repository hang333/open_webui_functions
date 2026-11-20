"""
title: Thinking
id: reasoning_effort_toggle
description: Thinking toggle with one level at a time; click its chat input badge to pick the level.
author: hang333
author_url: https://github.com/hang333
funding_url: https://github.com/hang333/open_webui_functions
license: MIT
version: 2.0.0
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, Field

# The levels offered in the picker (one at a time) and the `reasoning_effort` each one sends.
LEVELS: Final[dict[str, str]] = {
    '极速': 'minimal',
    '快速': 'low',
    '标准': 'medium',
    '深度': 'high',
    '极限': 'xhigh',
}
Level = Literal['极速', '快速', '标准', '深度', '极限']
DEFAULT_LEVEL: Final[Level] = '深度'

# Brain icon (stroke uses the text color, like Open WebUI's own icons).
ICON: Final = (
    'data:image/svg+xml;base64,'
    'PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tl'
    'PSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5k'
    'Ij48cGF0aCBkPSJNMTIgNWEzIDMgMCAxIDAtNS45OTcuMTI1IDQgNCAwIDAgMC0yLjUyNiA1Ljc3IDQgNCAwIDAgMCAuNTU2IDYuNTg4'
    'QTQgNCAwIDEgMCAxMiAxOFoiLz48cGF0aCBkPSJNMTIgNWEzIDMgMCAxIDEgNS45OTcuMTI1IDQgNCAwIDAgMSAyLjUyNiA1Ljc3IDQg'
    'NCAwIDAgMS0uNTU2IDYuNTg4QTQgNCAwIDEgMSAxMiAxOFoiLz48cGF0aCBkPSJNMTUgMTNhNC41IDQuNSAwIDAgMS0zLTQgNC41IDQu'
    'NSAwIDAgMS0zIDQiLz48cGF0aCBkPSJNMTcuNTk5IDYuNWEzIDMgMCAwIDAgLjM5OS0xLjM3NSIvPjxwYXRoIGQ9Ik02LjAwMyA1LjEy'
    'NUEzIDMgMCAwIDAgNi40MDEgNi41Ii8+PHBhdGggZD0iTTMuNDc3IDEwLjg5NmE0IDQgMCAwIDEgLjU4NS0uMzk2Ii8+PHBhdGggZD0i'
    'TTE5LjkzOCAxMC41YTQgNCAwIDAgMSAuNTg1LjM5NiIvPjxwYXRoIGQ9Ik02IDE4YTQgNCAwIDAgMS0xLjk2Ny0uNTE2Ii8+PHBhdGgg'
    'ZD0iTTE5Ljk2NyAxNy40ODRBNCA0IDAgMCAxIDE4IDE4Ii8+PC9zdmc+'
)


class Filter:
    """
    One thinking toggle with one level at a time. While it is on, the request's `reasoning_effort` is set to
    the user's chosen level; it overrides the chat's own parameter (Open WebUI applies model and chat
    parameters before inlet filters run).

    With "Show Input Selections" on, the active toggle is a badge in the chat input. Because the filter has user
    valves, clicking the badge opens the level picker (a single choice, so levels exclude each other) and the
    badge's x turns thinking control off for the chat. Make it a default filter of a model to have it on in
    every new chat.

    `reasoning_effort` is the OpenAI parameter: Gemini Manifold pipes map it to each model's nearest supported
    thinking level, OpenAI-compatible connections pass it on (models without reasoning support may reject it).
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        LEVEL: Level = Field(
            default=DEFAULT_LEVEL,
            description='Thinking level: 极速 (minimal), 快速 (low), 标准 (medium), 深度 (high) or 极限 (xhigh).'
            ' Models use their nearest supported level.',
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True  # Shown as an on/off switch in the chat input.
        self.icon = ICON

    def inlet(self, body: dict[str, Any], __user__: dict[str, Any] | None = None) -> dict[str, Any]:
        user_valves = (__user__ or {}).get('valves')
        level = user_valves.get('LEVEL') if isinstance(user_valves, dict) else getattr(user_valves, 'LEVEL', None)
        body['reasoning_effort'] = LEVELS.get(level or '', LEVELS[DEFAULT_LEVEL])
        return body
