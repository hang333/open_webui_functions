# `reasoning_effort_toggle.py` - Reasoning Effort toggle

A chat toggle that sets the request's `reasoning_effort` while it is switched on. Open WebUI's own model controls (a Thinking menu next to the model selector) cannot be added to pipe models, so this filter gives Gemini Manifold models, and any other model, a visible switch instead.

## How it works

- While the toggle is on, `body['reasoning_effort']` is set to the `LEVEL` valve (`minimal`, `low`, `medium`, `high` or `xhigh`).
- It overrides the chat's own `reasoning_effort` parameter: Open WebUI applies model and chat parameters before inlet filters run.
- If several of these toggles are on at once, the highest level wins (they share the request's `__metadata__`).
- Gemini Manifold (Plus) maps the level to each model's nearest supported thinking level (e.g. `xhigh` becomes `high` on models without `xhigh`). OpenAI-compatible connections pass the parameter on; models without reasoning support may ignore or reject it.

## Installation

Install the same file once per level, each with its own function id and name, then enable it and make it global (or attach it to the models you want):

| Name | Function id | `LEVEL` valve |
| --- | --- | --- |
| 深度思考 | `reasoning_effort_deep` | `xhigh` |
| 快速思考 | `reasoning_effort_fast` | `low` |

The toggles appear in the chat input's integrations menu; with both off, the model's default (or the pipe's `THINKING_CONFIG_RULES`) applies.
