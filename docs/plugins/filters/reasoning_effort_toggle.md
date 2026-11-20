# `reasoning_effort_toggle.py` - Thinking toggle

One chat toggle for the thinking level, with one level at a time. Open WebUI's own model controls (a Thinking menu next to the model selector) cannot be added to pipe models, and separate toggles per level cannot exclude each other, so this filter puts the level in its user valves instead.

## How it works

- While the toggle is on, `body['reasoning_effort']` is set to the user's level. It overrides the chat's own `reasoning_effort` parameter: Open WebUI applies model and chat parameters before inlet filters run.
- The level is a user valve with a single choice:

  | Level | `reasoning_effort` |
  | --- | --- |
  | 极速 | `minimal` |
  | 快速 | `low` |
  | 标准 | `medium` |
  | 深度 (default) | `high` |
  | 极限 | `xhigh` |

- Gemini Manifold (Plus) maps the value to each model's nearest supported thinking level (e.g. `xhigh` becomes `high` on models without `xhigh`, `minimal` becomes `low` on Gemini 4). OpenAI-compatible connections pass the parameter on; models without reasoning support may ignore or reject it.

## Using it from the chat input

1. Turn on **Settings → Interface → Show Input Selections**. The active toggle then shows as a badge in the chat input.
2. Click the badge to open the level picker (Open WebUI opens a filter's user valves when its badge is clicked). Hover the badge and click its x to turn the toggle off for the chat; the model then uses its own default.
3. To have thinking on in every new chat, make the filter a **default filter** of the models (model editor → Filters → Default Filters).

## Installation

Install it with the function id `reasoning_effort_toggle` (the file name), enable it and make it global (or attach it to the models you want).
