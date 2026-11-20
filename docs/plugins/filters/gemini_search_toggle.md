# `gemini_search_toggle.py` - Google Search toggle

A chat toggle for Gemini's native Google Search grounding with the Gemini Manifold pipes (the original and the Plus edition).

## How it works

- While the toggle is on, the filter sets the `google_search_tool` feature. This is the same flag the companion filter derives from Open WebUI's web search toggle, so **Open WebUI's own web search (and a search engine) does not have to be enabled**.
- The pipe then adds Google Search to the Gemini request, for models whose entry in the model table supports search grounding. Citations and search suggestions are shown by the companion filter as usual.
- Other providers and models that cannot ground ignore the flag; Open WebUI drops `body['features']` before calling OpenAI-compatible connections.

## Using it from the chat input

With **Settings → Interface → Show Input Selections** on, the active toggle shows as a badge in the chat input; hover it and click its x to turn search off for the chat. To have search on in every new chat, open the model editor → **Filters** and turn on **Start enabled in new chats** under this filter. The model still decides when to search; searches beyond the free allowance are billed (see the Gemini API pricing page).

## Installation

Install it with the function id `gemini_search_toggle` (the file name), enable it and make it global, or attach it to the Gemini models only. URL Context and Google Maps grounding have their own toggles (`gemini_url_context_toggle`, `gemini_map_grounding_toggle`).
