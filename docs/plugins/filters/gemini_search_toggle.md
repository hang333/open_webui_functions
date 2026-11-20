# `gemini_search_toggle.py` - Google Search toggle

A chat toggle for Gemini's native Google Search grounding with the Gemini Manifold pipes (the original and the Plus edition).

## How it works

- While the toggle is on, the filter sets the `google_search_tool` feature. This is the same flag the companion filter derives from Open WebUI's web search toggle, so **Open WebUI's own web search (and a search engine) does not have to be enabled**.
- The pipe then adds Google Search to the Gemini request, for models whose entry in the model table supports search grounding. Citations and search suggestions are shown by the companion filter as usual.
- Other providers and models that cannot ground ignore the flag; Open WebUI drops `body['features']` before calling OpenAI-compatible connections.

## Installation

Install it with the function id `gemini_search_toggle` (the file name), enable it and make it global, or attach it to the Gemini models only. URL Context and Google Maps grounding have their own toggles (`gemini_url_context_toggle`, `gemini_map_grounding_toggle`).
