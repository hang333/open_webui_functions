# `gemini_manifold.py` - Detailed Documentation

This document provides a comprehensive overview of the `gemini_manifold.py` Open WebUI plugin.

## Description

This is a manifold pipe function that adds support for Google's Gemini Studio API and Vertex AI into Open WebUI using `google-genai` SDK.

This branch turns the pipe into **Gemini Manifold Plus** (function id `gemini_manifold_plus`, version `2.2.0+plus.1`), based on Gemini Manifold google_genai 2.1.0. It has its own function id, so it can be installed next to the original pipe. A detailed description of the changes, the installation for Open WebUI 0.13.4 and the known limitations (in Simplified Chinese) is in [gemini_manifold_plus/说明.md](gemini_manifold_plus/说明.md).

## Features

Here's a breakdown of implemented and planned features for the Gemini Manifold plugin:

**Implemented Features:**

-   [x] Display thinking summary
-   [x] Thinking budget and thinking levels, fitted to each model (Gemini 4 `low`…`xhigh`, Gemini 3.x levels, Gemini 2.5 budgets); `reasoning_effort` is mapped onto them
-   [x] Reasoning toggle (Reason filter function required, see [it's doc](../filters/gemini_reasoning_toggle.md))
-   [x] Native image generation and editing (image output), with per-model aspect ratios and sizes, the reference image limit and Image Search grounding
-   [x] Document understanding (PDF and plaintext files). (Gemini Manifold Companion >= 1.4.0 filter required, see [it's doc](../filters/gemini_manifold_companion.md))
-   [x] Image input
-   [x] YouTube video input (automatically detects youtube.com and youtu.be URLs in messages)
-   [x] Video input support (other than YouTube URLs)
-   [x] Audio input support
-   [x] Google Files API
-   [x] Grounding with Google Search (Gemini Manifold Companion >= 1.2.0 required), with the Search Suggestions shown as an embed
-   [x] Grounding with Google Maps (Gemini Manifold Companion >= 1.7.0 required), with the Google Maps source attribution after the answer. If you want to toggle this just like reasoning then install [Google Maps Grounding](../../../plugins/filters/gemini_map_grounding_toggle.py) filter function.
-   [x] Display citations in the front-end. (Gemini Manifold Companion >= 1.5.0 required)
-   [x] Permissive safety settings (Gemini Manifold Companion >= 1.3.0 required)
-   [x] Each user can decide to use their own API key.
-   [x] Token usage data (thinking, cached and tool-use tokens included) and cost
-   [x] Code execution tool. (Gemini Manifold Companion >= 1.1.0 required)
-   [x] URL context tool (Gemini Manifold Companion >= 1.5.0 required if you want to see citations in the front-end). If you want to toggle this then install [URL Context](../../../plugins/filters/gemini_url_context_toggle.py) filter function.
-   [x] Streaming and non-streaming responses.
-   [x] Native tool calling through Open WebUI's tool loop, with thought signatures sent back exactly; built-in Google tools combined with functions on Gemini 3 and later
-   [x] Structured output from `response_format` (JSON schema)
-   [x] Media resolution, including per-part `ULTRA_HIGH` for images on Gemini 3 and later
-   [x] File Search tool (store names in a valve)
-   [x] Deep Research agents and Omni video generation (Interactions API), as opt-in models
-   [x] Ability to switch between paid and free API ([Gemini Paid API](../../../plugins/filters/gemini_paid_api.py) toggle)

## Installation

Paste `plugins/pipes/gemini_manifold.py` into a new function in Open WebUI (Admin Panel → Functions) with the function id `gemini_manifold_plus`, and install the companion filter from `plugins/filters/gemini_manifold_companion.py` (version 2.2.0 serves both pipes). The model table is embedded in the pipe, so nothing has to be hosted.

## Configuration

After installation, click the gear icon next to the `gemini_manifold_plus` function within Open WebUI. At a minimum, you must enter your Google Gemini API key. Other configurable options are also available on that settings page.

## Usage

If the whitelist and blacklist are configured to allow models, those models will appear in the Open WebUI model selection list. If valid credentials for both Gemini Developer API and Vertex AI are provided then the models get fetched from both sources and merged together. Models that are not chat models (speech, live, embedding, robotics, ...) are left out. To use a Gemini model, simply select it from the list and begin your chat.

## Troubleshooting

If you encounter issues, check the Open WebUI logs for error messages. The logs contain detailed information that should help pinpoint the problem. If you need further assistance, please open a new issue in this repository.

## Contributing

See `CONTRIBUTING.md`. For this plugin, I've also included several ideas in `TODO` comments within the code. These comments can serve as a starting point for contributions, but feel free to propose completely new features as well!

## License

MIT License. See the `LICENSE` file for details.
