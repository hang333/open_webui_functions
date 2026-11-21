# `youtube_watch.py` - YouTube Watch tool

A tool (Workspace → Tools) that lets the model decide to watch a YouTube video. A YouTube link in a message stays plain text; when the answer depends on what the video shows or says, the model calls `watch_youtube_video` with the link and its question, and gets Gemini's answer back as the tool result.

## Why a tool

- Gemini only understands a YouTube video when the video itself is in the request (`fileData` with the YouTube URL). Its URL context tool does not read YouTube videos.
- Gemini Manifold's `PARSE_YOUTUBE_URLS` sends every linked video along with the message, whether or not it is needed. Turn it off and attach this tool instead, so the video is only watched when the model asks for it.
- Any model that can call tools can use it, also one that is not Gemini.

## How it works

- `watch_youtube_video(url, question)` accepts watch, `youtu.be`, `shorts/`, `live/` and `embed/` links. It sends one `generateContent` request with the video (`video/mp4`, which Vertex AI requires) and a prompt asking for an answer with timestamps in the language of the question.
- The request is plain REST in camelCase, so proxies that read only camelCase (such as AxonHub) see all of it.
- Thought summaries are left out of the answer. Errors, blocked requests and empty answers go back to the model as text.
- Status messages show while the video is being watched.

## Valves

| Valve | Default | Meaning |
| --- | --- | --- |
| `MODEL` | `gemini-3.8-flash` | Gemini model that watches the video. |
| `API_KEY` | empty | Own key. Empty: the key of the Gemini Manifold pipe (paid key first). |
| `BASE_URL` | empty | API root without `/v1beta`. Empty: the pipe's base URL when its key is used, else Google. With its own key the tool never uses the pipe's base URL. |
| `PIPE_ID` | `gemini_manifold_plus` | Function id of the pipe whose key and base URL are used. |
| `MEDIA_RESOLUTION` | `low` | Frame resolution; low uses about a third of the tokens (a proxy may drop it). |
| `THINKING_LEVEL` | `low` | Thinking level of the watching model; empty uses the model's default. |
| `TIMEOUT_SECONDS` | `900` | How long to wait for the answer. |

Video tokens add up quickly (roughly 100 tokens per second of video at low resolution), so a long video costs accordingly.

## Installation

1. Workspace → Tools → import the file (or paste it) with the id `youtube_watch`.
2. Give users who should have it read access (tools are private by default).
3. Attach it to the models in the model editor (Tools), or enable it per chat.
4. Optionally turn off `PARSE_YOUTUBE_URLS` in Gemini Manifold, so links are no longer sent as video automatically.
