"""
title: YouTube Watch
id: youtube_watch
description: Lets the model watch a YouTube video with Gemini when the answer depends on what the video shows or says.
author: hang333
author_url: https://github.com/hang333
funding_url: https://github.com/hang333/open_webui_functions
license: MIT
version: 1.0.0
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Final, Literal

import aiohttp
from pydantic import BaseModel, Field

# The 11-character video id of watch, short, live, embed and youtu.be links (also m. and music. hosts).
YOUTUBE_ID: Final = re.compile(
    r'(?:https?://)?(?:www\.|m\.|music\.)?'
    r'(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|live/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})'
)
GOOGLE_BASE_URL: Final = 'https://generativelanguage.googleapis.com'
# Vertex AI rejects file data without a MIME type, and a proxy may send the request on to Vertex AI.
VIDEO_MIME_TYPE: Final = 'video/mp4'
PROMPT: Final = (
    'Watch the attached YouTube video and answer the request below from what is shown and said in it. '
    'Give timestamps (m:ss) for the parts you rely on. If the video does not cover the request, say so '
    'instead of guessing. Answer in the language of the request.\n\nRequest: {question}'
)


class Tools:
    """
    A tool that lets the model decide to watch a YouTube video. A link in a message is only text; when the answer
    depends on the video itself, the model calls `watch_youtube_video`, which sends the video to Gemini (video
    understanding, `fileData` with the YouTube URL) together with the model's question and returns the answer.

    Gemini's URL context tool cannot read YouTube videos, so this is the way for a model to look at one on demand.
    The request is plain REST with camelCase fields, which proxies such as AxonHub read, and the key and base URL
    of Gemini Manifold Plus are used unless the valves set their own.
    """

    class Valves(BaseModel):
        MODEL: str = Field(default='gemini-3.8-flash', description='Gemini model that watches the video.')
        API_KEY: str = Field(
            default='',
            description='Gemini API key. Empty: the key of the Gemini Manifold pipe (paid key first) is used.',
        )
        BASE_URL: str = Field(
            default='',
            description='Root of the Gemini API without /v1beta, e.g. a reverse proxy. Empty: the base URL of the'
            ' Gemini Manifold pipe when its key is used, else Google.',
        )
        PIPE_ID: str = Field(
            default='gemini_manifold_plus',
            description='Function id of the Gemini Manifold pipe whose key and base URL are used by default.',
        )
        MEDIA_RESOLUTION: Literal['low', 'medium', 'high'] = Field(
            default='low', description='Resolution of the video frames; low uses about a third of the tokens.'
        )
        THINKING_LEVEL: Literal['', 'minimal', 'low', 'medium', 'high'] = Field(
            default='low', description="Thinking level of the watching model; empty uses the model's default."
        )
        TIMEOUT_SECONDS: int = Field(default=900, description='How long to wait for the answer.')

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def watch_youtube_video(self, url: str, question: str, __event_emitter__: Any = None) -> str:
        """
        Watch a YouTube video and answer a question about its content: what is shown and said, a summary, or the
        details at a given time. Use it only when the answer depends on the video itself, not when a link is merely
        mentioned or the answer is already known. It takes a while and its cost grows with the video's length.

        :param url: Link to the YouTube video (youtube.com/watch?v=..., youtu.be/..., shorts/ or live/).
        :param question: What to find out from the video, e.g. "Summarize the main points" or "What is said at 2:30?".
        """
        match = YOUTUBE_ID.search(url or '')
        if not match:
            return f'Not a link to a YouTube video: {url}'
        video_url = f'https://www.youtube.com/watch?v={match.group(1)}'
        try:
            api_key, base_url = await self._connection()
        except LookupError as error:
            return str(error)

        await _status(__event_emitter__, f'Watching {video_url} with {self.valves.MODEL}...')
        endpoint = f'{base_url}/v1beta/models/{self.valves.MODEL}:generateContent'
        try:
            status, data = await self._post(endpoint, {'x-goog-api-key': api_key}, self._body(video_url, question))
        except Exception as error:  # Timeouts and connection errors go back to the model as text.
            await _status(__event_emitter__, 'Could not watch the video', done=True)
            return f'Could not watch the video: {type(error).__name__}: {error}'.rstrip(': ')

        if status != 200:
            await _status(__event_emitter__, 'Could not watch the video', done=True)
            return _error_message(status, data)
        await _status(__event_emitter__, f'Watched {video_url}', done=True)
        return _answer(data)

    def _body(self, video_url: str, question: str) -> dict[str, Any]:
        """The generateContent request, written in the REST API's own camelCase."""
        generation_config: dict[str, Any] = {
            'mediaResolution': f'MEDIA_RESOLUTION_{self.valves.MEDIA_RESOLUTION.upper()}'
        }
        if self.valves.THINKING_LEVEL:
            generation_config['thinkingConfig'] = {'thinkingLevel': self.valves.THINKING_LEVEL.upper()}
        return {
            'contents': [
                {
                    'role': 'user',
                    'parts': [
                        {'fileData': {'fileUri': video_url, 'mimeType': VIDEO_MIME_TYPE}},
                        {'text': PROMPT.format(question=question.strip() or 'Summarize the video.')},
                    ],
                }
            ],
            'generationConfig': generation_config,
        }

    async def _connection(self) -> tuple[str, str]:
        """
        The API key and base URL: the valves' own key with their base URL (or Google), else the Gemini Manifold
        pipe's key (paid first) with the valves' or the pipe's base URL.
        """
        if self.valves.API_KEY:
            return self.valves.API_KEY, (self.valves.BASE_URL or GOOGLE_BASE_URL).rstrip('/')
        pipe = await self._pipe_valves()
        api_key = pipe.get('GEMINI_PAID_API_KEY') or pipe.get('GEMINI_FREE_API_KEY')
        if not api_key:
            raise LookupError(
                f'No Gemini API key: set API_KEY in the YouTube Watch valves or a key in "{self.valves.PIPE_ID}".'
            )
        return api_key, (self.valves.BASE_URL or pipe.get('GEMINI_API_BASE_URL') or GOOGLE_BASE_URL).rstrip('/')

    async def _pipe_valves(self) -> dict[str, Any]:
        """The admin valves of the Gemini Manifold pipe, as Open WebUI stores them (decrypted)."""
        try:
            from open_webui.models.functions import Functions
        except ImportError:
            return {}
        valves = Functions.get_function_valves_by_id(self.valves.PIPE_ID)
        if inspect.isawaitable(valves):
            valves = await valves
        return valves or {}

    async def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[int, Any]:
        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=body) as response:
                try:
                    data = await response.json(content_type=None)
                except ValueError:
                    data = {'error': {'message': (await response.text())[:500]}}
                return response.status, data


async def _status(emitter: Any, description: str, done: bool = False) -> None:
    if emitter:
        await emitter({'type': 'status', 'data': {'description': description, 'done': done}})


def _error_message(status: int, data: Any) -> str:
    error = data.get('error') if isinstance(data, dict) else None
    message = error.get('message') if isinstance(error, dict) else None
    return f'Could not watch the video: HTTP {status}' + (f': {message}' if message else '')


def _answer(data: Any) -> str:
    """The answer's text without thought summaries, or why there is none."""
    candidates = data.get('candidates') if isinstance(data, dict) else None
    if not candidates:
        reason = ((data or {}).get('promptFeedback') or {}).get('blockReason') if isinstance(data, dict) else None
        return f'The video could not be watched: the request was blocked ({reason}).' if reason else 'No answer.'
    candidate = candidates[0] or {}
    parts = (candidate.get('content') or {}).get('parts') or []
    text = ''.join(part.get('text', '') for part in parts if isinstance(part, dict) and not part.get('thought'))
    if text.strip():
        return text.strip()
    return f'No answer (finish reason: {candidate.get("finishReason", "unknown")}).'
