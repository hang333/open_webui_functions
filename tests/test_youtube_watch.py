"""Tests of the YouTube Watch tool (plugins/tools/youtube_watch.py). The HTTP request is captured, not sent."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from plugins.tools import youtube_watch as yw

PIPE_VALVES = {'GEMINI_PAID_API_KEY': 'paid-key', 'GEMINI_API_BASE_URL': 'http://axonhub:8090/gemini/'}
ANSWER = {
    'candidates': [
        {
            'content': {'parts': [{'text': 'Planning...', 'thought': True}, {'text': 'It is a music video (0:00).'}]},
            'finishReason': 'STOP',
        }
    ]
}


class Recorder:
    """Stands in for the HTTP POST and records what would be sent."""

    def __init__(self, status: int = 200, data: Any = None, error: Exception | None = None):
        self.status, self.data, self.error = status, ANSWER if data is None else data, error
        self.requests: list[tuple[str, dict, dict]] = []

    async def __call__(self, url: str, headers: dict, body: dict):
        self.requests.append((url, headers, body))
        if self.error:
            raise self.error
        return self.status, self.data


def make_tools(monkeypatch, recorder: Recorder, pipe_valves: dict | None = None, **valves: Any) -> yw.Tools:
    tools = yw.Tools()
    tools.valves = yw.Tools.Valves(**valves)
    monkeypatch.setattr(tools, '_post', recorder)

    async def stored_pipe_valves():
        return dict(PIPE_VALVES if pipe_valves is None else pipe_valves)

    monkeypatch.setattr(tools, '_pipe_valves', stored_pipe_valves)
    return tools


def test_the_model_sees_two_parameters_and_when_to_use_the_tool():
    parameters = [name for name in inspect.signature(yw.Tools.watch_youtube_video).parameters if name != 'self']
    assert [name for name in parameters if not name.startswith('__')] == ['url', 'question']
    doc = yw.Tools.watch_youtube_video.__doc__
    assert 'only when the answer depends on the video itself' in doc
    assert ':param url:' in doc and ':param question:' in doc


@pytest.mark.asyncio
async def test_sends_the_video_in_camel_case_with_a_mime_type(monkeypatch):
    recorder = Recorder()
    tools = make_tools(monkeypatch, recorder)
    events = []

    async def emitter(event):
        events.append(event)

    answer = await tools.watch_youtube_video('https://youtu.be/dQw4w9WgXcQ?t=42', ' What is it? ', emitter)

    assert answer == 'It is a music video (0:00).'  # Thought summaries are left out.
    url, headers, body = recorder.requests[0]
    # The Gemini Manifold pipe's key and base URL (through AxonHub here); the trailing slash is dropped.
    assert url == 'http://axonhub:8090/gemini/v1beta/models/gemini-3.8-flash:generateContent'
    assert headers == {'x-goog-api-key': 'paid-key'}
    video, prompt = body['contents'][0]['parts']
    assert video == {'fileData': {'fileUri': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'mimeType': 'video/mp4'}}
    assert prompt['text'].endswith('Request: What is it?')
    assert body['generationConfig'] == {
        'mediaResolution': 'MEDIA_RESOLUTION_LOW',
        'thinkingConfig': {'thinkingLevel': 'LOW'},
    }
    assert [event['data']['done'] for event in events] == [False, True]


@pytest.mark.parametrize(
    'link',
    [
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'youtube.com/watch?feature=share&v=dQw4w9WgXcQ',
        'https://m.youtube.com/shorts/dQw4w9WgXcQ',
        'https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=x',
        'https://www.youtube.com/live/dQw4w9WgXcQ?si=abc',
        'https://www.youtube.com/embed/dQw4w9WgXcQ',
    ],
)
@pytest.mark.asyncio
async def test_accepts_the_usual_youtube_links(monkeypatch, link):
    recorder = Recorder()
    await make_tools(monkeypatch, recorder).watch_youtube_video(link, 'Summarize.')
    file_data = recorder.requests[0][2]['contents'][0]['parts'][0]['fileData']
    assert file_data['fileUri'] == 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'


@pytest.mark.asyncio
async def test_rejects_other_links_without_a_request(monkeypatch):
    recorder = Recorder()
    answer = await make_tools(monkeypatch, recorder).watch_youtube_video('https://vimeo.com/123', 'Summarize.')
    assert answer.startswith('Not a link to a YouTube video') and recorder.requests == []


@pytest.mark.asyncio
async def test_own_key_goes_to_its_own_base_url_or_google(monkeypatch):
    recorder = Recorder()
    await make_tools(monkeypatch, recorder, API_KEY='own-key').watch_youtube_video('youtu.be/dQw4w9WgXcQ', 'x')
    await make_tools(monkeypatch, recorder, API_KEY='own-key', BASE_URL='https://proxy.example').watch_youtube_video(
        'youtu.be/dQw4w9WgXcQ', 'x'
    )
    urls = [url for url, _, _ in recorder.requests]
    assert urls[0].startswith('https://generativelanguage.googleapis.com/v1beta/models/')  # Not the pipe's proxy.
    assert urls[1].startswith('https://proxy.example/v1beta/models/')
    assert {headers['x-goog-api-key'] for _, headers, _ in recorder.requests} == {'own-key'}


@pytest.mark.asyncio
async def test_falls_back_to_the_free_key_and_reports_a_missing_key(monkeypatch):
    recorder = Recorder()
    tools = make_tools(monkeypatch, recorder, pipe_valves={'GEMINI_FREE_API_KEY': 'free-key'})
    await tools.watch_youtube_video('youtu.be/dQw4w9WgXcQ', 'x')
    assert recorder.requests[0][0].startswith('https://generativelanguage.googleapis.com/')
    assert recorder.requests[0][1] == {'x-goog-api-key': 'free-key'}

    no_key = make_tools(monkeypatch, Recorder(), pipe_valves={})
    assert (await no_key.watch_youtube_video('youtu.be/dQw4w9WgXcQ', 'x')).startswith('No Gemini API key')


@pytest.mark.asyncio
async def test_thinking_level_can_be_left_to_the_model(monkeypatch):
    recorder = Recorder()
    await make_tools(monkeypatch, recorder, THINKING_LEVEL='', MEDIA_RESOLUTION='high').watch_youtube_video(
        'youtu.be/dQw4w9WgXcQ', 'x'
    )
    assert recorder.requests[0][2]['generationConfig'] == {'mediaResolution': 'MEDIA_RESOLUTION_HIGH'}


@pytest.mark.parametrize(
    ('recorder', 'expected'),
    [
        (
            Recorder(400, {'error': {'message': 'empty mimeType parameter in fileData'}}),
            'Could not watch the video: HTTP 400: empty mimeType parameter in fileData',
        ),
        (Recorder(error=TimeoutError()), 'Could not watch the video: TimeoutError'),
        (Recorder(data={'promptFeedback': {'blockReason': 'SAFETY'}}), 'blocked (SAFETY)'),
        (Recorder(data={'candidates': [{'finishReason': 'MAX_TOKENS'}]}), 'No answer (finish reason: MAX_TOKENS).'),
    ],
)
@pytest.mark.asyncio
async def test_failures_go_back_to_the_model_as_text(monkeypatch, recorder, expected):
    answer = await make_tools(monkeypatch, recorder).watch_youtube_video('youtu.be/dQw4w9WgXcQ', 'x')
    assert expected in answer
