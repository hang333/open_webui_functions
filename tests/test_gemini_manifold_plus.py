"""
Tests of Gemini Manifold Plus (plugins/pipes/gemini_manifold.py) and the companion filter changes.

The google-genai client is replaced by `FakeClient` (no network); the Open WebUI modules are the stand-ins in
tests/owui_stubs.py, patched into the plugin module per test. google-genai's own types are real, so requests
and responses are built and validated by the pinned SDK.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi.datastructures import State

import plugins.filters.gemini_manifold_companion as companion
import plugins.pipes.gemini_manifold as gm
from tests import owui_stubs

types = gm.types
Pipe = gm.Pipe
REPO = Path(__file__).resolve().parent.parent
USER = {'id': 'user-1', 'email': 'user@example.com', 'role': 'user', 'valves': None}
SIGNATURE = b'\x12\x34signature-of-fc1\xfe'


# region Helpers


class RecordingEmitter:
    """Records what the pipe shows through the companion filter's EventEmitter interface."""

    def __init__(self) -> None:
        self.start_time = time.monotonic()
        self.statuses: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def emit_status(self, description: str, done: bool = False, hidden: bool = False, **_: Any) -> None:
        self.statuses.append(description)

    def emit_toast(self, msg: str, type: str = 'info') -> None:
        self.toasts.append((type, msg))


class FakeModels:
    """`client.aio.models`: serves one prepared list of response chunks per request."""

    def __init__(self, rounds: list[list[types.GenerateContentResponse]], models: list[types.Model]):
        self.rounds = list(rounds)
        self.models = models
        self.calls: list[dict[str, Any]] = []

    async def generate_content_stream(self, **kwargs: Any):
        self.calls.append(kwargs)
        chunks = self.rounds.pop(0)

        async def stream():
            for chunk in chunks:
                yield chunk

        return stream()

    async def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        chunks = self.rounds.pop(0)
        parts = [part for chunk in chunks for part in chunk.parts or []]
        last = chunks[-1]
        candidate = last.candidates[0]
        return response(
            *parts, finish=candidate.finish_reason, usage=last.usage_metadata, grounding=candidate.grounding_metadata
        )

    async def list(self, config: Any = None):
        async def pager():
            for model in self.models:
                yield model

        return pager()


class FakeInteractions:
    """`client.aio.interactions`: `create` returns the first prepared interaction, `get` the following ones."""

    def __init__(self, interactions: list[Any]):
        self.interactions = list(interactions)
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []

    async def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return self.interactions.pop(0)

    async def get(self, id: str, **kwargs: Any) -> Any:
        return self.interactions.pop(0)

    async def cancel(self, id: str, **kwargs: Any) -> Any:
        self.cancelled.append(id)


class FakeClient:
    """Stands in for `genai.Client`; only the attributes the pipe uses."""

    def __init__(self, rounds=(), models=(), interactions=(), vertexai: bool = False):
        self.vertexai = vertexai
        self._api_client = SimpleNamespace(api_key='test-api-key')
        self.aio = SimpleNamespace(
            models=FakeModels(list(rounds), list(models)),
            interactions=FakeInteractions(list(interactions)),
        )


def response(*parts: types.Part, finish: Any = None, usage: Any = None, grounding: Any = None):
    """One GenerateContentResponse chunk with a single candidate."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role='model', parts=list(parts)),
                finish_reason=finish,
                grounding_metadata=grounding,
            )
        ],
        usage_metadata=usage,
    )


def usage(prompt=100, candidates=20, thoughts=30, cached=None, tool_prompt=None):
    total = prompt + candidates + thoughts + (tool_prompt or 0)
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        cached_content_token_count=cached,
        tool_use_prompt_token_count=tool_prompt,
        total_token_count=total,
    )


def tool_spec(name: str, **properties: Any) -> dict[str, Any]:
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': f'The {name} tool.',
            'parameters': {
                'type': 'object',
                'properties': {key: {'type': kind} for key, kind in properties.items()},
                'required': list(properties),
            },
        },
    }


def make_request(pipe: Pipe | None = None) -> SimpleNamespace:
    state = State()
    if pipe is not None:
        state.FUNCTIONS = {'gemini_manifold_plus': pipe}
    app = SimpleNamespace(state=state, url_path_for=lambda name, id: f'/api/v1/files/{id}/content')
    return SimpleNamespace(app=app)


def metadata(**extra: Any) -> dict[str, Any]:
    return {
        'chat_id': 'local:chat-1',
        'message_id': 'msg-1',
        'user_id': USER['id'],
        'session_id': 'session-1',
        'features': {},
        'params': {},
        **extra,
    }


async def collect(generator) -> list[dict]:
    return [chunk async for chunk in generator]


def deltas(chunks: list[dict]) -> list[dict]:
    return [chunk['choices'][0] for chunk in chunks if chunk.get('choices')]


def stored_content(detail: dict) -> types.Content:
    assert detail['format'] == gm.REPLAY_FORMAT
    return types.Content.model_validate_json(detail['data'])


def builder(messages, owui, *, model_id='gemini-4-hydrogen', meta=None, **kwargs) -> gm.GeminiContentBuilder:
    files_api_manager = SimpleNamespace(client=SimpleNamespace(vertexai=False))
    return gm.GeminiContentBuilder(
        messages_body=messages,
        metadata_body=meta or metadata(),
        user_data=USER,
        event_emitter=RecordingEmitter(),
        options=Pipe.Valves(),
        files_api_manager=files_api_manager,
        model_id=model_id,
        **kwargs,
    )


@pytest.fixture
def owui(monkeypatch) -> owui_stubs.OwuiModules:
    return owui_stubs.OwuiModules().patch_plugin(monkeypatch, gm)


@pytest_asyncio.fixture
async def pipe(owui, monkeypatch):
    """A pipe with a paid key whose genai client is the test's `FakeClient` (set `pipe.fake_client`)."""
    instance = Pipe()
    instance.valves = Pipe.Valves(GEMINI_PAID_API_KEY='paid-key')
    instance.fake_client = FakeClient()
    monkeypatch.setattr(gm.genai, 'Client', lambda **kwargs: instance.fake_client)
    Pipe._get_or_create_genai_client.cache_clear()
    yield instance
    Pipe._get_or_create_genai_client.cache_clear()
    await instance._get_genai_models.cache.clear()


def toggle(owui: owui_stubs.OwuiModules, *filter_ids: str) -> list[str]:
    """Installs global toggle filters and returns the `filter_ids` that switch them on."""
    for filter_id in filter_ids:
        owui.functions.functions[filter_id] = owui_stubs.FunctionModel(filter_id, is_active=True, is_global=True)
    return list(filter_ids)


# endregion Helpers


# region Thinking per model family


@pytest.mark.parametrize(
    'model_id, budget, level, expected',
    [
        # Gemini 4 Carbon/Argon: low|medium|high; minimal maps to low, xhigh to high; budgets become levels.
        ('gemini-4-carbon', None, 'XHIGH', {'thinking_level': 'HIGH'}),
        ('gemini-4-carbon', None, 'MINIMAL', {'thinking_level': 'LOW'}),
        ('gemini-4-argon', 8192, None, {'thinking_level': 'MEDIUM'}),
        ('gemini-4-carbon', 0, None, {'thinking_level': 'LOW'}),
        ('gemini-4-carbon', -1, None, {}),
        ('gemini-4-carbon', None, None, {}),
        # Gemini 4 Neon/Hydrogen add xhigh.
        ('gemini-4-hydrogen', None, 'XHIGH', {'thinking_level': 'XHIGH'}),
        ('gemini-4-neon', None, 'NONE', {'thinking_level': 'LOW'}),
        # Gemini 3.x Flash models.
        ('gemini-3.8-flash', None, 'MINIMAL', {'thinking_level': 'LOW'}),
        ('gemini-3.6-flash', None, 'MINIMAL', {'thinking_level': 'MINIMAL'}),
        ('gemini-3.6-flash', None, 'NONE', {'thinking_level': 'MINIMAL'}),
        ('gemini-3.5-flash-lite', 30000, None, {'thinking_level': 'HIGH'}),
        ('gemini-3.1-pro-preview', None, 'MINIMAL', {'thinking_level': 'LOW'}),
        # Image models: Nano Banana 2.1 has minimal|medium|high (a tie rounds up).
        ('gemini-nano-banana-2.1', None, 'LOW', {'thinking_level': 'MEDIUM'}),
        ('gemini-3.1-flash-image', None, 'MEDIUM', {'thinking_level': 'HIGH'}),
        # Gemini 2.5: budgets, levels mapped like the OpenAI compatibility layer, clamped to the model's range.
        ('gemini-2.5-flash', None, 'HIGH', {'thinking_budget': 24576}),
        ('gemini-2.5-flash', None, 'NONE', {'thinking_budget': 0}),
        ('gemini-2.5-flash', 99999, None, {'thinking_budget': 24576}),
        ('gemini-2.5-pro', None, 'NONE', {'thinking_budget': 128}),
        ('gemini-2.5-pro', 0, None, {'thinking_budget': 128}),
        ('gemini-2.5-pro', -1, None, {'thinking_budget': -1}),
        ('gemini-2.5-flash-lite', 100, None, {'thinking_budget': 512}),
        ('gemini-2.5-flash-lite', None, 'XHIGH', {'thinking_budget': 24576}),
        # Thinking without controls.
        ('gemini-3-pro-image', None, 'HIGH', {}),
    ],
)
def test_fit_thinking_per_model_family(model_id, budget, level, expected):
    fitted = Pipe()._fit_thinking(model_id, budget, level)
    assert {key: getattr(value, 'name', value) for key, value in fitted.items()} == expected
    assert not ('thinking_level' in fitted and 'thinking_budget' in fitted)


@pytest.mark.asyncio
async def test_reasoning_effort_reaches_gemini4_as_level_never_budget(owui):
    pipe = Pipe()
    body = {'model': 'gemini_manifold_plus.gemini-4-hydrogen', 'messages': [], 'reasoning_effort': 'xhigh'}
    options = gm._resolve_options(pipe.valves, None, USER['email'], body, metadata())
    config = await pipe._thinking_config(options, 'gemini-4-hydrogen', metadata())
    assert config.thinking_level == types.ThinkingLevel.XHIGH
    assert config.thinking_budget is None
    assert config.include_thoughts is True


@pytest.mark.asyncio
async def test_reasoning_toggle_off_asks_for_least_thinking(owui):
    pipe = Pipe()
    toggle(owui, 'gemini_reasoning_toggle')  # Installed, but not in this request's filter_ids: OFF.
    options = gm._resolve_options(pipe.valves, None, USER['email'], {'reasoning_effort': 'high'}, metadata())
    carbon = await pipe._thinking_config(options, 'gemini-4-carbon', metadata())
    flash = await pipe._thinking_config(options, 'gemini-2.5-flash', metadata())
    assert carbon.thinking_level == types.ThinkingLevel.LOW
    assert flash.thinking_budget == 0


@pytest.mark.asyncio
async def test_models_without_thinking_get_no_thinking_config(owui):
    pipe = Pipe()
    assert await pipe._thinking_config(Pipe.Valves(), 'gemini-2.5-flash-image', metadata()) is None
    assert await pipe._thinking_config(Pipe.Valves(), 'not-in-the-table', metadata()) is None


def test_invalid_reasoning_effort_is_ignored_instead_of_failing():
    pipe = Pipe()
    pipe.valves = Pipe.Valves(THINKING_CONFIG_RULES='LOW')
    options = gm._resolve_options(pipe.valves, None, USER['email'], {'reasoning_effort': 'turbo'}, metadata())
    assert options.THINKING_CONFIG_RULES == 'LOW'
    assert gm._resolve_options(pipe.valves, None, '', {'reasoning_effort': 'none'}, {}).THINKING_CONFIG_RULES == 'none'


def test_resolve_thinking_config_prefers_budget_rules_for_gemini_25():
    rules = '{"gemini-2.5": "HIGH", "gemini": 2048}'
    assert Pipe._resolve_thinking_config(rules, 'gemini-2.5-pro') == (2048, None)
    assert Pipe._resolve_thinking_config(rules, 'gemini-4-carbon') == (2048, None)
    assert Pipe._resolve_thinking_config('{"gemini-4": "medium", ".*": 0}', 'gemini-4-carbon') == (None, 'MEDIUM')


# endregion Thinking per model family


# region Request shape: sampling parameters, structured output, media resolution, prefill


@pytest.mark.asyncio
async def test_deprecated_sampling_parameters_and_candidate_count_are_never_sent(owui):
    pipe = Pipe()
    body = {'temperature': 0.2, 'top_p': 0.9, 'top_k': 40, 'n': 2, 'max_tokens': 1000, 'stop': ['END']}
    meta = metadata(canonical_model_id='gemini-4-carbon')
    options = gm._resolve_options(pipe.valves, None, '', body, meta)
    config = await pipe._build_gen_content_config(body, meta, options, RecordingEmitter(), gm.ToolBridge())
    assert (config.temperature, config.top_p, config.top_k, config.candidate_count) == (None, None, None, None)
    assert config.max_output_tokens == 1000
    assert config.stop_sequences == ['END']
    assert config.automatic_function_calling.disable is True


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id, keeps_tools', [('gemini-4-carbon', True), ('gemini-2.5-flash', False)])
async def test_json_schema_response_format_maps_to_structured_output(owui, model_id, keeps_tools):
    pipe = Pipe()
    schema = {'type': 'object', 'properties': {'title': {'type': 'string'}}, 'required': ['title']}
    body = {'response_format': {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': schema}}}
    meta = metadata(canonical_model_id=model_id)
    bridge = gm.ToolBridge([tool_spec('search_notes', query='string')])
    config = await pipe._build_gen_content_config(body, meta, Pipe.Valves(), RecordingEmitter(), bridge)
    assert config.response_mime_type == 'application/json'
    assert config.response_json_schema == schema
    declared = [tool for tool in config.tools or [] if tool.function_declarations]
    assert bool(declared) is keeps_tools


@pytest.mark.asyncio
async def test_json_object_response_format_sets_only_the_mime_type(owui):
    pipe = Pipe()
    meta = metadata(canonical_model_id='gemini-3.8-flash')
    body = {'response_format': {'type': 'json_object'}}
    config = await pipe._build_gen_content_config(body, meta, Pipe.Valves(), RecordingEmitter(), gm.ToolBridge())
    assert config.response_mime_type == 'application/json'
    assert config.response_json_schema is None


@pytest.mark.asyncio
async def test_media_resolution_global_and_ultra_high_per_image_part(owui):
    pipe = Pipe()
    meta = metadata(canonical_model_id='gemini-4-carbon')
    high = Pipe.Valves(MEDIA_RESOLUTION='HIGH')
    config = await pipe._build_gen_content_config({}, meta, high, RecordingEmitter(), gm.ToolBridge())
    assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH
    assert pipe._image_media_resolution(high, 'gemini-4-carbon') is None

    ultra = Pipe.Valves(MEDIA_RESOLUTION='ULTRA_HIGH')
    config = await pipe._build_gen_content_config({}, meta, ultra, RecordingEmitter(), gm.ToolBridge())
    assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH
    per_part = pipe._image_media_resolution(ultra, 'gemini-4-carbon')
    assert per_part.level == types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH
    assert pipe._image_media_resolution(ultra, 'gemini-2.5-flash') is None

    content_builder = builder([{'role': 'user', 'content': 'x'}], owui, image_media_resolution=per_part)
    image = await content_builder._create_genai_part_from_file_data(b'png', 'image/png', None, asyncio.Queue())
    pdf = await content_builder._create_genai_part_from_file_data(b'pdf', 'application/pdf', None, asyncio.Queue())
    # The temporary chat sends raw bytes; the image part carries the per-part resolution.
    assert image.media_resolution == per_part and pdf.media_resolution is None


@pytest.mark.asyncio
async def test_a_request_never_ends_with_a_model_turn(owui):
    messages = [
        {'role': 'user', 'content': 'Write a poem.'},
        {'role': 'assistant', 'content': 'Roses are red,'},  # Open WebUI's "Continue response".
    ]
    contents = await builder(messages, owui).build_contents()
    assert [c.role for c in contents] == ['user', 'model', 'user']
    assert contents[-1].parts[0].text == gm.CONTINUE_PROMPT


@pytest.mark.asyncio
async def test_last_user_turn_with_only_attachments_gets_text(owui):
    image = 'data:image/png;base64,iVBORw0KGgo='
    messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}}]}]
    contents = await builder(messages, owui).build_contents()
    assert contents[-1].parts[0].inline_data is not None
    assert contents[-1].parts[-1].text == gm.ATTACHMENTS_ONLY_PROMPT


# endregion Request shape


# region Native tool calling


def test_tool_bridge_declarations_names_and_schemas():
    tools = [
        tool_spec('search_web', query='string'),
        tool_spec('search_web', query='string'),  # Duplicate: skipped.
        tool_spec('server:tool.with dots'),
        {
            'type': 'function',
            'function': {
                'name': 'complex',
                'parameters': {
                    '$schema': 'http://json-schema.org/draft-07/schema#',
                    'type': 'object',
                    'properties': {'pair': {'type': 'array', 'items': [{'type': 'string'}, {'type': 'integer'}]}},
                    'required': ['pair', 'missing'],
                    'x-internal': True,
                },
            },
        },
    ]
    bridge = gm.ToolBridge(tools)
    names = [declaration.name for declaration in bridge.declarations]
    assert names[0] == 'search_web' and names[2] == 'complex' and len(names) == 3
    assert re.fullmatch(r'server_tool_with_dots_[0-9a-f]{8}', names[1])
    # Tools without properties are declared without parameters.
    assert bridge.declarations[1].parameters_json_schema is None
    schema = bridge.declarations[2].parameters_json_schema
    assert '$schema' not in schema and 'x-internal' not in schema
    assert schema['required'] == ['pair']
    assert schema['properties']['pair']['prefixItems'] == [{'type': 'string'}, {'type': 'integer'}]
    assert bridge.owui_name(names[1]) == 'server:tool.with dots'
    assert bridge.owui_name('default_api.search_web') == 'search_web'


@pytest.mark.parametrize(
    'tool_choice, mode, allowed',
    [
        ('none', 'NONE', None),
        ('required', 'ANY', None),
        ({'type': 'function', 'function': {'name': 'search_web'}}, 'ANY', ['search_web']),
        ({'type': 'function', 'function': {'name': 'undeclared'}}, None, None),
        ('auto', None, None),
        (None, None, None),
    ],
)
def test_tool_choice_mapping(tool_choice, mode, allowed):
    config = gm.ToolBridge([tool_spec('search_web', query='string')]).function_calling_config(tool_choice)
    assert (config.mode.name if config else None) == mode
    assert (config.allowed_function_names if config else None) == allowed


def test_tool_bridge_is_empty_in_legacy_mode_for_tasks_and_image_models():
    body = {'tools': [tool_spec('search_web', query='string')]}
    for meta, accepts in [
        ({'params': {'function_calling': 'legacy'}}, True),
        ({'task': 'title_generation'}, True),
        ({}, False),
    ]:
        assert gm.ToolBridge.for_request(body, meta, accepts_functions=accepts).declarations == []
    assert gm.ToolBridge.for_request(body, {'task': 'context_compaction'}, accepts_functions=True).declarations
    assert gm.ToolBridge.for_request(body, {'params': {}}, accepts_functions=True).declarations
    assert not Pipe()._accepts_functions('gemini-nano-banana-2.1')
    assert Pipe()._accepts_functions('gemini-4-carbon')


def tool_round_chunks() -> list[types.GenerateContentResponse]:
    """A Gemini 4 round: thought summary, text, then two parallel calls (only the first one signed)."""
    return [
        response(types.Part(text='**Planning**\nI should check both.', thought=True)),
        response(types.Part(text='Let me check')),
        response(types.Part(text=' both.')),
        response(
            types.Part(
                function_call=types.FunctionCall(id='fc-1', name='get_weather', args={'city': 'Paris'}),
                thought_signature=SIGNATURE,
            ),
            types.Part(function_call=types.FunctionCall(id='fc-2', name='get_time', args={'tz': 'CET'})),
            finish=types.FinishReason.STOP,
            usage=usage(),
        ),
    ]


TOOLS = [tool_spec('get_weather', city='string'), tool_spec('get_time', tz='string')]


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [True, False])
async def test_tool_round_returns_tool_calls_and_exact_signatures(pipe, stream):
    pipe.fake_client = FakeClient(rounds=[tool_round_chunks()])
    body = {
        'model': 'gemini_manifold_plus.gemini-4-hydrogen',
        'messages': [{'role': 'user', 'content': 'Weather and time in Paris?'}],
        'stream': stream,
        'tools': TOOLS,
    }
    result = await pipe.pipe(body, USER, make_request(), metadata())

    if stream:
        chunks = await collect(result)
        items = deltas(chunks)
        detail_at = next(i for i, d in enumerate(items) if d['delta'].get('reasoning_details'))
        calls_at = next(i for i, d in enumerate(items) if d['delta'].get('tool_calls'))
        # Open WebUI keeps reasoning_details with a round only if they arrive before its tool calls.
        assert detail_at < calls_at
        assert items[calls_at + 1]['finish_reason'] == 'tool_calls'
        detail = items[detail_at]['delta']['reasoning_details'][0]
        tool_calls = items[calls_at]['delta']['tool_calls']
        assert ''.join(d['delta'].get('reasoning_content') or '' for d in items).startswith('**Planning**')
        assert any('usage' in chunk for chunk in chunks)
    else:
        message = result['choices'][0]['message']
        assert result['choices'][0]['finish_reason'] == 'tool_calls'
        detail = message['reasoning_details'][0]
        tool_calls = message['tool_calls']
        assert message['content'] == 'Let me check both.'

    assert [(c['id'], c['function']['name'], json.loads(c['function']['arguments'])) for c in tool_calls] == [
        ('fc-1', 'get_weather', {'city': 'Paris'}),
        ('fc-2', 'get_time', {'tz': 'CET'}),
    ]
    parts = stored_content(detail).parts
    # Thought summaries are display-only; the stream fragments are merged; signatures stay on their part.
    assert [p.text for p in parts if p.text] == ['Let me check both.']
    assert [(p.function_call.id, p.thought_signature) for p in parts if p.function_call] == [
        ('fc-1', SIGNATURE),
        ('fc-2', None),
    ]

    request = pipe.fake_client.aio.models.calls[0]
    declared = [d.name for tool in request['config'].tools for d in tool.function_declarations or []]
    assert declared == ['get_weather', 'get_time']
    assert request['config'].tool_config is None


def continued_round_messages(detail: dict) -> list[dict]:
    """The messages Open WebUI sends back for the next round of its tool loop."""
    return [
        {'role': 'user', 'content': 'Weather and time in Paris?'},
        {
            'role': 'assistant',
            'content': 'Let me check both.',
            'tool_calls': [
                {
                    'id': 'fc-1',
                    'type': 'function',
                    'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
                },
                {'id': 'fc-2', 'type': 'function', 'function': {'name': 'get_time', 'arguments': '{"tz": "CET"}'}},
            ],
            'reasoning_details': [detail],
        },
        {'role': 'tool', 'tool_call_id': 'fc-1', 'content': '18°C and sunny'},
        {'role': 'tool', 'tool_call_id': 'fc-2', 'content': 'Error: time service unavailable'},
    ]


def round_detail(*parts: types.Part) -> dict:
    content = types.Content(role='model', parts=list(parts))
    return {'type': 'reasoning.encrypted', 'format': gm.REPLAY_FORMAT, 'index': 0, 'data': content.model_dump_json()}


PARALLEL_DETAIL = round_detail(
    types.Part(text='Let me check both.'),
    types.Part(
        function_call=types.FunctionCall(id='fc-1', name='get_weather', args={'city': 'Paris'}),
        thought_signature=SIGNATURE,
    ),
    types.Part(function_call=types.FunctionCall(id='fc-2', name='get_time', args={'tz': 'CET'})),
)


@pytest.mark.asyncio
async def test_parallel_calls_are_replayed_as_fc1_signed_fc2_fr1_fr2(owui):
    contents = await builder(continued_round_messages(PARALLEL_DETAIL), owui).build_contents()
    assert [c.role for c in contents] == ['user', 'model', 'user']
    model, results = contents[1], contents[2]
    assert model.parts[0].text == 'Let me check both.'
    calls = [p for p in model.parts if p.function_call]
    assert [(p.function_call.id, p.thought_signature) for p in calls] == [('fc-1', SIGNATURE), ('fc-2', None)]
    assert [
        (p.function_response.id, p.function_response.name, p.function_response.response) for p in results.parts
    ] == [
        ('fc-1', 'get_weather', {'output': '18°C and sunny'}),
        ('fc-2', 'get_time', {'error': 'Error: time service unavailable'}),
    ]


@pytest.mark.asyncio
async def test_sequential_calls_keep_one_signature_per_step(owui):
    second_signature = b'second-step'
    messages = [
        {'role': 'user', 'content': 'Plan my day.'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'id': 'a', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{}'}}],
            'reasoning_details': [
                round_detail(
                    types.Part(
                        function_call=types.FunctionCall(id='a', name='get_weather', args={}),
                        thought_signature=SIGNATURE,
                    )
                )
            ],
        },
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'rain'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'id': 'b', 'type': 'function', 'function': {'name': 'get_time', 'arguments': '{}'}}],
            'reasoning_details': [
                round_detail(
                    types.Part(
                        function_call=types.FunctionCall(id='b', name='get_time', args={}),
                        thought_signature=second_signature,
                    )
                )
            ],
        },
        {'role': 'tool', 'tool_call_id': 'b', 'content': '09:00'},
    ]
    contents = await builder(messages, owui).build_contents()
    assert [c.role for c in contents] == ['user', 'model', 'user', 'model', 'user']
    assert contents[1].parts[0].thought_signature == SIGNATURE
    assert contents[3].parts[0].thought_signature == second_signature
    assert contents[4].parts[0].function_response.id == 'b'


def hidden_call_messages() -> list[dict]:
    """A tool round that Open WebUI added itself (e.g. recalled memories): no reasoning_details."""
    return [
        {'role': 'user', 'content': 'What do you remember about me?'},
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {'id': 'fc_x', 'type': 'function', 'function': {'name': 'recall_memories', 'arguments': '{}'}}
            ],
        },
        {'role': 'tool', 'tool_call_id': 'fc_x', 'content': 'Likes tea.'},
    ]


@pytest.mark.asyncio
async def test_unsigned_calls_of_the_current_turn_get_the_documented_placeholder(owui):
    contents = await builder(hidden_call_messages(), owui).build_contents()
    call = contents[1].parts[0]
    assert call.function_call.name == 'recall_memories'
    # What the SDK sends on the wire is the literal placeholder.
    assert call.model_dump(mode='json')['thought_signature'] == gm.SKIP_THOUGHT_SIGNATURE

    # Gemini 2.5 does not validate signatures, and earlier turns are never validated.
    old_turn = [
        *hidden_call_messages(),
        {'role': 'assistant', 'content': 'You like tea.'},
        {'role': 'user', 'content': 'Thanks'},
    ]
    for model_id, messages in [('gemini-2.5-flash', hidden_call_messages()), ('gemini-4-hydrogen', old_turn)]:
        contents = await builder(messages, owui, model_id=model_id).build_contents()
        assert contents[1].parts[0].thought_signature is None


@pytest.mark.asyncio
async def test_unanswered_calls_and_orphan_results_are_dropped(owui):
    messages = continued_round_messages(PARALLEL_DETAIL)
    del messages[3]  # No result for fc-2.
    messages.append({'role': 'tool', 'tool_call_id': 'unknown', 'content': 'orphan'})
    contents = await builder(messages, owui).build_contents()
    assert [p.function_call.id for p in contents[1].parts if p.function_call] == ['fc-1']
    assert contents[1].parts[1].thought_signature == SIGNATURE
    assert [p.function_response.id for p in contents[2].parts] == ['fc-1']


@pytest.mark.asyncio
async def test_dropping_the_signed_call_moves_no_signature_but_adds_the_placeholder(owui):
    messages = continued_round_messages(PARALLEL_DETAIL)
    del messages[2]  # No result for the signed fc-1.
    contents = await builder(messages, owui).build_contents()
    calls = [p for p in contents[1].parts if p.function_call]
    assert [p.function_call.id for p in calls] == ['fc-2']
    assert calls[0].model_dump(mode='json')['thought_signature'] == gm.SKIP_THOUGHT_SIGNATURE


@pytest.mark.asyncio
async def test_edited_or_foreign_tool_rounds_are_rebuilt_from_tool_calls(owui):
    messages = continued_round_messages(PARALLEL_DETAIL)
    messages[1]['content'] = 'I edited this.'
    content_builder = builder(messages, owui)
    contents = await content_builder.build_contents()
    assert contents[1].parts[0].text == 'I edited this.'
    calls = [p for p in contents[1].parts if p.function_call]
    assert [(p.function_call.id, p.function_call.args) for p in calls] == [
        ('fc-1', {'city': 'Paris'}),
        ('fc-2', {'tz': 'CET'}),
    ]
    assert any('edit was detected' in msg for _, msg in content_builder.event_emitter.toasts)


@pytest.mark.asyncio
async def test_server_side_tool_parts_are_replayed_only_with_tool_context_circulation(owui):
    detail = round_detail(
        types.Part(
            tool_call=types.ToolCall(id='t1', tool_type=types.ToolType.GOOGLE_SEARCH_WEB, args={'queries': ['x']}),
            thought_signature=b'search',
        ),
        types.Part(
            tool_response=types.ToolResponse(id='t1', tool_type=types.ToolType.GOOGLE_SEARCH_WEB, response={}),
            thought_signature=b'result',
        ),
        types.Part(text='Found it.', thought_signature=b'text'),
    )
    messages = [
        {'role': 'user', 'content': 'Search x'},
        {'role': 'assistant', 'content': 'Found it.', 'reasoning_details': [detail]},
        {'role': 'user', 'content': 'More?'},
    ]
    with_circulation = await builder(messages, owui, replay_server_side_parts=True).build_contents()
    without = await builder(messages, owui).build_contents()
    assert [bool(p.tool_call or p.tool_response) for p in with_circulation[1].parts] == [True, True, False]
    assert [p.text for p in without[1].parts] == ['Found it.']
    assert without[1].parts[0].thought_signature == b'text'


@pytest.mark.asyncio
async def test_text_signatures_survive_citation_markers_and_maps_attribution(owui):
    detail = round_detail(types.Part(text='Paris is the capital.', thought_signature=b'sig'))
    attribution = gm._maps_attribution(maps_grounding())
    visible = f'Paris is the capital.[1]{attribution}'
    messages = [
        {'role': 'user', 'content': 'Capital?'},
        {'role': 'assistant', 'content': visible, 'reasoning_details': [detail]},
        {'role': 'user', 'content': 'Thanks'},
    ]
    contents = await builder(messages, owui).build_contents()
    assert contents[1].parts == [types.Part(text='Paris is the capital.', thought_signature=b'sig')]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'model_id, features, toggles, vertex, combined',
    [
        ('gemini-4-carbon', {'google_search_tool': True}, [], False, True),
        ('gemini-3.8-flash', {}, ['gemini_url_context_toggle'], False, True),
        ('gemini-2.5-flash', {'google_search_tool': True}, [], False, False),
        ('gemini-4-carbon', {'google_search_tool': True}, [], True, False),
    ],
)
async def test_builtin_tools_with_functions_need_gemini_3_and_circulation(
    owui, model_id, features, toggles, vertex, combined
):
    pipe = Pipe()
    meta = metadata(
        canonical_model_id=model_id,
        features=features,
        is_vertex_ai=vertex,
        filter_ids=toggle(owui, *toggles),
        tools={'get_weather': {'type': 'builtin'}},
    )
    bridge = gm.ToolBridge.for_request({'tools': TOOLS}, meta, accepts_functions=True)
    emitter = RecordingEmitter()
    config = await pipe._build_gen_content_config({}, meta, Pipe.Valves(), emitter, bridge)
    builtin = [tool for tool in config.tools if not tool.function_declarations]
    declared = [tool for tool in config.tools if tool.function_declarations]
    assert len(builtin) == 1
    assert bool(declared) is combined
    circulation = bool(config.tool_config and config.tool_config.include_server_side_tool_invocations)
    assert circulation is combined
    # Only Open WebUI's built-in tools were dropped, which is not worth a toast.
    assert emitter.toasts == []


@pytest.mark.asyncio
async def test_dropping_user_selected_tools_is_announced(owui):
    meta = metadata(
        canonical_model_id='gemini-2.5-pro', features={'google_search_tool': True}, tools={'crm': {'type': 'external'}}
    )
    bridge = gm.ToolBridge.for_request({'tools': [tool_spec('crm', q='string')]}, meta, accepts_functions=True)
    emitter = RecordingEmitter()
    await Pipe()._build_gen_content_config({}, meta, Pipe.Valves(), emitter, bridge)
    assert emitter.toasts and 'function tools' in emitter.toasts[0][1]


@pytest.mark.asyncio
async def test_background_tasks_get_no_tools_except_compaction_with_mode_none(owui):
    pipe = Pipe()
    body = {'tools': TOOLS}
    features = {'google_search_tool': True}
    title_meta = metadata(canonical_model_id='gemini-4-carbon', task='title_generation', features=features)
    bridge = gm.ToolBridge.for_request(body, title_meta, accepts_functions=True)
    config = await pipe._build_gen_content_config(body, title_meta, Pipe.Valves(), RecordingEmitter(), bridge)
    assert config.tools is None

    compaction_meta = metadata(canonical_model_id='gemini-4-carbon', task='context_compaction', features=features)
    bridge = gm.ToolBridge.for_request(body, compaction_meta, accepts_functions=True)
    config = await pipe._build_gen_content_config(body, compaction_meta, Pipe.Valves(), RecordingEmitter(), bridge)
    assert [tool.google_search for tool in config.tools] == [None]
    assert config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.NONE


@pytest.mark.asyncio
async def test_file_search_tool_and_its_precedence(owui):
    pipe = Pipe()
    options = Pipe.Valves(
        FILE_SEARCH_STORE_NAMES='fileSearchStores/a, fileSearchStores/b', FILE_SEARCH_METADATA_FILTER='year>2020'
    )
    meta = metadata(canonical_model_id='gemini-4-carbon')
    config = await pipe._build_gen_content_config({}, meta, options, RecordingEmitter(), gm.ToolBridge())
    assert config.tools[0].file_search.file_search_store_names == ['fileSearchStores/a', 'fileSearchStores/b']
    assert config.tools[0].file_search.metadata_filter == 'year>2020'

    meta['features'] = {'google_search_tool': True}
    emitter = RecordingEmitter()
    config = await pipe._build_gen_content_config({}, meta, options, emitter, gm.ToolBridge())
    assert [tool.file_search for tool in config.tools] == [None]
    assert 'File Search cannot be combined' in emitter.toasts[0][1]


# endregion Native tool calling


# region Usage


def test_usage_includes_thoughts_cache_and_tool_prompt_in_owui_shape():
    pipe = Pipe()
    meta = metadata(canonical_model_id='gemini-4-hydrogen', is_paid_api=True, cumulative_tokens=None)
    chunk = response(
        types.Part(text='x'), usage=usage(prompt=1000, candidates=200, thoughts=300, cached=400, tool_prompt=100)
    )
    data = pipe._get_usage_data(chunk, meta, time.monotonic())
    assert (data['input_tokens'], data['output_tokens'], data['total_tokens']) == (1100, 500, 1600)
    assert (data['prompt_tokens'], data['completion_tokens']) == (1100, 500)
    assert data['cache_read_input_tokens'] == 400
    assert data['prompt_tokens_details'] == {'cached_tokens': 400}
    assert data['completion_tokens_details'] == {'reasoning_tokens': 300}
    # Hydrogen: input 1.50, cached 0.075, output 7.50 per million; thinking is billed as output.
    expected_input = (700 * 1.50 + 400 * 0.075) / 1e6
    assert data['input_cost'] == pytest.approx(expected_input, abs=1e-6)
    assert data['output_cost'] == pytest.approx(500 * 7.50 / 1e6, abs=1e-6)
    assert data['total_cost'] == pytest.approx(expected_input + 500 * 7.50 / 1e6, abs=1e-6)
    assert 'cumulative_token_count' not in data


def test_usage_tiers_follow_the_prompt_size_and_free_tier_is_free():
    pipe = Pipe()
    long_prompt = response(types.Part(text='x'), usage=usage(prompt=300_000, candidates=1000, thoughts=0))
    data = pipe._get_usage_data(long_prompt, metadata(canonical_model_id='gemini-2.5-pro', is_paid_api=True), 0)
    # Over 200k tokens, the whole request is billed at the higher tier (2.50 input, 15.00 output).
    assert data['input_cost'] == pytest.approx(300_000 * 2.50 / 1e6)
    assert data['output_cost'] == pytest.approx(1000 * 15.00 / 1e6)
    free = pipe._get_usage_data(long_prompt, metadata(canonical_model_id='gemini-2.5-pro', is_paid_api=False), 0)
    assert free['total_cost'] == 0.0


def test_cumulative_usage_spans_the_rounds_of_a_tool_loop():
    pipe = Pipe()
    meta = metadata(
        canonical_model_id='gemini-2.5-flash', is_paid_api=False, cumulative_tokens=1000, cumulative_cost=0.0
    )
    first = pipe._get_usage_data(response(usage=usage(prompt=100, candidates=10, thoughts=0)), meta, 0)
    second = pipe._get_usage_data(response(usage=usage(prompt=200, candidates=20, thoughts=0)), meta, 0)
    assert first['cumulative_token_count'] == 1110
    assert second['cumulative_token_count'] == 1330


# endregion Usage


# region Grounding attributions


def search_grounding(
    html: str = '<style>.chip{}</style><div class="container"><a class="chip" href="https://www.google.com/search?q=x">x</a></div>',
):
    return types.GroundingMetadata(
        web_search_queries=['x'],
        search_entry_point=types.SearchEntryPoint(rendered_content=html),
        grounding_chunks=[
            types.GroundingChunk(web=types.GroundingChunkWeb(uri='https://example.com', title='Example'))
        ],
        grounding_supports=[
            types.GroundingSupport(
                segment=types.Segment(start_index=0, end_index=5, text='Paris'), grounding_chunk_indices=[0]
            )
        ],
    )


def maps_grounding():
    return types.GroundingMetadata(
        grounding_chunks=[
            types.GroundingChunk(
                maps=types.GroundingChunkMaps(
                    uri='https://maps.google.com/?cid=1',
                    title='Café [Central]',
                    place_answer_sources=types.GroundingChunkMapsPlaceAnswerSources(
                        review_snippets=[
                            types.GroundingChunkMapsPlaceAnswerSourcesReviewSnippet(
                                google_maps_uri='https://maps.google.com/review/1', title='Great coffee'
                            )
                        ]
                    ),
                )
            )
        ]
    )


@pytest.mark.asyncio
async def test_search_suggestions_are_rendered_unmodified_in_an_embed(pipe):
    html = search_grounding().search_entry_point.rendered_content
    rounds = [
        [
            response(
                types.Part(text='Paris is nice.'),
                finish=types.FinishReason.STOP,
                usage=usage(),
                grounding=search_grounding(),
            )
        ]
    ]
    pipe.fake_client = FakeClient(rounds=rounds)
    body = {
        'model': 'gemini_manifold_plus.gemini-4-carbon',
        'messages': [{'role': 'user', 'content': 'Paris?'}],
        'stream': True,
    }
    request = make_request()
    chunks = await collect(await pipe.pipe(body, USER, request, metadata(features={'google_search_tool': True})))
    events = [chunk['event'] for chunk in chunks if 'event' in chunk]
    assert len(events) == 1 and events[0]['type'] == 'embeds' and events[0]['data']['replace'] is True
    embed = events[0]['data']['embeds'][0]
    assert html in embed  # Byte for byte.
    assert '<base target="_blank">' in embed and 'iframe:height' in embed
    # Grounded text is stored for replay (the displayed text gets citation markers).
    assert any(d['delta'].get('reasoning_details') for d in deltas(chunks))
    # The final round's grounding is left for the companion filter's citations.
    assert request.app.state._state['grounding_local:chat-1_msg-1'] == search_grounding()


def test_search_suggestions_accumulate_per_response_up_to_five():
    pipe = Pipe()
    meta = metadata()
    for i in range(7):
        embeds = pipe._search_suggestions(search_grounding(f'<div>{i}</div>'), meta)
    assert len(embeds) == 5 and '<div>0</div>' in embeds[0] and '<div>4</div>' in embeds[-1]
    assert pipe._search_suggestions(types.GroundingMetadata(), meta) is None


def test_maps_sources_follow_the_text_with_google_maps_attribution():
    attribution = gm._maps_attribution(maps_grounding())
    assert attribution.startswith('\n\n---\n')
    lines = attribution.strip().split('\n')[1:]
    assert lines == [
        r'- [Café \[Central\]](https://maps.google.com/?cid=1) · Google' + '\u00a0Maps',
        '- [Great coffee](https://maps.google.com/review/1) · Google\u00a0Maps',
    ]
    assert gm._maps_attribution(search_grounding()) is None
    assert gm._comparable_text(f'Text.[1][2]{attribution}') == 'Text.'


@pytest.mark.asyncio
async def test_maps_attribution_is_streamed_right_after_the_round_text(pipe):
    rounds = [
        [response(types.Part(text='Try Café Central.'), finish=types.FinishReason.STOP, grounding=maps_grounding())]
    ]
    pipe.fake_client = FakeClient(rounds=rounds)
    body = {'model': 'gemini_manifold_plus.gemini-4-carbon', 'messages': [{'role': 'user', 'content': 'Coffee?'}]}
    chunks = await collect(await pipe.pipe(body, USER, make_request(), metadata()))
    contents = [d['delta']['content'] for d in deltas(chunks) if d['delta'].get('content')]
    assert contents[0] == 'Try Café Central.' and 'Google\u00a0Maps' in contents[1]


# endregion Grounding attributions


# region Image generation


@pytest.mark.parametrize(
    'model_id, resolution, aspect_ratio, expected, warned',
    [
        ('gemini-nano-banana-2.1', '4K', '21:9', {'image_size': '4K', 'aspect_ratio': '21:9'}, False),
        ('gemini-nano-banana-2.1', None, '8:1', {'aspect_ratio': '8:1'}, False),
        ('gemini-3.1-flash-image', '512PX', '1:4', {'image_size': '512', 'aspect_ratio': '1:4'}, False),
        ('gemini-3.1-flash-lite-image', '2K', '1:4', None, True),
        ('gemini-2.5-flash-image', '1K', '16:9', {'aspect_ratio': '16:9'}, True),
        ('gemini-3-pro-image', None, None, None, False),
    ],
)
def test_image_config_sends_only_supported_values(model_id, resolution, aspect_ratio, expected, warned):
    emitter = RecordingEmitter()
    options = Pipe.Valves(IMAGE_RESOLUTION=resolution, IMAGE_ASPECT_RATIO=aspect_ratio)
    config = Pipe()._image_config(model_id, options, emitter)
    assert (config.model_dump(exclude_none=True) if config else None) == expected
    assert bool(emitter.toasts) is warned


@pytest.mark.asyncio
async def test_image_search_grounding_on_models_that_support_it(owui):
    pipe = Pipe()
    for model_id, image_search in [('gemini-nano-banana-2.1', True), ('gemini-3-pro-image', False)]:
        meta = metadata(canonical_model_id=model_id, features={'google_search_tool': True})
        config = await pipe._build_gen_content_config({}, meta, Pipe.Valves(), RecordingEmitter(), gm.ToolBridge())
        search = config.tools[0].google_search
        assert (search.search_types is not None) is image_search
        assert config.response_modalities == ['TEXT', 'IMAGE']


@pytest.mark.asyncio
async def test_reference_images_beyond_the_model_limit_are_dropped_oldest_first(owui):
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}}
    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'first'}, image, image]},
        {'role': 'assistant', 'content': 'ok'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'second'}, image, image]},
    ]
    content_builder = builder(messages, owui, model_id='gemini-nano-banana-2.1', max_images=3)
    contents = await content_builder.build_contents()
    counts = [sum(1 for p in c.parts if p.inline_data) for c in contents]
    assert counts == [1, 0, 2]
    assert content_builder.event_emitter.toasts


@pytest.mark.asyncio
async def test_generated_images_are_stored_and_replayed_with_their_signature(pipe, owui):
    image_part = types.Part(
        inline_data=types.Blob(data=b'PNGDATA', mime_type='image/png'), thought_signature=b'img-sig'
    )
    rounds = [[response(types.Part(text='Here it is.'), image_part, finish=types.FinishReason.STOP)]]
    pipe.fake_client = FakeClient(rounds=rounds)
    meta = metadata(chat_id='chat-1')
    body = {'model': 'gemini_manifold_plus.gemini-nano-banana-2.1', 'messages': [{'role': 'user', 'content': 'A cat'}]}
    chunks = await collect(await pipe.pipe(body, USER, make_request(), meta))
    texts = ''.join(d['delta'].get('content') or '' for d in deltas(chunks))
    file_id = re.search(r'/api/v1/files/([\w-]+)/content', texts).group(1)
    assert owui.files.files[file_id].meta['content_type'] == 'image/png'
    assert owui.chats.chat_files[0]['file_id'] == file_id  # Linked to the chat like Open WebUI's own images.
    detail = next(d['delta']['reasoning_details'][0] for d in deltas(chunks) if d['delta'].get('reasoning_details'))
    stored = stored_content(detail).parts[1]
    assert stored.file_data.file_uri == f'/api/v1/files/{file_id}/content' and stored.thought_signature == b'img-sig'

    # The next turn (multi-turn editing) sends the image back as bytes with its signature.
    owui.storage.local_paths[owui.files.files[file_id].path] = __file__  # Any readable file stands in.
    messages = [
        {'role': 'user', 'content': 'A cat'},
        {'role': 'assistant', 'content': texts, 'reasoning_details': [detail]},
        {'role': 'user', 'content': 'Make it blue'},
    ]
    contents = await builder(messages, owui, model_id='gemini-nano-banana-2.1').build_contents()
    replayed = contents[1].parts[1]
    assert replayed.inline_data is not None and replayed.thought_signature == b'img-sig'


# endregion Image generation


# region Model list


@pytest.mark.asyncio
async def test_model_list_keeps_chat_and_image_models_and_adds_enabled_interactions_models(pipe):
    names = [
        'gemini-4-carbon',
        'gemini-3.8-flash',
        'gemini-nano-banana-2.1',
        'gemma-4-31b-it',
        'gemini-3.8-flash-tts',
        'gemini-3.8-live',
        'gemini-2.5-flash-native-audio-preview-12-2025',
        'gemini-3.5-transcribe',
        'gemini-embedding-2',
        'gemini-robotics-er-2-preview',
        'gemini-omni-1.2-flash',
        'gemini-decision-1',
        'lyria-3.5',
    ]
    models = [types.Model(name=f'models/{name}', supported_actions=['generateContent']) for name in names]
    models.append(types.Model(name='models/text-embedding-x', supported_actions=['embedContent']))
    pipe.fake_client = FakeClient(models=models)
    listed = [model['id'] for model in await pipe.pipes()]
    assert listed == ['gemini-4-carbon', 'gemini-3.8-flash', 'gemini-nano-banana-2.1', 'gemma-4-31b-it']

    await pipe._get_genai_models.cache.clear()
    pipe.valves.ENABLE_DEEP_RESEARCH = True
    pipe.valves.ENABLE_OMNI_VIDEO = True
    listed = [model['id'] for model in await pipe.pipes()]
    assert listed[-3:] == [*gm.DEEP_RESEARCH_AGENTS, gm.OMNI_VIDEO_MODEL]


@pytest.mark.parametrize(
    'model, expected',
    [
        ('gemini_manifold_plus.gemini-3.8-flash', 'gemini-3.8-flash'),
        ('my_custom_id.gemini-4-carbon', 'gemini-4-carbon'),
        ('gemini-3.8-flash', 'gemini-3.8-flash'),
    ],
)
def test_model_name_prefix_is_stripped_for_any_function_id(model, expected):
    assert Pipe._get_model_name({'model': model}) == expected


# endregion Model list


# region Interactions API models


def interaction(status: str, steps=(), usage=None, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(id='int-1', status=status, steps=list(steps), errors=None, usage=usage, **extra)


@pytest.mark.asyncio
async def test_deep_research_polls_shows_thoughts_and_returns_the_cited_report(pipe, monkeypatch):
    monkeypatch.setattr(gm.InteractionsRunner, 'POLL_INTERVAL', 0)
    thought = SimpleNamespace(type='thought', summary=[SimpleNamespace(text='Searching sources')])
    citation = SimpleNamespace(type='url_citation', url='https://example.com/a', title='Source A')
    report = SimpleNamespace(
        type='model_output', content=[SimpleNamespace(type='text', text='The report.', annotations=[citation])]
    )
    done = interaction(
        'completed',
        [thought, report],
        usage=SimpleNamespace(
            total_input_tokens=10,
            total_output_tokens=5,
            total_thought_tokens=3,
            total_tokens=18,
            total_cached_tokens=0,
        ),
    )
    pipe.valves.ENABLE_DEEP_RESEARCH = True
    pipe.fake_client = FakeClient(
        interactions=[interaction('in_progress'), interaction('in_progress', [thought]), done]
    )
    agent = 'deep-research-preview-04-2026'
    body = {'model': f'gemini_manifold_plus.{agent}', 'messages': [{'role': 'user', 'content': 'Research TPUs'}]}
    chunks = await collect(await pipe.pipe(body, USER, make_request(), metadata()))

    created = pipe.fake_client.aio.interactions.created[0]
    assert created['agent'] == agent and created['background'] is True
    assert created['input'] == [{'type': 'text', 'text': 'Research TPUs'}]
    items = deltas(chunks)
    assert items[0]['delta']['reasoning_details'][0] == {
        'type': 'reasoning.encrypted',
        'format': gm.INTERACTION_FORMAT,
        'index': 0,
        'data': 'int-1',
    }
    assert ''.join(d['delta'].get('reasoning_content') or '' for d in items) == 'Searching sources\n\n'
    assert [d['delta']['content'] for d in items if d['delta'].get('content')] == ['The report.']
    sources = [chunk['event'] for chunk in chunks if 'event' in chunk]
    assert sources[0]['type'] == 'source' and sources[0]['data']['source']['url'] == 'https://example.com/a'
    usage_chunk = next(chunk['usage'] for chunk in chunks if 'usage' in chunk)
    assert (usage_chunk['input_tokens'], usage_chunk['output_tokens']) == (10, 8)

    # A follow-up continues the interaction.
    follow_up = [
        *body['messages'],
        {
            'role': 'assistant',
            'content': 'The report.',
            'reasoning_details': [items[0]['delta']['reasoning_details'][0]],
        },
    ]
    assert gm._previous_interaction_id(follow_up) == 'int-1'


@pytest.mark.asyncio
async def test_interactions_models_run_only_when_the_admin_enabled_them(pipe):
    pipe.fake_client = FakeClient(interactions=[interaction('completed')])
    body = {
        'model': 'gemini_manifold_plus.deep-research-preview-04-2026',
        'messages': [{'role': 'user', 'content': 'x'}],
    }
    with pytest.raises(ValueError, match='ENABLE_DEEP_RESEARCH'):
        await pipe.pipe(body, USER, make_request(), metadata())
    assert pipe.fake_client.aio.interactions.created == []


@pytest.mark.asyncio
async def test_stopping_deep_research_cancels_the_background_task(pipe, monkeypatch):
    monkeypatch.setattr(gm.InteractionsRunner, 'POLL_INTERVAL', 0)
    pipe.valves.ENABLE_DEEP_RESEARCH = True
    pipe.fake_client = FakeClient(interactions=[interaction('in_progress')] * 50)

    async def stopped_get(id: str, **kwargs: Any):
        raise asyncio.CancelledError

    pipe.fake_client.aio.interactions.get = stopped_get
    body = {
        'model': 'gemini_manifold_plus.deep-research-max-preview-04-2026',
        'messages': [{'role': 'user', 'content': 'x'}],
    }
    with pytest.raises(asyncio.CancelledError):
        await collect(await pipe.pipe(body, USER, make_request(), metadata()))
    assert pipe.fake_client.aio.interactions.cancelled == ['int-1']


@pytest.mark.asyncio
async def test_omni_video_is_stored_attached_and_continuable(pipe, owui):
    import base64

    video = SimpleNamespace(data=base64.b64encode(b'MP4DATA').decode(), uri=None, mime_type='video/mp4')
    pipe.valves.ENABLE_OMNI_VIDEO = True
    pipe.valves.OMNI_VIDEO_DURATION = 8
    pipe.fake_client = FakeClient(interactions=[interaction('completed', output_video=video)])
    previous = {'type': 'reasoning.encrypted', 'format': gm.INTERACTION_FORMAT, 'index': 0, 'data': 'int-0'}
    body = {
        'model': f'gemini_manifold_plus.{gm.OMNI_VIDEO_MODEL}',
        'messages': [
            {'role': 'user', 'content': 'A drone shot'},
            {
                'role': 'assistant',
                'content': '<video>/api/v1/files/old/content</video>',
                'reasoning_details': [previous],
            },
            {'role': 'user', 'content': 'Make it longer'},
        ],
    }
    chunks = await collect(await pipe.pipe(body, USER, make_request(), metadata(chat_id='chat-1')))
    created = pipe.fake_client.aio.interactions.created[0]
    assert created['previous_interaction_id'] == 'int-0'
    assert created['response_format'] == {'type': 'video', 'delivery': 'uri', 'duration': '8s'}
    files_event = next(chunk['event'] for chunk in chunks if 'event' in chunk)
    entry = files_event['data']['files'][0]
    assert files_event['type'] == 'files' and entry['content_type'] == 'video/mp4'
    assert owui.storage.blobs[owui.files.files[entry['id']].path] == b'MP4DATA'
    assert f'<video>{entry["url"]}</video>' in [d['delta'].get('content') for d in deltas(chunks)]


# endregion Interactions API models


# region Companion filter


@pytest.mark.asyncio
async def test_companion_serves_the_plus_pipe_with_its_embedded_table(monkeypatch):
    def no_download(path):
        raise AssertionError('the YAML must not be downloaded for a pipe with an embedded table')

    monkeypatch.setattr(companion.Filter, '_load_model_config', staticmethod(no_download))
    plus = Pipe()
    request = make_request(plus)
    body = {'model': 'gemini_manifold_plus.gemini-4-carbon', 'features': {'web_search': True, 'code_interpreter': True}}
    meta = metadata()
    companion.Filter().inlet(body, request, meta, None, {'valves': None})
    assert meta['features'] == {
        'web_search': False,
        'google_search_tool': True,
        'code_interpreter': False,
        'google_code_execution': True,
        'gemini_manifold_companion_version': companion.VERSION,
        'upload_documents': False,
    }
    assert 'gemini_model_config' not in request.app.state._state


@pytest.mark.asyncio
async def test_companion_keeps_serving_the_original_pipe(monkeypatch):
    table = {'gemini-4-carbon': {'capabilities': {'search_grounding': False}}}
    monkeypatch.setattr(companion.Filter, '_load_model_config', staticmethod(lambda path: table))
    request = make_request()
    body = {'model': 'gemini_manifold_google_genai.gemini-4-carbon', 'features': {'web_search': True}}
    meta = metadata()
    companion.Filter().inlet(body, request, meta, None, {'valves': None})
    assert meta['features']['web_search'] is True  # Per the downloaded table, no Google Search here.
    assert request.app.state._state['gemini_model_config'] is table
    other = {'model': 'openai.gpt-x', 'features': {'web_search': True}}
    assert companion.Filter().inlet(other, request, metadata(), None, {'valves': None}) is other


def test_companion_permissive_safety_uses_the_documented_categories():
    settings = companion.Filter()._get_permissive_safety_settings()
    assert [s.category.name for s in settings] == [
        'HARM_CATEGORY_HARASSMENT',
        'HARM_CATEGORY_HATE_SPEECH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'HARM_CATEGORY_DANGEROUS_CONTENT',
    ]
    assert {s.threshold for s in settings} == {types.HarmBlockThreshold.OFF}


@pytest.mark.asyncio
async def test_companion_cites_only_the_last_round_of_a_tool_loop():
    request = make_request()
    grounding = types.GroundingMetadata(
        grounding_chunks=[types.GroundingChunk(web=types.GroundingChunkWeb(uri='https://example.com', title='E'))],
        grounding_supports=[
            types.GroundingSupport(
                segment=types.Segment(start_index=0, end_index=5, text='Paris'), grounding_chunk_indices=[0]
            )
        ],
    )
    request.app.state._state['grounding_chat-1_msg-1'] = grounding
    request.app.state._state['gemini_dummy_event_emitter'] = companion.EventEmitter(None)
    output = [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Let me search.'}]},
        {'type': 'function_call', 'call_id': 'a'},
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Paris is lovely.'}]},
    ]
    body = {'messages': [{'role': 'assistant', 'content': 'Let me search.\nParis is lovely.', 'output': output}]}
    filter_ = companion.Filter()

    async def no_sources(**kwargs):
        return None

    filter_._resolve_and_emit_sources = no_sources  # No URL resolution (network) in tests.
    await filter_.outlet(body, request, metadata(chat_id='chat-1'), None, {'valves': None})
    assert output[0]['content'][0]['text'] == 'Let me search.'
    assert output[2]['content'][0]['text'] == 'Paris[1] is lovely.'
    assert body['messages'][0]['content'] == 'Let me search.\nParis[1] is lovely.'


# endregion Companion filter


# region Packaging


def test_embedded_model_table_matches_the_yaml_file():
    yaml_text = (REPO / 'plugins' / 'pipes' / 'gemini_models.yaml').read_text(encoding='utf-8')
    assert gm._MODEL_TABLE_YAML.strip() == yaml_text.strip()


def test_no_text_that_open_webui_rewrites_as_an_import():
    """Open WebUI replaces 'from utils', 'from apps', 'from main' and 'from config' anywhere in a plugin."""
    source = (REPO / 'plugins' / 'pipes' / 'gemini_manifold.py').read_text(encoding='utf-8')
    hits = [m.group(0) for m in re.finditer(r'from (utils|apps|main|config)\b.*', source)]
    assert hits == ['from utils.manifold_types import (']  # Only inside `if TYPE_CHECKING:`.


def test_frontmatter_identifies_the_plus_edition():
    source = (REPO / 'plugins' / 'pipes' / 'gemini_manifold.py').read_text(encoding='utf-8')
    frontmatter = dict(line.split(': ', 1) for line in source.split('"""')[1].strip().splitlines())
    assert frontmatter['id'] == 'gemini_manifold_plus'
    assert frontmatter['version'] == gm.VERSION
    assert frontmatter['requirements'] == 'google-genai==3.3.0'
    assert all(len(line) <= 120 for line in source.split('"""')[1].splitlines())


# endregion Packaging
