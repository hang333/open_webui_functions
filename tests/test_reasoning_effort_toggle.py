"""Tests of the Thinking toggle filter (plugins/filters/reasoning_effort_toggle.py)."""

from __future__ import annotations

import pytest

from plugins.filters.reasoning_effort_toggle import DEFAULT_LEVEL, ICON, LEVELS, Filter


def test_is_a_toggle_with_an_icon_and_user_valves():
    flt = Filter()
    assert flt.toggle is True
    assert flt.icon == ICON and ICON.startswith('data:image/svg+xml;base64,')
    # User valves make Open WebUI open the level picker when the chat input badge is clicked.
    assert hasattr(Filter, 'UserValves')


def test_the_level_picker_is_a_single_choice_enum():
    schema = Filter.UserValves.model_json_schema()['properties']['LEVEL']
    assert schema['enum'] == list(LEVELS)  # Rendered as a dropdown: one level at a time.
    assert schema['default'] == DEFAULT_LEVEL


@pytest.mark.parametrize(('label', 'effort'), list(LEVELS.items()))
def test_sends_the_users_level_and_overrides_the_chat_parameter(label, effort):
    body = {'messages': [], 'reasoning_effort': 'none'}
    user = {'id': 'u', 'valves': Filter.UserValves(LEVEL=label)}
    assert Filter().inlet(body, user)['reasoning_effort'] == effort


def test_defaults_when_the_user_has_not_picked_a_level():
    expected = LEVELS[DEFAULT_LEVEL]
    assert Filter().inlet({'messages': []}, {'id': 'u', 'valves': Filter.UserValves()})['reasoning_effort'] == expected
    assert Filter().inlet({'messages': []}, {'id': 'u'})['reasoning_effort'] == expected
    assert Filter().inlet({'messages': []})['reasoning_effort'] == expected


def test_accepts_user_valves_as_a_dict_and_ignores_unknown_levels():
    assert Filter().inlet({'messages': []}, {'valves': {'LEVEL': '快速'}})['reasoning_effort'] == 'low'
    assert Filter().inlet({'messages': []}, {'valves': {'LEVEL': 'max'}})['reasoning_effort'] == LEVELS[DEFAULT_LEVEL]


def test_rejects_unknown_levels_in_the_picker():
    with pytest.raises(ValueError):
        Filter.UserValves(LEVEL='max')
