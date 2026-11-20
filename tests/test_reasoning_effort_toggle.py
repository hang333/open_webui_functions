"""Tests of the Reasoning Effort toggle filter (plugins/filters/reasoning_effort_toggle.py)."""

from __future__ import annotations

import pytest

from plugins.filters.reasoning_effort_toggle import ICON, Filter


def make(level: str) -> Filter:
    flt = Filter()
    flt.valves = Filter.Valves(LEVEL=level)
    return flt


def test_is_a_toggle_with_an_icon():
    flt = Filter()
    assert flt.toggle is True
    assert flt.icon == ICON and ICON.startswith('data:image/svg+xml;base64,')
    assert flt.valves.LEVEL == 'xhigh'


def test_sets_the_level_and_overrides_the_chat_parameter():
    body = {'model': 'm', 'messages': [], 'reasoning_effort': 'low'}
    assert make('xhigh').inlet(body, {})['reasoning_effort'] == 'xhigh'
    assert make('low').inlet({'messages': []}, {})['reasoning_effort'] == 'low'


@pytest.mark.parametrize('order', [('low', 'xhigh'), ('xhigh', 'low')])
def test_the_highest_level_wins_when_several_toggles_are_on(order):
    body: dict = {'messages': []}
    metadata: dict = {}  # Open WebUI passes the same __metadata__ dict to every filter of a request.
    for level in order:
        body = make(level).inlet(body, metadata)
    assert body['reasoning_effort'] == 'xhigh'


def test_works_without_metadata():
    assert make('medium').inlet({'messages': []})['reasoning_effort'] == 'medium'
    assert make('high').inlet({'messages': []}, None)['reasoning_effort'] == 'high'


def test_rejects_unknown_levels():
    with pytest.raises(ValueError):
        Filter.Valves(LEVEL='max')
