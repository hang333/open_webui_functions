"""Tests of the Google Search toggle filter (plugins/filters/gemini_search_toggle.py)."""

from __future__ import annotations

from plugins.filters.gemini_search_toggle import ICON, Filter


def test_is_a_toggle_with_an_icon():
    flt = Filter()
    assert flt.toggle is True
    assert flt.icon == ICON and ICON.startswith('data:image/svg+xml;base64,')


def test_sets_the_flag_in_a_features_dict_shared_with_metadata():
    body: dict = {'messages': []}
    metadata: dict = {'features': {}}  # Open WebUI's detached snapshot.
    body = Filter().inlet(body, metadata)
    assert body['features'] == {'google_search_tool': True}
    assert metadata['features'] is body['features']


def test_keeps_other_features_and_the_companions_shared_dict():
    features = {'web_search': False, 'gemini_manifold_companion_version': '2.2.0'}
    body: dict = {'messages': [], 'features': features}
    metadata: dict = {'features': features}
    body = Filter().inlet(body, metadata)
    assert body['features'] is features is metadata['features']
    assert features == {'web_search': False, 'gemini_manifold_companion_version': '2.2.0', 'google_search_tool': True}


def test_works_without_metadata():
    assert Filter().inlet({'messages': []})['features'] == {'google_search_tool': True}
