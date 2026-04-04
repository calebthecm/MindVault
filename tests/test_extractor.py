"""Tests for entity extraction — tests parsing logic, not the LLM call."""

from src.memory.extractor import deduplicate_entities, parse_entity_response


def test_parse_valid_json_array():
    raw = '[{"type": "project", "name": "MindVault", "value": "second brain"}]'
    result = parse_entity_response(raw)
    assert len(result) == 1
    assert result[0]["name"] == "MindVault"
    assert result[0]["type"] == "project"


def test_parse_json_with_surrounding_text():
    raw = 'Here are the entities:\n[{"type": "decision", "name": "charge retainer", "value": "$500/month"}]'
    result = parse_entity_response(raw)
    assert len(result) == 1
    assert result[0]["type"] == "decision"


def test_parse_empty_array():
    result = parse_entity_response("[]")
    assert result == []


def test_parse_invalid_returns_empty():
    result = parse_entity_response("not json at all")
    assert result == []


def test_parse_strips_bad_items():
    raw = '[{"type": "project"}, {"type": "fact", "name": "Python 3.14", "value": "installed"}]'
    result = parse_entity_response(raw)
    assert len(result) == 1
    assert result[0]["name"] == "Python 3.14"


def test_deduplicate_same_type_and_name():
    entities = [
        {"type": "project", "name": "MindVault", "value": "v1"},
        {"type": "project", "name": "mindvault", "value": "v2"},
        {"type": "project", "name": "BrainRAG", "value": "other"},
    ]
    result = deduplicate_entities(entities)
    assert len(result) == 2
    names = {e["name"] for e in result}
    assert "BrainRAG" in names


def test_deduplicate_different_types_kept():
    entities = [
        {"type": "project", "name": "MindVault", "value": ""},
        {"type": "decision", "name": "MindVault", "value": "ship it"},
    ]
    result = deduplicate_entities(entities)
    assert len(result) == 2
