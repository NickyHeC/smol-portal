"""Tests for dataset text extraction and schema validation."""

from __future__ import annotations

import pytest

from portal.data import DatasetSchemaError, extract_text


def test_text_field_wins_and_matches_legacy():
    # Historical precedence: `text` first, then `input`.
    assert extract_text({"text": "hello", "input": "ignored"}) == "hello"
    assert extract_text({"input": "fallback"}) == "fallback"


def test_additional_single_fields():
    assert extract_text({"content": "c"}) == "c"
    assert extract_text({"sentence": "s"}) == "s"
    assert extract_text({"document": "d"}) == "d"


def test_instruction_response_pair_is_combined():
    out = extract_text({"instruction": "Summarize.", "output": "A summary."})
    assert "Summarize." in out and "A summary." in out


def test_instruction_only():
    assert extract_text({"prompt": "Just a prompt"}) == "Just a prompt"


def test_chat_messages_joined():
    example = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    }
    assert extract_text(example) == "hi\nyo"


def test_empty_strings_are_skipped_not_returned():
    # A present-but-empty `text` should fall through to the next usable field.
    assert extract_text({"text": "   ", "input": "real"}) == "real"


def test_strict_raises_on_unknown_schema():
    with pytest.raises(DatasetSchemaError) as exc:
        extract_text({"label": 1, "foo": "bar"})
    # Error should name the available keys to aid debugging.
    assert "label" in str(exc.value) and "foo" in str(exc.value)


def test_non_strict_returns_empty():
    assert extract_text({"label": 1}, strict=False) == ""
