"""Dataset text extraction with explicit, validated field precedence.

The pipeline previously read training/calibration/eval text with
``example.get("text") or example.get("input", "")``. Any dataset that stored
its text under a different field (``instruction``, ``prompt``, ``messages``, …)
silently became an empty string, so training/eval ran on nothing and still
"succeeded". This module makes field resolution explicit and fails loudly when
no usable text is found, instead of degrading to empty strings.

This is intentionally a small, explicit resolver rather than a full
dataset-adapter registry (see ROADMAP Phase A2). It preserves the historical
precedence — ``text`` first, then ``input`` — so previously validated runs
(e.g. IMDB) are unaffected.
"""

from __future__ import annotations

from collections.abc import Mapping

# Single fields that already contain a complete text example, in priority order.
# ``text`` and ``input`` come first to preserve prior behaviour.
TEXT_FIELDS: tuple[str, ...] = ("text", "input", "content", "sentence", "document", "body")

# Instruction-style datasets: an instruction/prompt paired with a response.
INSTRUCTION_FIELDS: tuple[str, ...] = ("instruction", "prompt", "question")
RESPONSE_FIELDS: tuple[str, ...] = ("response", "output", "answer", "completion")


class DatasetSchemaError(ValueError):
    """Raised when a dataset example has no field we know how to read as text."""


def _first_nonempty_str(example: Mapping, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _messages_to_text(example: Mapping) -> str | None:
    """Join chat-style ``messages`` (list of ``{role, content}``) into one string."""
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    parts = [
        msg["content"]
        for msg in messages
        if isinstance(msg, Mapping) and isinstance(msg.get("content"), str)
    ]
    joined = "\n".join(part for part in parts if part.strip())
    return joined or None


def extract_text(example: Mapping, *, strict: bool = True) -> str:
    """Resolve a single training/eval text string from a dataset example.

    Resolution order:
        1. a complete single text field (``text``, ``input``, ``content``, …);
        2. chat-style ``messages`` joined into one string;
        3. an instruction field optionally combined with a response field.

    Args:
        example: one row of a HuggingFace dataset (a mapping of column → value).
        strict: when True (default) raise :class:`DatasetSchemaError` if nothing
            usable is found; when False return an empty string (legacy behaviour).

    Raises:
        DatasetSchemaError: if ``strict`` and no usable text field is present.
    """
    direct = _first_nonempty_str(example, TEXT_FIELDS)
    if direct is not None:
        return direct

    chat = _messages_to_text(example)
    if chat is not None:
        return chat

    instruction = _first_nonempty_str(example, INSTRUCTION_FIELDS)
    if instruction is not None:
        response = _first_nonempty_str(example, RESPONSE_FIELDS)
        return f"{instruction}\n\n{response}" if response is not None else instruction

    if strict:
        known = TEXT_FIELDS + ("messages",) + INSTRUCTION_FIELDS
        available = sorted(example.keys())
        raise DatasetSchemaError(
            "No usable text field found in dataset example. "
            f"Looked for {list(known)} (+ a response field). "
            f"Available keys: {available}. "
            "Pass a dataset that exposes one of these fields, or preprocess it "
            "so the text lives under 'text'."
        )
    return ""
