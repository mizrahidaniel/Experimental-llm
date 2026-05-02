"""Chat-template scaffolding: byte-level round-trip."""

import pytest

from sprl.inference.chat import (
    ASSISTANT_CLOSE,
    ASSISTANT_OPEN,
    SYSTEM_CLOSE,
    SYSTEM_OPEN,
    USER_CLOSE,
    USER_OPEN,
    apply_chat_template,
    parse_chat_response,
)


def test_apply_chat_template_basic():
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]
    out = apply_chat_template(msgs)
    assert SYSTEM_OPEN in out
    assert SYSTEM_CLOSE in out
    assert USER_OPEN in out
    assert USER_CLOSE in out
    assert out.endswith(ASSISTANT_OPEN)


def test_apply_chat_template_no_generation_prompt():
    msgs = [{"role": "user", "content": "hi"}]
    out = apply_chat_template(msgs, add_generation_prompt=False)
    assert not out.endswith(ASSISTANT_OPEN)
    assert out.endswith(USER_CLOSE)


def test_apply_chat_template_assistant_round_trip():
    msgs = [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]
    out = apply_chat_template(msgs, add_generation_prompt=False)
    assert b"pong" in out
    assert ASSISTANT_OPEN in out
    assert ASSISTANT_CLOSE in out


def test_apply_chat_template_bytes_content():
    msgs = [{"role": "user", "content": b"binary \x00\x01"}]
    out = apply_chat_template(msgs, add_generation_prompt=False)
    assert b"binary \x00\x01" in out


def test_apply_chat_template_unknown_role_raises():
    with pytest.raises(ValueError):
        apply_chat_template([{"role": "robot", "content": "hi"}])


def test_parse_chat_response_stops_at_assistant_close():
    full = b"<|user|>hi<|/user|><|assistant|>hello<|/assistant|>" + b"<|user|>more"
    prompt = b"<|user|>hi<|/user|><|assistant|>"
    reply = parse_chat_response(full, len(prompt))
    assert reply == b"hello"


def test_parse_chat_response_no_close_returns_tail():
    full = b"<|user|>hi<|/user|><|assistant|>partial reply"
    prompt = b"<|user|>hi<|/user|><|assistant|>"
    reply = parse_chat_response(full, len(prompt))
    assert reply == b"partial reply"
