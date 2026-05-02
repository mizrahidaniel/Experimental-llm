"""Pure byte-level chat-template scaffolding.

Turn markers are ASCII byte sequences (avoiding any non-ASCII overhead). The
markers below are unique enough to be unambiguous in plain text streams.

Format:
  <|system|> ... <|/system|>
  <|user|> ... <|/user|>
  <|assistant|> ... <|/assistant|>

`apply_chat_template(messages)` concatenates turns and leaves the prompt with
a trailing `<|assistant|>` so the model can complete the assistant's reply.
"""

from __future__ import annotations

from typing import List

SYSTEM_OPEN = b"<|system|>"
SYSTEM_CLOSE = b"<|/system|>"
USER_OPEN = b"<|user|>"
USER_CLOSE = b"<|/user|>"
ASSISTANT_OPEN = b"<|assistant|>"
ASSISTANT_CLOSE = b"<|/assistant|>"


def apply_chat_template(
    messages: List[dict],
    add_generation_prompt: bool = True,
) -> bytes:
    """Render `messages` as a byte string ready for the model.

    Args:
      messages: list of {"role": "system"|"user"|"assistant", "content": str|bytes}.
      add_generation_prompt: append a trailing `<|assistant|>` so the model
        knows the next bytes belong to the assistant's reply.
    """
    parts: List[bytes] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            content = content.encode("utf-8")
        if role == "system":
            parts.extend([SYSTEM_OPEN, content, SYSTEM_CLOSE])
        elif role == "user":
            parts.extend([USER_OPEN, content, USER_CLOSE])
        elif role == "assistant":
            parts.extend([ASSISTANT_OPEN, content, ASSISTANT_CLOSE])
        else:
            raise ValueError(f"unknown role: {role}")
    if add_generation_prompt:
        parts.append(ASSISTANT_OPEN)
    return b"".join(parts)


def parse_chat_response(stream: bytes, prompt_len: int) -> bytes:
    """Extract the assistant's reply bytes from a generation stream.

    Stops at `<|/assistant|>` if present; otherwise returns everything after
    the prompt.
    """
    tail = stream[prompt_len:]
    end = tail.find(ASSISTANT_CLOSE)
    return tail if end == -1 else tail[:end]
