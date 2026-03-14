"""
Conversation history management.

Converts ConversationItems to LLM messages format.
"""

from __future__ import annotations

from protocol.models import ConversationItem


def items_to_messages(items: list[ConversationItem]) -> list[dict]:
    """Convert conversation items to LLM messages format (OpenAI-style).

    Handles:
    - message items → user/assistant messages
    - function_call items → assistant messages with tool_calls
    - function_call_output items → tool messages
    """
    messages = []

    for item in items:
        if item.type == "message":
            if not item.role:
                continue

            # Build content from parts
            text_parts = []
            for part in item.content:
                if part.type in ("input_text", "text"):
                    if part.text:
                        text_parts.append(part.text)
                elif part.type in ("input_audio", "audio"):
                    # Audio transcription (input_audio = user, audio = assistant)
                    if part.transcript:
                        text_parts.append(part.transcript)

            if text_parts:
                messages.append({
                    "role": item.role,
                    "content": " ".join(text_parts),
                })

        elif item.type == "function_call":
            # Assistant made a function call
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.call_id or item.id,
                    "type": "function",
                    "function": {
                        "name": item.name or "",
                        "arguments": item.arguments or "{}",
                    },
                }],
            })

        elif item.type == "function_call_output":
            # Client provided function result
            messages.append({
                "role": "tool",
                "tool_call_id": item.call_id or "",
                "content": item.output or "",
            })

    return messages


def items_to_windowed_messages(
    items: list[ConversationItem],
    window: int = 6,
) -> list[dict]:
    """Convert items to messages using only the last `window` items.

    Ensures tool call / tool result pairs are not split: if the window
    boundary falls in the middle of a function_call → function_call_output
    pair, both items are included.
    """
    if len(items) <= window:
        return _clean_orphan_tool_messages(items_to_messages(items))

    # Take the last `window` items
    windowed = list(items[-window:])

    # Ensure we don't start with a function_call_output (orphan)
    while windowed and windowed[0].type == "function_call_output":
        # Pull the preceding function_call into the window
        first_idx = items.index(windowed[0])
        if first_idx > 0:
            windowed.insert(0, items[first_idx - 1])
        else:
            # No preceding item — drop the orphan
            windowed.pop(0)

    # Ensure we don't end with a function_call without its output
    if windowed and windowed[-1].type == "function_call":
        last_idx = items.index(windowed[-1])
        if last_idx + 1 < len(items):
            windowed.append(items[last_idx + 1])
        else:
            # No output — drop the dangling call
            windowed.pop()

    messages = items_to_messages(windowed)
    return _clean_orphan_tool_messages(messages)


def _clean_orphan_tool_messages(messages: list[dict]) -> list[dict]:
    """Remove tool messages whose function_call is not in the message list.

    Also removes function_call messages whose tool result is missing.
    This prevents LLM confusion from incomplete tool call sequences.
    """
    # Collect all tool_call IDs from assistant messages
    call_ids_with_call: set[str] = set()
    call_ids_with_result: set[str] = set()

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                call_ids_with_call.add(tc.get("id", ""))
        elif msg.get("role") == "tool":
            call_ids_with_result.add(msg.get("tool_call_id", ""))

    # Only keep complete pairs
    complete_ids = call_ids_with_call & call_ids_with_result

    cleaned = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Keep only if all tool_calls have results
            tc_ids = {tc.get("id", "") for tc in msg["tool_calls"]}
            if tc_ids <= complete_ids:
                cleaned.append(msg)
            # else: drop the incomplete function_call
        elif msg.get("role") == "tool":
            if msg.get("tool_call_id", "") in complete_ids:
                cleaned.append(msg)
            # else: drop the orphan tool result
        else:
            cleaned.append(msg)

    return cleaned
