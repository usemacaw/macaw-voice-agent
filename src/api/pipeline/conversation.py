"""
Conversation history management.

Converts ConversationItems to LLM messages format.
Supports token-budget windowing with pinning of important items.
"""

from __future__ import annotations

from protocol.models import ConversationItem


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for mixed pt/en text."""
    return max(1, len(text) // 4)


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


def items_to_budget_messages(
    items: list[ConversationItem],
    max_tokens: int = 4000,
    window_fallback: int = 8,
) -> list[dict]:
    """Convert items to messages using a token budget with smart pinning.

    Walks items from newest to oldest, accumulating estimated token count.
    Stops when the budget is exhausted.

    Pinning rules:
    - The first user message is always pinned (establishes topic/identity).
    - Tool call/result pairs are kept together.

    Falls back to item-count windowing if all items fit within budget.

    Args:
        items: Full conversation history.
        max_tokens: Maximum estimated tokens to include.
        window_fallback: Maximum items even if budget allows more.
    """
    if not items:
        return []

    n = len(items)
    if n <= 2:
        return _clean_orphan_tool_messages(items_to_messages(items))

    # Find the first user message for pinning
    first_user_idx: int | None = None
    for i, item in enumerate(items):
        if item.type == "message" and item.role == "user":
            first_user_idx = i
            break

    # Walk from newest to oldest, accumulating tokens
    selected_indices: set[int] = set()
    token_count = 0

    for i in range(n - 1, -1, -1):
        item = items[i]
        item_tokens = _item_token_estimate(item)

        if token_count + item_tokens > max_tokens and len(selected_indices) > 0:
            break

        selected_indices.add(i)
        token_count += item_tokens

        # If we selected a function_call_output, also select the preceding call
        if item.type == "function_call_output" and i > 0:
            prev = items[i - 1]
            if prev.type == "function_call" and i - 1 not in selected_indices:
                selected_indices.add(i - 1)
                token_count += _item_token_estimate(prev)

        # If we selected a function_call, also select the following output
        if item.type == "function_call" and i + 1 < n:
            nxt = items[i + 1]
            if nxt.type == "function_call_output" and i + 1 not in selected_indices:
                selected_indices.add(i + 1)
                token_count += _item_token_estimate(nxt)

        # Apply item count limit
        if len(selected_indices) >= window_fallback:
            break

    # Pin first user message
    if first_user_idx is not None and first_user_idx not in selected_indices:
        selected_indices.add(first_user_idx)

    # Build messages in original order
    selected_items = [items[i] for i in sorted(selected_indices)]
    messages = items_to_messages(selected_items)
    return _clean_orphan_tool_messages(messages)


def _item_token_estimate(item: ConversationItem) -> int:
    """Estimate token count for a single conversation item."""
    text = ""
    if item.type == "message":
        for part in item.content:
            if part.text:
                text += part.text
            if part.transcript:
                text += part.transcript
    elif item.type == "function_call":
        text = (item.name or "") + (item.arguments or "")
    elif item.type == "function_call_output":
        text = item.output or ""
    # Minimum overhead per message (role, formatting)
    return _estimate_tokens(text) + 4


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
