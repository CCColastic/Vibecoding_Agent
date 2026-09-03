from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence

from mini_agent.app import build_conversation_manager
from mini_agent.session import (
    ActiveConversation,
    ConversationManager,
    SessionNotFoundError,
)


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-agent")
    parser.add_argument("command", choices=["new", "sessions"])
    return parser


async def conversation_loop(
    conversation: ActiveConversation,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> None:
    while True:
        try:
            user_input = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return
        if user_input == "/exit":
            return
        if not user_input:
            continue

        was_new = conversation.session_id is None
        try:
            result = await conversation.send_message(user_input)
        except Exception as exc:
            output_fn(f"Error: {exc}")
            continue
        if was_new:
            output_fn(f"Session: {conversation.session_id}")
        output_fn(f"Agent: {result.final_answer or result.error}")


async def run_cli(
    argv: Sequence[str] | None = None,
    *,
    conversation_manager: ConversationManager | None = None,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        conversations = conversation_manager or build_conversation_manager()
    except (OSError, ValueError) as exc:
        output_fn(f"Error: {exc}")
        return 1

    if args.command == "new":
        conversation = conversations.new_conversation()
        await conversation_loop(
            conversation, input_fn=input_fn, output_fn=output_fn
        )
        return 0

    sessions = conversations.list_sessions()
    if not sessions:
        output_fn("No sessions found.")
        return 0
    for index, session in enumerate(sessions, start=1):
        output_fn(f"[{index}] {session.title}  {session.updated_at.isoformat()}")

    try:
        selection = input_fn("Select session: ").strip()
    except (EOFError, KeyboardInterrupt):
        output_fn("")
        return 0
    if not selection:
        return 0
    try:
        selected = sessions[int(selection) - 1]
    except (ValueError, IndexError):
        output_fn("Invalid session selection.")
        return 1

    try:
        conversation = conversations.resume_conversation(selected.id)
    except SessionNotFoundError as exc:
        output_fn(f"Error: {exc}")
        return 1
    await conversation_loop(conversation, input_fn=input_fn, output_fn=output_fn)
    return 0


def main() -> int:
    return asyncio.run(run_cli())
