"""Build chat SFT pairs from incoming Messages and the next sent reply burst.

Requires Full Disk Access. The output contains private message text and is ignored by git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from extract_texts import DB_PATH, MAX_MESSAGE_CHARS, apple_date_to_secs, resolve_text

OUT_PATH = Path(__file__).parent / "reply_sft_pairs.jsonl"
INCOMING_BURST_SECONDS = 15 * 60
REPLY_BURST_SECONDS = 5 * 60
MAX_INCOMING_MESSAGES = 6
MAX_PROMPT_CHARS = 1200


@dataclass(frozen=True)
class Message:
    chat_id: int
    at: float | None
    is_from_me: bool
    text: str | None


@dataclass(frozen=True)
class ReplyPair:
    chat_id: int
    incoming: tuple[str, ...]
    reply: tuple[str, ...]


def pair_messages(messages: Iterable[Message], max_reply_delay_seconds: int) -> Iterator[ReplyPair]:
    chat_id = None
    incoming: list[str] = []
    reply: list[str] = []
    last_incoming_at = None
    last_reply_at = None

    def completed() -> ReplyPair | None:
        if chat_id is not None and incoming and reply:
            return ReplyPair(chat_id, tuple(incoming), tuple(reply))
        return None

    for message in messages:
        if message.chat_id != chat_id:
            pair = completed()
            if pair:
                yield pair
            chat_id = message.chat_id
            incoming, reply = [], []
            last_incoming_at = last_reply_at = None

        if message.at is None or message.text is None:
            if not message.is_from_me:
                pair = completed()
                if pair:
                    yield pair
            incoming, reply = [], []
            last_incoming_at = last_reply_at = None
            continue

        if not message.is_from_me:
            pair = completed()
            if pair:
                yield pair
                incoming = []
            reply = []
            last_reply_at = None
            if (last_incoming_at is not None and
                    (message.at < last_incoming_at or
                     message.at - last_incoming_at > INCOMING_BURST_SECONDS)):
                incoming = []
            incoming.append(message.text)
            last_incoming_at = message.at
            while len(incoming) > MAX_INCOMING_MESSAGES or sum(map(len, incoming)) > MAX_PROMPT_CHARS:
                incoming.pop(0)
            continue

        if not incoming:
            continue
        if reply:
            if (last_reply_at is None or message.at < last_reply_at or
                    message.at - last_reply_at > REPLY_BURST_SECONDS):
                pair = completed()
                if pair:
                    yield pair
                incoming, reply = [], []
                last_incoming_at = last_reply_at = None
                continue
        elif (last_incoming_at is None or message.at < last_incoming_at or
              message.at - last_incoming_at > max_reply_delay_seconds):
            incoming = []
            last_incoming_at = None
            continue
        reply.append(message.text)
        last_reply_at = message.at

    pair = completed()
    if pair:
        yield pair


def read_messages(failed_decodes: list[int], db_path: Path) -> Iterator[Message]:
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / db_path.name
        for suffix in ("", "-wal", "-shm"):
            source = db_path.with_name(db_path.name + suffix)
            if source.exists():
                shutil.copy2(source, snapshot.with_name(snapshot.name + suffix))
        conn = sqlite3.connect(snapshot)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT cmj.chat_id AS chat_id, m.date AS date,
                       m.is_from_me AS is_from_me, m.text AS text,
                       m.attributedBody AS attributedBody
                FROM message m
                JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                WHERE m.associated_message_type = 0 AND m.item_type = 0
                ORDER BY cmj.chat_id, m.date, m.ROWID
                """
            )
            for row in rows:
                text = resolve_text(row, failed_decodes)
                if text is not None and len(text) > MAX_MESSAGE_CHARS:
                    text = None
                yield Message(row["chat_id"], apple_date_to_secs(row["date"]),
                              bool(row["is_from_me"]), text)
        finally:
            conn.close()


def make_record(pair: ReplyPair) -> dict:
    chat_key = hashlib.sha256(str(pair.chat_id).encode()).hexdigest()[:16]
    return {
        "conversation_id": chat_key,
        "prompt": [{"role": "user", "content": "\n".join(pair.incoming)}],
        "completion": [{"role": "assistant", "content": "\n".join(pair.reply)}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--max-reply-delay-minutes", type=int, default=120)
    parser.add_argument("--min-reply-words", type=int, default=3)
    args = parser.parse_args()
    if args.max_reply_delay_minutes <= 0 or args.min_reply_words <= 0:
        parser.error("delay and minimum reply words must be positive")
    if not args.db_path.exists():
        parser.error(f"Messages database not found at {args.db_path}")

    failed_decodes = [0]
    paired = written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                     prefix=".reply_sft_", delete=False) as output:
        temporary_output = Path(output.name)
    try:
        with temporary_output.open("w", encoding="utf-8") as output:
            for pair in pair_messages(read_messages(failed_decodes, args.db_path),
                                      args.max_reply_delay_minutes * 60):
                paired += 1
                if len(" ".join(pair.reply).split()) < args.min_reply_words:
                    continue
                output.write(json.dumps(make_record(pair), ensure_ascii=False) + "\n")
                written += 1
        os.replace(temporary_output, args.output)
    finally:
        temporary_output.unlink(missing_ok=True)
    print(f"Found {paired} reply pairs; wrote {written} -> {args.output}")
    if failed_decodes[0]:
        print(f"Could not decode {failed_decodes[0]} attributedBody blobs")


if __name__ == "__main__":
    main()
