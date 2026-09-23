"""
Extract sent iMessage/SMS text into a JSONL reference corpus for the Qwen
LoRA style-similarity reward (Phase 0).

Requires Full Disk Access for the terminal/IDE running this script:
System Settings -> Privacy & Security -> Full Disk Access.
"""

from __future__ import annotations  # `bytes | None` hints on macOS's Python 3.9

import json
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
OUT_PATH = Path(__file__).parent / "sent_texts.jsonl"

MAC_EPOCH_OFFSET = 978307200  # seconds between 1970-01-01 and 2001-01-01
OBJ_REPLACEMENT = "￼"  # inline placeholder where an attachment sat
BURST_GAP_SECONDS = 5 * 60  # gap after which a new burst starts even with no reply
MAX_MESSAGE_CHARS = 500  # drops pasted documents/essays, not real texting


def decode_attributed_body(blob: bytes | None) -> str | None:
    """attributedBody is a typedstream blob: after the NSString class name and
    5 header bytes comes a length (1 byte, or 0x81+u16, or 0x82+u32), then
    that many bytes of UTF-8 text."""
    if not blob:
        return None
    idx = blob.find(b"NSString")
    if idx == -1:
        return None
    p = idx + len(b"NSString") + 5
    if p >= len(blob) or blob[p - 1] != ord("+"):
        return None

    tag = blob[p]
    if tag == 0x81:
        length, start = int.from_bytes(blob[p + 1:p + 3], "little"), p + 3
    elif tag == 0x82:
        length, start = int.from_bytes(blob[p + 1:p + 5], "little"), p + 5
    else:
        length, start = tag, p + 1
    return blob[start:start + length].decode("utf-8", errors="replace")


def apple_date_to_secs(raw: int | None) -> float | None:
    if not raw:
        return None
    # Newer DBs store nanoseconds since 2001; older ones store seconds.
    return (raw / 1e9 if raw > 1e11 else raw) + MAC_EPOCH_OFFSET


def apple_date_to_iso(raw: int | None) -> str | None:
    secs = apple_date_to_secs(raw)
    return None if secs is None else datetime.fromtimestamp(secs, tz=timezone.utc).isoformat()


def open_snapshot(tmpdir: Path) -> sqlite3.Connection:
    for suffix in ("", "-wal", "-shm"):
        src = DB_PATH.with_name(DB_PATH.name + suffix)
        if src.exists():
            shutil.copy2(src, tmpdir / src.name)
    conn = sqlite3.connect(tmpdir / DB_PATH.name)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_text(row: sqlite3.Row, failed_decodes: list[int]) -> str | None:
    text = row["text"]
    if not text or not text.strip():
        text = decode_attributed_body(row["attributedBody"])
        if text is None and row["attributedBody"]:
            failed_decodes[0] += 1
    if not text:
        return None
    text = text.replace(OBJ_REPLACEMENT, "").strip()
    return text or None


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Messages database not found at {DB_PATH}")

    with tempfile.TemporaryDirectory() as tmp:
        conn = open_snapshot(Path(tmp))
        rows = conn.execute(
            """
            SELECT cmj.chat_id AS chat_id,
                   m.date AS date,
                   m.is_from_me AS is_from_me,
                   m.text AS text,
                   CASE WHEN m.is_from_me = 1 THEN m.attributedBody END AS attributedBody,
                   m.associated_message_type AS associated_message_type,
                   m.item_type AS item_type
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            ORDER BY cmj.chat_id ASC, m.date ASC
            """
        )

        failed_decodes = [0]
        dropped_oversized = [0]
        records = []
        current_chat = None
        burst_parts: list[str] = []
        burst_start_date = None
        last_sent_secs = None

        def flush_burst():
            if burst_parts:
                # Keep as a list, not "\n".join(...): a message can contain a
                # real typed newline, and joining would make that indistinguishable
                # from the boundary between two separate messages in the burst.
                records.append({
                    "date": apple_date_to_iso(burst_start_date),
                    "messages": list(burst_parts),
                })
            burst_parts.clear()

        for row in rows:
            if row["chat_id"] != current_chat:
                flush_burst()
                current_chat = row["chat_id"]
                burst_start_date = None
                last_sent_secs = None

            is_content = row["associated_message_type"] == 0 and row["item_type"] == 0

            if row["is_from_me"] == 0:
                if is_content:  # a real incoming message breaks the burst; their tapbacks don't
                    flush_burst()
                continue

            if not is_content:  # your own tapbacks/system rows: no new content, no break
                continue

            text = resolve_text(row, failed_decodes)
            if text is None:
                continue
            if len(text) > MAX_MESSAGE_CHARS:
                dropped_oversized[0] += 1
                continue

            secs = apple_date_to_secs(row["date"])
            if (burst_parts and secs is not None and last_sent_secs is not None
                    and secs - last_sent_secs > BURST_GAP_SECONDS):
                flush_burst()
            last_sent_secs = secs

            if not burst_parts:
                burst_start_date = row["date"]
            burst_parts.append(text)

        flush_burst()
        conn.close()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Extracted {len(records)} message bursts -> {OUT_PATH}")
    if failed_decodes[0]:
        print(f"Warning: {failed_decodes[0]} attributedBody blobs could not be decoded")
    if dropped_oversized[0]:
        print(f"Dropped {dropped_oversized[0]} messages over {MAX_MESSAGE_CHARS} chars (likely pasted content)")


if __name__ == "__main__":
    main()
