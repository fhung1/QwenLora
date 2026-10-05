"""Synthetic conversation checks for the reply-pair extractor."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_reply_sft_data import Message, make_record, pair_messages


class PairMessagesTests(unittest.TestCase):
    def test_bursts_chat_boundaries_and_nonreply(self):
        m = [
            Message(1, 0, False, "Are you free?"),
            Message(1, 30, False, "Tomorrow afternoon?"),
            Message(1, 60, True, "Yes,"),
            Message(1, 100, True, "after three."),
            Message(1, 120, False, "Great"),
            Message(1, 130, True, "See you then"),
            Message(2, 140, True, "Unprompted"),
            Message(2, 150, False, "Second chat"),
            Message(2, 160, True, "Second answer"),
        ]
        pairs = list(pair_messages(m, 7200))
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0].incoming, ("Are you free?", "Tomorrow afternoon?"))
        self.assertEqual(pairs[0].reply, ("Yes,", "after three."))
        self.assertEqual(pairs[1].reply, ("See you then",))
        self.assertEqual(pairs[2].chat_id, 2)
        self.assertEqual(make_record(pairs[0])["completion"],
                         [{"role": "assistant", "content": "Yes,\nafter three."}])

    def test_stale_incoming_and_attachment_barriers(self):
        m = [
            Message(1, 0, False, "Old question"),
            Message(1, 7300, True, "Too late"),
            Message(1, 7400, False, "Photo?"),
            Message(1, 7410, False, None),
            Message(1, 7420, True, "No context"),
            Message(1, 7500, False, "Real question"),
            Message(1, 7510, True, "First answer"),
            Message(1, 7900, True, "Separate follow up"),
        ]
        pairs = list(pair_messages(m, 7200))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].incoming, ("Real question",))
        self.assertEqual(pairs[0].reply, ("First answer",))

    def test_later_incoming_replaces_stale_burst(self):
        m = [Message(1, 0, False, "First topic"),
             Message(1, 1000, False, "New topic"),
             Message(1, 1010, True, "New reply")]
        pairs = list(pair_messages(m, 7200))
        self.assertEqual(pairs[0].incoming, ("New topic",))

    def test_cli_reads_messages_schema_and_writes_chat_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "chat.db"
            output = Path(tmp) / "pairs.jsonl"
            conn = sqlite3.connect(db)
            conn.executescript("""
                CREATE TABLE message (date INTEGER, is_from_me INTEGER, text TEXT,
                                      attributedBody BLOB, associated_message_type INTEGER,
                                      item_type INTEGER);
                CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
            """)
            for at, from_me, message in (
                (1000000000, 0, "Are you free tomorrow?"),
                (1000000030, 1, "Yes, after three."),
                (1000000040, 1, "Does that work?"),
            ):
                row = conn.execute("INSERT INTO message VALUES (?, ?, ?, NULL, 0, 0)",
                                   (at, from_me, message))
                conn.execute("INSERT INTO chat_message_join VALUES (1, ?)",
                             (row.lastrowid,))
            conn.commit()
            conn.close()
            script = Path(__file__).with_name("build_reply_sft_data.py")
            subprocess.run([sys.executable, str(script), "--db-path", str(db),
                            "--output", str(output)], check=True, capture_output=True)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["prompt"],
                             [{"role": "user", "content": "Are you free tomorrow?"}])
            self.assertEqual(records[0]["completion"],
                             [{"role": "assistant", "content": "Yes, after three.\nDoes that work?"}])


if __name__ == "__main__":
    unittest.main()
