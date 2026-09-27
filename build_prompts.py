"""
Build PPO training prompts from sent_texts.jsonl using continuation framing:
split each burst into a prompt and a target continuation, both joined with
"\n" regardless of whether a line break was a real send-boundary or an
in-message newline -- that distinction only matters in the stored source of
truth (sent_texts.jsonl), not in the disposable strings fed to the model.

Split point targets ~35% of the burst's own word count (a proxy for token
count -- no tokenizer dependency, consistent with word_count_score elsewhere),
clamped to hard floors on both sides: a prompt has to carry enough words to
be a real conditioning signal, and a continuation has to leave something
worth generating. The cut is made on the flat word stream across all of a
burst's messages, so it naturally lands on a real message boundary (a "\n"
in the output) whenever the word-count math allows it, and only splits a
single message mid-word when a message straddles the cut point -- no
separate boundary-snapping logic needed.

Bursts too short to clear both floors are skipped -- they still feed the
style centroid in reward.py, just not this prompt pool.

Output: prompts.jsonl, one {"prompt", "target_word_count"} object per line.
"""

import json
from pathlib import Path

IN_PATH = Path(__file__).parent / "sent_texts.jsonl"
OUT_PATH = Path(__file__).parent / "prompts.jsonl"

SPLIT_TARGET = 0.35
MIN_PROMPT_WORDS = 5
MIN_CONTINUATION_WORDS = 3
MIN_TOTAL_WORDS = MIN_PROMPT_WORDS + MIN_CONTINUATION_WORDS


def split_burst(messages: list[str]) -> tuple[str, str] | None:
    word_owner = [(w, i) for i, m in enumerate(messages) for w in m.split()]
    total = len(word_owner)
    if total < MIN_TOTAL_WORDS:
        return None

    split_idx = round(total * SPLIT_TARGET)
    split_idx = max(MIN_PROMPT_WORDS, min(split_idx, total - MIN_CONTINUATION_WORDS))

    def render(chunk: list[tuple[str, int]]) -> str:
        lines: list[list[str]] = []
        for word, msg_idx in chunk:
            if lines and lines[-1][0] == msg_idx:
                lines[-1][1].append(word)
            else:
                lines.append([msg_idx, [word]])
        return "\n".join(" ".join(words) for _, words in lines)

    prompt = render(word_owner[:split_idx])
    continuation = render(word_owner[split_idx:])
    return prompt, continuation


def main() -> None:
    examples = []
    with IN_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            split = split_burst(rec["messages"])
            if split is None:
                continue
            prompt, continuation = split
            examples.append({
                "prompt": prompt,
                "target_word_count": len(continuation.split()),
            })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Built {len(examples)} continuation prompts -> {OUT_PATH}")


if __name__ == "__main__":
    main()
