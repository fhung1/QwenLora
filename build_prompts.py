"""Split sent-message bursts into continuation prompts and target word counts."""

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
