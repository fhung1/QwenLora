"""Build raw prefix and continuation pairs from sent-message bursts."""

import json
import re
from itertools import chain
from pathlib import Path

from transformers import AutoTokenizer

from build_prompts import (
    MIN_CONTINUATION_WORDS,
    MIN_PROMPT_WORDS,
    MIN_TOTAL_WORDS,
    SPLIT_TARGET,
)

IN_PATH = Path(__file__).parent / "sent_texts.jsonl"
OUT_PATH = Path(__file__).parent / "sft_pairs.jsonl"
MODEL_NAME = "Qwen/Qwen3-1.7B"
EOS_TOKEN = "<|im_end|>"


def split_burst(messages: list[str], tokenizer) -> tuple[str, str] | None:
    text = "\n".join(messages)
    words = list(re.finditer(r"\S+", text))
    total = len(words)
    if total < MIN_TOTAL_WORDS:
        return None

    split_idx = round(total * SPLIT_TARGET)
    split_idx = max(MIN_PROMPT_WORDS, min(split_idx, total - MIN_CONTINUATION_WORDS))
    full_ids = tokenizer(text + EOS_TOKEN).input_ids
    candidates = chain(
        (split_idx,),
        range(split_idx + 1, total - MIN_CONTINUATION_WORDS + 1),
        range(split_idx - 1, MIN_PROMPT_WORDS - 1, -1),
    )
    for idx in candidates:
        cut = words[idx - 1].end()
        prompt = text[:cut]
        prompt_ids = tokenizer(prompt).input_ids
        if full_ids[:len(prompt_ids)] == prompt_ids:
            return prompt, text[cut:] + EOS_TOKEN
    return None


def main() -> None:
    count = 0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.eos_token != EOS_TOKEN:
        raise ValueError(f"Unexpected EOS token: {tokenizer.eos_token!r}")
    with IN_PATH.open(encoding="utf-8") as source, OUT_PATH.open("w", encoding="utf-8") as output:
        for line in source:
            split = split_burst(json.loads(line)["messages"], tokenizer)
            if split is None:
                continue
            prompt, completion = split
            output.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            count += 1
    print(f"Built {count} SFT pairs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
