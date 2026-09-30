"""Score generated text for length, banned phrases, and style similarity."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

SENT_TEXTS_PATH = Path(__file__).parent / "sent_texts.jsonl"

# Phrases to penalize in generated text.
BANNED_PHRASES = [
    "—",  # em dash
    "–",  # en dash
    "it's important to note",
    "it is important to note",
    "in conclusion",
    "furthermore",
    "moreover",
    "delve into",
    "boasts",
    "a testament to",
]


# Word count

def word_count_score(text: str, target: int) -> float:
    """Score proportional deviation from the target length."""
    n = len(text.split())
    if n == 0:
        return 0.0
    if target <= 0:
        return 1.0
    error = math.log(n / target)
    return 1.0 / (1.0 + (error / 0.75) ** 2)


# Banned phrases

def banned_phrase_score(text: str) -> float:
    """Return 1 if no banned phrase appears, otherwise 0."""
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in lowered:
            return 0.0
    return 1.0


# Style similarity

@dataclass
class StyleReference:
    embedder: SentenceTransformer
    centroid: torch.Tensor = field(init=False)

    def __post_init__(self):
        texts = []
        with SENT_TEXTS_PATH.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts.append("\n".join(rec["messages"]))
        embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        self.centroid = embeddings.mean(dim=0)

    def similarity_score(self, text: str) -> float:
        """Return style similarity for one text."""
        return self.similarity_scores([text])[0]

    def similarity_scores(self, texts: list[str]) -> list[float]:
        """Return style similarity for each text."""
        embs = self.embedder.encode(texts, convert_to_tensor=True)
        return util.cos_sim(embs, self.centroid.unsqueeze(0)).squeeze(1).tolist()


# Combined reward

WEIGHTS = {
    "word_count": 0.15,
    "banned_phrase": 0.30,
    "style_similarity": 0.55,
}


def _weighted_total(components: dict[str, float]) -> float:
    return sum(WEIGHTS[k] * v for k, v in components.items())


def compute_reward(
    text: str,
    target_word_count: int,
    style_ref: StyleReference,
) -> dict[str, float]:
    """Return component scores and their weighted total for one text."""
    wc = word_count_score(text, target_word_count)
    bp = banned_phrase_score(text)
    sim = style_ref.similarity_score(text)

    components = {
        "word_count": wc,
        "banned_phrase": bp,
        "style_similarity": sim,
    }
    return {**components, "total": _weighted_total(components)}


def make_grpo_reward_func(style_ref: StyleReference):
    """Build a batched GRPO reward function using target_word_count."""

    def reward_func(prompts, completions, target_word_count, **kwargs) -> list[float]:
        wc_scores = [word_count_score(c, t) for c, t in zip(completions, target_word_count)]
        bp_scores = [banned_phrase_score(c) for c in completions]
        sim_scores = style_ref.similarity_scores(completions)

        rewards = []
        for wc, bp, sim in zip(wc_scores, bp_scores, sim_scores):
            rewards.append(_weighted_total({
                "word_count": wc,
                "banned_phrase": bp,
                "style_similarity": sim,
            }))
        return rewards

    return reward_func


if __name__ == "__main__":
    sample = "yeah I can do 3pm, actually can we push to 4? something came up"
    print("word_count_score(target=15):", word_count_score(sample, target=15))
    print("banned_phrase_score:", banned_phrase_score(sample))
    print("banned_phrase_score (with em-dash):", banned_phrase_score(sample + " — no wait"))
