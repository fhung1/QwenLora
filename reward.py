"""
Phase 0 reward function for the Qwen humanization LoRA.

Pure functions over generated text, each independently testable before any
model touches PPO. Combined into one weighted reward at the bottom.

Scope decided 2026-09-23: word count is a soft factor (a target range, not
an exact match), banned-phrase and perplexity/burstiness are kept, style-
similarity is scored against sent_texts.jsonl.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

SENT_TEXTS_PATH = Path(__file__).parent / "sent_texts.jsonl"

# Common AI-tell phrases beyond the em-dash. Not exhaustive -- extend as
# patterns show up in Phase 3's qualitative read.
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


# ---------------------------------------------------------------------------
# Word count (soft)
# ---------------------------------------------------------------------------

def word_count_score(text: str, target: int, tolerance: float = 0.3) -> float:
    """1.0 at the target length, decaying linearly to 0.0 at `tolerance`
    fraction away (e.g. tolerance=0.3 means +/-30% of target scores 0)."""
    n = len(text.split())
    if target <= 0:
        return 1.0
    frac_off = abs(n - target) / target
    return max(0.0, 1.0 - frac_off / tolerance)


# ---------------------------------------------------------------------------
# Banned phrases
# ---------------------------------------------------------------------------

def banned_phrase_score(text: str) -> float:
    """1.0 if no banned phrase appears, else 0.0. Binary, not graded --
    a single em-dash is exactly the failure mode this exists to catch."""
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in lowered:
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Perplexity / burstiness
# ---------------------------------------------------------------------------

@dataclass
class ReferenceLM:
    """Wraps a frozen LM used only for scoring, not training. Intended to be
    the base Qwen3-1.7B with its LoRA adapter disabled (peft's adapter-
    disable trick) so this doesn't need a second model copy."""
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str = "mps"

    @torch.no_grad()
    def sentence_perplexities(self, text: str) -> list[float]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        perplexities = []
        for sent in sentences:
            ids = self.tokenizer(sent, return_tensors="pt").input_ids.to(self.device)
            if ids.shape[1] < 2:
                continue
            out = self.model(ids, labels=ids)
            perplexities.append(torch.exp(out.loss).item())
        return perplexities


def perplexity_burstiness_score(
    text: str, ref_lm: ReferenceLM, target_perplexity: float = 40.0
) -> tuple[float, float]:
    """Returns (perplexity_score, burstiness_score), each in [0, 1].
    perplexity_score peaks at target_perplexity (too low reads as robotic/
    predictable, too high reads as incoherent). burstiness_score rewards
    variance in per-sentence perplexity (human-typical) over uniformity
    (AI-typical)."""
    ppls = ref_lm.sentence_perplexities(text)
    if len(ppls) < 2:
        return 0.5, 0.0  # not enough sentences to judge either signal

    mean_ppl = sum(ppls) / len(ppls)
    perplexity_score = max(0.0, 1.0 - abs(mean_ppl - target_perplexity) / target_perplexity)

    variance = sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)
    std = variance ** 0.5
    # Normalize burstiness by mean so it's scale-free; squash to [0, 1].
    burstiness_score = min(1.0, (std / mean_ppl) if mean_ppl > 0 else 0.0)

    return perplexity_score, burstiness_score


# ---------------------------------------------------------------------------
# Style similarity
# ---------------------------------------------------------------------------

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
        emb = self.embedder.encode(text, convert_to_tensor=True)
        return util.cos_sim(emb, self.centroid).item()


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

WEIGHTS = {
    "word_count": 0.15,
    "banned_phrase": 0.20,
    "perplexity": 0.15,
    "burstiness": 0.15,
    "style_similarity": 0.35,
}


def compute_reward(
    text: str,
    target_word_count: int,
    ref_lm: ReferenceLM,
    style_ref: StyleReference,
) -> dict[str, float]:
    wc = word_count_score(text, target_word_count)
    bp = banned_phrase_score(text)
    ppl, burst = perplexity_burstiness_score(text, ref_lm)
    sim = style_ref.similarity_score(text)

    components = {
        "word_count": wc,
        "banned_phrase": bp,
        "perplexity": ppl,
        "burstiness": burst,
        "style_similarity": sim,
    }
    total = sum(WEIGHTS[k] * v for k, v in components.items())
    return {**components, "total": total}


def make_grpo_reward_func(ref_lm: ReferenceLM, style_ref: StyleReference):
    """Returns a reward function matching TRL's GRPOTrainer signature:
    (prompts, completions, **dataset_columns) -> list[float]. `target_word_count`
    arrives as a kwarg automatically since it's a column in prompts.jsonl."""

    def reward_func(prompts, completions, target_word_count, **kwargs) -> list[float]:
        rewards = []
        for completion, target in zip(completions, target_word_count):
            result = compute_reward(completion, target, ref_lm, style_ref)
            rewards.append(result["total"])
        return rewards

    return reward_func


if __name__ == "__main__":
    # Smoke test for the two components that need no model loading.
    sample = "yeah I can do 3pm, actually can we push to 4? something came up"
    print("word_count_score(target=15):", word_count_score(sample, target=15))
    print("banned_phrase_score:", banned_phrase_score(sample))
    print("banned_phrase_score (with em-dash):", banned_phrase_score(sample + " — no wait"))
