"""
Phase 0 reward function for the Qwen humanization LoRA.

Pure functions over generated text, each independently testable before any
model touches training. Combined into one weighted reward at the bottom.

Scope decided 2026-09-23: word count is a soft factor (a target range, not
an exact match), banned-phrase and perplexity/burstiness are kept, style-
similarity is scored against sent_texts.jsonl.

Reviewed and fixed 2026-09-27:
- sentence_perplexities/burstiness only ever split on '.', '!', '?' --  most
  of this corpus is unpunctuated texting, so real completions almost always
  hit the < 2 units fallback and silently got a constant default reward.
  _split_units now falls back through newline boundaries, then sentences,
  then fixed-size word chunks, so burstiness is measurable on real data.
- target_perplexity was a hardcoded guess (40.0); calibrate_target_perplexity
  measures it from the actual corpus instead.
- perplexity and style-similarity scoring were unbatched (one forward pass
  per completion) despite GRPO scoring num_generations completions per step;
  both are now batched in make_grpo_reward_func.
"""

from __future__ import annotations

import json
import math
import random
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

def word_count_score(text: str, target: int, tolerance: float = 0.3, min_sigma: float = 2.0) -> float:
    """Gaussian/MSE-based score: 1.0 at the target length, smoothly decaying
    with no hard cutoff. sigma = max(target * tolerance, min_sigma) -- the
    floor matters for short targets: the original linear version used a pure
    fractional tolerance, so for target=3 (17.9% of this corpus), 30% of 3 is
    under 1 word, making even n=2 or n=4 score exactly 0 -- effectively an
    exact-match requirement, the opposite of the "soft factor" it was meant
    to be. min_sigma guarantees a real decay curve regardless of how short
    the target is, while tolerance still dominates for longer targets."""
    n = len(text.split())
    if target <= 0:
        return 1.0
    sigma = max(target * tolerance, min_sigma)
    mse = (n - target) ** 2
    return math.exp(-mse / (2 * sigma ** 2))


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

MIN_PREDICTED_TOKENS = 4  # below this, per-unit perplexity is unreliable -- see batch_perplexities


def _split_units(text: str, chunk_words: int = 5) -> list[str]:
    """Split text into sub-units for per-unit perplexity. Falls back through
    three levels since most of this corpus has no terminal punctuation:
    1. newline boundaries (real message boundaries within a burst)
    2. sentence punctuation within each line
    3. fixed-size word chunks, for a single unpunctuated run of text
    Stops at the first level that yields >= 2 units."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    units = []
    for line in lines:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
        units.extend(sentences)
    if len(units) >= 2:
        return units

    words = text.split()
    if len(words) < 2:
        return [text] if text.strip() else []
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    return [c for c in chunks if c.strip()]


@dataclass
class ReferenceLM:
    """Wraps a frozen LM used only for scoring, not training. Intended to be
    a dedicated frozen copy of the base model, separate from the policy
    GRPOTrainer trains -- see train_grpo.py for why."""
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str = "mps"

    @torch.no_grad()
    def sentence_perplexities(self, text: str) -> list[float]:
        """Single-text path, used by compute_reward() for manual/smoke testing.
        For batched scoring during training, use batch_perplexities instead."""
        return self.batch_perplexities([text])[0]

    @torch.no_grad()
    def batch_perplexities(self, texts: list[str], max_batch_units: int = 32) -> list[list[float]]:
        """Per-text list of unit-level perplexities. Processes all_units in
        sub-batches of max_batch_units -- a single forward pass over hundreds
        of units at once was hitting MPS's 16GB budget (the cross-entropy
        reshape over [N*T, vocab_size] with a ~152k vocab gets large fast)."""
        all_units: list[str] = []
        owner: list[int] = []
        for i, text in enumerate(texts):
            units = _split_units(text)
            all_units.extend(units)
            owner.extend([i] * len(units))

        result: list[list[float]] = [[] for _ in texts]
        if not all_units:
            return result

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        for start in range(0, len(all_units), max_batch_units):
            batch_units = all_units[start:start + max_batch_units]
            batch_owner = owner[start:start + max_batch_units]

            enc = self.tokenizer(
                batch_units, return_tensors="pt", padding=True, truncation=True, max_length=64
            ).to(self.device)
            out = self.model(**enc)

            shift_logits = out.logits[:, :-1, :]
            shift_labels = enc.input_ids[:, 1:]
            shift_mask = enc.attention_mask[:, 1:].float()

            losses = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
            ).view(shift_labels.size())

            real_token_counts = shift_mask.sum(dim=1)
            seq_loss = (losses * shift_mask).sum(dim=1) / real_token_counts.clamp(min=1)
            seq_ppl = torch.exp(seq_loss).tolist()

            # Sequences under MIN_PREDICTED_TOKENS give degenerate perplexity --
            # confirmed empirically: a 2-token unit (1 prediction, no BOS, zero
            # context at position 0) scored in the millions across every phrase
            # tested, vs. sane values (~150-250) once there's real context. Units
            # this short are common in this corpus (e.g. a whole 2-word message),
            # so they're dropped here rather than silently corrupting the mean.
            for count, ppl, o in zip(real_token_counts.tolist(), seq_ppl, batch_owner):
                if count >= MIN_PREDICTED_TOKENS:
                    result[o].append(ppl)

        return result


def calibrate_target_perplexity(ref_lm: ReferenceLM, sample_size: int = 200) -> float:
    """Measures typical perplexity of the reference corpus under ref_lm, to
    use as target_perplexity instead of a hardcoded guess. Uses the median,
    not the mean -- real text has a heavy right tail (rare names, slang,
    emoji genuinely score high perplexity; this isn't a bug, but a handful
    of such units dragged the mean to ~10x the median in testing)."""
    with SENT_TEXTS_PATH.open(encoding="utf-8") as f:
        lines = f.readlines()
    random.seed(0)
    sample_lines = random.sample(lines, min(sample_size, len(lines)))
    texts = ["\n".join(json.loads(line)["messages"]) for line in sample_lines]

    per_text_ppls = ref_lm.batch_perplexities(texts)
    all_ppls = sorted(p for ppls in per_text_ppls for p in ppls)
    if not all_ppls:
        return 40.0
    return all_ppls[len(all_ppls) // 2]


def _score_perplexity_burstiness(ppls: list[float], target_perplexity: float) -> tuple[float, float]:
    """Perplexity and burstiness are decoupled: perplexity only needs one
    valid unit (a mean over however many exist), but burstiness genuinely
    needs >= 2 to measure variance at all. 44.3% of this corpus's targets
    are <= 5 words, where _split_units often returns exactly one whole-text
    unit -- treating that as "no signal" for perplexity too (the original
    behavior) was throwing away a perfectly good score on nearly half the
    data, not a real data limitation like the burstiness case is."""
    if len(ppls) == 0:
        return 0.5, 0.0  # nothing to measure at all

    mean_ppl = sum(ppls) / len(ppls)
    perplexity_score = max(0.0, 1.0 - abs(mean_ppl - target_perplexity) / target_perplexity)

    if len(ppls) < 2:
        return perplexity_score, 0.0  # real perplexity, but no variance to measure burstiness from

    variance = sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)
    std = variance ** 0.5
    burstiness_score = min(1.0, (std / mean_ppl) if mean_ppl > 0 else 0.0)

    return perplexity_score, burstiness_score


def perplexity_burstiness_score(
    text: str, ref_lm: ReferenceLM, target_perplexity: float = 40.0
) -> tuple[float, float]:
    """Single-text path for compute_reward()/manual testing. Returns
    (perplexity_score, burstiness_score), each in [0, 1]."""
    return _score_perplexity_burstiness(ref_lm.sentence_perplexities(text), target_perplexity)


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
        """Single-text path for compute_reward()/manual testing."""
        return self.similarity_scores([text])[0]

    def similarity_scores(self, texts: list[str]) -> list[float]:
        """Batched path, used by make_grpo_reward_func."""
        embs = self.embedder.encode(texts, convert_to_tensor=True)
        return util.cos_sim(embs, self.centroid.unsqueeze(0)).squeeze(1).tolist()


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


def _weighted_total(components: dict[str, float]) -> float:
    return sum(WEIGHTS[k] * v for k, v in components.items())


def compute_reward(
    text: str,
    target_word_count: int,
    ref_lm: ReferenceLM,
    style_ref: StyleReference,
    target_perplexity: float = 40.0,
) -> dict[str, float]:
    """Single-text path -- for manual testing (e.g. via QwenLora.py) or the
    smoke test below. make_grpo_reward_func uses its own batched path for
    actual training, not this function, to avoid per-completion model calls."""
    wc = word_count_score(text, target_word_count)
    bp = banned_phrase_score(text)
    ppl, burst = perplexity_burstiness_score(text, ref_lm, target_perplexity)
    sim = style_ref.similarity_score(text)

    components = {
        "word_count": wc,
        "banned_phrase": bp,
        "perplexity": ppl,
        "burstiness": burst,
        "style_similarity": sim,
    }
    return {**components, "total": _weighted_total(components)}


def make_grpo_reward_func(
    ref_lm: ReferenceLM, style_ref: StyleReference, target_perplexity: float | None = None
):
    """Returns a reward function matching TRL's GRPOTrainer signature:
    (prompts, completions, **dataset_columns) -> list[float]. `target_word_count`
    arrives as a kwarg automatically since it's a column in prompts.jsonl.

    Batches the two model-backed components (perplexity/burstiness, style-
    similarity) across the whole group of completions in one call each,
    rather than one forward pass per completion -- matters at GRPO's
    num_generations throughput."""
    if target_perplexity is None:
        target_perplexity = calibrate_target_perplexity(ref_lm)
        print(f"Calibrated target_perplexity from corpus: {target_perplexity:.2f}")

    def reward_func(prompts, completions, target_word_count, **kwargs) -> list[float]:
        wc_scores = [word_count_score(c, t) for c, t in zip(completions, target_word_count)]
        bp_scores = [banned_phrase_score(c) for c in completions]

        per_completion_ppls = ref_lm.batch_perplexities(completions)
        ppl_burst = [_score_perplexity_burstiness(p, target_perplexity) for p in per_completion_ppls]
        ppl_scores = [pb[0] for pb in ppl_burst]
        burst_scores = [pb[1] for pb in ppl_burst]

        sim_scores = style_ref.similarity_scores(completions)

        rewards = []
        for wc, bp, ppl, burst, sim in zip(wc_scores, bp_scores, ppl_scores, burst_scores, sim_scores):
            rewards.append(_weighted_total({
                "word_count": wc,
                "banned_phrase": bp,
                "perplexity": ppl,
                "burstiness": burst,
                "style_similarity": sim,
            }))
        return rewards

    return reward_func


if __name__ == "__main__":
    # Smoke test for the two components that need no model loading.
    sample = "yeah I can do 3pm, actually can we push to 4? something came up"
    print("word_count_score(target=15):", word_count_score(sample, target=15))
    print("banned_phrase_score:", banned_phrase_score(sample))
    print("banned_phrase_score (with em-dash):", banned_phrase_score(sample + " — no wait"))
    print()
    print("_split_units on unpunctuated single-line text (the real-world case):")
    print(" ", _split_units("yeah i can do 3pm actually can we push to 4 something came up"))
    print("_split_units on a multi-line burst:")
    print(" ", _split_units("wait actually\nnvm i figured it out\nall good now"))
