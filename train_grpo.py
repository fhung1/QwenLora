"""
Phase 1/2: GRPO training of the humanization LoRA on Qwen3-1.7B.

Switched from PPO to GRPO (2026-09-27) because TRL's current PPOTrainer
requires an nn.Module reward model, incompatible with our heuristic
reward.py; GRPOTrainer natively accepts a plain callable and needs no
separate value/critic model, which also helps on 16GB unified memory.

Design note: the perplexity/burstiness reward component needs a frozen
reference LM. Rather than reaching into GRPOTrainer's internally-managed,
peft-wrapped policy model to toggle its adapter off mid-reward-call, this
loads one separate, dedicated frozen copy of the base model purely for
reward scoring -- simpler and decoupled from the trainer's internals, at
the cost of one extra ~3.4GB model copy, which GRPO's lack of a critic
model leaves room for.
"""

from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import ReferenceLM, StyleReference, make_grpo_reward_func

MODEL_NAME = "Qwen/Qwen3-1.7B"
PROMPTS_PATH = Path(__file__).parent / "prompts.jsonl"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "lora_adapter"

print("Loading frozen reference model for reward scoring...")
ref_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to("mps")
ref_model.eval()
ref_lm = ReferenceLM(model=ref_model, tokenizer=ref_tokenizer, device="mps")

print("Embedding style-similarity reference corpus...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
style_ref = StyleReference(embedder=embedder)

reward_func = make_grpo_reward_func(ref_lm, style_ref)

print(f"Loading prompts from {PROMPTS_PATH}...")
dataset = load_dataset("json", data_files=str(PROMPTS_PATH), split="train")

print("Loading policy model...")
# Load and place explicitly, rather than passing MODEL_NAME as a string to
# GRPOTrainer and letting accelerate handle it -- with two 1.7B model copies
# in the same process, accelerate was falling back to meta-device lazy
# loading for this one, which then crashed trying to materialize it onto MPS
# ("Cannot copy out of meta tensor; no data!"). low_cpu_mem_usage=False
# forces real, immediate allocation instead of the meta-device placeholder path.
policy_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
policy_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", low_cpu_mem_usage=False
).to("mps")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

training_args = GRPOConfig(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=4,
    num_generations=4,       # group size GRPO compares within, per prompt --
                              # must evenly divide generation_batch_size (which
                              # defaults to per_device_train_batch_size); lowered
                              # from 8 rather than raising batch size, to keep
                              # memory down on 16GB unified memory
    max_completion_length=48,  # prompts.jsonl's target_word_count: p50=6,
                              # p75=10, p90=13, p99=25 words. 32 (used for the
                              # POC run) left completions/clipped_ratio at 1.0
                              # across all 5 steps -- nothing ever finished
                              # naturally. 48 tokens covers p99 with headroom
                              # for an EOS token, without 128's excess.
    learning_rate=1e-4,
    max_steps=5,              # proof-of-concept: does it run at all, end to end
    logging_steps=1,
    save_steps=5,
)

trainer = GRPOTrainer(
    model=policy_model,
    processing_class=policy_tokenizer,
    peft_config=peft_config,
    args=training_args,
    reward_funcs=reward_func,
    train_dataset=dataset,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    print(f"Saved LoRA adapter -> {OUTPUT_DIR}")
