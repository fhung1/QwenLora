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
    num_generations=8,       # group size GRPO compares within, per prompt
    max_completion_length=128,
    learning_rate=1e-4,
    num_train_epochs=1,      # start small -- Phase 1-style sanity run first
    logging_steps=5,
    save_steps=50,
)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    peft_config=peft_config,
    args=training_args,
    reward_funcs=reward_func,
    train_dataset=dataset,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    print(f"Saved LoRA adapter -> {OUTPUT_DIR}")
