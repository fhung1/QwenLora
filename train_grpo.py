"""Train a Qwen3-1.7B LoRA with GRPO."""

from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import StyleReference, make_grpo_reward_func

MODEL_NAME = "Qwen/Qwen3-1.7B"
PROMPTS_PATH = Path(__file__).parent / "prompts.jsonl"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "lora_adapter"

print("Embedding style-similarity reference corpus...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
style_ref = StyleReference(embedder=embedder)

reward_func = make_grpo_reward_func(style_ref)

print(f"Loading prompts from {PROMPTS_PATH}...")
dataset = load_dataset("json", data_files=str(PROMPTS_PATH), split="train")

print("Loading policy model...")
# Avoid meta-device loading when both model copies share MPS.
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
    num_generations=4,  # Matches the batch size.
    max_completion_length=48,  # Prior 32-token run clipped every completion.
    learning_rate=1e-4,
    max_steps=5,  # End-to-end smoke run.
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
