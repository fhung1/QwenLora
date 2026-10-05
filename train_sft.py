"""Train a Qwen3-1.7B LoRA on raw message continuations."""

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from build_sft_data import EOS_TOKEN, MODEL_NAME

DATA_PATH = Path(__file__).parent / "sft_pairs.jsonl"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "sft_lora_adapter"
MAX_LENGTH = 512


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.eos_token != EOS_TOKEN:
        raise ValueError(f"Unexpected EOS token: {tokenizer.eos_token!r}")

    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

    def fits_context(example: dict) -> bool:
        prompt_ids = tokenizer(example["prompt"]).input_ids
        full_ids = tokenizer(example["prompt"] + example["completion"]).input_ids
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("SFT pair has an unstable token boundary; rebuild sft_pairs.jsonl")
        return len(full_ids) <= MAX_LENGTH

    before = len(dataset)
    dataset = dataset.filter(fits_context)
    print(f"Kept {len(dataset)} of {before} pairs within {MAX_LENGTH} tokens")
    if len(dataset) < 2:
        raise ValueError("Need at least two SFT pairs after filtering")
    split = dataset.train_test_split(test_size=0.05, seed=42)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", low_cpu_mem_usage=False
    ).to("mps")
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=1e-4,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        eos_token=EOS_TOKEN,
        packing=False,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        peft_config=lora,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(OUTPUT_DIR))
    print(f"Saved SFT LoRA adapter -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
