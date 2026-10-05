"""Pilot Qwen3-1.7B LoRA SFT on incoming-message/reply pairs."""

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from build_sft_data import EOS_TOKEN, MODEL_NAME

DATA_PATH = Path(__file__).parent / "reply_sft_pairs.jsonl"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "reply_sft_lora_adapter"
MAX_LENGTH = 512


def cap_conversations(dataset, limit: int):
    if limit == 0:
        return dataset
    by_chat = defaultdict(list)
    for index, chat_id in enumerate(dataset["conversation_id"]):
        by_chat[chat_id].append(index)
    rng = random.Random(42)
    selected = []
    for indices in by_chat.values():
        rng.shuffle(indices)
        selected.extend(indices[:limit])
    return dataset.select(sorted(selected))


def format_pair(example, tokenizer):
    prompt = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        example["prompt"] + example["completion"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    if full.endswith(EOS_TOKEN + "\n"):
        full = full[:-1]
    if not full.startswith(prompt):
        return {"prompt": "", "completion": "", "valid": False}
    prompt_ids = tokenizer(prompt).input_ids
    full_ids = tokenizer(full).input_ids
    valid = (full_ids[:len(prompt_ids)] == prompt_ids
             and len(full_ids) <= MAX_LENGTH
             and full.endswith(EOS_TOKEN))
    return {"prompt": prompt, "completion": full[len(prompt):], "valid": valid}


def prepare_dataset(tokenizer, max_per_conversation: int):
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    total = len(dataset)
    dataset = cap_conversations(dataset, max_per_conversation)
    capped = len(dataset)
    formatted = dataset.map(
        lambda row: format_pair(row, tokenizer),
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Apply non-thinking Qwen chat template",
    )
    formatted = formatted.filter(lambda row: row["valid"])
    valid = len(formatted)
    formatted = formatted.remove_columns("valid")
    if valid < 20:
        raise ValueError("Too few valid chat pairs after tokenization")
    print(f"Loaded {total} pairs; kept {capped} after chat cap; {valid} fit tokenization/context", flush=True)
    split = formatted.train_test_split(test_size=0.05, seed=42)
    print(f"Train: {len(split['train'])}; validation: {len(split['test'])}", flush=True)
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-per-conversation", type=int, default=2000)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (args.max_steps != -1 and args.max_steps <= 0) or args.max_per_conversation < 0:
        parser.error("max-steps must be -1 or positive; max-per-conversation must be nonnegative")
    if args.learning_rate <= 0 or args.eval_steps <= 0:
        parser.error("learning-rate and eval-steps must be positive")
    if not DATA_PATH.exists():
        parser.error(f"Reply dataset not found: {DATA_PATH}")
    if args.init_adapter and not (args.init_adapter / "adapter_config.json").exists():
        parser.error(f"Adapter not found: {args.init_adapter}")
    if args.resume_from_checkpoint and not (args.resume_from_checkpoint / "trainer_state.json").exists():
        parser.error(f"Checkpoint not found: {args.resume_from_checkpoint}")
    if (not args.prepare_only and not args.resume_from_checkpoint and
            args.output_dir.exists() and any(args.output_dir.iterdir())):
        parser.error(f"Output directory already contains files: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.eos_token != EOS_TOKEN:
        raise ValueError(f"Unexpected EOS token: {tokenizer.eos_token!r}")
    split = prepare_dataset(tokenizer, args.max_per_conversation)
    if args.prepare_only:
        return

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", low_cpu_mem_usage=False,
    ).to("mps")
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
        lora = None
        print(f"Continuing trainable adapter from {args.init_adapter}", flush=True)
    else:
        lora = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        print("Starting a new adapter from base Qwen", flush=True)
    config = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        eos_token=EOS_TOKEN,
        packing=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model, args=config, processing_class=tokenizer, peft_config=lora,
        train_dataset=split["train"], eval_dataset=split["test"],
    )
    trainable = sum(param.numel() for param in trainer.model.parameters() if param.requires_grad)
    if trainable == 0:
        raise ValueError("No trainable adapter parameters")
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(split["train"]) / 4)
    print(f"Trainable parameters: {trainable}; planned steps: {planned_steps}", flush=True)
    prepared = trainer.train_dataset[0]
    labels = prepared.get("labels", [])
    first_target = next((i for i, value in enumerate(labels) if value != -100), None)
    if (first_target is None or first_target == 0 or
            any(value != -100 for value in labels[:first_target]) or
            labels[first_target:] != prepared["input_ids"][first_target:]):
        raise ValueError("TRL did not mask the prompt tokens from training loss")
    if prepared["input_ids"][-1] != tokenizer.eos_token_id or labels[-1] != tokenizer.eos_token_id:
        raise ValueError("The assistant end token is not included in completion loss")
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir))
    print(f"Saved reply SFT LoRA adapter -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
