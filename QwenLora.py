"""Compare base and LoRA outputs from one Qwen3-1.7B model.

Commands: /base, /lora, /both; /raw (training format), /chat; quit/exit.
"""

import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = Path(__file__).parent / "checkpoints" / os.environ.get("QWEN_ADAPTER", "lora_adapter")
# The output directory can exist before an adapter is saved.
ADAPTER_CONFIG_PATH = ADAPTER_PATH / "adapter_config.json"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to("mps")
base_model.eval()

model = base_model
has_lora = False
if ADAPTER_CONFIG_PATH.exists():
    from peft import PeftModel
    print(f"Found LoRA adapter at {ADAPTER_PATH}, loading...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    has_lora = True
    print("Adapter loaded. Use /base, /lora, or /both to pick what responds.")
else:
    print(f"No LoRA adapter found at {ADAPTER_PATH} -- base model only for now.")

mode = "both" if has_lora else "base"
fmt = "raw"  # Training format.
print(f"Ready (mode: {mode}, format: {fmt}). Type a prompt, or 'quit' to exit.\n")


def generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.8) -> str:
    if fmt == "chat":
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to("mps")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_base(prompt: str) -> str:
    if not has_lora:
        return generate(prompt)
    with model.disable_adapter():
        return generate(prompt)


def generate_lora(prompt: str) -> str:
    return generate(prompt)


if __name__ == "__main__":
    while True:
        line = input("> ").strip()
        if line.lower() in ("quit", "exit"):
            break
        if not line:
            continue
        if line.lower() in ("/base", "/lora", "/both"):
            requested = line.lower()[1:]
            if requested == "lora" and not has_lora:
                print("No LoRA adapter loaded -- staying in base mode.\n")
                continue
            mode = requested
            print(f"Mode set to: {mode}\n")
            continue
        if line.lower() in ("/raw", "/chat"):
            fmt = line.lower()[1:]
            print(f"Format set to: {fmt}\n")
            continue

        if mode == "base":
            print(f"\n[base] {generate_base(line)}\n")
        elif mode == "lora":
            print(f"\n[lora] {generate_lora(line)}\n")
        else:  # both
            print(f"\n[base] {generate_base(line)}")
            print(f"\n[lora] {generate_lora(line)}\n")
