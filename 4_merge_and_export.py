"""
Step 4: Merge Qwen2.5 LoRA adapters into the base model.

Merges the trained LoRA delta weights back into Qwen2.5-1.5B-Instruct
base weights, producing a single standalone model for faster inference.

Without merging: every inference call applies adapters on top of base (slower)
After merging  : one unified model, no adapter overhead (~10-15% faster inference)

Input : models/qwen_lora/          (LoRA adapters, ~80 MB)
Output: models/qwen_merged/        (full fp16 model, ~3.5 GB)

RAM requirement: ~8 GB CPU RAM (1.5B model in fp16 is much lighter than LLaMA 3)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./models/qwen_lora"
MERGED_DIR  = "./models/qwen_merged"

os.makedirs(MERGED_DIR, exist_ok=True)


def main():
    print("[Merge] Loading Qwen2.5-1.5B base model in fp16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype       = torch.float16,
        device_map        = "cpu",          # merge on CPU — avoids VRAM spikes
        trust_remote_code = True,
    )

    print(f"[Merge] Loading LoRA adapters from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    print("[Merge] Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"[Merge] Saving merged model to {MERGED_DIR}...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"[Merge] Done.")
    print(f"  Model saved to : {MERGED_DIR}")
    print(f"  Use this path in 5_inference.py: QWEN_DIR = '{MERGED_DIR}'")


if __name__ == "__main__":
    main()
