"""
Step 3: Fine-tune Qwen2.5-1.5B-Instruct with QLoRA.

Model architecture (Qwen2.5-1.5B-Instruct):
  - Hidden size       : 1536
  - Intermediate size : 8960
  - Num heads         : 12
  - Num KV heads      : 2   ← GQA (grouped query attention)
  - Num layers        : 28
  - Vocab size        : 151936
  - Context window    : 131072 (we use 1024 for training efficiency)
  - RoPE theta        : 1,000,000
  - Activation        : SiLU (gate_proj * up_proj → down_proj)

LoRA target modules (all linear projections in Qwen2 attention + MLP):
  Attention : q_proj, k_proj, v_proj, o_proj
  MLP       : gate_proj, up_proj, down_proj
  NOTE: q_proj=1536→1536, k_proj=1536→256, v_proj=1536→256 (due to GQA with 2 KV heads)

QLoRA config:
  - 4-bit NF4 quantisation of base weights
  - bfloat16 compute dtype (A-series GPUs) or float16 (older GPUs)
  - Double quantisation enabled (saves ~0.4 GB extra)
  - LoRA applied in bfloat16 on top of frozen 4-bit base

Input : data/processed/llama3_train.json  (LLaMA 3 format → reformatted at load time)
Output: models/qwen_lora/

Hardware:
  - 6 GB VRAM  : batch=1, grad_accum=16
  - 8 GB VRAM  : batch=2, grad_accum=8   ← default
  - 12+ GB VRAM: batch=4, grad_accum=4

Qwen2.5 chat template:
  <|im_start|>system
  {system}<|im_end|>
  <|im_start|>user
  {user}<|im_end|>
  <|im_start|>assistant
  {assistant}<|im_end|>

Response template for completion-only masking: "<|im_start|>assistant"
Everything before this in the loss is masked to -100.
"""

import os
import re
import json
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR  = "./models/qwen_lora"
TRAIN_FILE  = "./data/processed/llama3_train.json"
VAL_FILE    = "./data/processed/llama3_val.json"

MAX_SEQ_LEN  = 1024   # model supports 131k but 1024 is enough for privacy clauses
BATCH_SIZE   = 3      # increase to 4 if VRAM > 12 GB
GRAD_ACCUM   = 8      # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS       = 3
LR           = 2e-4
WARMUP_RATIO = 0.05
SEED         = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────
# QUANTISATION CONFIG
#
# NF4 (Normal Float 4) is optimal for normally distributed weights.
# Double quantisation quantises the quantisation constants themselves
# saving an extra ~0.4 GB on a 1.5B model.
# bfloat16 compute dtype: better numerical stability than float16
# on A-series GPUs (A10, A100). Use torch.float16 on RTX 30xx/20xx.
# ─────────────────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)


# ─────────────────────────────────────────────────────────────
# LORA CONFIG
#
# Qwen2.5-1.5B projection dimensions (with GQA, num_kv_heads=2):
#   q_proj  : 1536 → 1536   (12 heads × 128 dim)
#   k_proj  : 1536 → 256    (2 KV heads × 128 dim)
#   v_proj  : 1536 → 256    (2 KV heads × 128 dim)
#   o_proj  : 1536 → 1536
#   gate_proj: 1536 → 8960
#   up_proj  : 1536 → 8960
#   down_proj: 8960 → 1536
#
# r=16 is a good balance for a 1.5B model:
#   - r=8  : fewer params, faster, slightly weaker adaptation
#   - r=16 : ~18M trainable params (~1.2% of 1.5B) — sweet spot
#   - r=32 : more expressive but diminishing returns + slower
#
# lora_alpha=32 (= 2×r): standard scaling, keeps effective LR stable
# dropout=0.05: light regularisation, fine for small datasets
# ─────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    target_modules = [
        # Attention projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLP projections (SiLU gated MLP)
        "gate_proj", "up_proj", "down_proj",
    ],
    bias           = "none",     # don't train biases — saves params, no quality loss
    inference_mode = False,
)


# ─────────────────────────────────────────────────────────────
# LLAMA 3 FORMAT → QWEN FORMAT CONVERTER
#
# Preprocessor outputs LLaMA 3 format:
#   <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#   {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
#   {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#   {assistant}<|eot_id|>
#
# We parse the three content blocks and re-apply Qwen's native
# tokenizer.apply_chat_template(), which produces:
#   <|im_start|>system\n{system}<|im_end|>\n
#   <|im_start|>user\n{user}<|im_end|>\n
#   <|im_start|>assistant\n{assistant}<|im_end|>
# ─────────────────────────────────────────────────────────────

_SYSTEM_RE    = re.compile(
    r"<\|start_header_id\|>system<\|end_header_id\|>\n(.*?)<\|eot_id\|>", re.DOTALL
)
_USER_RE      = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\n(.*?)<\|eot_id\|>", re.DOTALL
)
_ASSISTANT_RE = re.compile(
    r"<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)<\|eot_id\|>", re.DOTALL
)


def llama3_to_messages(text: str) -> list[dict] | None:
    """Parse a LLaMA 3 formatted string into message dicts. Returns None if malformed."""
    sys_m  = _SYSTEM_RE.search(text)
    user_m = _USER_RE.search(text)
    asst_m = _ASSISTANT_RE.search(text)
    if not (sys_m and user_m and asst_m):
        return None
    return [
        {"role": "system",    "content": sys_m.group(1).strip()},
        {"role": "user",      "content": user_m.group(1).strip()},
        {"role": "assistant", "content": asst_m.group(1).strip()},
    ]


def convert_to_qwen(records: list[dict], tokenizer) -> list[dict]:
    """Convert list of LLaMA 3 records to Qwen chat format. Skips malformed records."""
    converted = []
    skipped   = 0
    for rec in records:
        messages = llama3_to_messages(rec["text"])
        if messages is None:
            skipped += 1
            continue
        # add_generation_prompt=False: include full assistant turn in the text
        # (we want the model to learn to generate the assistant content)
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = False,
        )
        converted.append({"text": formatted})
    if skipped > 0:
        print(f"  [WARN] Skipped {skipped} malformed records")
    print(f"  Converted {len(converted)} records to Qwen chat format")
    return converted


# ─────────────────────────────────────────────────────────────
# COMPLETION-ONLY COLLATOR
#
# Masks all tokens BEFORE the assistant response with -100 so
# the model only learns to generate the assistant turn.
#
# For Qwen2.5 the response template is exactly: "<|im_start|>assistant"
# This appears verbatim in every apply_chat_template() output.
# Everything up to and including this token sequence is masked.
# The newline after "assistant" is part of the response and IS predicted.
# ─────────────────────────────────────────────────────────────

class CompletionOnlyCollator:
    def __init__(self, response_template: str, tokenizer):
        self.tokenizer    = tokenizer
        self.template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        print(f"[Collator] Response template  : '{response_template}'")
        print(f"[Collator] Template token IDs : {self.template_ids}")
        # Verify the template decodes back correctly
        decoded = tokenizer.decode(self.template_ids)
        print(f"[Collator] Decoded back        : '{decoded}'")

    def __call__(self, features: list) -> dict:
        input_ids_list = [torch.tensor(f["input_ids"]) for f in features]
        labels_list    = []

        for input_ids in input_ids_list:
            labels = input_ids.clone()
            ids    = input_ids.tolist()
            tlen   = len(self.template_ids)
            found  = False

            for i in range(len(ids) - tlen + 1):
                if ids[i: i + tlen] == self.template_ids:
                    labels[: i + tlen] = -100   # mask system + user + template
                    found = True
                    break

            if not found:
                labels[:] = -100   # no assistant turn found → zero loss contribution

            labels_list.append(labels)

        pad_id = self.tokenizer.pad_token_id

        input_ids_padded = pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        labels_padded = pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids_padded != pad_id).long()

        return {
            "input_ids"     : input_ids_padded,
            "labels"        : labels_padded,
            "attention_mask": attention_mask,
        }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    # ── Tokenizer ──────────────────────────────────────────────
    print("[Qwen] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Qwen2.5 ships with a proper pad token ("<|endoftext|>", id=151643)
    # but set it explicitly in case it's None on some versions
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # right-pad for causal LM training

    print(f"  Vocab size     : {tokenizer.vocab_size}")
    print(f"  Pad token      : '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"  EOS token      : '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"  Chat template  : {tokenizer.chat_template[:80]}...")

    # Verify chat template produces expected Qwen format
    test_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True
    )
    assert "<|im_start|>" in test_prompt, "Unexpected chat template format!"
    print(f"  Template check : ✓ (<|im_start|> found in test prompt)")

    # ── Load + convert data ────────────────────────────────────
    print("\n[Qwen] Loading and converting datasets to Qwen chat format...")

    def load_and_convert(path: str) -> Dataset:
        with open(path) as f:
            records = json.load(f)
        if not records:
            raise ValueError(f"Empty file: {path}")
        converted = convert_to_qwen(records, tokenizer)
        print(f"  {path}: {len(records)} raw → {len(converted)} converted")
        if converted:
            print(f"  Format preview (first 400 chars):\n  {converted[0]['text'][:400]}\n")
        return Dataset.from_list(converted)

    train_ds = load_and_convert(TRAIN_FILE)
    val_ds   = load_and_convert(VAL_FILE)

    # ── Collator ───────────────────────────────────────────────
    # "<|im_start|>assistant" is the exact string that apply_chat_template
    # puts before every assistant turn in Qwen2.5 format
    response_template = "<|im_start|>assistant"
    collator = CompletionOnlyCollator(
        response_template = response_template,
        tokenizer         = tokenizer,
    )

    # ── Collator sanity check ──────────────────────────────────
    print("\n[Collator] Sanity check on first training sample...")
    sample_enc  = tokenizer(
        train_ds[0]["text"], max_length=MAX_SEQ_LEN,
        truncation=True, return_tensors="pt",
    )
    sample_feat  = {"input_ids": sample_enc["input_ids"][0].tolist()}
    sample_batch = collator([sample_feat])
    labels       = sample_batch["labels"][0]
    input_ids    = sample_batch["input_ids"][0]
    n_masked     = (labels == -100).sum().item()
    n_unmasked   = (labels != -100).sum().item()

    print(f"  Total tokens   : {len(labels)}")
    print(f"  Masked (prompt): {n_masked}")
    print(f"  Unmasked (resp): {n_unmasked}")

    if n_unmasked == 0:
        # Print debug info before raising
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        print("\n  [DEBUG] First 40 tokens:")
        for i, tok in enumerate(all_tokens[:40]):
            print(f"    {i:3d}  {repr(tok)}")
        template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        print(f"\n  [DEBUG] Template '{response_template}' encodes to: {template_ids}")
        print(f"  [DEBUG] Text sample: {train_ds[0]['text'][:500]}")
        raise RuntimeError(
            "\n[ERROR] ALL tokens are masked — response template not found in sample.\n"
            "This means apply_chat_template is not producing '<|im_start|>assistant'.\n"
            "Check tokenizer.chat_template or report the debug output above."
        )

    first_response_tokens = tokenizer.decode(
        input_ids[labels != -100][:80], skip_special_tokens=False
    )
    print(f"  First unmasked : '{first_response_tokens}'")
    print("  Masking OK ✓\n")

    # ── Tokenize datasets ──────────────────────────────────────
    def tokenize(example):
        return tokenizer(
            example["text"],
            max_length = MAX_SEQ_LEN,
            truncation = True,
            padding    = False,   # collator handles padding
        )

    print("[Qwen] Tokenizing datasets...")
    train_ds_tok = train_ds.map(
        tokenize, remove_columns=["text"], desc="Tokenizing train"
    )
    val_ds_tok = val_ds.map(
        tokenize, remove_columns=["text"], desc="Tokenizing val"
    )

    avg_len = sum(len(x["input_ids"]) for x in train_ds_tok) / len(train_ds_tok)
    print(f"  Avg token length (train): {avg_len:.0f} / {MAX_SEQ_LEN}")

    # ── Base model in 4-bit ────────────────────────────────────
    print("\n[Qwen] Loading Qwen2.5-1.5B-Instruct in 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
        torch_dtype         = torch.bfloat16,
        attn_implementation = "eager",  # use "flash_attention_2" if flash-attn installed
    )

    # prepare_model_for_kbit_training:
    #   - Enables gradient checkpointing inside 4-bit layers
    #   - Upcasts LayerNorm to float32 for stable training
    #   - Enables input embedding gradients
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing = True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: ~1.2% trainable (~18.5M / 1.54B params)

    # ── Training steps calculation ─────────────────────────────
    steps_per_epoch  = len(train_ds_tok) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps      = steps_per_epoch * EPOCHS
    warmup_steps     = max(1, int(total_steps * WARMUP_RATIO))
    eval_save_steps  = max(50, steps_per_epoch // 2)  # eval 2x per epoch

    print(f"\n[Config] Train samples     : {len(train_ds_tok)}")
    print(f"[Config] Val samples       : {len(val_ds_tok)}")
    print(f"[Config] Effective batch   : {BATCH_SIZE * GRAD_ACCUM}")
    print(f"[Config] Steps per epoch   : {steps_per_epoch}")
    print(f"[Config] Total steps       : {total_steps}")
    print(f"[Config] Warmup steps      : {warmup_steps}")
    print(f"[Config] Eval/save steps   : {eval_save_steps}")

    # ── Training args ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                    = OUTPUT_DIR,
        num_train_epochs              = EPOCHS,
        per_device_train_batch_size   = BATCH_SIZE,
        per_device_eval_batch_size    = BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        learning_rate                 = LR,
        warmup_steps                  = warmup_steps,
        lr_scheduler_type             = "cosine",
        # paged_adamw_8bit: offloads optimizer states to CPU when not in use
        # saves ~1 GB VRAM vs standard AdamW with minimal speed impact
        optim                         = "paged_adamw_8bit",
        bf16                          = True,     # bfloat16 training (A-series GPUs)
        # fp16                        = True,     # use instead on RTX 30xx/20xx
        eval_strategy                 = "steps",
        eval_steps                    = eval_save_steps,
        save_strategy                 = "steps",
        save_steps                    = eval_save_steps,
        save_total_limit              = 3,
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        logging_steps                 = 20,
        seed                          = SEED,
        report_to                     = "none",
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        dataloader_num_workers        = 2,
        # Remove the old eval_accumulation_steps to avoid OOM on large val sets
        eval_accumulation_steps       = 4,
    )

    # ── Trainer ────────────────────────────────────────────────
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds_tok,
        eval_dataset  = val_ds_tok,
        data_collator = collator,
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ──────────────────────────────────────────────────
    print("\n[Qwen] Starting QLoRA fine-tuning...")
    print(f"       Expected time: ~20-40 min (RTX 3060) | ~12 min (A100)\n")

    # Resume from checkpoint if one exists (handles interrupted runs)
    checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            checkpoint = os.path.join(OUTPUT_DIR, sorted(ckpts)[-1])
            print(f"  Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    # ── Save ───────────────────────────────────────────────────
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    best_loss = trainer.state.best_metric
    print(f"\n[Qwen] Training complete.")
    print(f"       Best eval loss  : {best_loss:.4f}")
    print(f"       LoRA adapters   : {OUTPUT_DIR}")
    print(f"       Next step       : python 4_merge_and_export.py")


if __name__ == "__main__":
    main()
