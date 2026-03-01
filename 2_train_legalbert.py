"""
Step 2 (v2): Fine-tune LegalBERT with multi-strategy imbalance handling.

Changes from v1:
  - FOCAL_GAMMA raised from 2.0 → 3.0 (global baseline)
  - Per-class gamma: FPCU, Third Party, Other get gamma=4.0
    These three classes had F1 < 0.60 at gamma=2. A higher gamma puts
    proportionally more loss weight on their hard/misclassified examples.
  - MAX_CAP in preprocessing was raised from 2000 → 3500, so FPCU and
    Third Party now have more training samples; the per-class gamma
    corrects the residual loss-level underrepresentation.

Three-layer defense against the 99:1 class imbalance:
  Layer 1: Data level  → sqrt-frequency rebalancing done in preprocessing
                         MAX_CAP=3500: FPCU keeps ~2800 samples (was 1473)
  Layer 2: Loss level  → Per-class Focal Loss
                         Weak classes: gamma=4 | Others: gamma=3
  Layer 3: Batch level → WeightedRandomSampler — balanced batches

Model : nlpaueb/legal-bert-base-uncased
Output: ./models/legalbert_classifier/
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

MODEL_NAME     = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR     = "./models/legalbert_classifier"
TRAIN_FILE     = "./data/processed/legalbert_train.json"
VAL_FILE       = "./data/processed/legalbert_val.json"
TEST_FILE      = "./data/processed/legalbert_test.json"
LABEL_MAP_FILE = "./data/processed/label_map.json"

MAX_LEN        = 512
BATCH_SIZE     = 16
GRAD_ACCUM     = 2           # effective batch size = 32
EPOCHS         = 16          # +2 extra epochs — more data needs more time
LR             = 2e-5
WARMUP_RATIO   = 0.1
WEIGHT_DECAY   = 0.01
FOCAL_GAMMA    = 3.0         # ← raised from 2.0 (global default for all classes)
SEED           = 42

# Per-class gamma overrides:
# Classes with F1 < 0.6 in v1 get a harder focal penalty (gamma=4)
# This forces the loss to focus even more on their misclassified examples.
# Classes not listed here use the global FOCAL_GAMMA=3.0 above.
WEAK_CLASS_GAMMA = {
    "First Party Collection/Use"        : 4.0,  # was F1=0.55, recall collapsed
    "Third Party Sharing/Collection"    : 4.0,  # was F1=0.56, confused with FPCU
    "Other"                             : 4.0,  # was F1=0.50, semantically incoherent
}

torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD LABEL MAP
# ─────────────────────────────────────────────────────────────

with open(LABEL_MAP_FILE) as f:
    lmap = json.load(f)

ID2LABEL   = {int(k): v for k, v in lmap["id2label"].items()}
LABEL2ID   = lmap["label2id"]
NUM_LABELS = lmap["num_labels"]

CLASS_WEIGHTS_DICT = lmap["class_weights"]
class_weights_list = [CLASS_WEIGHTS_DICT[str(i)] for i in range(NUM_LABELS)]

# Build per-class gamma tensor (shape: [NUM_LABELS])
# Each class gets either its weak-class override or the global FOCAL_GAMMA
per_class_gamma = []
for i in range(NUM_LABELS):
    label = ID2LABEL[i]
    gamma = WEAK_CLASS_GAMMA.get(label, FOCAL_GAMMA)
    per_class_gamma.append(gamma)
per_class_gamma_tensor = torch.tensor(per_class_gamma, dtype=torch.float)

print(f"[Config] {NUM_LABELS} classes, global focal gamma={FOCAL_GAMMA}")
print("[Config] Per-class gamma and weights:")
for i in range(NUM_LABELS):
    label  = ID2LABEL[i]
    w      = class_weights_list[i]
    g      = per_class_gamma[i]
    marker = " ← BOOSTED" if label in WEAK_CLASS_GAMMA else ""
    print(f"  {label:<45}  weight={w:.4f}  gamma={g}{marker}")


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class ClauseDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_len: int):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length  = self.max_len,
            truncation  = True,
            padding     = False,
        )
        enc["labels"] = rec["label_id"]
        return enc


# ─────────────────────────────────────────────────────────────
# PER-CLASS FOCAL LOSS
# ─────────────────────────────────────────────────────────────

class PerClassFocalLoss(nn.Module):
    """
    Focal Loss with:
      - Per-class gamma: weak classes get gamma=4 (higher focus on hard examples)
      - Per-class weights: residual class-frequency correction
    
    For sample i with true class c:
      FL_i = class_weight[c] * (1 - p_t_i)^gamma[c] * CE_i

    This is strictly more expressive than a single global gamma because
    semantically ambiguous classes (FPCU, Other) need more aggressive
    hard-example focusing than already well-separated classes (Policy Change).
    """
    def __init__(self,
                 class_weights: torch.Tensor,
                 per_class_gamma: torch.Tensor):
        super().__init__()
        self.class_weights    = class_weights       # shape: [num_labels]
        self.per_class_gamma  = per_class_gamma     # shape: [num_labels]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = logits.device

        # Unweighted CE per sample (no reduction)
        ce_loss = F.cross_entropy(
            logits, labels,
            weight    = self.class_weights.to(device),
            reduction = "none",
        )

        # p_t = exp(-CE) = the probability assigned to the true class
        p_t = torch.exp(-ce_loss)

        # Look up per-sample gamma from the true class label
        # gamma shape: [batch_size]
        sample_gamma = self.per_class_gamma.to(device)[labels]

        # Focal weight: (1 - p_t)^gamma — higher gamma = more focus on hard examples
        focal_weight = (1.0 - p_t) ** sample_gamma
        focal_loss   = focal_weight * ce_loss

        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────
# CUSTOM TRAINER
# ─────────────────────────────────────────────────────────────

class FocalLossTrainer(Trainer):
    def __init__(self, focal_loss_fn: PerClassFocalLoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────
# WEIGHTED RANDOM SAMPLER
# ─────────────────────────────────────────────────────────────

def make_weighted_sampler(records: list[dict]) -> WeightedRandomSampler:
    label_counts = {}
    for r in records:
        lid = r["label_id"]
        label_counts[lid] = label_counts.get(lid, 0) + 1

    sample_weights = [1.0 / label_counts[r["label_id"]] for r in records]
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(records),
        replacement = True,
    )


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds  = np.argmax(logits, axis=-1)

    acc         = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    macro_f1    = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    weighted_f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]

    per_class_f1 = f1_metric.compute(predictions=preds, references=labels, average=None)["f1"]
    per_class_metrics = {}
    for i, f1_val in enumerate(per_class_f1):
        short_name = ID2LABEL[i].replace("/", "_").replace(" ", "_")[:30]
        per_class_metrics[f"f1_{short_name}"] = round(f1_val, 4)

    return {
        "accuracy"   : acc,
        "macro_f1"   : macro_f1,
        "weighted_f1": weighted_f1,
        **per_class_metrics,
    }


# ─────────────────────────────────────────────────────────────
# CONFIDENCE-BASED POST-PROCESSING
# ─────────────────────────────────────────────────────────────

# FPCU ↔ Third Party is the most common confusion pair.
# If the model predicts FPCU or Third Party with softmax < CONF_THRESHOLD,
# we check the clause for third-party keywords and nudge toward Third Party.
# This is applied at inference and during test eval only — not during training.

CONF_THRESHOLD = 0.55
THIRD_PARTY_KEYWORDS = [
    "third party", "third-party", "third parties", "partners", "vendors",
    "advertisers", "affiliates", "service providers", "business partners",
    "analytics providers", "ad networks", "social media",
]
FPCU_ID         = LABEL2ID["First Party Collection/Use"]
THIRD_PARTY_ID  = LABEL2ID["Third Party Sharing/Collection"]

def postprocess_predictions(logits: np.ndarray, texts: list[str]) -> np.ndarray:
    """
    Apply confidence-threshold correction for the FPCU ↔ Third Party confusion pair.
    Only fires when:
      1. The top prediction is FPCU or Third Party
      2. The softmax confidence is below CONF_THRESHOLD (model is uncertain)
      3. The clause text contains explicit third-party keywords
    """
    import scipy.special as sps
    probs = sps.softmax(logits, axis=-1)
    preds = np.argmax(probs, axis=-1)

    corrected = 0
    for i, (pred, prob_row) in enumerate(zip(preds, probs)):
        confidence = prob_row[pred]
        if pred in (FPCU_ID, THIRD_PARTY_ID) and confidence < CONF_THRESHOLD:
            text_lower = texts[i].lower()
            if any(kw in text_lower for kw in THIRD_PARTY_KEYWORDS):
                preds[i] = THIRD_PARTY_ID
                corrected += 1

    if corrected > 0:
        print(f"  [PostProcess] Corrected {corrected} FPCU→ThirdParty predictions "
              f"(confidence < {CONF_THRESHOLD} + keyword match)")
    return preds


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────

def main():
    with open(TRAIN_FILE) as f:
        train_records = json.load(f)
    with open(VAL_FILE) as f:
        val_records = json.load(f)
    with open(TEST_FILE) as f:
        test_records = json.load(f)

    print(f"\n[Data] Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
    aug_count = sum(1 for r in train_records if r.get("augmented"))
    print(f"[Data] Augmented samples in train: {aug_count} ({aug_count/len(train_records)*100:.1f}%)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = ClauseDataset(train_records, tokenizer, MAX_LEN)
    val_ds   = ClauseDataset(val_records,   tokenizer, MAX_LEN)
    test_ds  = ClauseDataset(test_records,  tokenizer, MAX_LEN)
    collator = DataCollatorWithPadding(tokenizer)

    # Per-class focal loss
    cw = torch.tensor(class_weights_list, dtype=torch.float)
    focal_loss_fn = PerClassFocalLoss(
        class_weights   = cw,
        per_class_gamma = per_class_gamma_tensor,
    )
    print(f"\n[Loss] Using Per-Class Focal Loss")
    print(f"  Weak classes (FPCU, Third Party, Other): gamma=4.0")
    print(f"  All other classes: gamma={FOCAL_GAMMA}")

    sampler = make_weighted_sampler(train_records)
    print("[Sampler] WeightedRandomSampler active — balanced batches guaranteed")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels             = NUM_LABELS,
        id2label               = ID2LABEL,
        label2id               = LABEL2ID,
        ignore_mismatched_sizes= True,
    )

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "macro_f1",
        greater_is_better           = True,
        logging_steps               = 50,
        fp16                        = torch.cuda.is_available(),
        seed                        = SEED,
        report_to                   = "none",
        dataloader_num_workers      = 4,
        dataloader_drop_last        = True,
    )

    trainer = FocalLossTrainer(
        focal_loss_fn    = focal_loss_fn,
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        data_collator    = collator,
        compute_metrics  = compute_metrics,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Inject weighted sampler
    original_get_train_dl = trainer.get_train_dataloader

    def custom_train_dataloader():
        dl = original_get_train_dl()
        return DataLoader(
            dl.dataset,
            batch_size  = BATCH_SIZE,
            sampler     = sampler,
            collate_fn  = collator,
            num_workers = 4,
            pin_memory  = True,
        )

    trainer.get_train_dataloader = custom_train_dataloader

    print("\n[LegalBERT] Starting training...")
    print("[LegalBERT] Watch FPCU, Third Party, Other per-class F1 — target > 0.65 by epoch 5.")
    trainer.train(resume_from_checkpoint="./models/legalbert_classifier/checkpoint-3468")

    # ── Validation eval ──
    val_results = trainer.evaluate(val_ds)
    print("\n[LegalBERT] Validation results:")
    for k, v in sorted(val_results.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Test eval ──
    print("\n[LegalBERT] Running on held-out test set...")
    test_preds_out = trainer.predict(test_ds)
    raw_logits     = test_preds_out.predictions
    true_labels    = test_preds_out.label_ids
    label_names    = [ID2LABEL[i] for i in range(NUM_LABELS)]

    # Standard argmax predictions
    pred_labels_raw = np.argmax(raw_logits, axis=-1)

    # Post-processed predictions (FPCU ↔ Third Party confidence correction)
    test_texts = [r["text"] for r in test_records]
    pred_labels_pp = postprocess_predictions(raw_logits, test_texts)

    print("\n[LegalBERT] Classification report — RAW predictions:")
    print(classification_report(true_labels, pred_labels_raw, target_names=label_names, digits=4))

    print("\n[LegalBERT] Classification report — POST-PROCESSED predictions:")
    print(classification_report(true_labels, pred_labels_pp, target_names=label_names, digits=4))

    cm = confusion_matrix(true_labels, pred_labels_pp)
    cm_dict = {"labels": label_names, "matrix": cm.tolist()}

    # ── Save ──
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    report_raw = classification_report(
        true_labels, pred_labels_raw,
        target_names=label_names, digits=4, output_dict=True,
    )
    report_pp = classification_report(
        true_labels, pred_labels_pp,
        target_names=label_names, digits=4, output_dict=True,
    )

    all_results = {
        "val"                          : val_results,
        "confusion_matrix"             : cm_dict,
        "classification_report_raw"    : report_raw,
        "classification_report_postproc": report_pp,
        "config": {
            "focal_gamma_global"  : FOCAL_GAMMA,
            "weak_class_gamma"    : WEAK_CLASS_GAMMA,
            "conf_threshold"      : CONF_THRESHOLD,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[LegalBERT] Model + results saved to {OUTPUT_DIR}")

    # ── Imbalance check ──
    print("\n[Imbalance Check] Per-class F1 (post-processed) on test set:")
    problem_classes = []
    for label in label_names:
        f1      = report_pp.get(label, {}).get("f1-score", 0)
        support = report_pp.get(label, {}).get("support", 0)
        status  = "✓" if f1 >= 0.6 else "⚠ LOW"
        marker  = " ← was LOW in v1" if label in WEAK_CLASS_GAMMA else ""
        print(f"  {status}  {label:<45}  F1={f1:.4f}  (n={support}){marker}")
        if f1 < 0.6:
            problem_classes.append(label)

    if problem_classes:
        print(f"\n⚠ WARNING: Still low F1: {problem_classes}")
        print("  Next steps: check confusion matrix for specific misclassification patterns")
        print("  and consider label-specific data augmentation for these classes.")
    else:
        print("\n✓ All classes achieved F1 ≥ 0.6")


if __name__ == "__main__":
    main()
