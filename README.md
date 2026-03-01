# PrivacyGuard — AI Privacy Policy Risk Scanner

> AMD Slingshot Project   
> LegalBERT (Fine-Tuned) + Qwen 2.5 (QLoRA) · AWS SageMaker · DynamoDB Cache · Chrome Extension

PrivacyGuard is a Chrome extension that automatically analyses privacy policies and terms of service pages. It uses a two-model AI pipeline to classify risk clauses and generate plain-English explanations, backed by a cloud infrastructure that caches results so repeated visits cost nothing.

---

## How It Works

1. The side panel opens and immediately checks DynamoDB for a cached result (<7 days old) — if found, findings render instantly with no ML inference
2. If no cache: click Scan Policy — `content.js` strips PII in-browser and sends clean text chunks to the API
3. LegalBERT classifies every clause into one of 10 risk categories with a confidence score
4. Qwen 2.5 generates a plain-English explanation for the top risky clauses
5. Results are written to DynamoDB and returned to the extension
6. Findings are grouped by category and numbered — click Show explanation for full AI analysis per clause
7. View Summary tab shows collective analysis per category. Permissions tab shows active browser hardware permissions for the site

---


## Repository Structure

```
privacyguard/
│
├── 0_inspect_datasets.py    # Verify OPP-115 structure and class distribution
├── 1_preprocess.py          # Clean + split dataset, build label_map.json
├── 2_train_legalbert.py     # Fine-tune LegalBERT (Focal Loss + WeightedRandomSampler)
├── 3_train_qwen.py          # QLoRA fine-tune Qwen 2.5 1.5B with CompletionOnlyCollator
├── 4_merge_and_export.py    # Merge LoRA adapters into base model for deployment
├── 5_inference.py           # Local two-model pipeline (demo / text / html / batch)
├── local_server.py          # FastAPI server wrapping 5_inference.py for the extension
│
├── chrome_extension/
│   ├── manifest.json        # Permissions: sidePanel, storage, activeTab, scripting
│   ├── background.js        # Service worker — opens panel on toolbar icon click
│   ├── content.js           # Injected into page: extract text, scrub PII, highlight elements
│   ├── sidepanel.html       # UI layout: header, risk bar, Findings/Summary/Permissions tabs
│   └── sidepanel.js         # All UI logic: boot, cache check, scan, render, summary
│
├── models/                  # ← Download from HuggingFace (see Step 2 — too large for GitHub)
│   ├── legalbert_classifier/
│   └── qwen_merged/
│
├── requirements.txt
└── README.md
```

---

## Setup Guide

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- GPU with ≥ 6 GB VRAM recommended for Qwen 2.5 (RTX 4050 / 4060 )
- CPU-only mode works for LegalBERT classification only (`/analyze/fast` endpoint)
- Chrome v114 or later — required for the Side Panel API

---

### Step 1 — Create Conda Environment

```bash
conda create -n privacyguard python=3.10 -y
conda activate privacyguard
```

Install PyTorch — pick the line matching your CUDA version:

```bash
# CUDA 12.1  (RTX 30/40 series, A100, A10G — most common)
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# CUDA 11.8  (older GPUs)
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CPU only  (LegalBERT classification only, no Qwen 2.5 explanations)
conda install pytorch==2.2.2 torchvision==0.17.2 cpuonly -c pytorch -y
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

### Step 2 — Download Models from HuggingFace

The trained weights are hosted on HuggingFace because they are too large for GitHub (LegalBERT ~500 MB, Qwen 2.5 merged ~3 GB).

```bash
pip install huggingface_hub

# Download LegalBERT classifier
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vir101/privacyguard-legalbert',
    local_dir='./models/legalbert_classifier'
)
"

# Download Qwen 2.5 merged model
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vir101/privacyguard-qwen-merged',
    local_dir='./models/qwen_merged'
)
"
```


---

### Step 3 — Start the Local Inference Server

```bash
# local_server.py imports from inference.py — create it first
cp 5_inference.py inference.py

# Start the server (keep this terminal open while using the extension)
python local_server.py 
```

Expected output:

```

╔══════════════════════════════════════════════════╗
║        PrivacyGuard — Local Inference Server     ║
╚══════════════════════════════════════════════════╝

2026-03-01 15:37:34,358 [INFO] Loading LegalBERT from ./models/legalbert_classifier ...
Loading weights: 100%|██████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 1475.52it/s, Materializing param=classifier.weight]
2026-03-01 15:37:36,138 [INFO]   LegalBERT loaded on cuda
2026-03-01 15:37:36,139 [INFO] Loading Qwen2.5 from ./models/qwen_merged ...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|███████████████████████████████████████████████████████████████| 338/338 [00:01<00:00, 280.59it/s, Materializing param=model.norm.weight]
2026-03-01 15:37:40,205 [INFO]   Qwen2.5 loaded
2026-03-01 15:37:40,205 [INFO] ✓ Models ready — server accepting requests

Server ready!
```


---

### Step 4 — Load the Chrome Extension

1. Open Chrome → navigate to `chrome://extensions`
2. Enable Developer mode (toggle in the top-right corner)
3. Click Load unpacked → select the `chrome_extension/` folder
4. The PrivacyGuard icon appears in the Chrome toolbar

---

### Step 5 — Use the Extension

1. Navigate to any privacy policy page (e.g. `reddit.com/policies/privacy-policy`)
2. Click the PrivacyGuard toolbar icon — the side panel opens
3. Click Scan Policy
4. Findings appear grouped by category with numbered clauses
5. Click Show explanation to see the AI justification + key excerpt for any clause
6. Click View Summary for a collective analysis per risk category
7. Switch to Permissions to see active browser hardware permissions for the site

---

## Running Inference Without the Extension

```bash
# Built-in demo policy
python 5_inference.py --demo

# Your own text file
python 5_inference.py --text path/to/policy.txt

# HTML file
python 5_inference.py --html path/to/privacy_policy.html

# Classification only — fast, no Qwen 2.5
python 5_inference.py --demo --no-explain
```

---

## ML Models

### LegalBERT — Classifier

BERT-base (110M parameters) pre-trained on legal corpora (contracts, court opinions, legislation), then fine-tuned on the OPP-115 dataset — 115 real privacy policies with ~23,000 human-annotated clause labels.

| Parameter | Value |
|-----------|-------|
| Base model | `nlpaueb/legal-bert-base-uncased` |
| Dataset | OPP-115 — 115 policies, ~23,000 annotations |
| Loss function | Focal Loss — per-class gamma (4.0 for FPCU/Third Party, 3.0 default) |
| Sampling | WeightedRandomSampler — all 10 classes present in every batch |
| Batch size | 16 |
| Learning rate | 2e-5 with cosine schedule |
| Epochs | 12 with early stopping (patience 3) |
| Macro-F1 | ~71% across 10 severely imbalanced classes |
| Inference speed | ~2–5ms per clause on CPU |

The 10 risk categories:

| Category | Risk | What It Detects |
|----------|------|-----------------|
| Third Party Sharing/Collection | 🔴 HIGH | Data shared with advertisers, analytics partners, affiliates |
| Data Retention | 🔴 HIGH | How long data is kept, indefinite retention clauses |
| First Party Collection/Use | 🟡 MEDIUM | What data the service itself collects directly |
| User Choice/Control | 🟡 MEDIUM | Opt-out mechanisms, consent language |
| Data Security | 🟡 MEDIUM | Encryption, breach notification, security measures |
| Policy Change | 🟡 MEDIUM | How and when the policy can be updated |
| User Access, Edit and Deletion | 🟡 MEDIUM | Rights to access, correct, or delete personal data |
| Do Not Track | 🟡 MEDIUM | Response to browser DNT signals |
| International and Specific Audiences | 🟢 LOW | GDPR, COPPA, cross-border data transfers |
| Other | 🟢 LOW | Clauses that do not fit the above categories |

FPCU vs Third Party Confusion: the hardest classification problem — both categories use nearly identical vocabulary, the difference being "we collect" vs "our partners collect". Addressed at three levels: higher focal gamma during training, a 0.65 confidence threshold at inference, and a keyword scoring heuristic that counts `we collect / we use / we process` signals against `third party / partners / affiliates` signals to tip low-confidence predictions correctly.

---

### Qwen 2.5 1.5B — Explainer

QLoRA fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` to generate structured plain-English explanations for each risky clause found by LegalBERT.

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4 quantised) |
| Method | QLoRA — rank 16, alpha 32, dropout 0.05 |
| LoRA target layers | q/k/v/o_proj + gate/up/down_proj (all attention + FFN) |
| Trainable parameters | ~42M (0.52% of model) |
| Training samples | 7,207 train + 1,082 validation |
| Learning rate | 2e-4 cosine schedule, 5% warmup |
| Epochs | 3 with early stopping (patience 3) |
| Collator | CompletionOnlyCollator — loss on assistant response tokens only |
| Expected training time | ~3–5h on A100, ~4–6h on A10G, ~8–10h on T4 |
| Estimated training cost | ~$1.35–4.00 on g5.xlarge spot |

QLoRA explained: the base model is compressed to 4-bit NF4 (~3–4 GB VRAM) and frozen. Small trainable LoRA adapter matrices are injected alongside the frozen layers. Only the 42M adapter weights (~150 MB) are updated during training. At inference the adapters are merged back into the base model.

CompletionOnlyCollator: standard training computes loss on every token including the repeated system prompt boilerplate (identical across all 7,207 examples). The custom collator masks all tokens before the assistant response with `-100` so only the explanation itself contributes to the gradient — preventing the model from overfitting to the prompt template.

Each explanation output includes: `Clause Type`, `Risk Level`, `Analysis` (the justification shown in the UI), `Key Excerpt` (verbatim quote from the clause), and optionally `User Rights`.

---

## Dataset

OPP-115 (Online Privacy Policies) from Carnegie Mellon University — 115 real privacy policies with ~23,000 clause segments annotated by law students and privacy researchers. Available at [usableprivacy.org/data](https://usableprivacy.org/data).

Class imbalance is severe: 90:1 ratio between the most common class (User Access, 4,200 samples) and the rarest (Do Not Track, 47 samples). Addressed with three techniques applied together:

- Data level: sqrt-frequency scaling, MAX_CAP=3500 per class, MIN_FLOOR=300 per class, rule-based augmentation for rare classes
- Loss level: Focal Loss with gamma=4.0 for the rarest and most confused classes, 3.0 for the rest
- Sampling level: WeightedRandomSampler guarantees all 10 classes appear in every training batch regardless of their frequency

---

## Training From Scratch (Optional)

```bash
# Download OPP-115 from usableprivacy.org/data → place at data/opp115/

python 0_inspect_datasets.py    # verify data structure and class distribution
python 1_preprocess.py          # clean + split → data/processed/
python 2_train_legalbert.py     # ~1–2h on any GPU or CPU
python 3_train_qwen.py          # ~4–6h on A10G  (requires ≥8 GB VRAM)
python 4_merge_and_export.py    # merge LoRA adapters → models/qwen_merged/
python 5_inference.py --demo    # test the full pipeline end-to-end
```

---

## AWS Infrastructure

To use the AWS backend instead of `local_server.py`, update `API_URL` in `chrome_extension/sidepanel.js` with your API Gateway invoke URL.

| Service | Role | Config | Cost |
|---------|------|--------|------|
| SageMaker | LegalBERT endpoint | ml.m5.xlarge (CPU, 4 vCPUs, 16 GB) | ~$0.23/hr |
| SageMaker | Qwen 2.5 endpoint | ml.g5.xlarge (A10G 24 GB) | ~$1.41/hr |
| Lambda | Orchestration | Python 3.10, 512 MB, 5 min timeout | ~$0.0002/call |
| API Gateway | HTTPS route | HTTP API, POST /analyze | ~$0.001/req |
| DynamoDB | 7-day result cache | On-demand + TTL | ~$0.001/scan |
| S3 | Model artifact storage | Standard | ~$0.02 one-time |


---

## Troubleshooting

`ModuleNotFoundError: inference`
```bash
cp 5_inference.py inference.py
```

`FileNotFoundError: label_map.json`
```bash
mkdir -p data/processed
cp models/legalbert_classifier/label_map.json data/processed/label_map.json
```

CUDA out of memory — ensure the downloaded Qwen 2.5 model is the 4-bit quantised version from HuggingFace (loads in ~4 GB VRAM). If still OOM, use the `/analyze/fast` endpoint only — LegalBERT runs on CPU with zero VRAM requirement.

"No policy text found" — navigate to the actual privacy policy page, not the homepage. Text blocks under 80 characters are filtered out. JavaScript-rendered pages may need a moment to fully load before scanning.

CORS error in Chrome DevTools — confirm the server is running with `curl http://localhost:8000/health`. If using the `PRIVACY_EXT_ID` env var, check it matches the extension ID at `chrome://extensions`.

---

*PrivacyGuard v3.0 · AMD Slingshot · Confidential*
