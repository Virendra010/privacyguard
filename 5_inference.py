"""
Step 5: Inference Pipeline  (Qwen2.5 version)

Input:  A raw privacy policy or terms of service — either:
          --text  path/to/policy.txt
          --html  path/to/policy.html     (works with ToS;DR HTML files)
          --tosdr                         (runs on all data/tosdr/ HTML files)
          --demo                          (built-in example)

Pipeline:
  1. Extract and chunk text
  2. LegalBERT  → classify each chunk (clause type + risk level)
  3. Filter to HIGH/MEDIUM risk chunks
  4. Qwen2.5    → generate cited risk explanation per chunk
  5. Output structured JSON report

Usage:
  python 5_inference.py --demo
  python 5_inference.py --text my_policy.txt
  python 5_inference.py --html data/tosdr/aetna/Privacy\\ Policy.html
  python 5_inference.py --tosdr              # batch all ToS;DR HTMLs
  python 5_inference.py --demo --no-explain  # classification only (fast)
"""

import os, json, re, argparse
import torch
from pathlib import Path
from bs4 import BeautifulSoup
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

LEGALBERT_DIR  = "./models/legalbert_classifier"
QWEN_DIR       = "./models/qwen_merged"     # use qwen_lora if not merged yet
LABEL_MAP_FILE = "./data/processed/label_map.json"
TOSDR_ROOT     = "./data/tosdr"

RISK_THRESHOLD = {"high", "medium"}
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

with open(LABEL_MAP_FILE) as f:
    lmap = json.load(f)
ID2LABEL = {int(k): v for k, v in lmap["id2label"].items()}
RISK_MAP  = lmap["risk_map"]


# ─────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_text_from_html(html_path: str) -> str:
    """
    Extract clean body text from a policy HTML file.
    Removes nav/header/footer/scripts, collects paragraphs and list items.
    """
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup(["script","style","nav","header","footer","noscript","aside","form"]):
        tag.decompose()

    chunks = []
    for tag in soup.find_all(["p", "li", "h2", "h3"]):
        text = re.sub(r"\s+", " ", tag.get_text()).strip()
        if len(text) > 40:
            chunks.append(text)

    return "\n\n".join(chunks)


def chunk_text(text: str, max_words: int = 120) -> list[dict]:
    """
    Split policy text into chunks by paragraph → sentence if paragraph too long.
    Returns: [{ chunk_id, text }]
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []

    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            if len(para) > 40:
                chunks.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            window = []
            for sent in sentences:
                window.append(sent)
                if len(" ".join(window).split()) >= max_words:
                    text_chunk = " ".join(window)
                    if len(text_chunk) > 40:
                        chunks.append(text_chunk)
                    window = []
            if window:
                tail = " ".join(window)
                if len(tail) > 40:
                    chunks.append(tail)

    return [{"chunk_id": i, "text": c} for i, c in enumerate(chunks)]


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_legalbert():
    print("[Inference] Loading LegalBERT classifier...")
    tokenizer = AutoTokenizer.from_pretrained(LEGALBERT_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(LEGALBERT_DIR)
    model.eval()
    return tokenizer, model.to(DEVICE)


def load_qwen():
    """
    Load the fine-tuned Qwen2.5 model for explanation generation.
    Tries merged model first (qwen_merged/), falls back to adapter model (qwen_lora/).
    Qwen2.5-1.5B fits in fp16 on ~4 GB VRAM — no quantization needed at inference.
    """
    model_path = QWEN_DIR
    if not os.path.exists(model_path):
        fallback = "./models/qwen_lora"
        print(f"  [WARN] {QWEN_DIR} not found, trying {fallback}")
        model_path = fallback

    print(f"[Inference] Loading Qwen2.5 from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
    )
    model.eval()
    return tokenizer, model


# ─────────────────────────────────────────────────────────────
# LEGALBERT CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_chunks(chunks: list[dict], tokenizer, model, batch_size=32) -> list[dict]:
    """Add clause_type, risk_level, confidence to each chunk."""
    texts = [c["text"] for c in chunks]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            max_length    = 512,
            truncation    = True,
            padding       = True,
            return_tensors= "pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)
            preds  = torch.argmax(probs, dim=-1)
            confs  = probs.max(dim=-1).values

        for j, (pred, conf) in enumerate(zip(preds.cpu().tolist(), confs.cpu().tolist())):
            clause_type = ID2LABEL[pred]
            chunks[i + j]["clause_type"] = clause_type
            chunks[i + j]["risk_level"]  = RISK_MAP.get(clause_type, "low")
            chunks[i + j]["confidence"]  = round(conf, 4)

    return chunks


# ─────────────────────────────────────────────────────────────
# QWEN2.5 EXPLANATION GENERATION
#
# We use tokenizer.apply_chat_template() — the same method used
# during training — to format the prompt. This guarantees the
# prompt format exactly matches what the model was fine-tuned on.
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a privacy law and data protection expert. "
    "Analyze the provided clause from a privacy policy or terms of service. "
    "Your response must include:\n"
    "1. Data Practice Type – what kind of data practice this clause represents.\n"
    "2. Risk Level – HIGH, MEDIUM, or LOW, with a one-sentence justification.\n"
    "3. Data Details – what data is involved, and any third parties mentioned.\n"
    "4. User Rights – what control or rights the user has (or lacks).\n"
    "5. Key Excerpt – quote the specific phrase that most drives the risk assessment.\n"
    "Be concise and cite directly from the clause text."
)


def build_qwen_prompt(tokenizer, clause_text: str, clause_type: str, risk_level: str) -> str:
    """
    Build the inference prompt using Qwen's native apply_chat_template.
    add_generation_prompt=True adds '<|im_start|>assistant\n' at the end
    to tell the model to start generating the response.
    """
    user_msg = (
        f"Analyze this clause (pre-classified as '{clause_type}', "
        f"risk: {risk_level}):\n\n\"{clause_text}\""
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,   # appends <|im_start|>assistant\n
    )


def generate_explanations(
    risky_chunks:   list[dict],
    tokenizer,
    model,
    max_new_tokens: int = 300,
) -> list[dict]:
    for chunk in risky_chunks:
        prompt = build_qwen_prompt(
            tokenizer,
            chunk["text"],
            chunk["clause_type"],
            chunk["risk_level"],
        )
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens     = max_new_tokens,
                do_sample          = False,          # deterministic for analysis
                repetition_penalty = 1.1,
                eos_token_id       = tokenizer.eos_token_id,
                pad_token_id       = tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        new_ids     = out[0][enc["input_ids"].shape[1]:]
        explanation = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        chunk["explanation"] = explanation

    return risky_chunks


# ─────────────────────────────────────────────────────────────
# REPORT BUILDER
# ─────────────────────────────────────────────────────────────

def build_report(source_name: str, chunks: list[dict]) -> dict:
    high   = [c for c in chunks if c["risk_level"] == "high"]
    medium = [c for c in chunks if c["risk_level"] == "medium"]
    low    = [c for c in chunks if c["risk_level"] == "low"]

    def fmt(c):
        return {
            "clause_type" : c["clause_type"],
            "risk_level"  : c["risk_level"],
            "confidence"  : c["confidence"],
            "clause_text" : c["text"],
            "explanation" : c.get("explanation", ""),
        }

    return {
        "source"  : source_name,
        "summary" : {
            "total_clauses"       : len(chunks),
            "high_risk_clauses"   : len(high),
            "medium_risk_clauses" : len(medium),
            "low_risk_clauses"    : len(low),
            "overall_risk"        : "HIGH" if high else "MEDIUM" if medium else "LOW",
        },
        "high_risk_findings"   : [fmt(c) for c in high],
        "medium_risk_findings" : [fmt(c) for c in medium],
        "low_risk_findings"    : [fmt(c) for c in low],
    }


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def analyze(text: str, source_name: str = "policy", explain: bool = True) -> dict:
    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunks extracted")

    # Step 1: LegalBERT classification
    lb_tok, lb_model = load_legalbert()
    chunks = classify_chunks(chunks, lb_tok, lb_model)
    del lb_tok, lb_model
    torch.cuda.empty_cache()

    # Step 2: Filter risky chunks
    risky = [c for c in chunks if c["risk_level"] in RISK_THRESHOLD]
    print(f"  {len(risky)} risky chunks flagged (high/medium)")

    # Step 3: Qwen2.5 explanations
    if explain and risky:
        q_tok, q_model = load_qwen()
        risky = generate_explanations(risky, q_tok, q_model)
        risky_by_id = {c["chunk_id"]: c for c in risky}
        for c in chunks:
            if c["chunk_id"] in risky_by_id:
                c["explanation"] = risky_by_id[c["chunk_id"]].get("explanation", "")
        del q_tok, q_model
        torch.cuda.empty_cache()

    return build_report(source_name, chunks)


def print_report(report: dict):
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"REPORT: {report['source']}")
    print(f"{'='*60}")
    print(f"Clauses analyzed : {s['total_clauses']}")
    print(f"High risk        : {s['high_risk_clauses']}")
    print(f"Medium risk      : {s['medium_risk_clauses']}")
    print(f"Overall risk     : {s['overall_risk']}")

    for level, key in [("HIGH", "high_risk_findings"), ("MEDIUM", "medium_risk_findings")]:
        findings = report[key]
        if findings:
            print(f"\n── {level} RISK FINDINGS ──")
        for i, f in enumerate(findings, 1):
            print(f"\n  [{i}] {f['clause_type']}  (conf: {f['confidence']:.0%})")
            print(f"  Clause : {f['clause_text'][:150]}...")
            if f.get("explanation"):
                print(f"  Analysis:\n{f['explanation']}")


# ─────────────────────────────────────────────────────────────
# DEMO TEXT
# ─────────────────────────────────────────────────────────────

DEMO = """
We collect information you provide directly to us, such as when you create an account,
make a purchase, or contact us for support. This includes your name, email address,
phone number, and payment information.

We may share your personal information with third-party advertising and analytics partners.
These partners may use your information to send promotional materials. You cannot opt out
of this sharing if you wish to continue using our service.

We retain your personal data indefinitely, even after you close your account,
to comply with legal obligations and for analytics purposes.

We use industry-standard encryption during transmission. However, no method of
transmission over the Internet is 100% secure.

By continuing to use our service after changes to this policy are posted, you agree
to the updated terms. We may update these terms at any time without prior notice.
"""


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy Policy Risk Analyzer")
    parser.add_argument("--text",       help="Path to a plain text policy file")
    parser.add_argument("--html",       help="Path to an HTML policy file")
    parser.add_argument("--tosdr",      action="store_true",
                        help="Batch-analyze all HTML files in data/tosdr/")
    parser.add_argument("--demo",       action="store_true",
                        help="Run on built-in demo text")
    parser.add_argument("--no-explain", action="store_true",
                        help="Skip Qwen — classification only (much faster)")
    args = parser.parse_args()

    explain = not args.no_explain
    reports = []

    if args.demo:
        r = analyze(DEMO, "demo_policy", explain)
        reports.append(r)

    elif args.text:
        with open(args.text) as f:
            text = f.read()
        r = analyze(text, Path(args.text).stem, explain)
        reports.append(r)

    elif args.html:
        text = extract_text_from_html(args.html)
        r    = analyze(text, Path(args.html).stem, explain)
        reports.append(r)

    elif args.tosdr:
        html_files = sorted(Path(TOSDR_ROOT).rglob("*.html"))
        print(f"Found {len(html_files)} ToS;DR HTML files")
        for hp in html_files:
            print(f"\nAnalyzing: {hp}")
            text = extract_text_from_html(str(hp))
            name = f"{hp.parent.name}/{hp.stem}"
            r    = analyze(text, name, explain)
            reports.append(r)

    else:
        parser.print_help()
        exit(1)

    # Print reports and save JSON
    for r in reports:
        print_report(r)

    out_path = "./risk_report.json"
    with open(out_path, "w") as f:
        json.dump(reports if len(reports) > 1 else reports[0], f, indent=2)
    print(f"\n[Report saved to {out_path}]")
