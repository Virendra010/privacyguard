"""
local_server.py — Local inference server for PrivacyGuard Chrome Extension

Wraps your existing LegalBERT + Qwen2.5 pipeline in a FastAPI server
that speaks the exact same JSON API as the AWS Lambda function.
The Chrome extension needs zero changes except the API_URL in sidepanel.js.

Usage:
    pip install fastapi uvicorn
    python local_server.py

Then in chrome_extension/sidepanel.js change line 16 to:
    const API_URL = "http://localhost:8000/analyze";

The server loads models ONCE at startup and keeps them in RAM.
LegalBERT: ~500 MB RAM
Qwen2.5:   ~4 GB VRAM (GPU) or ~8 GB RAM (CPU, slow)
"""

import os
import re
import json
import time
import torch
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

# ─────────────────────────────────────────────────────────────
# CONFIG  — adjust paths if your folder structure differs
# ─────────────────────────────────────────────────────────────

LEGALBERT_DIR   = "./models/legalbert_classifier"
QWEN_DIR        = "./models/qwen_merged"
LABEL_MAP_FILE  = "./data/processed/label_map.json"

PORT            = 8000
MAX_LLM_CLAUSES = 10       # cap Qwen calls per request (cost/time control)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Simple in-memory cache (domain → report), survives for the server session
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("privacy-guard")

# ─────────────────────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────────────────────

def load_label_map():
    if os.path.exists(LABEL_MAP_FILE):
        with open(LABEL_MAP_FILE) as f:
            lmap = json.load(f)
        id2label = {int(k): v for k, v in lmap["id2label"].items()}
        risk_map = lmap["risk_map"]
    else:
        log.warning("label_map.json not found — using built-in defaults")
        id2label = {
            0: "First Party Collection/Use",
            1: "Third Party Sharing/Collection",
            2: "User Choice/Control",
            3: "Data Security",
            4: "Data Retention",
            5: "User Access, Edit and Deletion",
            6: "Policy Change",
            7: "Do Not Track",
            8: "International and Specific Audiences",
            9: "Other",
        }
        risk_map = {
            "Third Party Sharing/Collection"      : "high",
            "Data Retention"                      : "high",
            "First Party Collection/Use"          : "medium",
            "User Choice/Control"                 : "medium",
            "Data Security"                       : "medium",
            "Policy Change"                       : "medium",
            "User Access, Edit and Deletion"      : "medium",
            "Do Not Track"                        : "medium",
            "International and Specific Audiences": "low",
            "Other"                               : "low",
        }
    return id2label, risk_map

ID2LABEL, RISK_MAP = load_label_map()

THIRD_PARTY_KW = [
    "third party", "third-party", "third parties", "partners", "advertisers",
    "vendors", "affiliates", "service providers", "analytics providers",
]
FIRST_PARTY_KW = [
    "we collect", "we use", "we process", "we store",
    "we may collect", "we gather", "we receive", "we log",
]
label2id       = {v: k for k, v in ID2LABEL.items()}
FPCU_ID        = label2id.get("First Party Collection/Use", 0)
THIRD_PARTY_ID = label2id.get("Third Party Sharing/Collection", 1)

# ─────────────────────────────────────────────────────────────
# MODEL GLOBALS  (loaded once at startup)
# ─────────────────────────────────────────────────────────────

lb_tokenizer  = None
lb_model      = None
q_tokenizer   = None
q_model       = None

SYSTEM_PROMPT = (
    "You are a privacy law and data protection expert. "
    "Analyze the provided clause from a privacy policy or terms of service. "
    "Your response must include:\n"
    "1. Data Practice Type\n2. Risk Level with justification\n"
    "3. Data Details\n4. User Rights\n5. Key Excerpt\n"
    "Be concise and cite directly from the clause text."
)


def load_models(skip_llm: bool = False):
    global lb_tokenizer, lb_model, q_tokenizer, q_model

    log.info(f"Loading LegalBERT from {LEGALBERT_DIR} ...")
    lb_tokenizer = AutoTokenizer.from_pretrained(LEGALBERT_DIR)
    lb_model     = AutoModelForSequenceClassification.from_pretrained(LEGALBERT_DIR)
    lb_model.eval()
    lb_model.to(DEVICE)
    log.info(f"  LegalBERT loaded on {DEVICE}")

    if not skip_llm:
        qwen_path = QWEN_DIR if os.path.exists(QWEN_DIR) else "./models/qwen_lora"
        log.info(f"Loading Qwen2.5 from {qwen_path} ...")
        q_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        q_model     = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype       = torch.bfloat16,
            device_map        = "auto",
            trust_remote_code = True,
        )
        q_model.eval()
        log.info("  Qwen2.5 loaded")
    else:
        log.info("  Qwen2.5 skipped (--no-llm flag set)")

    log.info("✓ Models ready — server accepting requests")


# ─────────────────────────────────────────────────────────────
# PII SCRUBBING
# ─────────────────────────────────────────────────────────────

PII_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
    (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',                    '[PHONE]'),
    (r'\b\d{3}-\d{2}-\d{4}\b',                                 '[SSN]'),
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',           '[CARD]'),
    (r'utm_[a-z_]+=\S+',                                       ''),
    (r'[?&]fbclid=\S+',                                        ''),
]

def scrub_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────────────────────
# LEGALBERT CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_chunks(chunks: list[dict], batch_size: int = 32) -> list[dict]:
    texts = [c["text"] for c in chunks]
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = lb_tokenizer(
            batch,
            max_length    = 512,
            truncation    = True,
            padding       = True,
            return_tensors= "pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = lb_model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=-1)

        for j in range(len(batch)):
            pred = int(preds[j])
            conf = float(probs[j][pred])
            t    = batch[j].lower()

            # Post-processing: fix ambiguous FPCU vs Third Party predictions
            tp_score = sum(1 for kw in THIRD_PARTY_KW if kw in t)
            fp_score = sum(1 for kw in FIRST_PARTY_KW if kw in t)

            if pred in (FPCU_ID, THIRD_PARTY_ID) and conf < 0.65:
                pred = THIRD_PARTY_ID if tp_score > fp_score else FPCU_ID
            elif pred == FPCU_ID and tp_score >= 2 and fp_score == 0:
                pred = THIRD_PARTY_ID

            clause_type = ID2LABEL[pred]
            results.append({
                "clause_type": clause_type,
                "risk_level" : RISK_MAP.get(clause_type, "low"),
                "confidence" : round(conf, 4),
            })

    return results


# ─────────────────────────────────────────────────────────────
# QWEN EXPLANATION GENERATION
# ─────────────────────────────────────────────────────────────

def generate_explanation(clause_text: str, clause_type: str, risk_level: str,
                          max_new_tokens: int = 300) -> str:
    if q_model is None:
        return ""

    user_msg = (
        f"Analyze this clause (pre-classified as '{clause_type}', "
        f"risk: {risk_level}):\n\n\"{clause_text[:1500]}\""
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    prompt = q_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    enc = q_tokenizer(prompt, return_tensors="pt").to(
        next(q_model.parameters()).device
    )

    with torch.no_grad():
        out = q_model.generate(
            **enc,
            max_new_tokens     = max_new_tokens,
            do_sample          = False,
            repetition_penalty = 1.1,
            eos_token_id       = q_tokenizer.eos_token_id,
            pad_token_id       = q_tokenizer.eos_token_id,
        )

    new_ids = out[0][enc["input_ids"].shape[1]:]
    return q_tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# FULL ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────

def run_analysis(chunks: list[dict], domain: str) -> dict:
    # 1. Scrub PII
    clean_chunks = [{"id": c.get("id", i), "text": scrub_pii(c["text"])}
                    for i, c in enumerate(chunks)]

    # 2. Classify with LegalBERT
    log.info(f"Classifying {len(clean_chunks)} chunks for {domain}")
    classifications = classify_chunks(clean_chunks)

    annotated = []
    for chunk, cls in zip(clean_chunks, classifications):
        annotated.append({
            "chunk_id"   : chunk["id"],
            "text"       : chunk["text"],
            "clause_type": cls["clause_type"],
            "risk_level" : cls["risk_level"],
            "confidence" : cls["confidence"],
            "explanation": "",
        })

    # 3. Filter risky, sort high-first
    risky = [c for c in annotated if c["risk_level"] in ("high", "medium")]
    low   = [c for c in annotated if c["risk_level"] == "low"]
    risky.sort(key=lambda c: (c["risk_level"] != "high", -c["confidence"]))

    log.info(f"  {len(risky)} risky chunks ({sum(1 for c in risky if c['risk_level']=='high')} high)")

    # 4. Qwen explanations (capped)
    for_llm = risky[:MAX_LLM_CLAUSES]
    for chunk in for_llm:
        try:
            log.info(f"  Generating explanation for chunk {chunk['chunk_id']} ...")
            chunk["explanation"] = generate_explanation(
                chunk["text"], chunk["clause_type"], chunk["risk_level"]
            )
        except Exception as e:
            log.error(f"  Qwen failed for chunk {chunk['chunk_id']}: {e}")
            chunk["explanation"] = f"[Explanation unavailable: {str(e)[:100]}]"

    # 5. Build report
    def fmt(c):
        return {
            "chunk_id"   : c["chunk_id"],
            "clause_type": c["clause_type"],
            "risk_level" : c["risk_level"],
            "confidence" : c["confidence"],
            "clause_text": c["text"],
            "explanation": c["explanation"],
        }

    high_findings   = [fmt(c) for c in risky if c["risk_level"] == "high"]
    medium_findings = [fmt(c) for c in risky if c["risk_level"] == "medium"]

    return {
        "domain"      : domain,
        "cache_status": "live_scan",
        "scanned_at"  : datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_clauses"      : len(annotated),
            "high_risk_clauses"  : len(high_findings),
            "medium_risk_clauses": len(medium_findings),
            "low_risk_clauses"   : len(low),
            "overall_risk"       : "HIGH" if high_findings else "MEDIUM" if medium_findings else "LOW",
            "clauses_explained"  : len(for_llm),
        },
        "high_risk_findings"  : high_findings,
        "medium_risk_findings": medium_findings,
        "low_risk_findings"   : [fmt(c) for c in low],
    }


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="PrivacyGuard Local Server")

# Allow Chrome extension to call us (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Chrome extension origin is chrome-extension://...
    allow_methods     = ["POST", "OPTIONS"],
    allow_headers     = ["Content-Type"],
)


@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "legalbert"   : lb_model is not None,
        "qwen"        : q_model is not None,
        "device"      : DEVICE,
        "cached_domains": list(_cache.keys()),
    }


@app.post("/analyze")
async def analyze(request: Request):
    try:
        body   = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    url    = body.get("url", "")
    chunks = body.get("chunks", [])

    if not url or not chunks:
        return JSONResponse(
            {"error": "Missing 'url' or 'chunks'"}, status_code=400
        )

    # Extract domain for cache key
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
    except Exception:
        domain = url.lower().strip()

    log.info(f"Request: {domain} — {len(chunks)} chunks")

    # ── In-memory cache check ─────────────────────────────────
    with _cache_lock:
        if domain in _cache:
            log.info(f"  Cache HIT for {domain}")
            cached = dict(_cache[domain])
            cached["cache_status"] = "cached"
            return JSONResponse(cached)

    # ── Run analysis ──────────────────────────────────────────
    try:
        report = run_analysis(chunks, domain)
        with _cache_lock:
            _cache[domain] = report
        return JSONResponse(report)

    except Exception as e:
        log.error(f"Analysis failed for {domain}: {e}", exc_info=True)
        return JSONResponse({"error": str(e), "domain": domain}, status_code=500)


@app.delete("/cache")
def clear_cache():
    """Clear the in-memory cache (useful for testing)."""
    with _cache_lock:
        _cache.clear()
    return {"cleared": True}


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PrivacyGuard Local Inference Server")
    parser.add_argument("--port",   type=int, default=PORT,  help="Port to listen on (default 8000)")
    parser.add_argument("--no-llm", action="store_true",     help="Skip loading Qwen — classification only (much faster)")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════╗
║        PrivacyGuard — Local Inference Server     ║
╚══════════════════════════════════════════════════╝
""")

    load_models(skip_llm=args.no_llm)

    print(f"""
Server ready!

In chrome_extension/sidepanel.js, change line 16 to:
    const API_URL = "http://localhost:{args.port}/analyze";

Health check: http://localhost:{args.port}/health
Clear cache:  DELETE http://localhost:{args.port}/cache
""")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
