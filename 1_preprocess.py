"""
Step 1 (Final): Preprocess OPP-115 — correctly.

ROOT CAUSE OF PREVIOUS ERRORS:
  pd.read_csv(...) WITHOUT header=None treats row 0 as column names.
  OPP-115 has NO header row. So row 0 data values became column names,
  and the inferring fallback picked col 0 (annotation_id like 'C866') as "text"
  and col 1 (batch name like 'test_category_labeling_highlight') as "data_practice".
  Nothing matched OPP115_LABELS → 0 rows → crash.

ACTUAL COLUMN LAYOUT (from manual.txt, confirmed by inspection):
  col 0: annotation_id   e.g. 'C866' (consolidated) or '10' (singlet)
  col 1: batch_id        e.g. 'test_category_labeling_highlight'
  col 2: annotator_id    e.g. '84'
  col 3: policy_id       e.g. '3621'   ← maps to sanitized_policies filename
  col 4: segment_id      e.g. '0'      ← 0-indexed segment position
  col 5: category        e.g. 'Other', 'First Party Collection/Use'  ← LABEL
  col 6: attributes_json JSON blob with selectedText spans            ← TEXT SOURCE
  col 7: date            e.g. '4/14/15'
  col 8: url             e.g. 'https://www.reddit.com/help/privacypolicy'

TEXT STRATEGY (two-tier):
  Primary  → sanitized_policies/<policy_id>_*.txt
             Full segment text (col 4 = segment index in ||| separated file)
             Better training signal; longer context for the model
  Fallback → longest selectedText from attributes_json (col 6)
             Used when sanitized policy file not found

ERRANT SPAN FILTERING:
  documentation/errant_span_indexes/*.csv list annotation IDs with buggy spans.
  We drop those rows from consolidation data (they have wrong character indexes).
  Note: these are numeric IDs; IDs starting with 'C' are consolidated and
  are NOT in the errant list — only singlet numeric IDs are affected.

DIRECTORY STRUCTURE (case-sensitive, use exactly as shown):
  ./data/OPP-115/
    annotations/                      ← per-annotator CSVs (not used here)
    consolidation/
      threshold-0.5-overlap-similarity/   ← PRIMARY source (115 CSVs)
      threshold-0.75-overlap-similarity/  ← unused
      threshold-1.0-overlap-similarity/   ← unused
    sanitized_policies/                   ← full segment texts
    documentation/
      errant_span_indexes/               ← errant_span_indexes_block_*.csv

CHANGE v2:
  MAX_CAP raised from 2000 → 3500.
  Previously FPCU (4058 raw) was capped at ~1473 training samples, discarding
  2500+ examples covering diverse phrasings. This caused FPCU recall to collapse
  (precision=0.71, recall=0.44 on test). Raising the cap keeps more lexical
  diversity while Focal Loss + WeightedRandomSampler prevent class dominance.

OUTPUTS:
  ./data/processed/legalbert_train.json
  ./data/processed/legalbert_val.json
  ./data/processed/legalbert_test.json   (real samples only, held-out)
  ./data/processed/llama3_train.json
  ./data/processed/llama3_val.json
  ./data/processed/label_map.json        (id2label, class_weights, etc.)
"""

import os
import json
import re
import glob
import random
import warnings
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

os.makedirs("./data/processed", exist_ok=True)

OPP115_BASE = "./data/OPP-115"   # EXACT directory name — capital, hyphenated

# ─────────────────────────────────────────────────────────────
# LABEL / RISK DEFINITIONS
# ─────────────────────────────────────────────────────────────

OPP115_LABELS = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "Other",
    "User Choice/Control",
    "Data Security",
    "International and Specific Audiences",
    "User Access, Edit and Deletion",
    "Policy Change",
    "Data Retention",
    "Do Not Track",
]

ACTUAL_COUNTS = {     # from your inspect run
    "First Party Collection/Use"            : 8935,
    "Third Party Sharing/Collection"        : 5221,
    "Other"                                 : 3548,
    "User Choice/Control"                   : 1789,
    "Data Security"                         : 1008,
    "International and Specific Audiences"  : 939,
    "User Access, Edit and Deletion"        : 746,
    "Policy Change"                         : 548,
    "Data Retention"                        : 370,
    "Do Not Track"                          : 90,
}

LABEL2ID   = {l: i for i, l in enumerate(OPP115_LABELS)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(OPP115_LABELS)

RISK_MAP = {
    "First Party Collection/Use"            : "medium",
    "Third Party Sharing/Collection"        : "high",
    "Other"                                 : "low",
    "User Choice/Control"                   : "medium",
    "Data Security"                         : "medium",
    "International and Specific Audiences"  : "low",
    "User Access, Edit and Deletion"        : "medium",
    "Policy Change"                         : "medium",
    "Data Retention"                        : "high",
    "Do Not Track"                          : "medium",
}


# ─────────────────────────────────────────────────────────────
# REBALANCING TARGETS  (sqrt-frequency scaling)
# ─────────────────────────────────────────────────────────────

TOTAL_TARGET = 11000  # bumped up slightly to accommodate higher cap
MAX_CAP      = 3000   # ← CHANGED from 2000: keeps more FPCU/Third Party diversity
MIN_FLOOR    = 300

def compute_targets(counts: dict) -> dict:
    sqrt_c     = {k: np.sqrt(v) for k, v in counts.items()}
    total      = sum(sqrt_c.values())
    raw        = {k: int(v / total * TOTAL_TARGET) for k, v in sqrt_c.items()}
    return {k: max(MIN_FLOOR, min(MAX_CAP, v)) for k, v in raw.items()}

REBALANCE_TARGETS = compute_targets(ACTUAL_COUNTS)


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD ERRANT ANNOTATION IDs
# ─────────────────────────────────────────────────────────────

def load_errant_ids(base_path: str) -> set:
    errant_dir = os.path.join(base_path, "documentation", "errant_span_indexes")
    errant_ids = set()

    if not os.path.exists(errant_dir):
        print(f"  [WARN] errant_span_indexes dir not found at {errant_dir} — skipping filter")
        return errant_ids

    files = glob.glob(os.path.join(errant_dir, "*.csv"))
    if not files:
        print(f"  [WARN] No errant span CSV files found in {errant_dir}")
        return errant_ids

    for fpath in files:
        try:
            df = pd.read_csv(fpath, header=None, dtype=str, on_bad_lines="skip")
            ids = df.iloc[:, 0].dropna().str.strip().tolist()
            errant_ids.update(ids)
        except Exception as e:
            print(f"  [WARN] Could not read {os.path.basename(fpath)}: {e}")

    print(f"[Errant] Loaded {len(errant_ids)} errant annotation IDs to exclude")
    return errant_ids


# ─────────────────────────────────────────────────────────────
# STEP 2: LOAD SANITIZED POLICIES (for full segment text)
# ─────────────────────────────────────────────────────────────

def load_sanitized_policies(base_path: str) -> dict:
    san_dir = os.path.join(base_path, "sanitized_policies")
    policies = {}

    if not os.path.exists(san_dir):
        print(f"  [WARN] sanitized_policies/ not found — will fall back to selectedText only")
        return policies

    txt_files = glob.glob(os.path.join(san_dir, "*.txt"))
    print(f"[Sanitized] Loading {len(txt_files)} sanitized policy files...")

    for fpath in txt_files:
        fname = os.path.basename(fpath)
        match = re.match(r"^(\d+)_", fname)
        if not match:
            continue
        policy_id = match.group(1)

        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            segments = raw.split("|||")
            clean_segs = []
            for seg in segments:
                seg = re.sub(r"<[^>]+>", " ", seg)
                seg = re.sub(r"\s+", " ", seg).strip()
                clean_segs.append(seg)
            policies[policy_id] = clean_segs
        except Exception as e:
            print(f"  [WARN] Could not read {fname}: {e}")

    print(f"[Sanitized] Loaded {len(policies)} policies")
    return policies


# ─────────────────────────────────────────────────────────────
# STEP 3: EXTRACT TEXT FROM ATTRIBUTES JSON (fallback)
# ─────────────────────────────────────────────────────────────

def extract_selected_text(attributes_json: str) -> str:
    if not isinstance(attributes_json, str) or not attributes_json.strip().startswith("{"):
        return ""
    try:
        blob = json.loads(attributes_json)
        candidates = []
        for key, val in blob.items():
            if isinstance(val, dict):
                st = val.get("selectedText", "")
                if (isinstance(st, str)
                        and st.lower() not in ("null", "not selected", "not-selected", "")
                        and len(st) > 10):
                    candidates.append(st.strip())
        if not candidates:
            return ""
        return max(candidates, key=len)
    except (json.JSONDecodeError, AttributeError):
        return ""


# ─────────────────────────────────────────────────────────────
# STEP 4: LOAD CONSOLIDATION CSV FILES
# ─────────────────────────────────────────────────────────────

def load_consolidation(base_path: str,
                       sanitized: dict,
                       errant_ids: set) -> pd.DataFrame:
    consol_dir = os.path.join(
        base_path, "consolidation", "threshold-0.5-overlap-similarity"
    )

    if not os.path.exists(consol_dir):
        raise FileNotFoundError(
            f"\n[ERROR] Directory not found: {consol_dir}\n"
            f"Make sure your data folder is: {base_path}\n"
        )

    csv_files = glob.glob(os.path.join(consol_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {consol_dir}")

    print(f"\n[OPP-115] Found {len(csv_files)} consolidation CSV files")

    frames = []
    for fpath in sorted(csv_files):
        try:
            df = pd.read_csv(
                fpath,
                header       = None,   # ← CRITICAL: no header row in OPP-115
                dtype        = str,
                encoding     = "utf-8",
                on_bad_lines = "skip",
                quotechar    = '"',
            )
            if len(df.columns) >= 7:
                frames.append(df)
            else:
                print(f"  [WARN] Skipping {os.path.basename(fpath)}: only {len(df.columns)} cols")
        except Exception as e:
            print(f"  [WARN] Failed to read {os.path.basename(fpath)}: {e}")

    if not frames:
        raise RuntimeError("All CSV files failed to load. Check file encoding and format.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[OPP-115] Raw rows loaded: {len(combined)}")

    combined = combined.rename(columns={
        0: "annotation_id",
        1: "batch_id",
        2: "annotator_id",
        3: "policy_id",
        4: "segment_id",
        5: "category",
        6: "attributes_json",
        7: "date",
        8: "url",
    })

    before_errant = len(combined)
    combined = combined[~combined["annotation_id"].isin(errant_ids)]
    print(f"[OPP-115] After errant filter: {len(combined)} rows "
          f"(removed {before_errant - len(combined)})")

    combined["category"] = combined["category"].astype(str).str.strip()
    combined = combined[combined["category"].isin(OPP115_LABELS)]
    print(f"[OPP-115] After category filter: {len(combined)} rows")

    print("[OPP-115] Extracting segment text (sanitized_policies primary, JSON fallback)...")

    def get_text(row) -> str:
        pid  = str(row["policy_id"]).strip() if pd.notna(row["policy_id"]) else ""
        sid  = str(row["segment_id"]).strip() if pd.notna(row["segment_id"]) else ""
        json_blob = str(row["attributes_json"]) if pd.notna(row["attributes_json"]) else ""

        if pid in sanitized and sid.isdigit():
            segs = sanitized[pid]
            sidx = int(sid)
            if 0 <= sidx < len(segs) and len(segs[sidx]) > 30:
                return segs[sidx]

        return extract_selected_text(json_blob)

    combined["text"] = combined.apply(get_text, axis=1)

    combined = combined[combined["text"].str.len() > 30]
    combined = combined.drop_duplicates(subset=["text", "category"])
    combined = combined.rename(columns={"category": "data_practice"})
    combined = combined[["text", "data_practice", "annotation_id", "policy_id"]].reset_index(drop=True)

    print(f"\n[OPP-115] Final clean rows: {len(combined)}")
    print("[OPP-115] Category distribution:")
    for cat, cnt in combined["data_practice"].value_counts().items():
        print(f"  {cnt:>6}  {cat}")

    return combined


# ─────────────────────────────────────────────────────────────
# STEP 5: AUGMENTATION (for rare class oversampling)
# ─────────────────────────────────────────────────────────────

LEGAL_PARAPHRASES = [
    ("we may",                  "we can"),
    ("we will",                 "we shall"),
    ("in order to",             "to"),
    ("with respect to",         "regarding"),
    ("in the event",            "if"),
    ("prior to",                "before"),
    ("subsequent to",           "after"),
    ("in accordance with",      "according to"),
    ("at our discretion",       "as we see fit"),
    ("at our sole discretion",  "as we determine"),
    ("you acknowledge",         "you agree"),
    ("you understand",          "you acknowledge"),
    ("personal information",    "personal data"),
    ("personal data",           "personal information"),
    ("third parties",           "third-party companies"),
    ("third-party companies",   "third parties"),
    ("opt out",                 "opt-out"),
    ("opt-out",                 "opt out"),
    ("as described in",         "as outlined in"),
    ("as set forth in",         "as described in"),
    ("we collect",              "we gather"),
    ("we gather",               "we collect"),
    ("we store",                "we retain"),
    ("we retain",               "we store"),
    ("may be shared",           "can be shared"),
    ("may be used",             "can be used"),
]


def augment_text(text: str) -> str:
    t      = text
    swaps  = random.sample(LEGAL_PARAPHRASES,
                           k=min(len(LEGAL_PARAPHRASES), random.randint(2, 4)))
    changed = False
    for orig, repl in swaps:
        pat = re.compile(re.escape(orig), re.IGNORECASE)
        if pat.search(t):
            t       = pat.sub(repl, t, count=1)
            changed = True
    if not changed:
        t = t.replace("information", "data", 1) if "information" in t else t + " "
    return t


def oversample_to_target(records: list, target: int) -> list:
    if len(records) >= target:
        return records[:target]
    result   = list(records)
    pool     = list(records)
    attempts = 0
    while len(result) < target and attempts < target * 10:
        attempts += 1
        source  = random.choice(pool)
        new_rec = {**source, "text": augment_text(source["text"]), "augmented": True}
        result.append(new_rec)
    return result[:target]


# ─────────────────────────────────────────────────────────────
# STEP 6: REBALANCE
# ─────────────────────────────────────────────────────────────

def build_rebalanced_records(df: pd.DataFrame):
    stats       = {"before": {}, "after": {}, "augmented_added": {}}
    all_records = []

    for label in OPP115_LABELS:
        subset = df[df["data_practice"] == label].copy()
        recs   = subset.to_dict(orient="records")
        for r in recs:
            r["label_id"]   = LABEL2ID[label]
            r["risk_level"] = RISK_MAP[label]
            r["augmented"]  = False

        target  = REBALANCE_TARGETS[label]
        n_orig  = len(recs)
        stats["before"][label] = n_orig

        if n_orig == 0:
            print(f"  [WARN] Class '{label}' has 0 samples in loaded data!")
            stats["after"][label]          = 0
            stats["augmented_added"][label] = 0
            continue

        if n_orig >= target:
            sampled = random.sample(recs, target)
            n_aug   = 0
        else:
            sampled = oversample_to_target(recs, target)
            n_aug   = sum(1 for r in sampled if r.get("augmented"))

        stats["after"][label]           = len(sampled)
        stats["augmented_added"][label] = n_aug
        all_records.extend(sampled)

    random.shuffle(all_records)
    return all_records, stats


# ─────────────────────────────────────────────────────────────
# STEP 7: STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────

def stratified_split(records: list, val_ratio=0.12, test_ratio=0.08):
    """
    Augmented samples go ONLY into train.
    Val and test use ONLY real (non-augmented) samples.
    """
    real = [r for r in records if not r.get("augmented")]
    aug  = [r for r in records if r.get("augmented")]

    real_labels = [r["label_id"] for r in real]

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    tv_idx, test_idx = next(sss1.split(real, real_labels))

    test     = [real[i] for i in test_idx]
    tv_real  = [real[i] for i in tv_idx]
    tv_labels = [r["label_id"] for r in tv_real]

    val_frac = val_ratio / (1 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    tr_idx, v_idx = next(sss2.split(tv_real, tv_labels))

    train = [tv_real[i] for i in tr_idx] + aug   # augmented only in train
    val   = [tv_real[i] for i in v_idx]

    random.shuffle(train)
    return train, val, test


def print_split_stats(name: str, split: list):
    counts = Counter(r["data_practice"] for r in split)
    real   = sum(1 for r in split if not r.get("augmented"))
    aug    = sum(1 for r in split if r.get("augmented"))
    print(f"\n  {name} ({len(split)} total | {real} real | {aug} aug):")
    for label in OPP115_LABELS:
        n = counts.get(label, 0)
        print(f"    {n:>4}  {label}")


# ─────────────────────────────────────────────────────────────
# STEP 8: LLAMA 3 INSTRUCTION FORMAT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a privacy and legal expert. "
    "Analyze the given clause from a privacy policy or terms of service. "
    "Identify:\n"
    "1. The type of data practice or permission it represents.\n"
    "2. The risk level (high / medium / low) and why.\n"
    "3. What data is collected or shared and with whom.\n"
    "4. Any user rights or lack thereof mentioned.\n"
    "Be concise, cite the specific part of the clause that drove your assessment."
)

RISK_REASONS = {
    "Third Party Sharing/Collection"        : "user data is shared with external parties, reducing user control and potentially exposing data to entities with different privacy standards.",
    "Data Retention"                        : "data is retained for an extended or indefinite period, increasing exposure risk if the service is breached.",
    "First Party Collection/Use"            : "the service collects data directly from the user; risk depends on the breadth of collection and stated purpose.",
    "User Choice/Control"                   : "limited opt-out or preference controls restrict the user's ability to manage their data.",
    "Data Security"                         : "the security measures described (or absent) directly affect the risk of a data breach.",
    "Policy Change"                         : "the company can change terms without explicit user consent, affecting ongoing data use.",
    "User Access, Edit and Deletion"        : "restricting access and deletion rights limits user autonomy over personal data.",
    "Do Not Track"                          : "the service's handling of DNT signals determines whether user tracking preferences are respected.",
    "International and Specific Audiences"  : "data handling rules may vary across jurisdictions or specific user groups, affecting applicable rights.",
    "Other"                                 : "this clause has general implications for data handling practices.",
}


def record_to_llama3(rec: dict) -> dict:
    label   = rec["data_practice"]
    risk    = rec["risk_level"].upper()
    text    = rec["text"]
    excerpt = text[:200] + ("..." if len(text) > 200 else "")
    reason  = RISK_REASONS.get(label, "this clause affects user data handling.")

    user_msg = f"Analyze this privacy policy clause:\n\n\"{text}\""
    asst_msg = (
        f"**Clause Type:** {label}\n"
        f"**Risk Level:** {risk}\n\n"
        f"**Assessment:**\n"
        f"This clause falls under *{label}*. "
        f"Risk is **{risk}** because {reason}\n\n"
        f"**Key Excerpt:** \"{excerpt}\""
    )
    full = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{asst_msg}<|eot_id|>"
    )
    return {"text": full}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("OPP-115 Preprocessing Pipeline  [v2 — higher undersampling cap]")
    print("=" * 70)
    print(f"OPP-115 base path : {OPP115_BASE}")
    print(f"MAX_CAP           : {MAX_CAP}  (was 2000 — raised to keep FPCU diversity)")
    print(f"MIN_FLOOR         : {MIN_FLOOR}")
    print(f"TOTAL_TARGET      : {TOTAL_TARGET}")

    if not os.path.exists(OPP115_BASE):
        raise FileNotFoundError(
            f"\n[ERROR] '{OPP115_BASE}' not found.\n"
            "Place the OPP-115 folder at ./data/OPP-115\n"
            "Exact name matters: 'OPP-115' not 'opp115' or 'OPP115'"
        )

    errant_ids = load_errant_ids(OPP115_BASE)
    sanitized  = load_sanitized_policies(OPP115_BASE)
    df         = load_consolidation(OPP115_BASE, sanitized, errant_ids)

    if len(df) == 0:
        raise RuntimeError(
            "\n[ERROR] 0 rows after loading. Debug steps:\n"
            "1. Check that OPP-115/consolidation/threshold-0.5-overlap-similarity/*.csv exist\n"
            "2. Run: import pandas as pd\n"
            "        df = pd.read_csv('...303_reddit_com.csv', header=None, dtype=str)\n"
            "        print(df.iloc[:3, :9])\n"
            "   Column 5 should show category names."
        )

    print("\n" + "=" * 70)
    print("Rebalancing plan (original → target):")
    print("=" * 70)
    for label in OPP115_LABELS:
        orig   = ACTUAL_COUNTS[label]
        actual = len(df[df["data_practice"] == label])
        target = REBALANCE_TARGETS[label]
        action = "UNDERSAMPLE" if target < actual else "OVERSAMPLE"
        print(f"  {label:<45}  {actual:>5} → {target:>4}  {action}")

    print("\n[Rebalancing] Applying sqrt-frequency rebalancing + augmentation...")
    all_records, stats = build_rebalanced_records(df)
    print(f"\n  {'Label':<45}  Before  After  Augmented")
    for label in OPP115_LABELS:
        b = stats["before"].get(label, 0)
        a = stats["after"].get(label, 0)
        g = stats["augmented_added"].get(label, 0)
        print(f"  {label:<45}  {b:>5} → {a:>4}  (+{g})")

    print("\n[Split] Creating stratified train / val / test split...")
    train, val, test = stratified_split(all_records)
    print_split_stats("Train", train)
    print_split_stats("Val",   val)
    print_split_stats("Test",  test)

    train_labels = [r["label_id"] for r in train]
    cw = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=train_labels)
    print(f"\n[Class Weights] (residual correction for Focal Loss trainer):")
    for i, w in enumerate(cw):
        print(f"  {ID2LABEL[i]:<45}  {w:.4f}")

    lb_fields = ["text", "label_id", "data_practice", "risk_level", "augmented"]
    def keep(r): return {k: r.get(k, False) for k in lb_fields}

    for split_data, fname in [
        (train, "legalbert_train"),
        (val,   "legalbert_val"),
        (test,  "legalbert_test"),
    ]:
        with open(f"./data/processed/{fname}.json", "w") as f:
            json.dump([keep(r) for r in split_data], f, indent=2)

    ll_train = [record_to_llama3(r) for r in train if not r.get("augmented")]
    ll_val   = [record_to_llama3(r) for r in val   if not r.get("augmented")]
    with open("./data/processed/llama3_train.json", "w") as f:
        json.dump(ll_train, f, indent=2)
    with open("./data/processed/llama3_val.json", "w") as f:
        json.dump(ll_val, f, indent=2)
    print(f"\n[LLaMA 3] Instruction samples → train={len(ll_train)} val={len(ll_val)}")

    label_map = {
        "id2label"          : ID2LABEL,
        "label2id"          : LABEL2ID,
        "risk_map"          : RISK_MAP,
        "num_labels"        : NUM_LABELS,
        "class_weights"     : {str(i): float(w) for i, w in enumerate(cw)},
        "rebalance_targets" : REBALANCE_TARGETS,
        "actual_counts"     : ACTUAL_COUNTS,
        "stats"             : stats,
    }
    with open("./data/processed/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n[Done] All outputs saved to ./data/processed/")
    print("  legalbert_train/val/test.json  — test = real samples only (held-out)")
    print("  llama3_train/val.json          — natural samples only")
    print("  label_map.json                 — class_weights included for trainer")


if __name__ == "__main__":
    main()