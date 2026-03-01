"""
Step 0: Inspect actual dataset structures.
Run this first. Shows exactly what columns/fields you have.
"""

import os, csv, json
from pathlib import Path
from collections import Counter
from bs4 import BeautifulSoup

OPP115_ROOT = "./data/OPP-115"
TOSDR_ROOT  = "./data/tosdr"

# ─────────────────────────────────────────────────────────────
# OPP-115 CONFIRMED STRUCTURE (no header, 9 columns):
#   col 0  : annotator_id         (e.g. "C125", "2843")
#   col 1  : annotation_type      (always "test_category_labeling_highlight")
#   col 2  : annotator quality score (int as string)
#   col 3  : policy document id
#   col 4  : segment_id           (0-indexed paragraph reference)
#   col 5  : CATEGORY LABEL       ← training target
#   col 6  : JSON attributes blob ← training text lives here (nested)
#   col 7  : date
#   col 8  : source URL
#
# col 6 JSON — each key is an attribute name, value is a dict:
#   {
#     "User Type"                : {"selectedText": "...", "value": "...", ...},
#     "Personal Information Type": {"selectedText": "...", "value": "...", ...},
#     "Action First-Party"       : {"selectedText": "...", ...},
#     ...
#   }
#   selectedText = the actual highlighted clause text (training input)
#   value        = fine-grained attribute label (e.g. "Generic/Other", "Financial")
#
# Attribute keys by category:
#   First Party Collection/Use    -> User Type, Personal Information Type,
#                                    Action First-Party, Purpose, Identifiability,
#                                    Does/Does Not, Collection Mode
#   Third Party Sharing/Collection-> Third Party Entity, Action Third Party,
#                                    Personal Information Type, ...
#   Data Retention                -> Retention Period, Retention Purpose
#   Policy Change                 -> Change Type, User Choice, Notification Type
#   Data Security                 -> Security Measure
#   Other                         -> Other Type
#
# CONSOLIDATION FILES (same format, fewer rows, higher quality):
#   threshold-0.5  = ≥50% annotator agreement  ← USE THIS FOR TRAINING
#   threshold-0.75 = ≥75% agreement
#   threshold-1.0  = 100% agreement (smallest, highest quality)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# ToS;DR CONFIRMED STRUCTURE:
#   data/tosdr/{company_name}/{document_name}.html
#
# ⚠️  CRITICAL WARNING:
#   These HTML files are RAW POLICY TEXT ONLY.
#   They have ZERO labels, ZERO ratings, ZERO annotations.
#   The risk ratings ("bad","blocker","good","neutral") are stored
#   in the ToS;DR web database — NOT in these HTML files.
#
#   These files CANNOT be used as labeled training data directly.
#   To get labels, either:
#     A) Fetch from ToS;DR API  -> fetch_tosdr_ratings() below
#     B) Use as inference targets only (analyze with trained model)
# ─────────────────────────────────────────────────────────────

def inspect_opp115():
    ann_dir = Path(OPP115_ROOT) / "annotations"
    csv_files = list(ann_dir.glob("*.csv"))
    print(f"[OPP-115] Annotation files: {len(csv_files)}")

    all_rows = []
    for fp in csv_files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for row in csv.reader(f):
                if len(row) >= 6:
                    all_rows.append({
                        "annotator_id"   : row[0],
                        "policy_doc_id"  : row[3],
                        "segment_id"     : row[4],
                        "category"       : row[5],
                        "attributes_json": row[6] if len(row) > 6 else "",
                        "source_file"    : fp.name,
                    })

    print(f"[OPP-115] Total annotation rows : {len(all_rows)}")
    print(f"[OPP-115] Unique policy files   : {len(set(r['source_file'] for r in all_rows))}")
    print(f"[OPP-115] Category distribution :")
    for cat, n in Counter(r["category"] for r in all_rows).most_common():
        print(f"   {n:6d}  {cat}")

    # Consolidation row counts
    print("\n[OPP-115] Consolidation files:")
    for thresh in ["threshold-0.5-overlap-similarity",
                   "threshold-0.75-overlap-similarity",
                   "threshold-1.0-overlap-similarity"]:
        p = Path(OPP115_ROOT) / "consolidation" / thresh
        if p.exists():
            files = list(p.glob("*.csv"))
            total = sum(
                sum(1 for _ in csv.reader(open(fp, encoding="utf-8", errors="ignore")))
                for fp in files
            )
            print(f"   {thresh}: {len(files)} files, {total} rows")
        else:
            print(f"   {thresh}: directory not found")
    return all_rows


def inspect_tosdr():
    root = Path(TOSDR_ROOT)
    if not root.exists():
        print(f"[ToS;DR] Root not found: {TOSDR_ROOT}")
        return

    companies = sorted([d for d in root.iterdir() if d.is_dir()])
    html_files = list(root.rglob("*.html"))
    print(f"\n[ToS;DR] Companies : {len(companies)}")
    print(f"[ToS;DR] HTML files: {len(html_files)}")
    print(f"\n⚠️  These HTMLs are raw scraped policy pages — NO labels attached.")

    for company in companies[:6]:
        docs = list(company.glob("*.html"))
        print(f"\n  {company.name}/")
        for d in docs:
            soup = BeautifulSoup(open(d, errors="ignore").read(), "html.parser")
            for tag in soup(["script","style","nav","header","footer","noscript"]):
                tag.decompose()
            paras = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
            print(f"    {d.name}  ({len(paras)} paragraphs)")
            print(f"      First para: {paras[1][:100] if len(paras) > 1 else 'N/A'}")


def fetch_tosdr_ratings(save_path="./data/tosdr_api_cases.json"):
    """
    Fetch labeled clauses from ToS;DR public API.
    This is required to use ToS;DR as TRAINING data.
    Saves to save_path.

    API returns:
      { "parameters": { "cases": [
          { "id", "title", "description", "classification",
            "service": { "name": "..." }, "quote" }
      ]}}

    classification values: "good" | "bad" | "blocker" | "neutral"
    quote = the actual clause text
    """
    import urllib.request

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_cases, page = [], 1

    print("[ToS;DR API] Fetching labeled cases (requires internet)...")
    while True:
        url = f"https://api.tosdr.org/case/v1/?page={page}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  Stopped at page {page}: {e}")
            break

        cases = data.get("parameters", {}).get("cases", [])
        if not cases:
            break

        for c in cases:
            quote = c.get("quote", "").strip()
            if not quote:
                continue
            all_cases.append({
                "service"    : c.get("service", {}).get("name", "Unknown"),
                "title"      : c.get("title", ""),
                "description": c.get("description", ""),
                "rating"     : c.get("classification", ""),
                "quote"      : quote,
            })
        print(f"  Page {page}: {len(cases)} cases (running total: {len(all_cases)})")
        page += 1

    with open(save_path, "w") as f:
        json.dump(all_cases, f, indent=2)

    print(f"\nSaved {len(all_cases)} labeled cases to {save_path}")
    print("Rating distribution:", dict(Counter(c["rating"] for c in all_cases)))
    return all_cases


if __name__ == "__main__":
    print("=" * 60)
    print("OPP-115")
    print("=" * 60)
    inspect_opp115()

    print("\n" + "=" * 60)
    print("ToS;DR")
    print("=" * 60)
    inspect_tosdr()

    print("\n" + "=" * 60)
    print("TRAINING PLAN")
    print("=" * 60)
    print("""
  LegalBERT:
    Primary  -> OPP-115 consolidation/threshold-0.5 (consensus labels, higher quality)
    Secondary-> OPP-115 annotations/ (all annotators, more data)

  LLaMA 3 (instruction tuning):
    Primary  -> OPP-115 consolidation/threshold-0.5 (text + rich attribute JSON
                  used to build detailed explanations)
    Optional -> ToS;DR API cases (fetch with fetch_tosdr_ratings() if internet available)

  ToS;DR HTML files:
    -> Use as INFERENCE TARGETS after training, not as training data.
    -> OR call fetch_tosdr_ratings() to get labeled quotes from the API.
""")
