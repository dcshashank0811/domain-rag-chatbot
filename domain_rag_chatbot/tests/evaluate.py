"""Run tests/test_questions.csv against the PDFs in documents/.

Usage (from the project root):  python tests/evaluate.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rag_pipeline import answer_question, build_index  # noqa: E402

REFUSAL = "I could not find this information in the uploaded documents."
TOP_K = 5


def main() -> int:
    pdf_paths = sorted((ROOT / "documents").glob("*.pdf"))
    if not pdf_paths:
        print("No PDFs found in documents/. Add sample PDFs first.")
        return 1

    print(f"Building index from {len(pdf_paths)} PDF(s)...")
    index = build_index([{"name": p.name, "bytes": p.read_bytes()} for p in pdf_paths])

    with open(ROOT / "tests" / "test_questions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    passed = failed = skipped = 0
    for row in rows:
        question = row["question"]
        expected = (row.get("expected_source") or "").strip()
        page = (row.get("expected_page") or "").strip()

        if not expected:
            skipped += 1
            print(f"[SKIP] {question}")
            continue

        result = answer_question(index, question, top_k=TOP_K)

        if expected.lower() == "not available":
            ok = REFUSAL in result["answer"]
            detail = "refused" if ok else "did not refuse (needs GROQ_API_KEY to test)"
        else:
            hits = [
                s for s in result["sources"]
                if s["document"] == expected and (not page or str(s["page"]) == page)
            ]
            ok = bool(hits)
            target = f"{expected} p.{page}" if page else expected
            detail = f"found {target}" if ok else f"{target} not in top {TOP_K}"

        passed += ok
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question} -> {detail}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
