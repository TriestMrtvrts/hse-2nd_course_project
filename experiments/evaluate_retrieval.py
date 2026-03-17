import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from RAG_CODE.rag.rag_inference import (
    EMBEDDING_MODEL,
    load_question_pool,
    retrieve_matches,
    score_question_retrieval,
    setup_database,
    summarize_retrieval_scores,
)


BASE_DIR = Path(__file__).resolve().parent
QUESTION_POOL_PATH = BASE_DIR / "question_pool.csv"
RESULTS_DIR = BASE_DIR / "results"
DETAILED_RESULTS_PATH = RESULTS_DIR / "retrieval_eval_results.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "retrieval_eval_summary.json"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    db = setup_database()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    question_rows = load_question_pool(str(QUESTION_POOL_PATH))

    scored_rows = []
    for question_row in question_rows:
        matches = retrieve_matches(question_row["question"], db, embedding_model, k=5)
        scored_rows.append(score_question_retrieval(question_row, matches))

    if scored_rows:
        fieldnames = list(scored_rows[0].keys())
        with open(DETAILED_RESULTS_PATH, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(scored_rows)

    summary = summarize_retrieval_scores(scored_rows)
    with open(SUMMARY_RESULTS_PATH, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    db.close()

    print(f"Detailed results saved to: {DETAILED_RESULTS_PATH}")
    print(f"Summary saved to: {SUMMARY_RESULTS_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
