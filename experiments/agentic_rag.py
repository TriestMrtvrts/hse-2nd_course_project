import csv
from dataclasses import replace
from pathlib import Path

from experiment_utils import (
    QUESTION_POOL_PATH,
    RESULTS_DIR,
    build_context,
    build_retrievers,
    call_llm,
    default_retrieval_experiments,
    ensure_results_dir,
    generate_llm_answer,
    load_corpus_documents,
    load_csv_rows,
    reciprocal_rank_fusion,
    retrieve_with_config,
)


def generate_subqueries(question: str) -> list[str]:
    prompt = (
        "Разложи пользовательский вопрос на 2-3 коротких поисковых запроса для RAG-поиска. "
        "Верни только сами запросы, каждый с новой строки, без нумерации.\n\n"
        f"Вопрос: {question}"
    )
    response = call_llm(prompt, temperature=0.0)
    queries = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
    return queries[:3] or [question]


def main():
    results_dir = ensure_results_dir(RESULTS_DIR)
    corpus_documents = load_corpus_documents()
    question_rows = load_csv_rows(QUESTION_POOL_PATH)
    boundary_rows = [row for row in question_rows if row.get("split") == "analysis"]

    base_config = replace(default_retrieval_experiments()[0], name="agentic_dense_analysis", split="analysis")
    chunk_cache = {}
    dense_cache = {}
    sparse_cache = {}
    reranker_cache = {}
    _, dense_retriever, sparse_retriever, reranker = build_retrievers(
        config=base_config,
        chunk_cache=chunk_cache,
        dense_cache=dense_cache,
        sparse_cache=sparse_cache,
        reranker_cache=reranker_cache,
        corpus_documents=corpus_documents,
    )

    rows = []
    markdown_parts = ["# AgenticRAG Pilot", ""]
    for row in boundary_rows:
        subqueries = generate_subqueries(row["question"])
        rankings = []
        for subquery in subqueries:
            rankings.append(
                retrieve_with_config(
                    subquery,
                    base_config,
                    dense_retriever=dense_retriever,
                    sparse_retriever=sparse_retriever,
                    reranker=reranker,
                )[:3]
            )
        fused_matches = reciprocal_rank_fusion(rankings, top_k=5)
        answer = generate_llm_answer(
            row["question"],
            fused_matches,
            top_k=3,
            temperature=0.1,
            cited=True,
        )
        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "subqueries": " | ".join(subqueries),
                "top_documents": " | ".join(match["document_name"] for match in fused_matches[:5]),
                "top_sections": " | ".join(match["section"] for match in fused_matches[:5]),
                "answer": answer,
            }
        )

        markdown_parts.append(f"## Q{row['id']}")
        markdown_parts.append(f"**Question:** {row['question']}")
        markdown_parts.append("")
        markdown_parts.append(f"**Subqueries:** {'; '.join(subqueries)}")
        markdown_parts.append("")
        markdown_parts.append(f"**Top documents:** {' | '.join(match['document_name'] for match in fused_matches[:5])}")
        markdown_parts.append("")
        markdown_parts.append("**Answer:**")
        markdown_parts.append(answer)
        markdown_parts.append("")
        markdown_parts.append("**Context preview:**")
        markdown_parts.append(build_context(fused_matches, 3))
        markdown_parts.append("")

    csv_path = results_dir / "agentic_rag_analysis.csv"
    md_path = results_dir / "agentic_rag_analysis.md"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text("\n".join(markdown_parts), encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
