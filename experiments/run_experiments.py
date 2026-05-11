import json
from dataclasses import replace

import pandas as pd

from experiment_utils import (
    GENERATION_EVAL_PATH,
    QUESTION_POOL_PATH,
    RESULTS_DIR,
    build_context,
    build_retrievers,
    citation_present,
    default_generation_experiments,
    default_retrieval_experiments,
    ensure_results_dir,
    filter_question_rows,
    generate_extractive_answer,
    generate_llm_answer,
    grounding_overlap,
    load_corpus_documents,
    load_csv_rows,
    render_latex_table,
    render_markdown_table,
    retrieve_with_config,
    rouge_l_f1,
    score_retrieval,
    summarize_retrieval_experiment,
)


RETRIEVAL_SUMMARY_COLUMNS = [
    "experiment",
    "family",
    "split",
    "document_hit@1",
    "document_hit@3",
    "document_mrr",
    "document_ndcg@3",
    "section_hit@1",
    "section_hit@3",
]

GENERATION_SUMMARY_COLUMNS = [
    "experiment",
    "questions",
    "failed_items",
    "rouge_l_f1",
    "grounding_overlap",
    "citation_rate",
]


def run_retrieval_suite(results_dir):
    corpus_documents = load_corpus_documents()
    question_rows = load_csv_rows(QUESTION_POOL_PATH)

    chunk_cache = {}
    dense_cache = {}
    sparse_cache = {}
    reranker_cache = {}

    detailed_rows = []
    summary_rows = []
    validation_configs = default_retrieval_experiments()
    config_by_name = {}

    for config in validation_configs:
        config_by_name[config.name] = config
        experiment_rows = []
        dataset_rows = filter_question_rows(question_rows, config.split)
        _, dense_retriever, sparse_retriever, reranker = build_retrievers(
            config=config,
            chunk_cache=chunk_cache,
            dense_cache=dense_cache,
            sparse_cache=sparse_cache,
            reranker_cache=reranker_cache,
            corpus_documents=corpus_documents,
        )

        for question_row in dataset_rows:
            matches = retrieve_with_config(
                question_row["question"],
                config,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                reranker=reranker,
            )
            scored_row = score_retrieval(question_row, matches, top_k=config.top_k)
            scored_row.update(
                {
                    "experiment": config.name,
                    "family": config.family,
                    "split": config.split,
                }
            )
            experiment_rows.append(scored_row)
            detailed_rows.append(scored_row)

        summary_rows.append(summarize_retrieval_experiment(config, experiment_rows))

    summary_df = pd.DataFrame(summary_rows)
    validation_df = summary_df.loc[summary_df["split"] == "validation"].copy()
    validation_df = validation_df.sort_values(
        by=["document_hit@3", "document_hit@1", "document_mrr"],
        ascending=[False, False, False],
    )
    best_validation_name = validation_df.iloc[0]["experiment"]
    best_validation_config = config_by_name[best_validation_name]

    test_configs = [
        replace(default_retrieval_experiments()[0], name="baseline_dense_test", split="test"),
        replace(best_validation_config, name=f"{best_validation_name.replace('_validation', '')}_test", split="test"),
    ]
    unique_test_configs = []
    seen_test_names = set()
    for config in test_configs:
        if config.name in seen_test_names:
            continue
        unique_test_configs.append(config)
        seen_test_names.add(config.name)
    test_configs = unique_test_configs

    for config in test_configs:
        config_by_name[config.name] = config
        experiment_rows = []
        dataset_rows = filter_question_rows(question_rows, config.split)
        _, dense_retriever, sparse_retriever, reranker = build_retrievers(
            config=config,
            chunk_cache=chunk_cache,
            dense_cache=dense_cache,
            sparse_cache=sparse_cache,
            reranker_cache=reranker_cache,
            corpus_documents=corpus_documents,
        )
        for question_row in dataset_rows:
            matches = retrieve_with_config(
                question_row["question"],
                config,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                reranker=reranker,
            )
            scored_row = score_retrieval(question_row, matches, top_k=config.top_k)
            scored_row.update(
                {
                    "experiment": config.name,
                    "family": "test",
                    "split": config.split,
                }
            )
            experiment_rows.append(scored_row)
            detailed_rows.append(scored_row)
        summary_rows.append(summarize_retrieval_experiment(config, experiment_rows))

    best_test_config = test_configs[-1]
    boundary_rows = filter_question_rows(question_rows, "analysis")
    _, dense_retriever, sparse_retriever, reranker = build_retrievers(
        config=best_test_config,
        chunk_cache=chunk_cache,
        dense_cache=dense_cache,
        sparse_cache=sparse_cache,
        reranker_cache=reranker_cache,
        corpus_documents=corpus_documents,
    )
    boundary_details = []
    for question_row in boundary_rows:
        matches = retrieve_with_config(
            question_row["question"],
            best_test_config,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker=reranker,
        )
        scored_row = score_retrieval(question_row, matches, top_k=best_test_config.top_k)
        scored_row.update(
            {
                "experiment": best_test_config.name,
                "family": "analysis",
                "split": "analysis",
            }
        )
        boundary_details.append(scored_row)

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    boundary_df = pd.DataFrame(boundary_details)

    summary_path = results_dir / "retrieval_experiment_summary.csv"
    detailed_path = results_dir / "retrieval_experiment_details.csv"
    boundary_path = results_dir / "boundary_retrieval_analysis.csv"
    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    boundary_df.to_csv(boundary_path, index=False)

    table_parts = ["# Retrieval Experiments", ""]
    for family in ["preprocessing", "chunk_size", "chunking", "embedder", "retriever"]:
        family_df = summary_df.loc[summary_df["family"] == family]
        if family_df.empty:
            continue
        table_parts.append(f"## {family}")
        table_parts.append(
            render_markdown_table(
                family_df.sort_values(by=["document_hit@3", "document_hit@1"], ascending=[False, False]),
                RETRIEVAL_SUMMARY_COLUMNS,
            )
        )
        table_parts.append("")

    test_df = summary_df.loc[summary_df["split"] == "test"]
    table_parts.append("## held_out_test")
    table_parts.append(render_markdown_table(test_df, RETRIEVAL_SUMMARY_COLUMNS))
    table_parts.append("")

    latex_parts = ["% Retrieval experiment tables", ""]
    for family in ["preprocessing", "chunk_size", "chunking", "embedder", "retriever"]:
        family_df = summary_df.loc[summary_df["family"] == family]
        if family_df.empty:
            continue
        latex_parts.append(f"% {family}")
        latex_parts.append(render_latex_table(family_df, RETRIEVAL_SUMMARY_COLUMNS))
        latex_parts.append("")

    retrieval_tables_md = results_dir / "retrieval_tables.md"
    retrieval_tables_tex = results_dir / "retrieval_tables.tex"
    retrieval_tables_md.write_text("\n".join(table_parts), encoding="utf-8")
    retrieval_tables_tex.write_text("\n".join(latex_parts), encoding="utf-8")

    metadata = {
        "best_validation_experiment": best_validation_name,
        "best_test_experiment": best_test_config.name,
        "summary_csv": str(summary_path),
        "detailed_csv": str(detailed_path),
        "boundary_csv": str(boundary_path),
    }
    (results_dir / "retrieval_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return best_test_config, metadata


def run_generation_suite(results_dir, best_retrieval_config):
    if not GENERATION_EVAL_PATH.exists():
        return {"status": "skipped", "reason": "generation_eval_set.csv not found"}

    corpus_documents = load_corpus_documents()
    question_rows = load_csv_rows(GENERATION_EVAL_PATH)

    chunk_cache = {}
    dense_cache = {}
    sparse_cache = {}
    reranker_cache = {}
    _, dense_retriever, sparse_retriever, reranker = build_retrievers(
        config=best_retrieval_config,
        chunk_cache=chunk_cache,
        dense_cache=dense_cache,
        sparse_cache=sparse_cache,
        reranker_cache=reranker_cache,
        corpus_documents=corpus_documents,
    )

    generation_configs = default_generation_experiments(best_retrieval_config.name)

    detailed_rows = []
    summary_rows = []

    for generation_config in generation_configs:
        experiment_rows = []
        for question_row in question_rows:
            matches = retrieve_with_config(
                question_row["question"],
                best_retrieval_config,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                reranker=reranker,
            )
            llm_error = ""
            try:
                if generation_config.method == "extractive":
                    answer = generate_extractive_answer(question_row["question"], matches[: generation_config.top_k])
                elif generation_config.method == "llm":
                    answer = generate_llm_answer(
                        question_row["question"],
                        matches,
                        top_k=generation_config.top_k,
                        temperature=generation_config.temperature,
                        cited=False,
                    )
                elif generation_config.method == "llm_cited":
                    answer = generate_llm_answer(
                        question_row["question"],
                        matches,
                        top_k=generation_config.top_k,
                        temperature=generation_config.temperature,
                        cited=True,
                    )
                else:
                    raise ValueError(f"Unsupported generation method: {generation_config.method}")
            except Exception as exc:
                answer = "LLM generation failed for this item."
                llm_error = str(exc)

            context = build_context(matches, generation_config.top_k)
            experiment_row = dict(question_row)
            experiment_row.update(
                {
                    "experiment": generation_config.name,
                    "retrieval_experiment": generation_config.retrieval_experiment_name,
                    "answer": answer,
                    "context": context,
                    "llm_error": llm_error,
                    "rouge_l_f1": rouge_l_f1(question_row["reference_answer"], answer),
                    "grounding_overlap": grounding_overlap(answer, context),
                    "citation_present": citation_present(answer),
                }
            )
            experiment_rows.append(experiment_row)
            detailed_rows.append(experiment_row)

        summary_rows.append(
            {
                "experiment": generation_config.name,
                "questions": len(experiment_rows),
                "failed_items": sum(1 for row in experiment_rows if row["llm_error"]),
                "rouge_l_f1": sum(row["rouge_l_f1"] for row in experiment_rows) / len(experiment_rows),
                "grounding_overlap": sum(row["grounding_overlap"] for row in experiment_rows) / len(experiment_rows),
                "citation_rate": sum(row["citation_present"] for row in experiment_rows) / len(experiment_rows),
            }
        )

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)
    detailed_df.to_csv(results_dir / "generation_experiment_details.csv", index=False)
    summary_df.to_csv(results_dir / "generation_experiment_summary.csv", index=False)

    markdown = [
        "# Generation Experiments",
        "",
        render_markdown_table(summary_df, GENERATION_SUMMARY_COLUMNS),
        "",
    ]
    latex = [
        "% Generation experiment tables",
        "",
        render_latex_table(summary_df, GENERATION_SUMMARY_COLUMNS),
        "",
    ]
    (results_dir / "generation_tables.md").write_text("\n".join(markdown), encoding="utf-8")
    (results_dir / "generation_tables.tex").write_text("\n".join(latex), encoding="utf-8")

    return {"status": "completed", "summary_csv": str(results_dir / "generation_experiment_summary.csv")}


def main():
    results_dir = ensure_results_dir(RESULTS_DIR)
    best_retrieval_config, retrieval_metadata = run_retrieval_suite(results_dir)

    run_report = {
        "retrieval": retrieval_metadata,
    }
    try:
        run_report["generation"] = run_generation_suite(results_dir, best_retrieval_config)
    except Exception as exc:
        run_report["generation"] = {
            "status": "skipped",
            "reason": str(exc),
        }

    (results_dir / "experiment_run_report.json").write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
