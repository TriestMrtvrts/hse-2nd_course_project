import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RAG_CODE.rag.model_utils import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    resolve_model_name_or_path,
)
from RAG_CODE.rag.prep_rag_data import (  # noqa: E402
    create_chunks,
    extract_header_from_text,
    preprocess_markdown,
)
from RAG_CODE.rag.rag_inference import (  # noqa: E402
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    expected_document_matches,
)


CORPUS_DIR = PROJECT_ROOT / "RAG_CODE" / "rag" / "converted_md"
QUESTION_POOL_PATH = PROJECT_ROOT / "experiments" / "question_pool.csv"
GENERATION_EVAL_PATH = PROJECT_ROOT / "experiments" / "generation_eval_set.csv"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def normalize_text_key(value: str) -> str:
    if not isinstance(value, str):
        return ""
    normalized = re.sub(r"[^a-zа-яё0-9]+", " ", value.lower(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def parse_multi_value(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def expected_section_matches(expected_section: str, candidate_section: str) -> bool:
    expected = normalize_text_key(expected_section)
    candidate = normalize_text_key(candidate_section)
    if not expected or not candidate:
        return False
    return expected in candidate or candidate in expected


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())


def content_tokens(text: str) -> list[str]:
    stopwords = {
        "и", "в", "во", "на", "по", "с", "со", "к", "ко", "от", "до", "за",
        "из", "у", "а", "но", "или", "ли", "не", "что", "как", "это", "то",
        "для", "при", "об", "обо", "о", "же", "бы", "быть", "есть", "если",
        "так", "его", "ее", "их", "мы", "вы", "они", "он", "она", "я",
    }
    return [token for token in tokenize(text) if token not in stopwords and len(token) > 2]


def format_metric(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    document_name: str
    source_path: str
    section: str
    text: str


@dataclass(frozen=True)
class RetrievalExperimentConfig:
    name: str
    family: str
    split: str
    preprocessing: str = "clean"
    chunking_method: str = "header_recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 256
    embedder: str = DEFAULT_EMBEDDING_MODEL
    retrieval_mode: str = "dense"
    top_k: int = 5
    candidate_k: int = 12
    dense_weight: float = 0.6
    reranker: str | None = None


@dataclass(frozen=True)
class GenerationExperimentConfig:
    name: str
    retrieval_experiment_name: str
    method: str
    temperature: float = 0.0
    top_k: int = 4


def load_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_corpus_documents(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    documents = []
    for path in sorted(corpus_dir.glob("*.md")):
        documents.append(
            {
                "path": path,
                "document_name": path.stem,
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return documents


def build_chunks(
    corpus_documents: list[dict],
    preprocessing: str,
    chunking_method: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    chunk_id = 1
    all_chunks = []
    for document in corpus_documents:
        processed_text = preprocess_markdown(document["text"], preprocessing)
        splits = create_chunks(
            processed_text,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for split in splits:
            section = (
                split.metadata.get("Header 3")
                or split.metadata.get("Header 2")
                or split.metadata.get("Header 1")
                or split.metadata.get("header")
                or extract_header_from_text(split.page_content)
            )
            all_chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_name=document["document_name"],
                    source_path=str(document["path"]),
                    section=section or "Full Document",
                    text=split.page_content.strip(),
                )
            )
            chunk_id += 1
    return all_chunks


def prepare_query_text(embedder_name: str, query: str) -> str:
    if "e5" in embedder_name.lower():
        return f"query: {query}"
    return query


def prepare_passage_text(embedder_name: str, passage: str) -> str:
    if "e5" in embedder_name.lower():
        return f"passage: {passage}"
    return passage


class DenseRetriever:
    def __init__(self, chunks: list[ChunkRecord], embedder_name: str):
        self.chunks = chunks
        self.embedder_name = embedder_name
        self.model = SentenceTransformer(resolve_model_name_or_path(embedder_name))
        chunk_texts = [prepare_passage_text(embedder_name, chunk.text) for chunk in chunks]
        self.embeddings = self.model.encode(
            chunk_texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int) -> list[dict]:
        query_embedding = self.model.encode(
            [prepare_query_text(self.embedder_name, query)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        matches = []
        for index in top_indices:
            chunk = self.chunks[int(index)]
            matches.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_name": chunk.document_name,
                    "source": chunk.source_path,
                    "section": chunk.section,
                    "text": chunk.text,
                    "score": float(scores[int(index)]),
                }
            )
        return matches


class BM25Retriever:
    def __init__(self, chunks: list[ChunkRecord], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lengths = np.array([len(document) for document in self.documents], dtype=float)
        self.average_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        self.term_frequencies = [Counter(document) for document in self.documents]

        document_frequencies = Counter()
        for document in self.documents:
            document_frequencies.update(set(document))

        total_documents = max(len(self.documents), 1)
        self.idf = {
            token: math.log(1 + (total_documents - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequencies.items()
        }

    def search(self, query: str, top_k: int) -> list[dict]:
        query_tokens = tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=float)
        for token in query_tokens:
            if token not in self.idf:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(token, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[index] / max(self.average_doc_length, 1.0)
                )
                scores[index] += self.idf[token] * (frequency * (self.k1 + 1)) / denominator

        top_indices = np.argsort(scores)[::-1][:top_k]
        matches = []
        for index in top_indices:
            chunk = self.chunks[int(index)]
            matches.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_name": chunk.document_name,
                    "source": chunk.source_path,
                    "section": chunk.section,
                    "text": chunk.text,
                    "score": float(scores[int(index)]),
                }
            )
        return matches


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = CrossEncoder(resolve_model_name_or_path(model_name), max_length=512)

    def rerank(self, query: str, matches: list[dict], top_k: int) -> list[dict]:
        if not matches:
            return []
        scores = self.model.predict([(query, match["text"]) for match in matches])
        reranked = []
        for match, score in zip(matches, scores):
            if isinstance(score, (list, tuple, np.ndarray)):
                score_value = float(score[-1])
            else:
                score_value = float(score)
            enriched_match = dict(match)
            enriched_match["rerank_score"] = score_value
            reranked.append(enriched_match)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k]


def reciprocal_rank_fusion(rankings: list[list[dict]], top_k: int, k_constant: int = 60) -> list[dict]:
    if not rankings:
        return []

    merged_scores = defaultdict(float)
    merged_matches = {}
    for ranking in rankings:
        for rank, match in enumerate(ranking, start=1):
            chunk_id = match["chunk_id"]
            merged_scores[chunk_id] += 1.0 / (k_constant + rank)
            merged_matches[chunk_id] = match

    ordered_chunk_ids = sorted(merged_scores, key=merged_scores.get, reverse=True)[:top_k]
    fused = []
    for chunk_id in ordered_chunk_ids:
        match = dict(merged_matches[chunk_id])
        match["score"] = merged_scores[chunk_id]
        fused.append(match)
    return fused


def ndcg_from_rank(rank: int | None) -> float:
    if not rank:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def summarize_scores(scored_rows: list[dict], metric_names: Iterable[str]) -> dict:
    metric_names = list(metric_names)
    summary = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in scored_rows
            if row.get(metric_name, "") not in ("", None)
        ]
        summary[metric_name] = mean(values) if values else None
    return summary


def score_retrieval(question_row: dict, matches: list[dict], top_k: int) -> dict:
    expected_document = question_row.get("expected_document", "")
    expected_sections = parse_multi_value(question_row.get("expected_section", ""))

    document_ranks = []
    section_ranks = []
    for index, match in enumerate(matches, start=1):
        if expected_document_matches(expected_document, match["document_name"]):
            document_ranks.append(index)
            if expected_sections and any(
                expected_section_matches(expected_section, match["section"])
                for expected_section in expected_sections
            ):
                section_ranks.append(index)

    top_documents = " | ".join(match["document_name"] for match in matches[:top_k])
    top_sections = " | ".join(match["section"] for match in matches[:top_k])
    top_scores = " | ".join(format_metric(match["score"]) for match in matches[:top_k])
    top_chunk_ids = " | ".join(str(match["chunk_id"]) for match in matches[:top_k])

    scored_row = dict(question_row)
    scored_row.update(
        {
            "top_documents": top_documents,
            "top_sections": top_sections,
            "top_scores": top_scores,
            "top_chunk_ids": top_chunk_ids,
            "top_chunks": " ||| ".join(" ".join(match["text"].split()) for match in matches[:top_k]),
            "document_first_relevant_rank": document_ranks[0] if document_ranks else "",
            "section_first_relevant_rank": section_ranks[0] if section_ranks else "",
            "document_hit@1": int(any(rank <= 1 for rank in document_ranks)),
            "document_hit@3": int(any(rank <= 3 for rank in document_ranks)),
            "document_mrr": 1 / document_ranks[0] if document_ranks else 0.0,
            "document_ndcg@3": ndcg_from_rank(document_ranks[0]) if document_ranks and document_ranks[0] <= 3 else 0.0,
            "section_hit@1": int(any(rank <= 1 for rank in section_ranks)) if expected_sections else "",
            "section_hit@3": int(any(rank <= 3 for rank in section_ranks)) if expected_sections else "",
        }
    )
    return scored_row


def filter_question_rows(question_rows: list[dict], split: str) -> list[dict]:
    return [row for row in question_rows if row.get("split") == split]


def summarize_retrieval_experiment(config: RetrievalExperimentConfig, scored_rows: list[dict]) -> dict:
    auto_rows = [row for row in scored_rows if row.get("evaluation_mode", "auto") == "auto"]
    section_rows = [row for row in auto_rows if row.get("section_hit@1", "") != ""]

    summary = {
        "experiment": config.name,
        "family": config.family,
        "split": config.split,
        "preprocessing": config.preprocessing,
        "chunking_method": config.chunking_method,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedder": config.embedder,
        "retrieval_mode": config.retrieval_mode,
        "reranker": config.reranker or "",
        "evaluated_questions": len(auto_rows),
        "question_types": ", ".join(sorted(set(row.get("question_type", "") for row in auto_rows))),
    }
    summary.update(
        summarize_scores(
            auto_rows,
            [
                "document_hit@1",
                "document_hit@3",
                "document_mrr",
                "document_ndcg@3",
            ],
        )
    )
    summary.update(
        summarize_scores(
            section_rows,
            [
                "section_hit@1",
                "section_hit@3",
            ],
        )
    )
    return summary


def call_llm(prompt: str, temperature: float) -> str:
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    last_error = None
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — аккуратный ассистент по документам ВШЭ. "
                            "Отвечай только по данному контексту, не придумывай факты, "
                            "если данных не хватает — так и скажи."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
    raise last_error


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def build_context(matches: list[dict], top_k: int) -> str:
    blocks = []
    for index, match in enumerate(matches[:top_k], start=1):
        snippet = match["text"][:1200]
        blocks.append(
            f"[{index}] Документ: {match['document_name']}\n"
            f"Раздел: {match['section']}\n"
            f"Фрагмент: {snippet}"
        )
    return "\n\n".join(blocks)


def generate_extractive_answer(question: str, matches: list[dict]) -> str:
    if not matches:
        return "В базе знаний не нашлось релевантного фрагмента для уверенного ответа."

    query_terms = set(content_tokens(question))
    best_sentence = None
    best_overlap = -1
    for sentence in sentence_split(matches[0]["text"]):
        overlap = len(query_terms & set(content_tokens(sentence)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
    if best_sentence:
        return f"{best_sentence} [Источник: {matches[0]['document_name']} | {matches[0]['section']}]"
    return f"{matches[0]['text'][:320]} [Источник: {matches[0]['document_name']} | {matches[0]['section']}]"


def generate_llm_answer(question: str, matches: list[dict], top_k: int, temperature: float, cited: bool) -> str:
    if not matches:
        return "В базе знаний не нашлось релевантного фрагмента для уверенного ответа."

    citation_instruction = (
        "В конце каждого смыслового абзаца добавляй ссылку в формате "
        "[Источник: <документ> | <раздел>]."
        if cited
        else "Если данных недостаточно, прямо скажи об этом."
    )
    prompt = (
        f"Вопрос пользователя: {question}\n\n"
        f"Контекст из базы знаний:\n{build_context(matches, top_k)}\n\n"
        "Сформулируй короткий, точный и проверяемый ответ на русском языке. "
        f"{citation_instruction}"
    )
    return call_llm(prompt, temperature=temperature)


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    dp = [[0] * (len(hypothesis_tokens) + 1) for _ in range(len(reference_tokens) + 1)]
    for i, reference_token in enumerate(reference_tokens, start=1):
        for j, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            if reference_token == hypothesis_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def grounding_overlap(answer: str, context: str) -> float:
    answer_terms = set(content_tokens(answer))
    if not answer_terms:
        return 0.0
    context_terms = set(content_tokens(context))
    return len(answer_terms & context_terms) / len(answer_terms)


def citation_present(answer: str) -> int:
    return int("[Источник:" in answer)


def supported_sentence_rate(answer: str, context: str, embedder: SentenceTransformer) -> float:
    answer_sentences = sentence_split(answer)
    context_sentences = sentence_split(context)
    if not answer_sentences:
        return 0.0
    if not context_sentences:
        return 0.0

    sentence_embeddings = embedder.encode(answer_sentences, normalize_embeddings=True, show_progress_bar=False)
    context_embeddings = embedder.encode(context_sentences, normalize_embeddings=True, show_progress_bar=False)

    supported_sentences = 0
    for sentence_embedding in sentence_embeddings:
        similarities = np.dot(context_embeddings, sentence_embedding)
        if float(np.max(similarities)) >= 0.55:
            supported_sentences += 1
    return supported_sentences / len(answer_sentences)


def default_retrieval_experiments() -> list[RetrievalExperimentConfig]:
    return [
        RetrievalExperimentConfig(
            name="baseline_dense_validation",
            family="reference",
            split="validation",
        ),
        RetrievalExperimentConfig(
            name="preprocess_raw_validation",
            family="preprocessing",
            split="validation",
            preprocessing="raw",
        ),
        RetrievalExperimentConfig(
            name="preprocess_clean_validation",
            family="preprocessing",
            split="validation",
            preprocessing="clean",
        ),
        RetrievalExperimentConfig(
            name="chunk_small_validation",
            family="chunk_size",
            split="validation",
            chunk_size=768,
            chunk_overlap=128,
        ),
        RetrievalExperimentConfig(
            name="chunk_medium_validation",
            family="chunk_size",
            split="validation",
            chunk_size=1024,
            chunk_overlap=256,
        ),
        RetrievalExperimentConfig(
            name="chunk_large_validation",
            family="chunk_size",
            split="validation",
            chunk_size=1536,
            chunk_overlap=384,
        ),
        RetrievalExperimentConfig(
            name="chunk_plain_recursive_validation",
            family="chunking",
            split="validation",
            chunking_method="plain_recursive",
        ),
        RetrievalExperimentConfig(
            name="chunk_header_recursive_validation",
            family="chunking",
            split="validation",
            chunking_method="header_recursive",
        ),
        RetrievalExperimentConfig(
            name="embedder_deepvk_validation",
            family="embedder",
            split="validation",
            embedder="deepvk/USER-bge-m3",
        ),
        RetrievalExperimentConfig(
            name="embedder_minilm_validation",
            family="embedder",
            split="validation",
            embedder="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        RetrievalExperimentConfig(
            name="embedder_e5_validation",
            family="embedder",
            split="validation",
            embedder="intfloat/multilingual-e5-small",
        ),
        RetrievalExperimentConfig(
            name="retriever_dense_validation",
            family="retriever",
            split="validation",
            retrieval_mode="dense",
        ),
        RetrievalExperimentConfig(
            name="retriever_bm25_validation",
            family="retriever",
            split="validation",
            retrieval_mode="bm25",
        ),
        RetrievalExperimentConfig(
            name="retriever_ensemble_validation",
            family="retriever",
            split="validation",
            retrieval_mode="ensemble",
        ),
        RetrievalExperimentConfig(
            name="retriever_ensemble_rerank_validation",
            family="retriever",
            split="validation",
            retrieval_mode="ensemble",
            reranker=DEFAULT_RERANKER_MODEL,
        ),
    ]


def default_generation_experiments(best_retrieval_name: str) -> list[GenerationExperimentConfig]:
    return [
        GenerationExperimentConfig(
            name="extractive_top1_generation",
            retrieval_experiment_name=best_retrieval_name,
            method="extractive",
            top_k=1,
        ),
        GenerationExperimentConfig(
            name="llm_strict_generation",
            retrieval_experiment_name=best_retrieval_name,
            method="llm",
            temperature=0.0,
            top_k=3,
        ),
        GenerationExperimentConfig(
            name="llm_cited_generation",
            retrieval_experiment_name=best_retrieval_name,
            method="llm_cited",
            temperature=0.2,
            top_k=3,
        ),
    ]


def ensure_results_dir(results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def build_retrievers(
    config: RetrievalExperimentConfig,
    chunk_cache: dict,
    dense_cache: dict,
    sparse_cache: dict,
    reranker_cache: dict,
    corpus_documents: list[dict],
):
    chunk_key = (
        config.preprocessing,
        config.chunking_method,
        config.chunk_size,
        config.chunk_overlap,
    )
    if chunk_key not in chunk_cache:
        chunk_cache[chunk_key] = build_chunks(
            corpus_documents,
            preprocessing=config.preprocessing,
            chunking_method=config.chunking_method,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    chunks = chunk_cache[chunk_key]

    dense_retriever = None
    sparse_retriever = None
    reranker = None

    if config.retrieval_mode in {"dense", "ensemble"}:
        dense_key = (chunk_key, config.embedder)
        if dense_key not in dense_cache:
            dense_cache[dense_key] = DenseRetriever(chunks, config.embedder)
        dense_retriever = dense_cache[dense_key]

    if config.retrieval_mode in {"bm25", "ensemble"}:
        if chunk_key not in sparse_cache:
            sparse_cache[chunk_key] = BM25Retriever(chunks)
        sparse_retriever = sparse_cache[chunk_key]

    if config.reranker:
        if config.reranker not in reranker_cache:
            reranker_cache[config.reranker] = CrossEncoderReranker(config.reranker)
        reranker = reranker_cache[config.reranker]

    return chunks, dense_retriever, sparse_retriever, reranker


def retrieve_with_config(
    query: str,
    config: RetrievalExperimentConfig,
    dense_retriever: DenseRetriever | None,
    sparse_retriever: BM25Retriever | None,
    reranker: CrossEncoderReranker | None,
) -> list[dict]:
    if config.retrieval_mode == "dense":
        matches = dense_retriever.search(query, config.candidate_k if reranker else config.top_k)
    elif config.retrieval_mode == "bm25":
        matches = sparse_retriever.search(query, config.candidate_k if reranker else config.top_k)
    elif config.retrieval_mode == "ensemble":
        dense_matches = dense_retriever.search(query, config.candidate_k)
        sparse_matches = sparse_retriever.search(query, config.candidate_k)
        matches = reciprocal_rank_fusion([dense_matches, sparse_matches], config.candidate_k)
    else:
        raise ValueError(f"Unsupported retrieval mode: {config.retrieval_mode}")

    if reranker:
        matches = reranker.rerank(query, matches, config.top_k)
    else:
        matches = matches[: config.top_k]
    return matches


def render_markdown_table(dataframe: pd.DataFrame, columns: list[str]) -> str:
    prepared = dataframe.loc[:, columns].copy()
    for column in prepared.columns:
        prepared[column] = prepared[column].apply(format_metric if column not in {"experiment", "family", "split", "preprocessing", "chunking_method", "embedder", "retrieval_mode", "reranker"} else str)
    return prepared.to_markdown(index=False)


def render_latex_table(dataframe: pd.DataFrame, columns: list[str]) -> str:
    prepared = dataframe.loc[:, columns].copy()
    for column in prepared.columns:
        prepared[column] = prepared[column].apply(format_metric if column not in {"experiment", "family", "split", "preprocessing", "chunking_method", "embedder", "retrieval_mode", "reranker"} else str)
    return prepared.to_latex(index=False, escape=True)
