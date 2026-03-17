import os
import sqlite3
import logging
import csv
from textwrap import dedent
import sqlite_vec
from sqlite_vec import serialize_float32
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
import time
from datetime import datetime
from RAG_CODE.rag.prompt import prompt_lia

prompt_summarize = '''
Проанализируй следующие вопросы и выдели наиболее распространенные проблемы.
Формат вывода:
1. Представь каждую итоговую проблему, как вопрос, подчеркивающий ее.
2. Представь получившиеся данные в виде упорядоченного списка.
'''


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DB_NAME = os.path.join(BASE_DIR, "hse.sqlite3")
LOG_DB_NAME = "log.sqlite3"
EMBEDDING_MODEL = 'deepvk/USER-bge-m3'
DEFAULT_RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
DEFAULT_DISTANCE_THRESHOLD = float(os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "1.01"))


def setup_database():
    db = sqlite3.connect(DATA_DB_NAME)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db

def setup_log_database():
    log_db = sqlite3.connect(LOG_DB_NAME)
    log_db.execute('''CREATE TABLE IF NOT EXISTS logs
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       timestamp TEXT,
                       chat_id TEXT,
                       question TEXT,
                       answer TEXT,
                       context TEXT)''')
    return log_db

def normalize_document_name(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return os.path.basename(value).strip().lower()


def expected_document_matches(expected_document: str, candidate_document: str) -> bool:
    expected = normalize_document_name(expected_document)
    candidate = normalize_document_name(candidate_document)
    if not expected or expected == "ambiguous":
        return False
    if expected == candidate:
        return True
    candidate_without_ext, _ = os.path.splitext(candidate)
    expected_without_ext, _ = os.path.splitext(expected)
    return expected in candidate or expected_without_ext in candidate_without_ext


def _normalize_match_metadata(section: str, source: str):
    section = section or ""
    source = source or ""
    section_looks_like_path = os.path.exists(section) or os.path.sep in section
    source_looks_like_path = os.path.exists(source) or os.path.sep in source

    if section_looks_like_path and not source_looks_like_path:
        return source, section
    return section, source


def retrieve_matches(
    query: str,
    db,
    embedding_model,
    k: int = DEFAULT_RETRIEVAL_K,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
):
    query_embedding = list(embedding_model.encode([query], normalize_embeddings=True))[0]
    rows = db.execute(
        """
    SELECT
        chunk_embeddings.id,
        distance,
        text, 
        meta_data_h,
        meta_data_source
    FROM chunk_embeddings
    LEFT JOIN chunks ON chunks.id = chunk_embeddings.id
    WHERE embedding MATCH ? AND k = ? AND distance <= 1.01
    ORDER BY distance
        """,
        [serialize_float32(query_embedding), k],
    ).fetchall()
    matches = []
    for row in rows:
        if row[1] > distance_threshold:
            continue
        section, source = _normalize_match_metadata(row[3], row[4])
        matches.append(
            {
                "chunk_id": row[0],
                "distance": row[1],
                "text": row[2],
                "section": section,
                "source": source,
                "document_name": os.path.basename(source) if source else "",
            }
        )
    return matches


def retrieve_context(
    query: str,
    db,
    embedding_model,
    k: int = DEFAULT_RETRIEVAL_K,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
):
    matches = retrieve_matches(query, db, embedding_model, k=k, distance_threshold=distance_threshold)
    context = "\n\nКонтекст:\n" + "\n-----\n".join([item["text"] for item in matches])
    meta_datas = [item["section"] for item in matches]
    sources = [item["source"] for item in matches]
    return context, meta_datas, sources


def load_question_pool(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def score_question_retrieval(
    question_row: dict,
    matches,
    top_k_values=(1, 3),
):
    expected_document = question_row.get("expected_document", "")
    relevant_ranks = []
    document_names = [match["document_name"] for match in matches]
    for index, document_name in enumerate(document_names, start=1):
        if expected_document_matches(expected_document, document_name):
            relevant_ranks.append(index)

    scored = dict(question_row)
    scored["top_1_document"] = document_names[0] if document_names else ""
    scored["top_3_documents"] = " | ".join(document_names[:3])
    scored["top_3_sections"] = " | ".join(match["section"] for match in matches[:3])
    scored["top_3_distances"] = " | ".join(f"{match['distance']:.4f}" for match in matches[:3])
    scored["top_3_chunk_ids"] = " | ".join(str(match["chunk_id"]) for match in matches[:3])
    scored["top_3_chunks"] = " ||| ".join(" ".join(match["text"].split()) for match in matches[:3])
    scored["first_relevant_rank"] = str(relevant_ranks[0]) if relevant_ranks else ""
    scored["mrr"] = 1 / relevant_ranks[0] if relevant_ranks else 0.0

    for k in top_k_values:
        scored[f"document_hit@{k}"] = int(any(rank <= k for rank in relevant_ranks))
    return scored


def summarize_retrieval_scores(scored_rows, top_k_values=(1, 3)):
    evaluated_rows = [row for row in scored_rows if row.get("expected_document", "").strip().lower() != "ambiguous"]
    summary = {
        "evaluated_questions": len(evaluated_rows),
        "skipped_questions": len(scored_rows) - len(evaluated_rows),
    }
    if not evaluated_rows:
        for k in top_k_values:
            summary[f"document_hit@{k}"] = 0.0
        summary["mrr"] = 0.0
        return summary

    for k in top_k_values:
        summary[f"document_hit@{k}"] = sum(int(row[f"document_hit@{k}"]) for row in evaluated_rows) / len(evaluated_rows)
    summary["mrr"] = sum(float(row["mrr"]) for row in evaluated_rows) / len(evaluated_rows)

    categories = {}
    for row in evaluated_rows:
        category = row.get("question_type", "unknown")
        categories.setdefault(category, []).append(row)
    for category, rows in categories.items():
        prefix = f"by_type.{category}"
        for k in top_k_values:
            summary[f"{prefix}.document_hit@{k}"] = sum(int(row[f"document_hit@{k}"]) for row in rows) / len(rows)
        summary[f"{prefix}.mrr"] = sum(float(row["mrr"]) for row in rows) / len(rows)
    return summary

def call_model(prompt: str, messages=None, temp=0.2):
    request_messages = list(messages) if messages else []
    if not request_messages or request_messages[-1].get("content") != prompt:
        request_messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=request_messages,
                temperature=temp,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logging.warning(
                "LLM request failed (attempt %s/%s, base_url=%s, model=%s): %s",
                attempt + 1,
                max_retries,
                LLM_BASE_URL,
                LLM_MODEL,
                exc,
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return (
                    "Извините, не удалось получить ответ от LLM. "
                    f"Проверьте, что Ollama запущена на {LLM_BASE_URL} "
                    f"и модель '{LLM_MODEL}' доступна."
                )

def ask_question(query: str, db, embedding_model, conversation_history, log_db, chat_id) -> str:
    context, meta_datas, sources = retrieve_context(query, db, embedding_model)
    prompt = dedent(f"""
    Используй следующую информацию:

    ```
    {context}
    ```

    чтобы ответить на вопрос:
    {query}
    """)
    conversation_history.append({"role": "user", "content": prompt})
    if len(sources) > 0 or len(query) < 2:
        response = call_model(prompt, conversation_history, temp=0.2)
    else:
        response = "Мне очень хочется помочь вам с вашим вопросом, но, к сожалению, я не нашла нужную информацию в предоставленных документах. Возможно, запрос можно переформулировать или уточнить детали, чтобы я могла более точно и эффективно обработать его. Буду рада, если вы подскажете, что именно вас интересует, и я постараюсь найти подходящий ответ!"
    conversation_history.append({"role": "assistant", "content": response})
    
    # Log the question and answer
    log_db.execute('INSERT INTO logs (timestamp, chat_id, question, answer, context) VALUES (?, ?, ?, ?, ?)',
                   (datetime.now().isoformat(), chat_id, query, response, json.dumps(context)))
    log_db.commit()
    
    return response, context, meta_datas, sources


def ask_question_creative(query: str, db, embedding_model, conversation_history, log_db, chat_id) -> str:
    context, meta_datas, sources = retrieve_context(query, db, embedding_model)
    prompt = dedent(f"""
    Используй следующую информацию:

    ```
    {context}
    ```

    чтобы креативно ответить на вопрос:
    {query}
    """)
    conversation_history.append({"role": "user", "content": prompt})
    if len(sources) > 0 or len(query) < 2:
        response = call_model(prompt, conversation_history, temp=0.5)
    else:
        response = "Мне очень хочется помочь вам с вашим вопросом, но, к сожалению, я не нашла нужную информацию в предоставленных документах. Возможно, запрос можно переформулировать или уточнить детали, чтобы я могла более точно и эффективно обработать его. Буду рада, если вы подскажете, что именно вас интересует, и я постараюсь найти подходящий ответ!"
    conversation_history.append({"role": "assistant", "content": response})
    
    # Log the question and answer
    log_db.execute('INSERT INTO logs (timestamp, chat_id, question, answer, context) VALUES (?, ?, ?, ?, ?)',
                   (datetime.now().isoformat(), chat_id, query, response, json.dumps(context)))
    log_db.commit()
    
    return response, context, meta_datas, sources


#########
def get_relevant_problems(questions):
    user_prompt = '\n'.join(questions)
    prompt = prompt_summarize + '\n' + user_prompt
    relevant_problems = call_model(prompt, [])
    return relevant_problems.splitlines()
def get_uncertain_questions(problems, db, embedding_model, thr = 0.5):
    need_clarification = []
    for problem in problems:
        q_data = retrieve_context(retrieve_context(problem, db, embedding_model))
        # pseudo code
        if q_data['distances'][0] < thr:
            need_clarification.append(problem)
    return need_clarification
#########



def main():
    db = setup_database()
    log_db = setup_log_database()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    conversation_history = [{"role": "system", "content": prompt_lia}]
    chat_id = datetime.now().strftime("%Y%m%d%H%M%S")

    print("Добро пожаловать! Задавайте ваши вопросы. Для выхода введите 'выход'.")
    print("Для очистки диалога введите 'очисти'.")

    while True:
        query = input("\nВаш вопрос: ")
        if query.lower() == 'выход':
            break
        elif query.lower() == 'очисти':
            conversation_history = [{"role": "system", "content": prompt_lia}]
            chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
            print("Диалог очищен.")
            continue

        response, context, meta_datas, sources = ask_question(query, db, embedding_model, conversation_history, log_db, chat_id)
        print('\nОтвет:')
        print(response)
        print('\nОткуда взята информация:')
        print(', '.join(set(meta_datas)))
        print('\nИсточники:')
        print(', '.join(set(sources)))

    db.close()
    log_db.close()
    print("Спасибо за использование нашей системы. До свидания!")

if __name__ == "__main__":
    main()
