# HSE RAG Bot

Telegram-бот для работы с нормативными документами НИУ ВШЭ на основе retrieval-augmented generation.

В проекте есть три слоя:
- продуктовый бот на `aiogram`;
- retrieval / indexing пайплайн на `sqlite-vec`;
- экспериментальный контур для сравнения preprocessing, chunking, embedders, sparse/dense retrieval, reranking и generation.

## 1. Установка

Из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r RAG_CODE/rag/requirements.txt
pip install watchdog
```

Для конвертации `.docx` в markdown нужен `pandoc`:

```bash
brew install pandoc
```

## 2. Корпус документов

Исходные файлы лежат в:

```bash
RAG_CODE/rag/data/
```

Сконвертированные markdown-файлы:

```bash
RAG_CODE/rag/converted_md/
```

SQLite-база знаний:

```bash
RAG_CODE/rag/hse.sqlite3
```

## 3. Пересборка индекса

По умолчанию индексатор использует:
- предобработку `clean`;
- метод чанкирования `header_recursive`;
- `chunk_size=1024`;
- `chunk_overlap=256`;
- embedding-модель `deepvk/USER-bge-m3`.

Полная пересборка:

```bash
source .venv/bin/activate
rm -f RAG_CODE/rag/hse.sqlite3
python3 RAG_CODE/rag/prep_rag_data.py
```

Можно переопределять параметры через переменные окружения:

```bash
export PREPROCESSING_PROFILE=clean
export CHUNKING_METHOD=header_recursive
export CHUNK_SIZE=1024
export CHUNK_OVERLAP=256
export EMBEDDING_MODEL=deepvk/USER-bge-m3
python3 RAG_CODE/rag/prep_rag_data.py
```

## 4. Retrieval и generation эксперименты

Локальные модели в `.hf_models/` в репозиторий не входят. Если их нет, перед воспроизведением экспериментов нужно скачать модели вручную:

```bash
source .venv/bin/activate

.venv/bin/huggingface-cli download deepvk/USER-bge-m3 --local-dir .hf_models/deepvk_USER-bge-m3
.venv/bin/huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir .hf_models/paraphrase-multilingual-MiniLM-L12-v2
.venv/bin/huggingface-cli download intfloat/multilingual-e5-small --local-dir .hf_models/multilingual-e5-small
.venv/bin/huggingface-cli download amberoad/bert-multilingual-passage-reranking-msmarco --local-dir .hf_models/bert-multilingual-passage-reranking-msmarco
```

Для generation-экспериментов и `AgenticRAG` также нужна локальная LLM через `Ollama`:

```bash
ollama pull llama3.1:8b
ollama serve
```

Основной запуск:

```bash
source .venv/bin/activate
python3 experiments/run_experiments.py
```

Скрипт сравнивает:
- `raw` vs `clean` preprocessing;
- разные размеры чанка;
- `header_recursive` vs `plain_recursive`;
- несколько embedders;
- `dense`, `BM25`, `ensemble`, `ensemble + rerank`;
- generation baseline и LLM-based ответы, если доступен локальный Ollama.

Основные артефакты сохраняются в:

```bash
experiments/results/retrieval_experiment_summary.csv
experiments/results/retrieval_experiment_details.csv
experiments/results/retrieval_tables.md
experiments/results/boundary_retrieval_analysis.csv
experiments/results/generation_experiment_summary.csv
experiments/results/generation_experiment_details.csv
experiments/results/generation_tables.md
experiments/results/experiment_run_report.json
```

`experiments/evaluate_retrieval.py` оставлен как совместимый алиас на основной экспериментальный запуск.

Отдельный запуск pilot `AgenticRAG` для пограничных вопросов:

```bash
source .venv/bin/activate
python3 experiments/agentic_rag.py
```

Артефакты `AgenticRAG`:

```bash
experiments/results/agentic_rag_analysis.csv
experiments/results/agentic_rag_analysis.md
```

Если нужно полностью воспроизвести контур с нуля, порядок действий такой:

```bash
source .venv/bin/activate
pip install -r RAG_CODE/rag/requirements.txt
pip install watchdog
brew install pandoc

.venv/bin/huggingface-cli download deepvk/USER-bge-m3 --local-dir .hf_models/deepvk_USER-bge-m3
.venv/bin/huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir .hf_models/paraphrase-multilingual-MiniLM-L12-v2
.venv/bin/huggingface-cli download intfloat/multilingual-e5-small --local-dir .hf_models/multilingual-e5-small
.venv/bin/huggingface-cli download amberoad/bert-multilingual-passage-reranking-msmarco --local-dir .hf_models/bert-multilingual-passage-reranking-msmarco

ollama pull llama3.1:8b
ollama serve
```

В другом терминале:

```bash
source .venv/bin/activate
python3 RAG_CODE/rag/prep_rag_data.py
python3 experiments/run_experiments.py
python3 experiments/agentic_rag.py
```

## 5. Датасеты и дизайн

Описание retrieval/generation наборов:

```bash
experiments/dataset_card.md
```

Краткий design doc с бизнес- и техническими целями:

```bash
docs/design_doc.md
```

## 6. Локальная LLM через Ollama

Проект ожидает OpenAI-совместимый endpoint Ollama:

```bash
export LLM_BASE_URL=http://127.0.0.1:11434/v1
export LLM_API_KEY=ollama
export LLM_MODEL=llama3.1:8b
```

Запуск сервера:

```bash
ollama serve
```

Проверка списка моделей:

```bash
ollama list
```

## 7. Запуск бота

Перед запуском задайте токен Telegram:

```bash
source .venv/bin/activate
export BOT_TOKEN='YOUR_TELEGRAM_BOT_TOKEN'
export LLM_BASE_URL='http://127.0.0.1:11434/v1'
export LLM_API_KEY='ollama'
export LLM_MODEL='llama3.1:8b'
export TOKENIZERS_PARALLELISM=false
python3 bot.py
```

## 8. Экспериментальный режим в боте

В главном меню бота доступна кнопка:

```bash
🧪 Eval retrieval
```

После нажатия можно ввести:
- произвольный вопрос;
- или `id` из `experiments/question_pool.csv`.

Бот вернёт:
- ожидаемый документ, если введён `id`;
- top-k retrieval;
- названия документов;
- секции;
- score найденных чанков;
- исходные файлы-источники.
