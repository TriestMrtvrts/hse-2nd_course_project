# HSE RAG Bot

Telegram-бот для работы с нормативными документами НИУ ВШЭ на основе retrieval-augmented generation.

В проекте используются:
- база знаний на SQLite + `sqlite-vec`
- эмбеддинги `deepvk/USER-bge-m3`
- LLM через Ollama
- Telegram-бот на `aiogram`


## 1. Установка окружения

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

## 2. Обновление и загрузка данных

Исходные документы для индексации лежат в:

```bash
RAG_CODE/rag/data/
```

Сконвертированные markdown-файлы сохраняются в:

```bash
RAG_CODE/rag/converted_md/
```

Векторная база знаний:

```bash
RAG_CODE/rag/hse.sqlite3
```

### Полная пересборка базы знаний

Если вы добавили или заменили документы и хотите собрать базу заново:

```bash
source .venv/bin/activate
rm -f RAG_CODE/rag/hse.sqlite3
python3 RAG_CODE/rag/prep_rag_data.py
```

### Обновление базы при добавлении новых файлов

Если нужен режим отслеживания новых документов в папке `data/`:

```bash
source .venv/bin/activate
python3 update_db.py
```

## 3. Запуск модели через Ollama

Убедитесь, что `ollama` установлена:

```bash
ollama --version
```

Запустите Ollama:

```bash
ollama serve
```

В отдельном терминале загрузите модель, например:

```bash
ollama pull llama3.1:8b
```

По умолчанию проект ожидает:

```bash
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.1:8b
```

## 4. Запуск бота

Перед запуском задайте Telegram-токен и параметры LLM:

```bash
source .venv/bin/activate

export BOT_TOKEN='YOUR_TELEGRAM_BOT_TOKEN'
export LLM_BASE_URL='http://127.0.0.1:11434/v1'
export LLM_API_KEY='ollama'
export LLM_MODEL='llama3.1:8b'
export TOKENIZERS_PARALLELISM=false
```

После этого запустите бота:

```bash
python3 bot.py
```

## 5. Оценка retrieval

Тестовый набор вопросов лежит в:

```bash
experiments/question_pool.csv
```

Чтобы прогнать автоматическую оценку retrieval:

```bash
source .venv/bin/activate
python3 experiments/evaluate_retrieval.py
```

Результаты будут сохранены в:

```bash
experiments/results/retrieval_eval_results.csv
experiments/results/retrieval_eval_summary.json
```

## 6. Экспериментальный режим в боте

В главном меню бота доступна кнопка:

```bash
🧪 Eval retrieval
```

После нажатия бот попросит ввести:
- произвольный вопрос для retrieval-проверки;
- или `id` из `experiments/question_pool.csv`.

В ответ бот показывает:
- ожидаемый документ, если введён `id` из тестового набора;
- top-k retrieval;
- названия документов;
- секции;
- distance для найденных чанков;
- исходные файлы, использованные как источник.

Также по-прежнему доступна slash-команда:

```bash
/eval <вопрос>
```

или

```bash
/eval <id_из_question_pool.csv>
```
