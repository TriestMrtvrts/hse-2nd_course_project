import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = PROJECT_ROOT / ".hf_models"

DEFAULT_EMBEDDING_MODEL = "deepvk/USER-bge-m3"
DEFAULT_RERANKER_MODEL = "amberoad/bert-multilingual-passage-reranking-msmarco"

MODEL_ALIASES = {
    "deepvk/USER-bge-m3": LOCAL_MODEL_DIR / "deepvk_USER-bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": LOCAL_MODEL_DIR / "paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small": LOCAL_MODEL_DIR / "multilingual-e5-small",
    "amberoad/bert-multilingual-passage-reranking-msmarco": LOCAL_MODEL_DIR / "bert-multilingual-passage-reranking-msmarco",
}


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    if not model_name_or_path:
        return model_name_or_path

    explicit_path = Path(model_name_or_path).expanduser()
    if explicit_path.exists():
        return str(explicit_path)

    alias_path = MODEL_ALIASES.get(model_name_or_path)
    if alias_path and alias_path.exists():
        return str(alias_path)

    guessed_local_path = LOCAL_MODEL_DIR / model_name_or_path.replace("/", "_")
    if guessed_local_path.exists():
        return str(guessed_local_path)

    return model_name_or_path


def get_embedding_model_name_or_path() -> str:
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return resolve_model_name_or_path(model_name)


def get_reranker_model_name_or_path() -> str:
    model_name = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    return resolve_model_name_or_path(model_name)
