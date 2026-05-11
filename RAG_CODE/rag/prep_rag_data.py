import sqlite3
import os
import re
from typing import List

import sqlite_vec
from sqlite_vec import serialize_float32
from tqdm import tqdm
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

try:
    from .anything_to_md import convert_file_to_markdown
    from .model_utils import DEFAULT_EMBEDDING_MODEL, get_embedding_model_name_or_path
except ImportError:
    from anything_to_md import convert_file_to_markdown  # type: ignore
    from model_utils import DEFAULT_EMBEDDING_MODEL, get_embedding_model_name_or_path  # type: ignore

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "converted_md")
DB_NAME = os.path.join(BASE_DIR, "hse.sqlite3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
PREPROCESSING_PROFILE = os.getenv("PREPROCESSING_PROFILE", "clean")
CHUNKING_METHOD = os.getenv("CHUNKING_METHOD", "header_recursive")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "256"))
HEADER_LEVELS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def extract_header_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped.startswith("**") and stripped.endswith("**"):
            return stripped.strip("* ").strip()
        if re.match(r"^\d+(\.\d+)*\.?\s+\*\*.+\*\*$", stripped):
            return re.sub(r"^\d+(\.\d+)*\.?\s+", "", stripped).strip("* ").strip()
        if re.match(r"^\d+(\.\d+)*\.?\s+\S+", stripped):
            return stripped
    return "Full Document"

def preprocess_markdown(text: str, profile: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\u00a0", " ")
    if profile == "raw":
        return cleaned.strip()

    cleaned = re.sub(r"\[\]\{#[^}]+\}", "", cleaned)
    cleaned = re.sub(r"^\[\^[^\]]+\]:.*$", "", cleaned, flags=re.MULTILINE)

    filtered_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if stripped.startswith("[") and "](#" in stripped:
            continue
        filtered_lines.append(line.rstrip())
    cleaned = "\n".join(filtered_lines)

    trim_markers = [
        "# Используемые понятия и сокращения",
        "**ПРАВИЛА ВНУТРЕННЕГО РАСПОРЯДКА ОБУЧАЮЩИХСЯ**",
        "## I. Общие положения",
    ]
    for marker in trim_markers:
        position = cleaned.find(marker)
        if position != -1:
            cleaned = cleaned[position:]
            break

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_numbered_sections(data: str) -> List[Document]:
    section_heading_pattern = re.compile(
        r"^\s*(\d+\.\s+\*\*.+\*\*|\*\*\d+\.\s+.+\*\*)\s*$"
    )
    sections = []
    current_lines = []
    current_header = None

    for line in data.splitlines():
        stripped = line.strip()
        if section_heading_pattern.match(stripped):
            if current_lines:
                sections.append(
                    Document(
                        page_content="\n".join(current_lines).strip(),
                        metadata={"header": current_header or extract_header_from_text("\n".join(current_lines))},
                    )
                )
            current_lines = [line]
            current_header = extract_header_from_text(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            Document(
                page_content="\n".join(current_lines).strip(),
                metadata={"header": current_header or extract_header_from_text("\n".join(current_lines))},
            )
        )
    return [section for section in sections if section.page_content]


def split_markdown_sections(data: str) -> List[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(HEADER_LEVELS, strip_headers=False)
    header_splits = markdown_splitter.split_text(data)
    if header_splits and (len(header_splits) > 1 or any(document.metadata for document in header_splits)):
        return header_splits
    numbered_sections = split_numbered_sections(data)
    if len(numbered_sections) > 1:
        return numbered_sections
    return [Document(page_content=data, metadata={"header": extract_header_from_text(data)})]


def create_chunks(
    data: str,
    chunking_method: str = CHUNKING_METHOD,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    if chunking_method == "section_only":
        return split_markdown_sections(data)

    if chunking_method == "plain_recursive":
        base_documents = [Document(page_content=data, metadata={"header": extract_header_from_text(data)})]
    else:
        base_documents = split_markdown_sections(data)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for base_document in base_documents:
        for chunk_text in text_splitter.split_text(base_document.page_content):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=dict(base_document.metadata),
                )
            )
    return chunks

def setup_database(db):
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        meta_data_h TEXT,
        meta_data_source TEXT
    );
    """)

    db.execute("""
    CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        text TEXT,
        meta_data_h TEXT,
        meta_data_source TEXT,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    );
    """)
    db.commit()

def create_embeddings_table(db, embedding_size):
    db.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
        id INTEGER PRIMARY KEY,
        embedding FLOAT[{embedding_size}]
    );
    """)
    db.commit()

def save_chunks(db, chunks: List[str], meta_data: List[dict], model, document_id: int):
    try:
        chunk_embeddings = list(model.encode(chunks, normalize_embeddings=True))
        for chunk, embedding, meta in zip(chunks, chunk_embeddings, meta_data):
            result = db.execute(
                "INSERT INTO chunks(document_id, text, meta_data_h, meta_data_source) VALUES(?, ?, ?, ?)", 
                (
                    document_id,
                    chunk,
                    meta.get("header", "Full Document"),
                    meta.get("source", ""),
                )
            )
            chunk_id = result.lastrowid
            db.execute(
                "INSERT INTO chunk_embeddings(id, embedding) VALUES (?, ?)",
                [chunk_id, serialize_float32(embedding)],
            )
        db.commit()
    except Exception as exc:
        print(f"Failed to save chunks for document_id={document_id}: {exc}")
        raise

def main():
    try:
        # Directory containing files to process
        input_dir = SOURCE_DIR
        output_dir = OUTPUT_DIR

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize SentenceTransformer model
        model = SentenceTransformer(get_embedding_model_name_or_path())

        # Setup database
        db_path = os.path.abspath(DB_NAME)
        db = sqlite3.connect(DB_NAME)
        setup_database(db)

        # Process each file in the input directory
        for root, _, files in os.walk(input_dir):
            visible_files = [file for file in files if not file.startswith('.')]
            for file in tqdm(visible_files, desc="Processing files"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_path_dir):
                    os.makedirs(output_path_dir)

                output_file = os.path.splitext(file)[0] + '.md'
                output_path = os.path.join(output_path_dir, output_file)

                # Convert file to markdown
                try:
                    convert_file_to_markdown(input_path, output_path)
                except Exception as exc:
                    print(f"Failed to convert {input_path}: {exc}")
                    continue

                # Read markdown file
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        data = preprocess_markdown(f.read(), PREPROCESSING_PROFILE)
                except Exception as exc:
                    print(f"Failed to read {output_path}: {exc}")
                    continue

                # Create chunks
                splits = create_chunks(
                    data,
                    chunking_method=CHUNKING_METHOD,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )

                # Create embeddings
                try:
                    embeddings = model.encode([chunk.page_content for chunk in splits], normalize_embeddings=True)
                except Exception as exc:
                    print(f"Failed to create embeddings for {input_path}: {exc}")
                    continue

                # Create embeddings table if not exists
                if root == input_dir and file == visible_files[0]:  # Assuming embedding size is consistent
                    if len(embeddings) > 0 and hasattr(embeddings[0], 'shape'):
                        embedding_size = embeddings[0].shape[0]
                    else:
                        embedding_size = len(embeddings[0])
                    create_embeddings_table(db, embedding_size)

                # Insert document and get document_id
                meta_document = {
                    "source": input_path,
                    "description": "Full Document"
                }
                try:
                    cursor = db.execute(
                        "INSERT INTO documents(text, meta_data_h, meta_data_source) VALUES(?, ?, ?)", 
                        (
                            data,
                            meta_document.get("description", "Full Document"),
                            meta_document.get("source", ""),
                        )
                    )
                    document_id = cursor.lastrowid
                    db.commit()
                except Exception as exc:
                    print(f"Failed to insert document {input_path}: {exc}")
                    continue

                # Prepare metadata for chunks
                try:
                    chunks_text = [chunk.page_content for chunk in splits]
                    chunks_meta = []
                    for chunk in splits:
                        header = (
                            chunk.metadata.get("Header 3")
                            or chunk.metadata.get("Header 2")
                            or chunk.metadata.get("Header 1")
                            or chunk.metadata.get("header")
                            or extract_header_from_text(chunk.page_content)
                        )
                        chunks_meta.append({"source": input_path, "header": header})
                except Exception as exc:
                    print(f"Failed to prepare chunks for {input_path}: {exc}")
                    continue

                # Save chunks and their embeddings
                save_chunks(db, chunks_text, chunks_meta, model, document_id)
                print(f"Indexed {file} with {len(chunks_text)} chunks")

    except Exception as exc:
        print(f"prep_rag_data failed: {exc}")
        raise
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    main()
