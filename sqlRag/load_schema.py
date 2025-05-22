import os
import re
import json
import hashlib
from dotenv import load_dotenv
from langchain_chroma import Chroma
from collections import defaultdict
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader

# Load environment variables
load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"]

# Path
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR2", "sqlRag/Docs")
HASH_STORE_PATH = os.getenv("HASH_STORE_PATH2", "sqlRag/processed_database.json")


# Initialize embedding model
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=api_key
)

# Initialize vectorstore
vectorstore = Chroma(
    collection_name="SqlSchema",
    embedding_function=embed_model,
    persist_directory="./schema_db",
)

# === Helper Functions === #


def get_file_hash(file_path):
    """Generate MD5 hash for a given file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_processed_files():
    """Load existing hash record from disk."""
    if os.path.exists(HASH_STORE_PATH):
        try:
            with open(HASH_STORE_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Failed to parse processed_files.json. Starting fresh.")
            return {}
    return {}


def save_processed_files(data):
    """Persist hash record to disk."""
    with open(HASH_STORE_PATH, "w") as f:
        json.dump(data, f, indent=4)


def load_documents(processed_files):
    """Load and filter documents based on file hash (per file, not per page)."""
    loader = DirectoryLoader(
        path=DOCUMENTS_DIR, glob="*.docx", loader_cls=Docx2txtLoader
    )
    docs = loader.load()

    # Group documents by source file path
    file_to_docs = defaultdict(list)
    for doc in docs:
        file_path = os.path.normpath(doc.metadata["source"])
        file_to_docs[file_path].append(doc)

    new_docs = []

    for file_path, doc_pages in file_to_docs.items():
        file_hash = get_file_hash(file_path)

        if file_path not in processed_files:
            print(f"New file found: {file_path}")
            new_docs.extend(doc_pages)
            processed_files[file_path] = file_hash

        elif processed_files[file_path] != file_hash:
            print(f"File updated: {file_path}, re-indexing...")
            vectorstore.delete(where={"source": file_path})
            new_docs.extend(doc_pages)
            processed_files[file_path] = file_hash

    return new_docs


def split_documents(documents):
    """Custom split: Each table section becomes a separate document chunk."""
    table_docs = []

    for doc in documents:
        raw_text = doc.page_content
        file_source = doc.metadata.get("source", "unknown")

        table_splits = re.split(
            r"(?:^|\n)(Table: .+?)(?=\nTable: |\Z)", raw_text, flags=re.DOTALL
        )

        for section in table_splits:
            section = section.strip()
            if not section or not section.startswith("Table:"):
                continue

            match = re.match(r"Table:\s*(\w+)", section)
            table_name = match.group(1) if match else "unknown_table"

            table_docs.append(
                Document(
                    page_content=section,
                    metadata={"source": file_source, "table": table_name},
                )
            )

    return table_docs


def clean_table_text(doc: Document) -> Document:
    """Normalize the page_content formatting of a schema Document."""
    text = doc.page_content

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n+", "\n", text)
    lines = text.split("\n")

    merged_lines = []

    for i in range(len(lines)):
        line = lines[i].strip()
        if line.endswith(":") or len(line) > 80:
            merged_lines.append(line)
        else:
            # Combine with previous if it's short and not a section
            if merged_lines:
                merged_lines[-1] += " " + line
            else:
                merged_lines.append(line)
    cleaned_text = "\n".join(merged_lines)

    return Document(page_content=cleaned_text, metadata=doc.metadata)


# === Main Workflow === #


def main():
    processed_files = load_processed_files()
    new_docs = load_documents(processed_files)

    if not new_docs:
        print("No new or changed documents to process.")
        return
    split_docs = split_documents(new_docs)
    cleaned_docs = [clean_table_text(doc) for doc in split_docs]

    # Add file path info explicitly to metadata for future delete operations
    for doc in cleaned_docs:
        doc.metadata["source"] = doc.metadata.get("source", "unknown")

    vectorstore.add_documents(cleaned_docs)
    save_processed_files(processed_files)

    print(f"Successfully added {len(cleaned_docs)} new chunks to vectorstore.")


if __name__ == "__main__":
    main()
