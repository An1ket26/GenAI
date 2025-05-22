# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.environ["GOOGLE_API_KEY"]

# loader = DirectoryLoader(path="Books", glob="*.pdf", loader_cls=PyPDFLoader)

# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
#     add_start_index=True,
# )
# split_docs = text_splitter.split_documents(docs)

# embed_model = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004", google_api_key=api_key
# )


# vectorstore = Chroma(
#     collection_name="DataEng",
#     embedding_function=embed_model,
#     persist_directory="./my_chroma_db",
# )

# vectorstore.add_documents(documents=split_docs)


import os
import json
import hashlib
from collections import defaultdict
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Load environment variables
load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"]

# Path
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "Books")
HASH_STORE_PATH = os.getenv("HASH_STORE_PATH", "processed_files.json")


# Initialize embedding model
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=api_key
)

# Initialize vectorstore
vectorstore = Chroma(
    collection_name="DataEng",
    embedding_function=embed_model,
    persist_directory="./my_chroma_db",
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
    loader = DirectoryLoader(path=DOCUMENTS_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
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
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    return text_splitter.split_documents(documents)


# === Main Workflow === #


def main():
    processed_files = load_processed_files()
    new_docs = load_documents(processed_files)

    if not new_docs:
        print("No new or changed documents to process.")
        return
    #testing
    # for doc in new_docs:
    #     print(f"Path: {doc.metadata['source']}")
    #     print(f"Content (first 300 chars):\n{doc.page_content[:3000]}")
    split_docs = split_documents(new_docs)

    # Add file path info explicitly to metadata for future delete operations
    for doc in split_docs:
        doc.metadata["source"] = doc.metadata.get("source", "unknown")

    vectorstore.add_documents(split_docs)
    save_processed_files(processed_files)

    print(f"Successfully added {len(split_docs)} new chunks to vectorstore.")


if __name__ == "__main__":
    main()
