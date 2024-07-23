import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
import os 
from dotenv import load_dotenv
load_dotenv()


def main(CHROMA_PATH, DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Restart the Database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing DB")
        clear_database(CHROMA_PATH)

    documents = load_documents(DATA_PATH)
    chunks = split_documents(CHUNK_SIZE, CHUNK_OVERLAP, documents)
    print("Splitting Documents Done...")
    print("\n")
    add_to_chroma(chunks, CHROMA_PATH)

def load_documents(DATA_PATH:str):
    print("Loading Documents...")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    print("Loading Documents Done...")
    print("\n")
    return document_loader.load()

def split_documents(CHUNK_SIZE:int, CHUNK_OVERLAP: int, documents: list[Document]):
    print("Splitting Documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], CHROMA_PATH: str):
    print("Adding Documents to DB...")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding New Documents: {len(new_chunks)}")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunks_ids)
        db.persist()

    else:
        print("No new document to add")
        print("\n")

    print("Added to DB...")
    print("\n")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database(CHROMA_PATH:str):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    CHROMA_PATH = os.getenv("CHROMA_PATH")
    DATA_PATH = os.getenv("DATA_PATH")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
    main(CHROMA_PATH, DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP)