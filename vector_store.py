import os
import requests
import chromadb
import uuid

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")


def get_embedding(text: str):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "text-embedding-3-small",
            "input": text,
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def add_chunks(chunks, filename: str):
    embeddings = [get_embedding(chunk) for chunk in chunks]

    ids = [f"{filename}_{uuid.uuid4()}" for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": filename} for _ in chunks],
    )


def search_chunks(query, filename: str | None = None):
    query_embedding = get_embedding(query)

    where_filter = {"source": filename} if filename else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where=where_filter,
    )

    return results["documents"][0]


def list_documents():
    results = collection.get()

    sources = {}

    for metadata in results.get("metadatas", []):
        if metadata and "source" in metadata:
            source = metadata["source"]
            sources[source] = sources.get(source, 0) + 1

    return [
        {"filename": filename, "chunks": count}
        for filename, count in sources.items()
    ]