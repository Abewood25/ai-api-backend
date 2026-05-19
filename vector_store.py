import os
import requests
import chromadb

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

def add_chunks(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
    )

def search_chunks(query):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    return results["documents"][0]