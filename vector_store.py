import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.Client()

collection = client.get_or_create_collection("documents")

model = SentenceTransformer("all-MiniLM-L6-v2")

def add_chunks(chunks):
    embeddings = model.encode(chunks).tolist()

    ids = [f"id_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

def search_chunks(query):
    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results["documents"][0]