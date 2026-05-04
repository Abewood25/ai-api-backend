import os
import math
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b)

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def get_context(question: str) -> str:
    try:
        with open("data.txt", "r") as f:
            docs = [line.strip() for line in f.readlines() if line.strip()]

        question_embedding = get_embedding(question)

        scored_docs = []
        for doc in docs:
            doc_embedding = get_embedding(doc)
            score = cosine_similarity(question_embedding, doc_embedding)
            scored_docs.append((score, doc))

        scored_docs.sort(reverse=True)

        top_docs = [doc for score, doc in scored_docs[:3]]

        return "\n".join(top_docs) if top_docs else "No relevant context found."

    except Exception as e:
        return f"Error loading context: {str(e)}"