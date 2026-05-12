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
            docs = f.readlines()

        for doc in docs:
            if any(word.lower() in doc.lower() for word in question.split()):
                return doc.strip()

        return "No relevant context found."

    except Exception as e:
        return f"Error loading context: {str(e)}"