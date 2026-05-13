
import math


def get_context(question: str) -> str:
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            docs = f.readlines()

        for doc in docs:
            if any(word.lower() in doc.lower() for word in question.split()):
                return doc.strip()

        return "No relevant context found."

    except Exception as e:
        return f"Error loading context: {str(e)}"