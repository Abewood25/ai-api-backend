def get_context(question: str) -> str:
    try:
        with open("data.txt", "r") as f:
            docs = f.readlines()

        best_match = ""
        best_score = 0

        for doc in docs:
            score = sum(word.lower() in doc.lower() for word in question.split())
            if score > best_score:
                best_score = score
                best_match = doc.strip()

        return best_match if best_match else "No relevant context found."

    except Exception as e:
        return f"Error loading context: {str(e)}"