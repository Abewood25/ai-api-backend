from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from rag import get_context

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

class AskRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "AI API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/openai-test")
def openai_test():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say hello"}
            ]
        )

        return {
            "ok": True,
            "response": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }

@app.post("/ask")
def ask(request: AskRequest):
    try:
        context = get_context(request.question)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Use the provided context to answer the question."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {request.question}"
                }
            ]
        )

        return {
            "question": request.question,
            "context_used": context,
            "answer": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "error": str(e)
        }

