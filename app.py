from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_context
from openai import APIConnectionError, APIStatusError, RateLimitError
import os


# ✅ DEFINE APP FIRST
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Then define models
class AskRequest(BaseModel):
    question: str

# ✅ Then OpenAI client
client = OpenAI()

# ✅ Then routes
@app.get("/openai-test")
def openai_test():
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )
        return {"ok": True, "text": resp.choices[0].message.content}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    
@app.get("/")
def root():
    return {"message": "AI API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask(request: AskRequest):
    context = get_context(request.question)

    try:
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

