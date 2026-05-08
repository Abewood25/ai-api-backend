from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_context
from openai import OpenAI
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
@app.get("/")
def root():
    return {"message": "AI API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask(request: AskRequest):
    context = get_context(request.question)

    print(f"QUESTION: {request.question}")
    print(f"CONTEXT USED: {context}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Use the provided context to answer the question. If unsure, say you don’t know."
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

