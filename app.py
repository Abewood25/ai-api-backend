from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
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

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"OpenAI error: {str(e)}"

    return {
        "question": request.question,
        "context_used": context,
        "answer": answer
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        with open("uploaded.pdf", "wb") as f:
            f.write(contents)

        reader = PdfReader("uploaded.pdf")

        extracted_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                extracted_text += text + "\n"

        with open("data.txt", "w", encoding="utf-8") as f:
            f.write(extracted_text)

        return {
            "message": "PDF uploaded and processed successfully"
        }

    except Exception as e:
        return {
            "error": str(e)
        }