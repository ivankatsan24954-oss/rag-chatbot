"""
RAG-чатбот для бизнеса — MVP.

Эндпоинты:
  POST /api/documents          — загрузить документ (txt/pdf/docx)
  GET  /api/documents          — список загруженных документов
  DELETE /api/documents/{id}   — удалить документ и его чанки
  POST /api/chat                — задать вопрос боту
  GET  /api/analytics           — базовая статистика по диалогам

Запуск:
  uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import ingest
from retriever import TfidfRetriever
from llm import generate_answer

app = FastAPI(title="RAG Chatbot MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для демо; в проде — конкретный домен клиента
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = TfidfRetriever()


@app.on_event("startup")
def startup():
    db.init_db()
    _rebuild_index()


def _rebuild_index():
    """Пересобрать TF-IDF индекс после загрузки/удаления документа."""
    chunks = db.get_all_chunks()
    retriever.fit(chunks)


class ChatRequest(BaseModel):
    question: str


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = ingest.parse_file(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = ingest.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Не удалось извлечь текст из файла")

    doc_id = db.add_document(file.filename)
    db.add_chunks(doc_id, chunks)
    _rebuild_index()

    return {"id": doc_id, "filename": file.filename, "chunks_created": len(chunks)}


@app.get("/api/documents")
def get_documents():
    return db.list_documents()


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: int):
    db.delete_document(document_id)
    _rebuild_index()
    return {"status": "deleted"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    relevant_chunks = retriever.search(req.question, top_k=4)
    answer, answered = generate_answer(req.question, relevant_chunks)

    db.log_chat(
        question=req.question,
        answer=answer,
        used_chunk_ids=[c["id"] for c in relevant_chunks],
        answered=answered,
    )

    return {
        "answer": answer,
        "sources": [
            {"filename": c["filename"], "score": round(c["score"], 3)}
            for c in relevant_chunks
        ],
    }


@app.get("/api/analytics")
def analytics():
    return db.get_analytics()


# Отдаём статический демо-виджет из /frontend по корневому пути
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
