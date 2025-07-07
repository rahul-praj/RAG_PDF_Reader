# backend/app/api/endpoints.py

from fastapi import APIRouter, UploadFile, File, Form
from services.agentic_chunker import (
    save_uploaded_file,
    load_pdf,
    chunk_and_embed,
    retrieve_and_generate
)
import os

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index document for retrieval."""
    try:
        temp_path = save_uploaded_file(file)
        docs = load_pdf(temp_path)
        await chunk_and_embed(docs)
        os.remove(temp_path)
        return {"status": "success", "message": f"{file.filename} uploaded and indexed."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/query")
async def query_rag(query: str = Form(...)):
    """Query indexed documents and generate answer."""
    try:
        answer = await retrieve_and_generate(query)
        return {"answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}
