from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import uvicorn
import os
import tempfile
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from docx import Document

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)
documents = []

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("page-1-uploader.html", {"request": request})

@app.post("/api/upload")
async def upload_file(document: UploadFile = File(...)):
    import tempfile
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, document.filename)

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(document.file, f)

    def extract_docx_text(path):
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    text = extract_docx_text(tmp_path)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)

    global documents
    documents.extend(chunks)
    index.add(np.array(embeddings, dtype="float32"))
    return JSONResponse({"status": "uploaded"})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    query_embedding = model.encode([request.message])[0]
    D, I = index.search(np.array([query_embedding], dtype="float32"), k=3)
    retrieved = [documents[i] for i in I[0]]

    context = "\n\n".join(retrieved)
    reply = f"Context:\n{context}\n\nAnswer: This is a simulated response based on retrieved context."
    return JSONResponse({"reply": reply})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True)
