from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import uvicorn
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from docx import Document
import tempfile
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

def search_documents(query: str) -> str:
    query_embedding = model.encode([query])[0]
    # Reduce k to 2 or 3 to limit the amount of text
    D, I = index.search(np.array([query_embedding], dtype="float32"), k=3)
    results = [documents[i] for i in I[0]]
    # Optionally, truncate each chunk to a max length
    max_chunk_length = 1000  # characters
    results = [chunk[:max_chunk_length] for chunk in results]
    return "\n\n".join(results) if results else "No relevant content found."

from langchain.agents import Tool

tools = [
    Tool(
        name="DocumentSearch",
        func=search_documents,
        description="Searches the uploaded documents for relevant information."
    )
]

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Enable CORS for all domains for simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates directory for serving the HTML UI
templates = Jinja2Templates(directory="templates")

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize a FAISS index with L2 (Euclidean) distance
index = faiss.IndexFlatL2(384)

# Store document chunks for retrieval
documents = []

# Define the expected format for chat requests
class ChatRequest(BaseModel):
    message: str

# Set up memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", k=3)  # Only keep last 3 exchanges

# Set up OpenAI chat model with API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# Create agent using initialize_agent (not wrapping in AgentExecutor manually)
agent_executor = initialize_agent(
    tools=tools,  # <-- Now using your tool
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Serve the HTML UI
@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("page-1-uploader.html", {"request": request})

# Upload and process a document
@app.post("/api/upload")
async def upload_file(document: UploadFile = File(...)):
    # Save uploaded file to a temporary location
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, document.filename)

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(document.file, f)

    # Extract text content from a .docx file
    def extract_docx_text(path):
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    # Read and split document text into chunks
    text = extract_docx_text(tmp_path)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)

    # Store chunks and their embeddings
    global documents
    documents.extend(chunks)
    index.add(np.array(embeddings, dtype="float32"))
    return JSONResponse({"status": "uploaded"})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Use the agent_executor to process the user's message
    response = agent_executor.run(request.message)
    return JSONResponse({"reply": response})

# Define route for chat/query endpoint
@app.post("/api/chat-old")
async def chat(request: ChatRequest):
    # Use the raw user message as the query
    query = f"{request.message}"

    # Create embedding for the user's query
    query_embedding = model.encode([query])[0]

    # Search the FAISS index for the 10 most similar document chunks
    D, I = index.search(np.array([query_embedding], dtype="float32"), k=10)

    # Retrieve the actual text chunks using the indices
    retrieved = [documents[i] for i in I[0]]

    # Filter to include only chunks containing 'functional requirement'
    filtered = [chunk for chunk in retrieved if 'functional requirement' in chunk.lower()]

    # Join filtered chunks or fallback to message if none found
    context = "\n\n".join(filtered) if filtered else "No specific functional requirements found."

    # Generate a simulated response using filtered context
    reply = f"Context:\n{context}\n\nAnswer: This is a simulated response filtered for Functional Requirements."

    # Return the response as JSON
    return JSONResponse({"reply": reply})

# Start the FastAPI app using Uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True)
