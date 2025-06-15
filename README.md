# Document Chat Application

A FastAPI-based application that allows users to upload documents, process them, and ask questions about their content using embeddings and similarity search.

## Features

- Upload DOCX documents for processing
- Extract text from documents
- Generate embeddings from text chunks
- Query documents using natural language
- Interactive chat interface

## Requirements

- Python 3.8+
- FastAPI
- Sentence Transformers
- FAISS
- LangChain
- And other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/user-stories-agentic-ai.git
cd user-stories-agentic-ai
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
user-stories-agentic-ai/
├── main.py                # Main FastAPI application
├── requirements.txt       # Project dependencies
├── templates/             # HTML templates
│   └── page-1-uploader.html  # Upload interface
└── uploads/               # Directory for uploaded files (created at runtime)
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn main:app --host 127.0.0.1 --port 9001 --reload
```

2. Open your browser and navigate to:
```
http://127.0.0.1:9001
```

If port 9001 is already in use, you can use a different port:
```bash
uvicorn main:app --host 127.0.0.1 --port 9002 --reload
```

## Using the Application

1. **Upload a Document**:
   - Click on the file input field to select a DOCX file
   - Click the "Upload" button
   - Wait for the confirmation message

2. **Chat with Your Document**:
   - Type a question in the input field at the bottom of the page
   - Press Enter or click "Send"
   - View the AI's response based on the document content

## API Endpoints

- `GET /`: Main application interface
- `POST /api/upload`: Upload and process a document
- `POST /api/chat`: Send a message and get a response

## Troubleshooting

- If you get a "socket address already in use" error, try using a different port
- Ensure you have all dependencies installed correctly
- Check that the `templates` directory contains the required HTML files

## Dependencies

The main dependencies for this project are:
- fastapi==0.110.0
- uvicorn==0.29.0
- sentence-transformers==2.6.1
- faiss-cpu==1.7.4
- langchain==0.1.20
- numpy==1.26.4
- python-multipart==0.0.7
- python-docx==0.8.11