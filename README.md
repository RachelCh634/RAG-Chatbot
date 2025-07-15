## PDF RAG API

A FastAPI-based system for PDF processing and intelligent question-answering using Retrieval-Augmented Generation (RAG).

### Features

- PDF validation and processing
- Vector database storage (Pinecone/local)
- Semantic search
- AI-powered question answering
- Chat with context memory

### API Endpoints

### 1. Home
- **Endpoint:** `GET /`
- **Description:** Health check - confirms API is running
- **Response:** Status message

### 2. Validate PDF
- **Endpoint:** `POST /validate-pdf`
- **Description:** Validates uploaded PDF file format and extracts basic file information
- **Input:** PDF file upload
- **Response:** Validation status and file metadata

### 3. Upload PDF
- **Endpoint:** `POST /upload-pdf`
- **Description:** Processes PDF, extracts text, and stores vectors in database
- **Input:** PDF file upload
- **Response:** Upload confirmation with chunk and vector statistics

### 4. Search Documents
- **Endpoint:** `POST /search`
- **Description:** Performs semantic search across stored documents
- **Input:** JSON with `query` (string) and `top_k` (number of results)
- **Response:** Ranked search results with relevance scores

### 5. Ask Question
- **Endpoint:** `POST /ask`
- **Description:** Answers questions using document context and AI
- **Input:** JSON with `query` (question string)
- **Response:** AI-generated answer with confidence score and sources

### 6. Chat with Context
- **Endpoint:** `POST /chat`
- **Description:** Conversational interface with memory and document context
- **Input:** JSON with `query` (message) and `history` (conversation history)
- **Response:** Contextual AI response

## Usage Examples

### Upload a PDF
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@document.pdf"
```

### Search Documents
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}'
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main conclusion?"}'
```

### Chat
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain this concept", "history": []}'
```

## Running the API

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Components

- **PDFProcessor:** Handles PDF validation and text extraction
- **VectorService:** Manages vector storage and similarity search
- **AIService:** Provides AI-powered responses using Qwen3 model