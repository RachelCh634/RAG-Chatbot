
# Construction Estimation Chatbot

This project is a RAG-based intelligent chatbot that extracts door and window schedules from architectural floorplans in PDF format, calculates areas, and provides cost estimations using an LLM (Qwen3 via OpenRouter) with vector search (Pinecone).

---

## Tech Stack

- **FastAPI** – Backend API
- **Streamlit** – Web-based client
- **PaddleOCR** – For scanned PDF text extraction
- **OpenCLIP** – Embedding model for text chunks
- **Pinecone** – Vector database
- **LangChain** – Retrieval-Augmented Generation orchestration
- **OpenRouter (Qwen3)** – LLM provider
- **Docker** – Containerization

---

## Folder Structure

```
.
├── main.py                   # FastAPI app (PDF upload, chat endpoints)
├── streamlit_app.py          # Streamlit UI client
├── door_schedule_parser.py   # Door parsing & cost calculation logic
├── pdf_processor.py          # OCR & PDF text extraction
├── vector_service.py         # Embedding + Pinecone vector database
├── ai_service.py             # LLM integration & RAG orchestration
├── models.py                 # Data models & schemas
├── config.py                 # Configuration settings
├── docker-compose.yml        # Docker orchestration
├── Dockerfile                # Container build instructions
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

---

## Setup Instructions

### 1. Create `.env` file
```env
PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
```

### 2. Build and run the project
```bash
docker-compose up --build
```

---

## Deliverables (as per requirements)

- Working FastAPI app with OCR & vector search
- LangChain RAG pipeline with Qwen3
- Docker + `.env` config
- Streamlit interface
- Research document
- README file
- Lucidchart diagrams
- Sample PDF + test flow

---

## Troubleshooting

| Problem                          | Solution                              |
|----------------------------------|----------------------------------------|
| Pinecone not responding          | Check `.env` and index status          |
| OCR returns empty text           | Make sure PDF isn't encrypted or low-quality |
| LLM gives poor answers           | Check vector chunks or prompt formatting |
| Docker build fails               | Rebuild with `--no-cache`              |

---