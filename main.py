from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from models import (
    SearchQuery, QuestionRequest, ChatRequest,
    ValidationResponse, UploadResponse, SearchResponse, 
    AnswerResponse, ChatResponse, SearchStatusResponse
)

from pdf_processor import PDFProcessor
from vector_service import VectorService
from ai_service import AIService

app = FastAPI(title="PDF RAG API", description="API for PDF processing and Q&A")

pdf_processor = PDFProcessor()
vector_service = VectorService()
ai_service = AIService()

@app.get("/")
def home():
    return {"message": "PDF RAG API is running"}

@app.post("/validate-pdf", response_model=ValidationResponse)
async def validate_pdf(file: UploadFile = File(...)):
    """PDF Validation"""
    try:
        file_content = await file.read()
        pdf_processor.validate_file(file.filename, file_content)
        file_info = pdf_processor.get_file_info(file.filename, file_content)
        
        return ValidationResponse(
            status="success",
            message="Valid PDF file",
            file_info=file_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploading PDF to Vector Database"""
    try:
        file_content = await file.read()
        pdf_processor.validate_file(file.filename, file_content)
        full_text, num_pages = pdf_processor.extract_text_from_pdf(file_content)
        
        if isinstance(full_text, list):
            full_text = " ".join(full_text)
        elif full_text is None:
            full_text = ""
        
        result = vector_service.store_vectors(file.filename, full_text)
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {file.filename}",
            "filename": result.get("filename", file.filename),
            "chunks_stored": result.get("chunks_stored", 0),
            "total_vectors": result.get("total_vectors", 0),
            "upload_success": result.get("upload_success", True)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_query: SearchQuery):
    """Searching Documents"""
    try:
        print(f"🔍 Search request: '{search_query.query}'")
        
        results = vector_service.search_vectors(search_query.query, search_query.top_k)
        
        search_method = "pinecone" if vector_service.pinecone_available else "json_local"
        
        return SearchResponse(
            status="success",
            message="Search completed",
            query=search_query.query,
            results=results,
            total_found=len(results),
            search_method=search_method
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Smart question and answer"""
    try:
        print(f"❓ Question: '{request.query}'")
        search_results = vector_service.search_vectors(request.query, top_k=3)
        
        if not search_results:
            return AnswerResponse(
                status="success",
                query=request.query,
                answer="I couldn't find relevant information in the document to answer your question.",
                confidence="low",
                model_used="qwen3",
                sources_used=0,
                sources=[]
            )
        
        context_chunks = [result['text'] for result in search_results]
        answer = ai_service.generate_answer_from_context(request.query, context_chunks)
        sources = [
            {
                "filename": result['filename'],
                "chunk_index": result['chunk_index'],
                "relevance_score": round(result['score'], 3)
            }
            for result in search_results
        ]
        
        confidence = ai_service.determine_confidence(search_results[0]['score'])
        
        return AnswerResponse(
            status="success",
            query=request.query,
            answer=answer,
            confidence=confidence,
            model_used="qwen3",
            sources_used=len(search_results),
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_context(request: ChatRequest):
    """Chat with context memory"""
    try:
        print(f"💬 Chat: '{request.query}'")
        
        search_results = vector_service.search_vectors(request.query, top_k=3)
        
        if not search_results:
            return ChatResponse(
                query=request.query,
                answer="I don't have information about that in the document.",
                model_used="qwen3",
                context_used=0,
                relevance_score=0.0
            )
        
        context = "\n\n".join([result['text'] for result in search_results[:2]])
        answer = ai_service.chat_with_context(request.query, context, request.history)
        
        return ChatResponse(
            query=request.query,
            answer=answer,
            model_used="qwen3",
            context_used=len(search_results),
            relevance_score=round(search_results[0]['score'], 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)