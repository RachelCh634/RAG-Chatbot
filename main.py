from fastapi import FastAPI, File, UploadFile, HTTPException
from models import ChatRequest, ChatResponse
from pdf_processor import PDFProcessor
from vector_service import VectorService
from ai_service import AIService  
from door_schedule_parser import parse_door_schedule, create_door_embeddings_text, generate_door_summary

app = FastAPI(title="PDF RAG API", description="API for PDF processing and Q&A")

pdf_processor = PDFProcessor()
vector_service = VectorService()
ai_service = AIService(vector_service)  

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploading PDF to Vector Database"""
    try:
        file_content = await file.read()
        pdf_processor.validate_file(file.filename, file_content)
        full_text, num_pages = pdf_processor.extract_text(file_content)
        
        if isinstance(full_text, list):
            full_text = " ".join(full_text)
        elif full_text is None:
            full_text = ""
        
        doors = parse_door_schedule(full_text)
        door_summary = generate_door_summary(doors) if doors else None
        if doors:
            door_embeddings = create_door_embeddings_text(doors)
            combined_text = full_text + "\n\n" + "\n".join(door_embeddings)
        else:
            combined_text = full_text
            door_embeddings = []
        
        result = vector_service.store_vectors(file.filename, combined_text)
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {file.filename}",
            "filename": result.get("filename", file.filename),
            "chunks_stored": result.get("chunks_stored", 0),
            "total_vectors": result.get("total_vectors", 0),
            "upload_success": result.get("upload_success", True),
            "doors_found": len(doors),
            "door_summary": door_summary,
            "doors": doors[:5] if doors else []  
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_context(request: ChatRequest):
    """Chat with context memory using LangChain"""
    try:
        print(f"💬 LangChain Chat: '{request.query}'")
        
        search_results = vector_service.search_vectors(request.query, top_k=3)
        
        if not search_results:
            return ChatResponse(
                query=request.query,
                answer="I don't have information about that in the document.",
                model_used="qwen3_langchain",
                context_used=0,
                relevance_score=0.0
            )
        
        context = "\n\n".join([result['text'] for result in search_results[:2]])
        answer = ai_service.chat_with_context(request.query, context)
        
        return ChatResponse(
            query=request.query,
            answer=answer,
            model_used="qwen3_langchain",
            context_used=len(search_results),
            relevance_score=round(search_results[0]['score'], 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain chat failed: {str(e)}")


@app.post("/clear-memory")
async def clear_conversation_memory():
    """Clear conversation memory"""
    try:
        ai_service.clear_memory()
        return {"status": "success", "message": "Conversation memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


@app.get("/conversation-history")
async def get_conversation_history():
    """Get conversation history"""
    try:
        history = ai_service.get_conversation_history()
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.post("/clear_all_vectors")
def clear_all_vectors():
    vector_service.clear_all_vectors()
    return {"status": "success", "message": "All vectors cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)