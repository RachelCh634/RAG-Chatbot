from fastapi import FastAPI, File, UploadFile, HTTPException
from models import ChatRequest, ChatResponse
from pdf_processor import PDFProcessor
from vector_service import VectorService
from ai_service import AIService  
from door_schedule_parser import parse_doors_dynamic

app = FastAPI(title="PDF RAG API", description="API for PDF processing and Q&A")

server_ready = False
pdf_processor = PDFProcessor()
vector_service = VectorService()
ai_service = AIService(vector_service) 
server_ready = True

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploading PDF to Vector Database with improved table extraction"""
    try:
        file_content = await file.read()
        pdf_processor.validate_file(file.filename, file_content)
        
        # שימוש בשיטה המשופרת
        full_text, extracted_tables = pdf_processor.extract_text_and_tables(file_content, force_ocr=True)
        
        print(f"Extracted text length: {len(full_text)}")
        print(f"Number of tables found: {len(extracted_tables)}")
        
        # הדפסת מידע על הטבלאות
        for i, table in enumerate(extracted_tables):
            print(f"Table {i+1}: {table.shape[0]} rows × {table.shape[1]} columns")
        
        if isinstance(full_text, list):
            full_text = " ".join(full_text)
        elif full_text is None:
            full_text = ""
        
        # פרסור דלתות
        door_result = parse_doors_dynamic(full_text)  
        doors = door_result.get("doors", [])
        door_summary = door_result.get("summary", {})
        print(f"Found {len(doors)} doors")
        print(f"Total cost: {door_summary.get('total_cost', 0)}")
        
        # הכנת טקסט מורחב עם דלתות
        if doors:
            door_texts = [f"Door {d['id']}: {d['size']} {d['material']} {d['operation']}" for d in doors]
            combined_text = full_text + "\n\n" + "\n".join(door_texts)
        else:
            combined_text = full_text
        
        # שמירה ב-vector database
        result = vector_service.store_vectors(file.filename, combined_text)
        
        # הכנת תקציר טבלאות למענה
        tables_summary = []
        for i, table in enumerate(extracted_tables):
            tables_summary.append({
                "table_id": i + 1,
                "rows": table.shape[0],
                "columns": table.shape[1],
                "column_names": list(table.columns)[:5],  # רק 5 הראשונות
                "preview": table.head(3).to_dict('records') if not table.empty else []
            })
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {file.filename}",
            "filename": result.get("filename", file.filename),
            "chunks_stored": result.get("chunks_stored", 0),
            "total_vectors": result.get("total_vectors", 0),
            "upload_success": result.get("upload_success", True),
            "doors_found": len(doors),
            "door_summary": door_summary,
            "doors": doors[:5] if doors else [],
            "tables_found": len(extracted_tables),
            "tables_summary": tables_summary[:3] 
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

@app.get("/healthcheck")
async def health_check():
    """Check if server is initialized and ready"""
    if server_ready:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Server is not ready yet")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)