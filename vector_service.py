import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

class VectorService:
    """Unified Vector Management Service with Pinecone and JSON fallback"""
    
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        
        # Try to initialize Pinecone
        self.pinecone_available = False
        self.pc = None
        self.index = None
        self.index_name = "pdf-rag-index"
        
        self._setup_pinecone()
    
    def _setup_pinecone(self):
        """Setup Pinecone connection"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                print("PINECONE_API_KEY not found in environment variables")
                return
            
            self.pc = Pinecone(api_key=api_key)
            
            # Check if index exists, create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print("Waiting for index to be ready...")
                time.sleep(10)
            
            self.index = self.pc.Index(self.index_name)
            self.pinecone_available = True
            print(f"Pinecone connected to index: {self.index_name}")
            
        except Exception as e:
            print(f"Pinecone unavailable: {e}")
            print("Will use JSON fallback for storage")
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for texts"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def store_vectors(self, filename: str, full_text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store vectors with Pinecone fallback to JSON"""
        print(f"📄 Processing: {filename}")
        
        # Split text into chunks
        chunks = self.split_text_into_chunks(full_text)
        print(f"Created {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        print(f"Created {len(embeddings)} embeddings")
        
        # Try Pinecone first
        if self.pinecone_available:
            try:
                return self._store_vectors_pinecone(filename, chunks, embeddings, metadata)
            except Exception as e:
                print(f"❌ Pinecone storage failed: {e}")
                print("Falling back to JSON storage...")
        
        # Fallback to JSON
        return self._store_vectors_json(filename, chunks, embeddings, metadata)
    
    def _store_vectors_pinecone(self, filename: str, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store vectors in Pinecone"""
        vectors_to_upsert = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{filename}_{uuid.uuid4().hex[:8]}_{i}"
            
            vector_metadata = {
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
                "chunk_size": len(chunk),
                **(metadata or {})
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": vector_metadata
            })
        
        print(f"Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
        self.index.upsert(vectors=vectors_to_upsert)
        
        print(f"Stored {len(vectors_to_upsert)} vectors in Pinecone")
        
        return {
            "status": "success",
            "message": "PDF processed and stored in Pinecone",
            "filename": filename,
            "chunks_stored": len(chunks),
            "total_vectors": len(vectors_to_upsert),
            "storage_method": "pinecone"
        }
    
    def _store_vectors_json(self, filename: str, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store vectors in JSON file"""
        print("Saving to JSON backup...")
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{filename}_{uuid.uuid4().hex[:8]}_{i}"
            
            vector_metadata = {
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
                "chunk_size": len(chunk),
                **(metadata or {})
            }
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": vector_metadata
            })
        
        backup_file = f"vectors_backup_{filename}_{int(time.time())}.json"
        backup_data = {
            "filename": filename,
            "timestamp": time.time(),
            "chunks": chunks,
            "vectors": vectors
        }
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vectors saved to: {backup_file}")
        
        return {
            "status": "success",
            "message": "PDF processed and saved to JSON",
            "filename": filename,
            "chunks_stored": len(chunks),
            "total_vectors": len(vectors),
            "storage_method": "json_backup",
            "backup_file": backup_file
        }
    
    def search_vectors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search vectors with Pinecone fallback to JSON"""
        print(f"Searching for: '{query}'")
        
        # Try Pinecone first
        if self.pinecone_available:
            try:
                return self._search_pinecone(query, top_k)
            except Exception as e:
                print(f"Pinecone search failed: {e}")
                print("Falling back to JSON search...")
        
        # Fallback to JSON
        return self._search_json_vectors(query, top_k)
    
    def _search_pinecone(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search vectors in Pinecone"""
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        processed_results = []
        for match in results["matches"]:
            result = {
                "id": match["id"],
                "score": match["score"],
                "text": match["metadata"]["text"],
                "filename": match["metadata"]["filename"],
                "chunk_index": match["metadata"]["chunk_index"]
            }
            processed_results.append(result)
        
        print(f"Found {len(processed_results)} matches in Pinecone")
        return processed_results
    
    def _search_json_vectors(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search vectors in JSON files"""
        try:
            backup_file = self._find_latest_backup_file()
            if not backup_file:
                print("No backup files found")
                return []
            
            print(f"Searching in local file: {backup_file}")
            
            vectors = self._load_vectors_from_json(backup_file)
            if not vectors:
                return []
            
            query_embedding = self.embedding_model.encode([query])[0]
            
            similarities = []
            for vector_data in vectors:
                vector_embedding = vector_data['values']
                similarity = cosine_similarity(
                    [query_embedding], 
                    [vector_embedding]
                )[0][0]
                
                similarities.append({
                    'id': vector_data['id'],
                    'score': float(similarity),
                    'text': vector_data['metadata']['text'],
                    'filename': vector_data['metadata']['filename'],
                    'chunk_index': vector_data['metadata']['chunk_index']
                })
            
            similarities.sort(key=lambda x: x['score'], reverse=True)
            results = similarities[:top_k]
            
            print(f"Found {len(results)} matches in JSON")
            return results
            
        except Exception as e:
            print(f"JSON search failed: {e}")
            return []
    
    def _find_latest_backup_file(self) -> Optional[str]:
        """Find the latest backup file"""
        backup_files = [f for f in os.listdir('.') 
        if f.startswith('vectors_backup_') and f.endswith('.json')]
        
        if not backup_files:
            return None
        
        return max(backup_files, key=lambda x: os.path.getctime(x))
    
    def _load_vectors_from_json(self, json_file: str) -> List[Dict]:
        """Load vectors from JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            vectors = data.get('vectors', [])
            print(f"Loaded {len(vectors)} vectors from {json_file}")
            return vectors
            
        except Exception as e:
            print(f"Failed to load JSON: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        backup_file = self._find_latest_backup_file()
        
        status = {
            "pinecone_available": self.pinecone_available,
            "backup_file_exists": backup_file is not None,
            "backup_file": backup_file,
            "primary_storage": "pinecone" if self.pinecone_available else "json_local",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": self.dimension
        }
        
        if backup_file:
            vectors = self._load_vectors_from_json(backup_file)
            status["vectors_count"] = len(vectors)
            
        if self.pinecone_available:
            try:
                index_stats = self.index.describe_index_stats()
                status["pinecone_vectors_count"] = index_stats.get("total_vector_count", 0)
            except:
                pass
        
        return status
    
    def clear_all_vectors(self):
        """Clear all vectors from both Pinecone and JSON files"""
        print("Clearing all vectors...")
        
        # Clear Pinecone
        if self.pinecone_available:
            try:
                self.index.delete(delete_all=True)
                print("✅ Pinecone vectors cleared")
            except Exception as e:
                print(f"❌ Failed to clear Pinecone: {e}")
        
        # Clear JSON files
        backup_files = [f for f in os.listdir('.') 
        if f.startswith('vectors_backup_') and f.endswith('.json')]
        
        for file in backup_files:
            try:
                os.remove(file)
                print(f"✅ Removed {file}")
            except Exception as e:
                print(f"❌ Failed to remove {file}: {e}")
        
        print("All vectors cleared!")