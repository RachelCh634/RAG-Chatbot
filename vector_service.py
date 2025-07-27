import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional
import torch
import open_clip
from dotenv import load_dotenv
import re
import numpy as np

load_dotenv()

class VectorService:
    """Optimized Vector Management Service with Pinecone and JSON fallback using OpenCLIP"""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        # Initialize OpenCLIP model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained, 
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        try:
            with torch.no_grad():
                sample_text = self.tokenizer(["test"])
                sample_features = self.model.encode_text(sample_text.to(self.device))
                
                if hasattr(sample_features, 'shape'):
                    self.dimension = sample_features.shape[-1]
                else:
                    print("Warning: Could not determine dimension from model, using default")
                    self.dimension = 512
        except Exception as e:
            print(f"Error determining model dimension: {e}")
            if "ViT-B-32" in model_name:
                self.dimension = 512
            elif "ViT-L-14" in model_name:
                self.dimension = 768
            else:
                self.dimension = 512  
            print(f"Using default dimension: {self.dimension}")
        
        print(f"Model: {model_name}, Embedding dimension: {self.dimension}")
        
        self.embedding_cache = {}
        
        self.pinecone_available = False
        self.pc = None
        self.index = None
        self.index_name = "pdf-rag-index"
        
        self._setup_pinecone()
    
    def _setup_pinecone(self):
        """Setup Pinecone connection with improved initialization"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                print("PINECONE_API_KEY not found in environment variables")
                return
            
            self.pc = Pinecone(api_key=api_key)
            
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
                while not self.pc.describe_index(self.index_name).status['ready']:
                    print("Index not ready yet, waiting...")
                    time.sleep(5)
                print("Index is ready!")
            
            self.index = self.pc.Index(self.index_name)
            self.pinecone_available = True
            print(f"Pinecone connected to index: {self.index_name}")
            
        except Exception as e:
            print(f"Pinecone unavailable: {e}")
            print("Will use JSON fallback for storage")
    
    def preprocess_text(self, text: str) -> str:
        """Lightweight preprocessing"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Faster chunking with simpler logic"""
        if not text or not text.strip():
            return []
            
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenCLIP with caching"""
        if not texts:
            return []
            
        embeddings = []
        texts_to_encode = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
                cache_indices.append(i)
            else:
                texts_to_encode.append(text)
        
        if texts_to_encode:
            batch_size = 32  
            new_embeddings = []
            
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(texts_to_encode), batch_size):
                    batch = texts_to_encode[i:i + batch_size]
                    
                    try:
                        tokens = self.tokenizer(batch).to(self.device)
                        
                        batch_embeddings = self.model.encode_text(tokens)
                        
                        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
                        
                        batch_embeddings = batch_embeddings.cpu().numpy()
                        new_embeddings.extend(batch_embeddings)
                        
                    except Exception as e:
                        print(f"Error creating embeddings for batch: {e}")
                        for _ in batch:
                            new_embeddings.append([0.0] * self.dimension)
            
            for text, embedding in zip(texts_to_encode, new_embeddings):
                self.embedding_cache[text] = embedding.tolist()
            
            text_idx = 0
            for i, text in enumerate(texts):
                if i not in cache_indices:
                    embeddings.insert(i, new_embeddings[text_idx].tolist())
                    text_idx += 1
        
        return embeddings
    
    def store_vectors(self, filename: str, full_text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store vectors with Pinecone fallback to JSON"""
        print(f"Processing: {filename}")
        
        if not full_text or not full_text.strip():
            return {
                "status": "error",
                "message": "Empty text provided",
                "filename": filename
            }
        
        chunks = self.split_text_into_chunks(full_text)
        print(f"Created {len(chunks)} chunks")
        
        if not chunks:
            return {
                "status": "error",
                "message": "No chunks created from text",
                "filename": filename
            }
        
        embeddings = self.create_embeddings(chunks)
        print(f"Created {len(embeddings)} embeddings")
        
        if self.pinecone_available:
            try:
                return self._store_vectors_pinecone(filename, chunks, embeddings, metadata)
            except Exception as e:
                print(f"Pinecone storage failed: {e}")
                print("Falling back to JSON storage...")
        
        return self._store_vectors_json(filename, chunks, embeddings, metadata)
    
    def _store_vectors_pinecone(self, filename: str, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store vectors in Pinecone with consistency checking"""
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
        
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Uploaded {len(vectors_to_upsert)} vectors to Pinecone")
        
        self._wait_for_consistency(expected_count=len(vectors_to_upsert))
        
        return {
            "status": "success",
            "message": "PDF processed and stored in Pinecone (synced)",
            "filename": filename,
            "chunks_stored": len(chunks),
            "total_vectors": len(vectors_to_upsert),
            "storage_method": "pinecone"
        }
    
    def _wait_for_consistency(self, expected_count: int, max_wait: int = 30):
        """Wait for Pinecone to sync the vectors"""
        print("Waiting for Pinecone consistency...")
        
        for attempt in range(max_wait):
            try:
                stats = self.index.describe_index_stats()
                current_count = stats.get('total_vector_count', 0)
                
                print(f"Attempt {attempt + 1}: {current_count} vectors in index")
                
                if current_count >= expected_count:
                    print("Pinecone synced successfully!")
                    return True
                    
                time.sleep(1)
                
            except Exception as e:
                print(f"Consistency check failed: {e}")
                time.sleep(1)
        
        print("Consistency check timed out, but vectors might still be syncing")
        return False
    
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
    
    def search_vectors(self, query: str, top_k: int = 5, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Search with retry logic for consistency issues"""
        if not query or not query.strip():
            return []
            
        print(f"Searching for: '{query}'")
        
        if self.pinecone_available:
            for attempt in range(max_retries):
                try:
                    results = self._search_pinecone(query, top_k)
                    
                    if results or attempt == max_retries - 1:
                        return results
                    
                    print(f"No results on attempt {attempt + 1}, retrying...")
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Pinecone search failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        print("Falling back to JSON search...")
                        return self._search_json_vectors(query, top_k)
        
        return self._search_json_vectors(query, top_k)
    
    def _search_pinecone(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search vectors in Pinecone with consistency check"""
        
        try:
            stats = self.index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            
            if vector_count == 0:
                print("Index appears empty, might be consistency issue")
                return []
                
            print(f"Index contains {vector_count} vectors")
            
        except Exception as e:
            print(f"Could not check index stats: {e}")
        
        if query in self.embedding_cache:
            query_embedding = self.embedding_cache[query]
        else:
            try:
                self.model.eval()
                with torch.no_grad():
                    tokens = self.tokenizer([query]).to(self.device)
                    query_features = self.model.encode_text(tokens)
                    query_features = torch.nn.functional.normalize(query_features, dim=-1)
                    query_embedding = query_features.cpu().numpy()[0].tolist()
                    self.embedding_cache[query] = query_embedding
            except Exception as e:
                print(f"Error creating query embedding: {e}")
                return []
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        processed_results = []
        for match in results.get("matches", []):
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
        """Optimized JSON search with numpy"""
        try:
            backup_file = self._find_latest_backup_file()
            if not backup_file:
                print("No backup files found")
                return []
            
            vectors = self._load_vectors_from_json(backup_file)
            if not vectors:
                return []
            
            if query in self.embedding_cache:
                query_embedding = self.embedding_cache[query]
            else:
                try:
                    self.model.eval()
                    with torch.no_grad():
                        tokens = self.tokenizer([query]).to(self.device)
                        query_features = self.model.encode_text(tokens)
                        query_features = torch.nn.functional.normalize(query_features, dim=-1)
                        query_embedding = query_features.cpu().numpy()[0].tolist()
                        self.embedding_cache[query] = query_embedding
                except Exception as e:
                    print(f"Error creating query embedding: {e}")
                    return []
            
            query_vec = np.array(query_embedding)
            vector_embeddings = np.array([v['values'] for v in vectors])
            
            similarities = np.dot(vector_embeddings, query_vec)
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'id': vectors[idx]['id'],
                    'score': float(similarities[idx]),
                    'text': vectors[idx]['metadata']['text'],
                    'filename': vectors[idx]['metadata']['filename'],
                    'chunk_index': vectors[idx]['metadata']['chunk_index']
                })
            
            print(f"Found {len(results)} matches in JSON")
            return results
            
        except Exception as e:
            print(f"JSON search failed: {e}")
            return []
    
    def _find_latest_backup_file(self) -> Optional[str]:
        """Find the latest backup file"""
        try:
            backup_files = [f for f in os.listdir('.') 
            if f.startswith('vectors_backup_') and f.endswith('.json')]
            
            if not backup_files:
                return None
            
            return max(backup_files, key=lambda x: os.path.getctime(x))
        except Exception as e:
            print(f"Error finding backup files: {e}")
            return None
    
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
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        print("Embedding cache cleared")
    
    def clear_all_vectors(self):
        """Clear all vectors from both Pinecone and JSON files"""
        print("Clearing all vectors...")
        
        if self.pinecone_available:
            try:
                self.index.delete(delete_all=True)
                print("Pinecone vectors cleared")
            except Exception as e:
                print(f"Failed to clear Pinecone: {e}")
        
        try:
            backup_files = [f for f in os.listdir('.') 
            if f.startswith('vectors_backup_') and f.endswith('.json')]
            
            for file in backup_files:
                try:
                    os.remove(file)
                    print(f"Removed {file}")
                except Exception as e:
                    print(f"Failed to remove {file}: {e}")
        except Exception as e:
            print(f"Error accessing backup files: {e}")
        
        self.clear_cache()
        print("All vectors cleared!")