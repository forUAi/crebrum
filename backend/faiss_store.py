"""
Memory Store implementation using FAISS for vector search and BM25 for text search
"""

import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import uuid
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch

logger = logging.getLogger("cerebrum.memory_store")


class Memory:
    """Represents a single memory entry"""
    
    def __init__(self, text: str, metadata: Dict[str, Any], memory_id: str = None):
        self.id = memory_id or str(uuid.uuid4())
        self.text = text
        self.metadata = metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class MemoryStore:
    """
    Hybrid memory store combining FAISS vector search with BM25 text search
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model: Optional[SentenceTransformer] = None
        self.embedding_dim: int = 384  # Default for all-MiniLM-L6-v2
        
        # FAISS index for vector similarity
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        
        # BM25 for text search
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        
        # Memory storage
        self.memories: List[Memory] = []
        self.id_to_index: Dict[str, int] = {}
        
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the memory store with embedding model and indices"""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            logger.info(f"Initializing memory store with model: {self.embedding_model_name}")
            
            # Initialize sentence transformer model
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.embedding_model = await loop.run_in_executor(
                    None, 
                    lambda: SentenceTransformer(self.embedding_model_name)
                )
                
                # Get actual embedding dimension
                test_embedding = await self._encode_text("test")
                self.embedding_dim = len(test_embedding)
                
                logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
            
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product similarity
            
            # Initialize empty BM25
            self.bm25 = None
            self.bm25_corpus = []
            
            self._initialized = True
            logger.info("Memory store initialized successfully")
    
    async def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        # Run encoding in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        )
        return embedding
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        return text.lower().split()
    
    async def add_memory(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a new memory to the store"""
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            # Create memory object
            memory = Memory(text, metadata)
            
            # Generate embedding
            try:
                embedding = await self._encode_text(text)
                embedding = embedding.reshape(1, -1).astype(np.float32)
                
                # Add to FAISS index
                self.faiss_index.add(embedding)
                
                # Add to BM25 corpus
                tokens = self._tokenize_text(text)
                self.bm25_corpus.append(tokens)
                
                if self.bm25 is None:
                    self.bm25 = BM25Okapi(self.bm25_corpus)
                else:
                    self.bm25 = BM25Okapi(self.bm25_corpus)
                
                # Store memory
                memory_index = len(self.memories)
                self.memories.append(memory)
                self.id_to_index[memory.id] = memory_index
                
                logger.info(f"Added memory {memory.id} to store")
                return memory.id
                
            except Exception as e:
                logger.error(f"Failed to add memory: {e}")
                raise
    
    async def search_memories(
        self, 
        query: str, 
        top_k: int = 5, 
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and BM25 text search
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.memories:
            return []
        
        async with self._lock:
            try:
                # Vector search with FAISS
                query_embedding = await self._encode_text(query)
                query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
                
                # Get all similarities from FAISS
                similarities, indices = self.faiss_index.search(query_embedding, len(self.memories))
                vector_scores = similarities[0]
                
                # BM25 search
                query_tokens = self._tokenize_text(query)

                # Safe fallback if BM25 is None
                bm25_scores = np.zeros(len(self.memories))
                if self.bm25:
                    bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Normalize scores to [0, 1] range
                if len(vector_scores) > 1:
                    vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-8)
                else:
                    vector_scores = np.array([1.0])
                
                if len(bm25_scores) > 1:
                    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
                else:
                    bm25_scores = np.array([1.0])
                
                # Combine scores
                combined_scores = vector_weight * vector_scores + bm25_weight * bm25_scores
                
                # Get top-k results
                top_indices = np.argsort(combined_scores)[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    if idx < len(self.memories):
                        memory = self.memories[idx]
                        result = memory.to_dict()
                        result["relevance_score"] = float(combined_scores[idx])
                        result["vector_score"] = float(vector_scores[idx])
                        result["bm25_score"] = float(bm25_scores[idx])
                        results.append(result)
                
                logger.info(f"Search for '{query}' returned {len(results)} results")
                return results
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        if not self._initialized:
            await self.initialize()
        
        return {
            "total_memories": len(self.memories),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "bm25_corpus_size": len(self.bm25_corpus),
            "embedding_dimension": self.embedding_dim,
            "embedding_model": self.embedding_model_name,
            "initialized": self._initialized
        }
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID"""
        if memory_id in self.id_to_index:
            memory_index = self.id_to_index[memory_id]
            if memory_index < len(self.memories):
                return self.memories[memory_index].to_dict()
        return None
    
memory_store = MemoryStore()