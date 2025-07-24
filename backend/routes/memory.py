"""
Memory management routes for Cerebrum
Handles memory storage, retrieval, and search functionality
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
import asyncio

from faiss_store import MemoryStore

logger = logging.getLogger("cerebrum.memory")

# Global memory store instance
memory_store: Optional[MemoryStore] = None

# Pydantic models
class MemoryCreate(BaseModel):
    """Request model for creating a new memory"""
    text: str = Field(..., min_length=1, max_length=10000, description="Memory content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()
    
    @validator('metadata')
    def validate_metadata(cls, v):
        # Ensure metadata doesn't contain reserved keys
        reserved_keys = {'id', 'created_at', 'updated_at', 'text'}
        if any(key in reserved_keys for key in v.keys()):
            raise ValueError(f'Metadata cannot contain reserved keys: {reserved_keys}')
        return v


class MemoryResponse(BaseModel):
    """Response model for memory operations"""
    id: str
    text: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    relevance_score: Optional[float] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None


class MemoryCreateResponse(BaseModel):
    """Response model for memory creation"""
    id: str
    message: str


class MemorySearchResponse(BaseModel):
    """Response model for memory search"""
    query: str
    results: List[MemoryResponse]
    total_results: int
    search_time_ms: float


class MemoryStatsResponse(BaseModel):
    """Response model for memory statistics"""
    total_memories: int
    faiss_index_size: int
    bm25_corpus_size: int
    embedding_dimension: int
    embedding_model: str
    initialized: bool


# Router instance
router = APIRouter()


async def initialize_memory_store() -> None:
    """Initialize the global memory store"""
    global memory_store
    if memory_store is None:
        memory_store = MemoryStore()
        await memory_store.initialize()


def get_memory_store() -> MemoryStore:
    """Get the memory store instance"""
    if memory_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory store not initialized"
        )
    return memory_store


@router.post("/", response_model=MemoryCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(memory_data: MemoryCreate) -> MemoryCreateResponse:
    """
    Create a new memory entry
    
    Stores the provided text and metadata, generating embeddings for vector search
    and indexing for BM25 text search.
    """
    try:
        store = get_memory_store()
        
        # Add memory to store
        memory_id = await store.add_memory(
            text=memory_data.text,
            metadata=memory_data.metadata
        )
        
        logger.info(f"Created memory {memory_id}")
        
        return MemoryCreateResponse(
            id=memory_id,
            message="Memory created successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to create memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create memory: {str(e)}"
        )


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(5, ge=1, le=50, description="Maximum number of results to return"),
    vector_weight: float = Query(0.7, ge=0.0, le=1.0, description="Weight for vector similarity"),
    bm25_weight: float = Query(0.3, ge=0.0, le=1.0, description="Weight for BM25 text search")
) -> MemorySearchResponse:
    """
    Search memories using hybrid vector + BM25 approach
    
    Combines semantic similarity (via sentence transformers) with lexical matching (BM25)
    for comprehensive memory retrieval.
    """
    import time
    
    # Validate weights sum to 1.0
    if abs(vector_weight + bm25_weight - 1.0) > 0.01:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="vector_weight and bm25_weight must sum to 1.0"
        )
    
    try:
        store = get_memory_store()
        
        start_time = time.time()
        
        # Perform search
        search_results = await store.search_memories(
            query=q,
            top_k=limit,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Convert to response models
        memory_responses = [
            MemoryResponse(**result) for result in search_results
        ]
        
        logger.info(f"Search for '{q}' returned {len(memory_responses)} results in {search_time_ms:.2f}ms")
        
        return MemorySearchResponse(
            query=q,
            results=memory_responses,
            total_results=len(memory_responses),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats() -> MemoryStatsResponse:
    """
    Get memory store statistics
    
    Returns information about the current state of the memory store including
    total memories, index sizes, and configuration details.
    """
    try:
        store = get_memory_store()
        
        stats = await store.get_stats()
        
        return MemoryStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str) -> MemoryResponse:
    """
    Retrieve a specific memory by ID
    """
    try:
        store = get_memory_store()
        
        memory_data = await store.get_memory_by_id(memory_id)
        
        if memory_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory with id {memory_id} not found"
            )
        
        return MemoryResponse(**memory_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory: {str(e)}"
        )


@router.get("/", response_model=List[MemoryResponse])
async def list_memories(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of memories to return"),
    offset: int = Query(0, ge=0, description="Number of memories to skip")
) -> List[MemoryResponse]:
    """
    List all memories with pagination
    
    Returns a paginated list of all stored memories, ordered by creation date (newest first).
    """
    try:
        store = get_memory_store()
        
        # Get all memories (in a real system, this would be optimized with proper pagination)
        all_memories = store.memories
        
        # Sort by creation date (newest first)
        sorted_memories = sorted(all_memories, key=lambda m: m.created_at, reverse=True)
        
        # Apply pagination
        paginated_memories = sorted_memories[offset:offset + limit]
        
        # Convert to response models
        memory_responses = [
            MemoryResponse(**memory.to_dict()) for memory in paginated_memories
        ]
        
        logger.info(f"Listed {len(memory_responses)} memories (offset: {offset}, limit: {limit})")
        
        return memory_responses
        
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list memories: {str(e)}"
        )