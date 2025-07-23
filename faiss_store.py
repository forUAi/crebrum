"""
Cerebrum - Second Brain Memory System
====================================

A fully self-contained memory store using FAISS for vector similarity,
BM25 for keyword matching, and SQLite for metadata persistence.

Google Colab Setup:
------------------
!pip install sentence-transformers faiss-cpu rank-bm25

Usage Example:
-------------
from faiss_store import MemoryStore

# Initialize the memory store
store = MemoryStore()

# Add memories
store.add_memory("Python is a programming language", {"source": "wiki", "tags": ["programming"]})
store.add_memory("Machine learning uses algorithms", {"source": "book", "tags": ["AI", "ML"]})

# Search memories
results = store.search_memory("programming language", k=3)
for result in results:
    print(f"Score: {result['score']:.3f} - {result['text']}")
"""

import os
import json
import pickle
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("📦 Install with: pip install sentence-transformers faiss-cpu rank-bm25")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryStore:
    """
    A hybrid memory store combining FAISS vector similarity and BM25 keyword matching.
    
    Features:
    - Vector embeddings via sentence-transformers
    - FAISS index for fast similarity search
    - BM25 for keyword-based relevance
    - SQLite for metadata persistence
    - Hybrid scoring: 0.6 * faiss_sim + 0.4 * bm25_score
    """
    
    def __init__(self, data_dir: str = "data", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the MemoryStore.
        
        Args:
            data_dir: Directory to store index and database files
            model_name: Sentence transformer model name
        """
        self.data_dir = data_dir
        self.model_name = model_name
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # File paths
        self.index_path = os.path.join(self.data_dir, "index.faiss")
        self.ids_path = os.path.join(self.data_dir, "index.ids")
        self.db_path = os.path.join(self.data_dir, "memory.db")
        self.bm25_path = os.path.join(self.data_dir, "bm25.pkl")
        
        # Initialize components
        self._init_embedding_model()
        self._init_database()
        self._load_or_create_index()
        self._load_or_create_bm25()
        
        logger.info(f"✅ MemoryStore initialized with {self.get_memory_count()} memories")
    
    def _init_embedding_model(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"🤖 Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded, embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def _init_database(self) -> None:
        """Initialize SQLite database with memory table."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Create memories table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT,
                    tags TEXT,  -- JSON string
                    relevance_score REAL DEFAULT 0.0,
                    pinned INTEGER DEFAULT 0,
                    metadata TEXT  -- JSON string
                )
            """)
            self.conn.commit()
            logger.info("✅ Database initialized")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _load_or_create_index(self) -> None:
        """Load existing FAISS index or create a new one."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
                # Load existing index
                self.index = faiss.read_index(self.index_path)
                with open(self.ids_path, 'rb') as f:
                    self.memory_ids = pickle.load(f)
                logger.info(f"✅ Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
                self.memory_ids = []
                logger.info("✅ Created new FAISS index")
        except Exception as e:
            logger.error(f"❌ FAISS index initialization failed: {e}")
            # Fallback to new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.memory_ids = []
    
    def _load_or_create_bm25(self) -> None:
        """Load existing BM25 index or create a new one."""
        try:
            if os.path.exists(self.bm25_path):
                with open(self.bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.bm25 = bm25_data['bm25']
                    self.bm25_corpus = bm25_data['corpus']
                logger.info(f"✅ Loaded existing BM25 index with {len(self.bm25_corpus)} documents")
            else:
                self.bm25 = None
                self.bm25_corpus = []
                logger.info("✅ Created new BM25 index")
        except Exception as e:
            logger.error(f"❌ BM25 index loading failed: {e}")
            self.bm25 = None
            self.bm25_corpus = []
    
    def _save_index(self) -> None:
        """Save FAISS index and memory IDs to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.ids_path, 'wb') as f:
                pickle.dump(self.memory_ids, f)
            logger.info("✅ FAISS index saved")
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index: {e}")
    
    def _save_bm25(self) -> None:
        """Save BM25 index to disk."""
        try:
            with open(self.bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'corpus': self.bm25_corpus
                }, f)
            logger.info("✅ BM25 index saved")
        except Exception as e:
            logger.error(f"❌ Failed to save BM25 index: {e}")
    
    def _update_bm25(self) -> None:
        """Rebuild BM25 index from current corpus."""
        if self.bm25_corpus:
            tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new memory to the store.
        
        Args:
            text: The memory text content
            metadata: Optional metadata dictionary
        
        Returns:
            str: The generated memory ID
        """
        if not text.strip():
            raise ValueError("Memory text cannot be empty")
        
        try:
            # Generate unique ID
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Process metadata
            if metadata is None:
                metadata = {}
            
            source = metadata.get('source', '')
            tags = json.dumps(metadata.get('tags', []))
            relevance_score = metadata.get('relevance_score', 0.0)
            pinned = 1 if metadata.get('pinned', False) else 0
            metadata_json = json.dumps(metadata)
            
            # Generate embedding
            embedding = self.embedding_model.encode([text], normalize_embeddings=True)[0]
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            self.memory_ids.append(memory_id)
            
            # Add to BM25 corpus
            self.bm25_corpus.append(text)
            self._update_bm25()
            
            # Store in database
            self.conn.execute("""
                INSERT INTO memories (memory_id, text, timestamp, source, tags, 
                                    relevance_score, pinned, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, text, timestamp, source, tags, relevance_score, pinned, metadata_json))
            self.conn.commit()
            
            # Save indices
            self._save_index()
            self._save_bm25()
            
            logger.info(f"✅ Added memory: {memory_id[:8]}... - {text[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Failed to add memory: {e}")
            raise
    
    def search_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories using hybrid FAISS + BM25 scoring.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of memory dictionaries with scores
        """
        if not query.strip():
            return []
        
        if self.index.ntotal == 0:
            logger.info("No memories to search")
            return []
        
        try:
            # Get more candidates for better hybrid scoring
            search_k = min(max(k * 3, 20), self.index.ntotal)
            
            # FAISS vector search
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            faiss_scores, faiss_indices = self.index.search(
                query_embedding.reshape(1, -1), search_k
            )
            
            # BM25 keyword search
            bm25_scores = {}
            if self.bm25 is not None:
                query_tokens = query.lower().split()
                bm25_doc_scores = self.bm25.get_scores(query_tokens)
                # Normalize BM25 scores to 0-1 range
                if len(bm25_doc_scores) > 0:
                    max_bm25 = max(bm25_doc_scores)
                    if max_bm25 > 0:
                        for i, score in enumerate(bm25_doc_scores):
                            if i < len(self.memory_ids):
                                bm25_scores[self.memory_ids[i]] = score / max_bm25
            
            # Combine scores and get memory details
            results = []
            seen_ids = set()
            
            for i, (faiss_idx, faiss_score) in enumerate(zip(faiss_indices[0], faiss_scores[0])):
                if faiss_idx == -1 or faiss_idx >= len(self.memory_ids):
                    continue
                
                memory_id = self.memory_ids[faiss_idx]
                if memory_id in seen_ids:
                    continue
                seen_ids.add(memory_id)
                
                # Get memory from database
                cursor = self.conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                if not row:
                    continue
                
                # Calculate hybrid score
                faiss_sim = float(faiss_score)  # Already normalized (cosine similarity)
                bm25_score = bm25_scores.get(memory_id, 0.0)
                hybrid_score = 0.6 * faiss_sim + 0.4 * bm25_score
                
                # Build result
                result = {
                    'memory_id': memory_id,
                    'text': row['text'],
                    'timestamp': row['timestamp'],
                    'source': row['source'] or '',
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'relevance_score': row['relevance_score'],
                    'pinned': bool(row['pinned']),
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'score': hybrid_score,
                    'faiss_score': faiss_sim,
                    'bm25_score': bm25_score
                }
                results.append(result)
            
            # Sort by hybrid score and return top k
            results.sort(key=lambda x: x['score'], reverse=True)
            final_results = results[:k]
            
            logger.info(f"🔍 Search '{query}' returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_memory_count(self) -> int:
        """Get the total number of memories stored."""
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"❌ Failed to get memory count: {e}")
            return 0
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        try:
            cursor = self.conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'memory_id': row['memory_id'],
                'text': row['text'],
                'timestamp': row['timestamp'],
                'source': row['source'] or '',
                'tags': json.loads(row['tags']) if row['tags'] else [],
                'relevance_score': row['relevance_score'],
                'pinned': bool(row['pinned']),
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            }
        except Exception as e:
            logger.error(f"❌ Failed to get memory {memory_id}: {e}")
            return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if successful."""
        try:
            # Remove from database
            cursor = self.conn.execute(
                "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
            )
            if cursor.rowcount == 0:
                logger.warning(f"Memory {memory_id} not found")
                return False
            
            self.conn.commit()
            
            # Rebuild indices (simple approach - could be optimized)
            self._rebuild_indices()
            
            logger.info(f"✅ Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete memory {memory_id}: {e}")
            return False
    
    def _rebuild_indices(self) -> None:
        """Rebuild FAISS and BM25 indices from database."""
        try:
            # Clear current indices
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.memory_ids = []
            self.bm25_corpus = []
            
            # Rebuild from database
            cursor = self.conn.execute("SELECT memory_id, text FROM memories ORDER BY timestamp")
            rows = cursor.fetchall()
            
            if rows:
                texts = [row['text'] for row in rows]
                memory_ids = [row['memory_id'] for row in rows]
                
                # Rebuild FAISS
                embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
                self.index.add(embeddings)
                self.memory_ids = memory_ids
                
                # Rebuild BM25
                self.bm25_corpus = texts
                self._update_bm25()
                
                # Save indices
                self._save_index()
                self._save_bm25()
            
            logger.info(f"✅ Rebuilt indices with {len(self.memory_ids)} memories")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild indices: {e}")
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recently added memories."""
        try:
            cursor = self.conn.execute("""
                SELECT * FROM memories 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'memory_id': row['memory_id'],
                    'text': row['text'],
                    'timestamp': row['timestamp'],
                    'source': row['source'] or '',
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'relevance_score': row['relevance_score'],
                    'pinned': bool(row['pinned']),
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent memories: {e}")
            return []
    
    def close(self) -> None:
        """Close database connection and save indices."""
        try:
            self._save_index()
            self._save_bm25()
            self.conn.close()
            logger.info("✅ MemoryStore closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing MemoryStore: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Cerebrum Memory System Test")
    print("=" * 40)
    
    # Initialize store
    store = MemoryStore()
    
    # Add sample memories
    sample_memories = [
        ("Python is a high-level programming language known for its simplicity", 
         {"source": "wiki", "tags": ["programming", "python"]}),
        ("Machine learning algorithms can learn patterns from data automatically", 
         {"source": "textbook", "tags": ["AI", "ML", "algorithms"]}),
        ("Neural networks are inspired by biological neurons in the brain", 
         {"source": "research", "tags": ["AI", "neuroscience"]}),
        ("FAISS is a library for efficient similarity search of dense vectors", 
         {"source": "documentation", "tags": ["vector-search", "faiss"]}),
        ("SQLite is a lightweight database engine perfect for embedded applications", 
         {"source": "manual", "tags": ["database", "sqlite"]})
    ]
    
    print(f"\n📝 Adding {len(sample_memories)} sample memories...")
    for text, metadata in sample_memories:
        memory_id = store.add_memory(text, metadata)
        print(f"   Added: {memory_id[:8]}...")
    
    # Test search
    print(f"\n🔍 Testing search functionality...")
    queries = ["programming language", "machine learning", "database"]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = store.search_memory(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.3f} | {result['text'][:60]}...")
    
    # Show stats
    print(f"\n📊 Store Statistics:")
    print(f"   Total memories: {store.get_memory_count()}")
    print(f"   FAISS vectors: {store.index.ntotal}")
    print(f"   BM25 documents: {len(store.bm25_corpus)}")
    
    print("\n✅ Test completed successfully!")
    store.close()