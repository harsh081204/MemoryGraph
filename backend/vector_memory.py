"""
Vector memory module for semantic similarity search and retrieval.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorMemory:
    """Vector-based memory for semantic similarity search."""
    
    def __init__(self, persist_path: Optional[str] = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector memory.
        
        Args:
            persist_path: Path to pickle file for persistence
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = []  # List of embedding vectors
        self.memories = []    # List of memory objects
        self.persist_path = persist_path
        self._unsaved_changes = False
        
        if persist_path:
            self._load_memories()
    
    def _load_memories(self):
        """Load memories from disk."""
        try:
            path = Path(self.persist_path)
            if path.exists():
                with open(self.persist_path, "rb") as f:
                    data = pickle.load(f)
                    self.embeddings = data.get("embeddings", [])
                    self.memories = data.get("memories", [])
                logger.info(f"Loaded {len(self.memories)} vector memories from {self.persist_path}")
            else:
                logger.info("No existing vector memories found. Starting fresh.")
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Corrupted vector memory file: {e}. Starting fresh.")
            self.embeddings = []
            self.memories = []
        except Exception as e:
            logger.error(f"Unexpected error loading vector memories: {e}")
            self.embeddings = []
            self.memories = []
    
    def add_memory(self, text: str, metadata: Dict = None) -> int:
        """
        Add a new memory with vector embedding.
        
        Args:
            text: Text content to store
            metadata: Optional metadata dictionary
            
        Returns:
            Index of the added memory
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for vector memory")
            return -1
        
        # Generate embedding
        embedding = self.model.encode([text.strip()])[0]
        
        # Create memory object
        memory = {
            "text": text.strip(),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "index": len(self.memories)
        }
        
        # Store
        self.memories.append(memory)
        self.embeddings.append(embedding)
        self._unsaved_changes = True
        
        logger.info(f"Added vector memory: {text[:50]}...")
        return len(self.memories) - 1
    
    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Search for similar memories using cosine similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar memories with similarity scores
        """
        if not self.memories:
            logger.info("No memories available for search")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top results above threshold
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                memory = self.memories[i].copy()
                memory["similarity"] = float(similarity)
                results.append(memory)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def get_context_from_similar(self, query: str, top_k: int = 3) -> str:
        """
        Get contextual information from similar memories.
        
        Args:
            query: Search query
            top_k: Number of similar memories to include
            
        Returns:
            Formatted context string
        """
        similar_memories = self.search_similar(query, top_k=top_k)
        
        if not similar_memories:
            return "No similar context found."
        
        context_parts = []
        for memory in similar_memories:
            context_parts.append(f"Related: {memory['text']} (similarity: {memory['similarity']:.2f})")
        
        return "\n".join(context_parts)
    
    def save_memories(self) -> bool:
        """Save memories to disk."""
        if not self.persist_path:
            logger.warning("No persist_path set. Vector memories not saved.")
            return False
        
        if not self._unsaved_changes:
            logger.debug("No vector memory changes to save")
            return True
        
        try:
            # Ensure directory exists
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "embeddings": self.embeddings,
                "memories": self.memories
            }
            
            with open(self.persist_path, "wb") as f:
                pickle.dump(data, f)
            self._unsaved_changes = False
            logger.info(f"Vector memories saved to {self.persist_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save vector memories: {e}")
            return False
    
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes
    
    def get_stats(self) -> Dict:
        """Get vector memory statistics."""
        return {
            "total_memories": len(self.memories),
            "embedding_dimension": len(self.embeddings[0]) if self.embeddings else 0,
            "unsaved_changes": self._unsaved_changes
        }
    
    def clear_memories(self):
        """Clear all memories (use with caution)."""
        self.memories = []
        self.embeddings = []
        self._unsaved_changes = True
        logger.info("All vector memories cleared")
