"""
Embeddings Module

Handles semantic embedding generation and management using sentence transformers.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages semantic embeddings for candidate profiles.
    
    Uses sentence transformers to convert text profiles into
    dense vector representations for similarity matching.
    
    Example:
        manager = EmbeddingManager()
        embeddings = manager.encode_profiles(profiles)
        query_emb = manager.encode_query("Python backend developer")
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        show_progress: bool = True
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Sentence transformer model identifier
            show_progress: Whether to show encoding progress bar
        """
        self.model_name = model_name
        self.show_progress = show_progress
        self._model = None
        
        # Cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model
    
    def profile_to_text(self, profile: Dict[str, Any]) -> str:
        """
        Convert a profile dictionary to searchable text representation.
        
        Args:
            profile: Profile dictionary
            
        Returns:
            Text representation for embedding
        """
        meta = profile.get("metadata", {})
        tech = profile.get("technical", {})
        collab = profile.get("collaboration", {})
        person = profile.get("personality", {})
        
        parts = [
            f"Role: {profile.get('role', meta.get('dev_type', 'Unknown'))}",
            f"Experience: {meta.get('work_experience_years', '0')} years",
            f"Industry: {meta.get('industry', 'General')}",
            f"Technical Skills: {', '.join(tech.get('skills', []))}",
            f"Tools: {', '.join(tech.get('tools', []))}",
            f"Team Role: {person.get('Belbin_team_role', 'Unknown')}",
            f"Communication: {collab.get('communication_style', 'Standard')}",
            f"Leadership: {collab.get('leadership_preference', 'Neutral')}"
        ]
        
        return " ".join(parts)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text string.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        return self.model.encode([text], show_progress_bar=False)[0]
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple text strings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
        
        return self.model.encode(texts, show_progress_bar=self.show_progress)
    
    def encode_profiles(
        self,
        profiles: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Encode a list of profiles into embeddings.
        
        Args:
            profiles: List of profile dictionaries
            use_cache: Whether to use/update embedding cache
            
        Returns:
            Array of embedding vectors (one per profile)
        """
        if not profiles:
            return np.array([])
        
        texts = []
        indices_to_encode = []
        cached_embeddings = []
        
        for i, profile in enumerate(profiles):
            profile_id = profile.get("id", str(i))
            
            if use_cache and profile_id in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[profile_id]))
            else:
                texts.append(self.profile_to_text(profile))
                indices_to_encode.append((i, profile_id))
        
        # Encode new profiles
        if texts:
            logger.info(f"Encoding {len(texts)} profiles...")
            new_embeddings = self.encode_texts(texts)
            
            # Update cache
            if use_cache:
                for (i, profile_id), emb in zip(indices_to_encode, new_embeddings):
                    self._embedding_cache[profile_id] = emb
        
        # Combine cached and new embeddings in correct order
        result = np.zeros((len(profiles), self.model.get_sentence_embedding_dimension()))
        
        for i, emb in cached_embeddings:
            result[i] = emb
        
        if texts:
            for (i, _), emb in zip(indices_to_encode, new_embeddings):
                result[i] = emb
        
        return result
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.
        
        Args:
            query: Search query string
            
        Returns:
            Query embedding vector
        """
        return self.encode_text(query)
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()
