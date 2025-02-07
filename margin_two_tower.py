import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import re  # Add at top with other imports
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.query_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.doc_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def compute_similarity(self, query_vec, doc_vec):
        # Normalize vectors
        query_vec = nn.functional.normalize(query_vec, p=2, dim=1)
        doc_vec = nn.functional.normalize(doc_vec, p=2, dim=1)
        
        if self.training:
            # During training, return vectors for loss calculation
            return query_vec, doc_vec
        else:
            # During inference, calculate and return similarity
            similarity = torch.matmul(query_vec, doc_vec.t())
            return similarity

    def encode_query(self, query_emb):
        # For backwards compatibility with validation script
        return self.encode(query_emb, self.query_encoder)
    
    def encode_doc(self, doc_emb):
        # For backwards compatibility with validation script
        return self.encode(doc_emb, self.doc_encoder)

    def encode(self, emb, encoder):
        _, hidden = encoder(emb)
        # Concatenate last layer's bidirectional states
        vec = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.projection(vec)
    
    def forward(self, query_emb, doc_emb):
        query_vec = self.encode(query_emb, self.query_encoder)
        doc_vec = self.encode(doc_emb, self.doc_encoder)
        
        return self.compute_similarity(query_vec, doc_vec)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):  # Lower temperature
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query_vec, doc_vec):
        # L2 normalize
        query_vec = nn.functional.normalize(query_vec, p=2, dim=1)
        doc_vec = nn.functional.normalize(doc_vec, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(query_vec, doc_vec.t()) / self.temperature
        
        # Regular cross entropy with temperature scaling
        labels = torch.arange(len(query_vec), device=query_vec.device)
        return nn.functional.cross_entropy(sim_matrix, labels)

class SimpleDataset(Dataset):
    def __init__(self, queries: List[str], docs: List[str], word2vec, max_length: int = 30):
        super().__init__()
        self.queries = queries
        self.docs = docs
        self.word2vec = word2vec
        self.max_length = max_length
        self.embedding_dim = word2vec.vector_size
    
    @staticmethod
    def text_to_embedding(text: str, word2vec, max_length: int = 30) -> torch.Tensor:
        embedding_dim = word2vec.vector_size
        
        # Store original tokens for exact matching
        original_tokens = text.lower().split()
        
        # Basic structural markers
        text = re.sub(r'\b(is|are|refers?\s+to)\s+(?:a|an|the)\b', 'IS', text.lower())
        text = re.sub(r'\b(contains?|has|have|includes?)\b', 'HAS', text.lower())
        text = re.sub(r'\b(part|component|element)\s+of\b', 'PART_OF', text.lower())
        
        # Simple relationship markers
        text = re.sub(r'\b(controls?|regulates?|manages?)\b', 'CONTROLS', text.lower())
        text = re.sub(r'\b(functions?|works?|operates?)\b', 'FUNCTIONS', text.lower())
        
        # Keep numbers but normalize format
        text = re.sub(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', r'\1_\2', text.lower())
        
        # Process words - combine original and processed tokens
        words = text.split()
        embeddings = []
        
        # Try both original and processed tokens
        for i, word in enumerate(words):
            try:
                # Try original token first for better exact matching
                if i < len(original_tokens):
                    try:
                        emb = word2vec[original_tokens[i]]
                        embeddings.append(emb)
                    except KeyError:
                        pass
                
                # Then try processed token if different
                if word != original_tokens[i]:
                    try:
                        emb = word2vec[word]
                        embeddings.append(emb)
                    except KeyError:
                        pass
                    
            except KeyError:
                continue
        
        if not embeddings:
            embeddings = [np.zeros(embedding_dim)]
        
        embeddings = np.array(embeddings)
        
        # Handle variable length with padding/truncation
        if len(embeddings) < max_length:
            padding = np.zeros((max_length - len(embeddings), embedding_dim))
            embeddings = np.vstack([embeddings, padding])
        else:
            embeddings = embeddings[:max_length]
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_emb = self.text_to_embedding(self.queries[idx], self.word2vec, self.max_length)
        doc_emb = self.text_to_embedding(self.docs[idx], self.word2vec, self.max_length)
        return query_emb, doc_emb 