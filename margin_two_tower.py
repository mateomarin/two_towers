import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import re  # Add at top with other imports

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.query_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,  # Keep 2 layers for better sequence understanding
            batch_first=True,
            bidirectional=True,
            dropout=0.1  # Keep dropout for regularization
        )
        
        self.doc_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Simpler but effective projection layers
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.doc_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def encode_query(self, query_emb):
        _, hidden = self.query_encoder(query_emb)
        # Concatenate last layer's bidirectional states
        query_vec = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.query_proj(query_vec)
    
    def encode_doc(self, doc_emb):
        _, hidden = self.doc_encoder(doc_emb)
        # Concatenate last layer's bidirectional states
        doc_vec = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.doc_proj(doc_vec)
    
    def forward(self, query_emb, doc_emb):
        query_vec = self.encode_query(query_emb)
        doc_vec = self.encode_doc(doc_emb)
        
        # Normalize vectors
        query_vec = nn.functional.normalize(query_vec, p=2, dim=1)
        doc_vec = nn.functional.normalize(doc_vec, p=2, dim=1)
        
        if self.training:
            # During training, return vectors for loss calculation
            return query_vec, doc_vec
        else:
            # During inference, calculate and return similarity
            cosine_sim = torch.matmul(query_vec, doc_vec.t())
            
            # Calculate embedding-level similarity
            query_flat = query_emb.view(-1, query_emb.size(-1))
            doc_flat = doc_emb.view(-1, doc_emb.size(-1))
            
            query_norms = torch.sum(query_flat ** 2, dim=1, keepdim=True)
            doc_norms = torch.sum(doc_flat ** 2, dim=1, keepdim=True)
            
            batch_size = query_emb.size(0)
            query_norms = query_norms.view(batch_size, -1).mean(dim=1, keepdim=True)
            doc_norms = doc_norms.view(batch_size, -1).mean(dim=1, keepdim=True)
            
            l2_sim = 1.0 / (1.0 + query_norms + doc_norms.t())
            
            similarity = cosine_sim + 0.1 * l2_sim
            return similarity

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
        
        # Preserve numbers but normalize format
        text = re.sub(r'\$(\d+(?:\.\d+)?)', r'PRICE \1', text.lower())
        text = re.sub(r'(\d+(?:\.\d+)?)\s*(°c|celsius)', r'\1 CELSIUS', text.lower())
        text = re.sub(r'(\d+(?:\.\d+)?)\s*(km|miles?|ft|feet)', r'\1 DISTANCE', text.lower())
        
        # Location handling (less aggressive)
        text = re.sub(r'\b(located|situated|found)\s+(in|at|near)\b', 'LOCATION', text.lower())
        text = re.sub(r'\b(north|south|east|west)\s+of\b', 'DIRECTION_OF', text.lower())
        
        # Definition handling (preserve terms)
        text = re.sub(r'\b(what\s+is|define|meaning\s+of)\s+(\w+)\b', r'DEFINE \2', text.lower())
        text = re.sub(r'\b(refers?\s+to|known\s+as)\b', 'MEANS', text.lower())
        
        # Query type markers
        text = re.sub(r'^(how|what|where|when|why|who)\b', r'QUERY_\1', text.lower())
        text = re.sub(r'^(is|are|can|does|do)\b', 'QUERY_YN', text.lower())  # Yes/No questions
        
        # Time expressions
        text = re.sub(r'\b(\d+)\s*(year|month|week|day|hour)s?\b', r'\1 TIME_\2', text.lower())
        
        # Process words
        embeddings = []
        words = text.split()
        words = words[:max_length]
        
        for word in words:
            try:
                # Try exact match first
                emb = word2vec[word]
                embeddings.append(emb)
            except KeyError:
                try:
                    # Try without special chars
                    clean_word = re.sub(r'[^\w\s]', '', word)
                    emb = word2vec[clean_word]
                    embeddings.append(emb)
                except KeyError:
                    # Handle compound words
                    parts = word.split('_')
                    if len(parts) > 1:
                        part_embs = []
                        for part in parts:
                            try:
                                emb = word2vec[part]
                                part_embs.append(emb)
                            except KeyError:
                                continue
                        if part_embs:
                            emb = np.mean(part_embs, axis=0)
                            embeddings.append(emb)
        
        if not embeddings:
            embeddings = [np.zeros(embedding_dim)]
            
        embeddings = np.array(embeddings)
        
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