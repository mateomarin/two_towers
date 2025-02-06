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
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        self.doc_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Project to hidden_dim with single layer
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.doc_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
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
        
        # Normalize numbers and currency
        text = re.sub(r'\$?\d+(?:\.\d+)?(?:k|m|b|bn)?\b', 'NUMBER', text.lower())
        text = re.sub(r'\b(usd|inr|rs|gbp|eur|dollars?|rupees?)\b', 'CURRENCY', text.lower())
        
        # Normalize units
        text = re.sub(r'\b(kg|kgs|kilometer|km|cm|mm|ml|lb|lbs|ft|feet|mph|kmh|celsius|fahrenheit)\b', 'UNIT', text.lower())
        text = re.sub(r'\b(mg|mcg|ml|gram|grams)\b', 'MEDICAL_UNIT', text.lower())
        
        # Normalize time
        text = re.sub(r'\b(years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b', 'TIME_UNIT', text.lower())
        
        # Normalize percentages
        text = re.sub(r'\d+%', 'PERCENTAGE', text.lower())
        
        # Normalize locations
        text = re.sub(r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b', 'STREET', text.lower())
        text = re.sub(r'\b(po box|p\.o\. box)\b', 'POBOX', text.lower())
        
        # Normalize measurements
        text = re.sub(r'\b(square|sq|cubic|cu)\b', 'MEASURE_TYPE', text.lower())
        text = re.sub(r'\b(length|width|height|depth)\b', 'DIMENSION', text.lower())
        
        # Normalize semantic variations
        text = re.sub(r'\b(cost|price|fee|charge|rate)\b', 'COST', text.lower())
        text = re.sub(r'\b(per|every|each|a)\s+(hour|hr|day|month|year)\b', 'PER_TIME', text.lower())
        text = re.sub(r'\b(use|utilize|employ|apply)\b', 'USE', text.lower())
        text = re.sub(r'\b(process|procedure|method|technique)\b', 'PROCESS', text.lower())
        
        # Normalize definitions
        text = re.sub(r'\b(is|are|refers to|defined as|means)\b', 'IS', text.lower())
        text = re.sub(r'\b(consists of|composed of|made of|contains)\b', 'CONTAINS', text.lower())
        
        # Normalize quantities
        text = re.sub(r'\b(about|approximately|around|roughly)\b', 'APPROXIMATELY', text.lower())
        text = re.sub(r'\b(between|from|range)\b', 'RANGE', text.lower())
        
        # Improved word handling
        embeddings = []
        words = re.findall(r'\w+|\S', text.lower())
        words = words[:max_length]
        
        for word in words:
            try:
                # Try exact match first
                emb = word2vec[word]
                embeddings.append(emb)
            except KeyError:
                try:
                    # Try without punctuation
                    clean_word = re.sub(r'[^\w\s]', '', word)
                    emb = word2vec[clean_word]
                    embeddings.append(emb)
                except KeyError:
                    # Try subwords
                    subwords = word.split('-')
                    if len(subwords) > 1:
                        subword_embs = []
                        for subword in subwords:
                            try:
                                emb = word2vec[subword]
                                subword_embs.append(emb)
                            except KeyError:
                                continue
                        if subword_embs:
                            # Average the subword embeddings
                            emb = np.mean(subword_embs, axis=0)
                            embeddings.append(emb)
                        continue
                    
                    # Try word parts (for compound words)
                    parts = re.findall(r'[a-z]+', word.lower())
                    if len(parts) > 1:
                        part_embs = []
                        for part in parts:
                            try:
                                emb = word2vec[part]
                                part_embs.append(emb)
                            except KeyError:
                                continue
                        if part_embs:
                            # Average the part embeddings
                            emb = np.mean(part_embs, axis=0)
                            embeddings.append(emb)
                        continue
        
        if not embeddings:
            embeddings = [np.zeros(embedding_dim)]
            
        embeddings = np.array(embeddings)
        
        if len(embeddings) < max_length:
            padding = np.zeros((max_length - len(embeddings), embedding_dim))
            embeddings = np.vstack([embeddings, padding])
        else:
            embeddings = embeddings[:max_length]
            
        # Add specificity score
        specificity_words = ['specifically', 'exactly', 'precisely', 'namely']
        specificity_score = sum(1 for w in words if w in specificity_words)
        
        # Modify embeddings based on specificity
        if specificity_score > 0:
            embeddings *= (1 + 0.1 * specificity_score)
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_emb = self.text_to_embedding(self.queries[idx], self.word2vec, self.max_length)
        doc_emb = self.text_to_embedding(self.docs[idx], self.word2vec, self.max_length)
        return query_emb, doc_emb 