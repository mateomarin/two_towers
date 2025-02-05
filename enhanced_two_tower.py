import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as api
from datasets import load_dataset
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import os
from datetime import datetime

class EnhancedTwoTower(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        # Same base architecture as SimpleTwoTower
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
        
        # Enhanced projection networks
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.doc_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def encode_query(self, query_emb):
        _, query_hidden = self.query_encoder(query_emb)
        query_hidden = torch.cat((query_hidden[-2], query_hidden[-1]), dim=1)
        query_vec = self.query_proj(query_hidden)
        return nn.functional.normalize(query_vec, p=2, dim=1)
    
    def encode_doc(self, doc_emb):
        _, doc_hidden = self.doc_encoder(doc_emb)
        doc_hidden = torch.cat((doc_hidden[-2], doc_hidden[-1]), dim=1)
        doc_vec = self.doc_proj(doc_hidden)
        return nn.functional.normalize(doc_vec, p=2, dim=1)
    
    def forward(self, query_emb, doc_emb):
        query_vec = self.encode_query(query_emb)
        doc_vec = self.encode_doc(doc_emb)
        return query_vec, doc_vec

class MarginRankingLoss(nn.Module):
    def __init__(self, margin=0.2, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, query_vec, pos_doc_vec, neg_doc_vec=None):
        # If no explicit negatives, use in-batch negatives
        if neg_doc_vec is None:
            sim_matrix = torch.matmul(query_vec, pos_doc_vec.t()) / self.temperature
            pos_mask = torch.eye(len(query_vec), device=query_vec.device)
            
            # Add margin to negative pairs
            sim_matrix = sim_matrix - self.margin * (1 - pos_mask)
            
            labels = torch.arange(len(query_vec), device=query_vec.device)
            loss = nn.functional.cross_entropy(sim_matrix, labels)
            return loss
        else:
            # Reshape neg_doc_vec to match query_vec
            batch_size = query_vec.size(0)
            num_negatives = neg_doc_vec.size(0) // batch_size
            
            # Reshape query_vec to compare with each negative
            query_vec_expanded = query_vec.unsqueeze(1).expand(-1, num_negatives, -1)
            neg_doc_vec = neg_doc_vec.view(batch_size, num_negatives, -1)
            
            # Compute similarities
            pos_sim = torch.nn.functional.cosine_similarity(query_vec, pos_doc_vec)
            neg_sim = torch.nn.functional.cosine_similarity(
                query_vec_expanded, 
                neg_doc_vec, 
                dim=2
            ).mean(dim=1)  # Average over negatives
            
            # Compute loss
            loss = torch.mean(torch.clamp(self.margin - pos_sim + neg_sim, min=0))
            return loss

def get_hard_negatives(query_vec, doc_vecs, positive_idx, k=5):
    with torch.no_grad():
        similarities = torch.nn.functional.cosine_similarity(
            query_vec.unsqueeze(0), 
            doc_vecs
        )
        # Zero out the positive example
        similarities[positive_idx] = -1
        # Get top k most similar docs that aren't the positive
        hard_neg_indices = torch.topk(similarities, k=k).indices
    return hard_neg_indices

class EnhancedDataset(Dataset):
    def __init__(self, queries: List[str], docs: List[str], word2vec, max_length: int = 30):
        super().__init__()
        self.queries = queries
        self.docs = docs
        self.word2vec = word2vec
        self.max_length = max_length
        self.embedding_dim = word2vec.vector_size
    
    def text_to_embedding(self, text: str) -> torch.Tensor:
        words = text.lower().split()[:self.max_length]
        embeddings = []
        
        for word in words:
            try:
                emb = self.word2vec[word]
                embeddings.append(emb)
            except KeyError:
                continue
        
        if not embeddings:
            embeddings = [np.zeros(self.embedding_dim)]
            
        embeddings = np.array(embeddings)
        
        if len(embeddings) < self.max_length:
            padding = np.zeros((self.max_length - len(embeddings), self.embedding_dim))
            embeddings = np.vstack([embeddings, padding])
        else:
            embeddings = embeddings[:self.max_length]
            
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_emb = self.text_to_embedding(self.queries[idx])
        doc_emb = self.text_to_embedding(self.docs[idx])
        return query_emb, doc_emb

def main():
    # Implementation of training loop with hard negative mining
    pass

if __name__ == "__main__":
    main() 