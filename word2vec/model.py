# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=150):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings with a smaller range
        nn.init.uniform_(self.in_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.out_embeddings.weight, -0.1, 0.1)
        
        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, target_words, context_words, neg_words):
        """
        Compute loss using InfoNCE-style contrastive learning
        """
        # Get embeddings and normalize
        target_embeds = self.in_embeddings(target_words)  # [batch_size, embed_dim]
        context_embeds = self.out_embeddings(context_words)  # [batch_size, embed_dim]
        neg_embeds = self.out_embeddings(neg_words)  # [batch_size, n_neg, embed_dim]
        
        # L2 normalize all embeddings
        target_embeds = F.normalize(target_embeds, dim=1)
        context_embeds = F.normalize(context_embeds, dim=1)
        neg_embeds = F.normalize(neg_embeds, dim=2)
        
        # Temperature parameter
        temperature = torch.exp(self.log_temperature)
        
        # Compute positive and negative scores
        pos_scores = torch.sum(target_embeds * context_embeds, dim=1) / temperature
        
        # Reshape for batch matrix multiplication
        target_embeds_expanded = target_embeds.unsqueeze(1)  # [batch_size, 1, embed_dim]
        neg_scores = torch.bmm(neg_embeds, target_embeds_expanded.transpose(1, 2)).squeeze(2)
        neg_scores = neg_scores / temperature  # [batch_size, n_neg]
        
        # Compute InfoNCE loss
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch_size, 1 + n_neg]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # Positive is at index 0
        
        loss = F.cross_entropy(logits, labels)
        
        return loss