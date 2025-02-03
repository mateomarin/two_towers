# two_tower_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings, freeze_embeddings=True):
        super(TwoTowerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.query_encoder = nn.GRU(input_size=embedding_dim,
                                    hidden_size=hidden_dim,
                                    batch_first=True)
        self.doc_encoder = nn.GRU(input_size=embedding_dim,
                                  hidden_size=hidden_dim,
                                  batch_first=True)
    
    def encode_query(self, query_tokens):
        emb = self.embedding(query_tokens)
        _, h_n = self.query_encoder(emb)
        return F.normalize(h_n.squeeze(0), p=2, dim=1)
    
    def encode_doc(self, doc_tokens):
        emb = self.embedding(doc_tokens)
        _, h_n = self.doc_encoder(emb)
        return F.normalize(h_n.squeeze(0), p=2, dim=1)
    
    def forward(self, query_tokens, pos_doc_tokens, neg_doc_tokens):
        q_enc = self.encode_query(query_tokens)
        pos_enc = self.encode_doc(pos_doc_tokens)
        neg_enc = self.encode_doc(neg_doc_tokens)
        return q_enc, pos_enc, neg_enc

if __name__ == "__main__":
    import torch
    dummy_query = torch.randint(0, 100, (4, 10))
    dummy_input = torch.randint(0, 100, (4, 10))
    model = TwoTowerModel(vocab_size=100, embedding_dim=300, hidden_dim=128,
                          pretrained_embeddings=torch.randn(100, 300))
    q, p, n = model(dummy_query, dummy_input, dummy_input)
    print("Query encoding shape:", q.shape)