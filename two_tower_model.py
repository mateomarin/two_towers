# two_tower_model.py
import torch
import torch.nn.functional as F

class TwoTowerModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings, freeze_embeddings=True):
        super(TwoTowerModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        self.query_tower = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.doc_tower = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

    def encode_query(self, query_tokens):
        emb = self.embedding(query_tokens)
        _, h_n = self.query_tower(emb)
        return F.normalize(h_n.squeeze(0), p=2, dim=1)

    def encode_doc(self, doc_tokens):
        emb = self.embedding(doc_tokens)
        _, h_n = self.doc_tower(emb)
        return F.normalize(h_n.squeeze(0), p=2, dim=1)
    
    def forward(self, query_tokens, pos_doc_tokens, neg_doc_tokens):
        query_emb = self.encode_query(query_tokens)
        pos_doc_emb = self.encode_doc(pos_doc_tokens)
        neg_doc_emb = self.encode_doc(neg_doc_tokens)

        return query_emb, pos_doc_emb, neg_doc_emb
    
if __name__ == "__main__":
    # Test model with dummy inputs
    dummy_input = torch.randint(0, 100, (4, 10))
    model = TwoTowerModel(vocab_size=100, embedding_dim=300, hidden_dim=128, pretrained_embeddings=torch.randn(100, 300))
    q, p, n = model(dummy_input, dummy_input, dummy_input)
    print("Query embedding shape:", q.shape)
