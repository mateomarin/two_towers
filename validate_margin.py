import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from margin_two_tower import TwoTowerModel, SimpleDataset
from tqdm import tqdm
import argparse
import random
from utils import load_or_cache_word2vec, load_or_cache_msmarco

def compute_mrr(model, query, relevant_docs, candidate_docs, word2vec, device):
    """Compute MRR using cosine similarity"""
    model.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        # Get query embedding
        query_emb = SimpleDataset.text_to_embedding(query, word2vec)
        query_emb = query_emb.unsqueeze(0).to(device)
        query_vec = model.encode_query(query_emb)
        
        # Process documents in batches
        batch_size = 128
        similarities = []
        
        for i in range(0, len(candidate_docs), batch_size):
            batch_docs = candidate_docs[i:i + batch_size]
            batch_embs = []
            
            # Create document embeddings
            for doc in batch_docs:
                doc_emb = SimpleDataset.text_to_embedding(doc, word2vec)
                batch_embs.append(doc_emb)
            
            # Stack and process batch
            doc_embs = torch.stack(batch_embs).to(device)
            doc_vecs = model.encode_doc(doc_embs)
            
            # Compute similarities for batch
            batch_sims = torch.nn.functional.cosine_similarity(
                query_vec.unsqueeze(1),  # [1, 1, dim]
                doc_vecs.unsqueeze(0),   # [1, batch, dim]
                dim=2
            ).squeeze()
            
            similarities.extend(batch_sims.cpu().tolist())
    
    # Sort and find first relevant doc
    doc_scores = list(zip(candidate_docs, similarities))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (doc, score) in enumerate(doc_scores, 1):
        if doc in relevant_docs:
            return 1.0 / rank, doc_scores[:3]
    
    return 0.0, doc_scores[:3]

def filter_validation_docs(val_docs, min_length=20, max_length=1000):
    """Filter validation docs to remove very short or very long passages"""
    return [
        doc for doc in val_docs 
        if min_length <= len(doc.split()) <= max_length
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    # Load cached embeddings and dataset
    word2vec = load_or_cache_word2vec()
    validation = load_or_cache_msmarco('validation')
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoTowerModel(embedding_dim=300, hidden_dim=512).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare validation data
    val_docs = []
    query_to_relevant = {}
    
    print(f"\nProcessing validation samples...")
    for sample in tqdm(validation):
        query = sample.get("query", "")
        if not query or "passages" not in sample:
            continue
            
        passages = sample["passages"]
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        
        relevant = set()
        for text, selected in zip(passage_texts, is_selected):
            if text not in val_docs:
                val_docs.append(text)
            if selected == 1:
                relevant.add(text)
        if relevant:
            query_to_relevant[query] = relevant
    
    print(f"Built index with {len(val_docs)} documents")
    
    # Test queries
    print("\nTesting with MS MARCO validation queries...")
    num_test_queries = 20
    total_mrr = 0
    queries_evaluated = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "validation_results.txt"), "w") as f:
        # Select random queries that have answers
        valid_samples = [
            s for s in validation 
            if s.get("query") and (s.get("answers") or s.get("wellFormedAnswers"))
        ]
        random.shuffle(valid_samples)
        
        for sample in valid_samples[:num_test_queries]:
            query = sample.get("query", "")
            answers = sample.get("answers", [])
            wellFormedAnswers = sample.get("wellFormedAnswers", [])
            answer = wellFormedAnswers[0] if wellFormedAnswers else answers[0]
            
            # Find passage containing the answer
            relevant_docs = query_to_relevant.get(query, set())
            ground_truth_passage = None
            for passage in relevant_docs:
                if answer.lower() in passage.lower():
                    ground_truth_passage = passage
                    break
            
            # Use first relevant passage if answer not found
            if not ground_truth_passage and relevant_docs:
                ground_truth_passage = next(iter(relevant_docs))
            
            mrr, top_results = compute_mrr(model, query, relevant_docs, val_docs, word2vec, device)
            total_mrr += mrr
            queries_evaluated += 1
            
            # Log results
            print(f"\nQuery: {query}")
            print(f"Ground Truth Answer: {answer}")
            print(f"Ground Truth Passage: {ground_truth_passage[:100]}...")
            print(f"MRR: {mrr:.4f}")
            for rank, (doc, score) in enumerate(top_results, 1):
                is_relevant = "✓" if doc in relevant_docs else " "
                print(f"{rank}. [{is_relevant}] ({score:.4f}) {doc[:100]}...")
                
            # Write to file
            f.write(f"\nQuery: {query}\n")
            f.write(f"Ground Truth Answer: {answer}\n")
            f.write(f"Ground Truth Passage: {ground_truth_passage}\n")
            f.write(f"MRR: {mrr:.4f}\n")
            for rank, (doc, score) in enumerate(top_results, 1):
                is_relevant = "✓" if doc in relevant_docs else " "
                f.write(f"{rank}. [{is_relevant}] ({score:.4f}) {doc[:100]}...\n")
    
        avg_mrr = total_mrr / queries_evaluated
        print(f"\nAverage MRR over {queries_evaluated} queries: {avg_mrr:.4f}")
        f.write(f"\nAverage MRR over {queries_evaluated} queries: {avg_mrr:.4f}\n")

if __name__ == "__main__":
    main() 