import torch
from torch.utils.data import DataLoader
import gensim.downloader as api
from datasets import load_dataset
from tqdm import tqdm
import os
from enhanced_two_tower import EnhancedTwoTower
from simple_two_tower import SimpleTwoTower
from tabulate import tabulate

def load_and_evaluate_model(model_class, model_path, word2vec, val_dataset, num_queries=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(embedding_dim=300, hidden_dim=512).to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Build validation index
    val_docs = []
    query_to_relevant = {}
    
    for sample in tqdm(val_dataset.select(range(10000))):
        query = sample.get("query", "")
        if not query or "passages" not in sample:
            continue
            
        passages = sample["passages"]
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        
        query_to_relevant[query] = []
        for text, selected in zip(passage_texts, is_selected):
            if text not in val_docs:
                val_docs.append(text)
            if selected == 1:
                query_to_relevant[query].append(text)
    
    # Evaluate
    test_count = 0
    mrr_sum = 0
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_10 = 0
    
    for sample in val_dataset:
        query = sample.get("query", "")
        if not query in query_to_relevant or not query_to_relevant[query]:
            continue
            
        relevant_docs = set(query_to_relevant[query])
        results = search_ms_marco(query, model, word2vec, val_docs, device)
        
        # Calculate metrics
        mrr = 0
        found_in_top_1 = False
        found_in_top_3 = False
        found_in_top_10 = False
        
        for rank, (doc, _) in enumerate(results, 1):
            if doc in relevant_docs:
                if mrr == 0:
                    mrr = 1.0 / rank
                if rank == 1:
                    found_in_top_1 = True
                if rank <= 3:
                    found_in_top_3 = True
                if rank <= 10:
                    found_in_top_10 = True
        
        mrr_sum += mrr
        recall_at_1 += int(found_in_top_1)
        recall_at_3 += int(found_in_top_3)
        recall_at_10 += int(found_in_top_10)
        
        test_count += 1
        if test_count >= num_queries:
            break
    
    return {
        'MRR': mrr_sum / test_count,
        'R@1': recall_at_1 / test_count,
        'R@3': recall_at_3 / test_count,
        'R@10': recall_at_10 / test_count
    }

def search_ms_marco(query, model, word2vec, val_docs, device, k=10):
    # Implementation similar to previous search function
    pass

def main():
    print("Loading Word2Vec embeddings...")
    word2vec = api.load('word2vec-google-news-300')
    
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    val_dataset = dataset["validation"]
    
    # Evaluate both models
    simple_results = load_and_evaluate_model(
        SimpleTwoTower,
        "output/run_YYYYMMDD_HHMMSS/best_model.pt",  # Update path
        word2vec,
        val_dataset
    )
    
    enhanced_results = load_and_evaluate_model(
        EnhancedTwoTower,
        "output/enhanced_run_YYYYMMDD_HHMMSS/best_model.pt",  # Update path
        word2vec,
        val_dataset
    )
    
    # Print comparison
    headers = ['Model', 'MRR', 'Recall@1', 'Recall@3', 'Recall@10']
    table = [
        ['Simple', simple_results['MRR'], simple_results['R@1'], 
         simple_results['R@3'], simple_results['R@10']],
        ['Enhanced', enhanced_results['MRR'], enhanced_results['R@1'],
         enhanced_results['R@3'], enhanced_results['R@10']]
    ]
    
    print("\nModel Comparison:")
    print(tabulate(table, headers=headers, floatfmt='.4f'))

if __name__ == "__main__":
    main() 