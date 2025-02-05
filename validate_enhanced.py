import torch
from torch.utils.data import DataLoader
import gensim.downloader as api
from datasets import load_dataset
from tqdm import tqdm
import os
from enhanced_two_tower import EnhancedTwoTower, EnhancedDataset

def encode_text(text: str, encoder_type: str, model, word2vec, device) -> torch.Tensor:
    dataset = EnhancedDataset([text], [text], word2vec)
    emb = dataset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        if encoder_type == 'query':
            vec = model.encode_query(emb)
        else:
            vec = model.encode_doc(emb)
    return vec.cpu()

def evaluate_model(model_path: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedTwoTower(embedding_dim=300, hidden_dim=512).to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Loading Word2Vec embeddings...")
    word2vec = api.load('word2vec-google-news-300')
    
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    val_dataset = dataset["validation"]
    
    print("\nPreparing MS MARCO validation index...")
    val_docs = []
    val_doc_to_queries = {}
    query_to_relevant = {}
    
    num_val_docs = 10000
    for sample in tqdm(val_dataset.select(range(num_val_docs))):
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
                if text not in val_doc_to_queries:
                    val_doc_to_queries[text] = []
                val_doc_to_queries[text].append(query)
    
    print(f"Built index with {len(val_docs)} documents")
    
    print("Encoding validation documents...")
    doc_encodings = []
    batch_size = 128
    for i in tqdm(range(0, len(val_docs), batch_size)):
        batch_docs = val_docs[i:i + batch_size]
        batch_encodings = []
        for doc in batch_docs:
            doc_vec = encode_text(doc, 'doc', model, word2vec, device)
            batch_encodings.append(doc_vec)
        doc_encodings.extend(batch_encodings)
    doc_encodings = torch.cat(doc_encodings, dim=0)
    
    def search_ms_marco(query: str, k: int = 5):
        query_vec = encode_text(query, 'query', model, word2vec, device)
        similarities = torch.nn.functional.cosine_similarity(query_vec, doc_encodings)
        top_k = torch.topk(similarities, k=k)
        results = []
        for idx, score in zip(top_k.indices, top_k.values):
            results.append((val_docs[idx], score.item()))
        return results
    
    # Test queries
    print("\nTesting with MS MARCO validation queries...")
    num_test_queries = 20  # Increased number of test queries
    test_count = 0
    mrr_sum = 0
    
    with open(os.path.join(output_dir, "validation_results.txt"), "w") as f:
        f.write("Enhanced Model Validation Results\n")
        f.write("==============================\n\n")
        
        for sample in val_dataset:
            query = sample.get("query", "")
            if not query in query_to_relevant or not query_to_relevant[query]:
                continue
                
            relevant_docs = set(query_to_relevant[query])
            results = search_ms_marco(query, k=10)
            
            f.write(f"\nQuery: {query}\n")
            f.write(f"Number of relevant documents: {len(relevant_docs)}\n")
            
            # Calculate MRR
            mrr = 0
            for rank, (doc, score) in enumerate(results, 1):
                is_relevant = "✓" if doc in relevant_docs else " "
                f.write(f"{rank}. [{is_relevant}] ({score:.4f}) {doc[:200]}...\n")
                
                if doc in relevant_docs and mrr == 0:
                    mrr = 1.0 / rank
            
            mrr_sum += mrr
            test_count += 1
            
            print(f"\nQuery: {query}")
            print(f"MRR: {mrr:.4f}")
            for rank, (doc, score) in enumerate(results[:3], 1):
                is_relevant = "✓" if doc in relevant_docs else " "
                print(f"{rank}. [{is_relevant}] ({score:.4f}) {doc[:100]}...")
            
            if test_count >= num_test_queries:
                break
        
        avg_mrr = mrr_sum / test_count
        print(f"\nAverage MRR over {test_count} queries: {avg_mrr:.4f}")
        f.write(f"\nAverage MRR over {test_count} queries: {avg_mrr:.4f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_model(args.model_path, args.output_dir) 