from utils import load_or_cache_msmarco

def load_ms_marco_train():
    """
    Load and prepare MS MARCO training dataset.
    Returns:
        queries: List of query strings
        docs: List of document strings
    """
    dataset = load_or_cache_msmarco('train')
    
    queries = []
    docs = []
    
    print("Processing training samples...")
    for sample in dataset:
        query = sample.get("query", "")
        if not query or "passages" not in sample:
            continue
            
        passages = sample["passages"]
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        
        for text, selected in zip(passage_texts, is_selected):
            if selected == 1:
                queries.append(query)
                docs.append(text)
    
    return queries, docs 