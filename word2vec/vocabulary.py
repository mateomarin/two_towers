import os
import re
from collections import Counter
import numpy as np
from nltk.stem import PorterStemmer

class Vocabulary:
    def __init__(self, filepath, min_count=20, subsample_t=1e-5):
        """
        Builds vocabulary from a given text file with subsampling.
        """
        self.filepath = filepath
        self.min_count = min_count
        self.subsample_t = subsample_t
        self.stemmer = PorterStemmer()
        
        # Common words to exclude from similarity calculations
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over',
            'after', 'that', 'this', 'these', 'those', 'it', 'its', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'can', 'could', 'will', 'would', 'should', 'must',
            'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
            'they', 'them', 'their', 'we', 'us', 'our', 'i', 'me', 'my', 'mine',
            'who', 'which', 'what', 'where', 'when', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such'
        }

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Read and preprocess the text
        with open(filepath, 'r', encoding="utf-8") as f:
            text = f.read().strip()
        
        self.processed_tokens = self._preprocess(text, "processed-dictionary.txt")

        # Build vocabulary dictionary
        word_freq = Counter(self.processed_tokens)
        
        # Filter out stop words and apply minimum count
        filtered_words = {
            w: c for w, c in word_freq.items() 
            if c >= self.min_count and w not in self.stop_words
        }
        
        # Sort by frequency for better embedding learning
        sorted_vocab = sorted(
            filtered_words.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        self.dictionary = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
        self.indices_to_tokens = {idx: word for word, idx in self.dictionary.items()}
        self.word_frequencies = {word: count for word, count in filtered_words.items()}

        print(f"Built vocabulary of size: {len(self.dictionary)}")
        
        # Print some statistics
        total_tokens = sum(word_freq.values())
        kept_tokens = sum(filtered_words.values())
        print(f"Total tokens: {total_tokens}")
        print(f"Tokens after filtering: {kept_tokens}")
        print(f"Token retention rate: {kept_tokens/total_tokens:.2%}")

        with open("dictionary.txt", "w", encoding="utf-8") as file:
            file.write(str(self.dictionary))

    def clean_text(self, text):
        """
        Applies cleaning steps (similar to _preprocess) without file I/O.
        """
        # Preprocessing steps
        text = re.sub(r"n't\b", " not", text)
        text = re.sub(r"'s\b", " is", text)
        text = re.sub(r"'re\b", " are", text)
        
        # Replace mentions, hashtags, URLs
        text = re.sub(r"@\w+", " <MENTION> ", text)
        text = re.sub(r"#\w+", " <HASHTAG> ", text)
        text = re.sub(r"http\S+", " <URL> ", text)
        
        # Replace punctuation with tokens
        text = text.replace(".", " <PERIOD> ")
        text = text.replace(",", " <COMMA> ")
        text = text.replace('"', " <QUOTATION_MARK> ")
        text = text.replace(";", " <SEMICOLON> ")
        text = text.replace("!", " <EXCLAMATION_MARK> ")
        text = text.replace("?", " <QUESTION_MARK> ")
        text = text.replace("(", " <LEFT_PAREN> ")
        text = text.replace(")", " <RIGHT_PAREN> ")
        text = text.replace("[", " <LEFT_BRACKET> ")
        text = text.replace("]", " <RIGHT_BRACKET> ")
        text = text.replace("{", " <LEFT_BRACE> ")
        text = text.replace("}", " <RIGHT_BRACE> ")
        text = text.replace("/", " <SLASH> ")
        text = text.replace("\\", " <BACKSLASH> ")
        text = text.replace("-", " <HYPHEN> ")
        text = text.replace("--", " <DOUBLE_HYPHEN> ")
        text = text.replace(":", " <COLON> ")
        text = text.replace("+", " <PLUS> ")
        text = text.replace("*", " <ASTERISK> ")
        text = text.replace("&", " <AMPERSAND> ")
        text = re.sub(r"\d+", " <DIGIT> ", text)
        text = re.sub(r"[^\x00-\x7F]+", " <UNICODE> ", text)
        
        # Remove any leftover special characters and extra whitespace
        text = re.sub(r"[^\w\s<>]", "", text)
        text = re.sub(r"\s+", " ", text)
        
        # Lowercase everything
        text = text.lower()
        
        # Tokenize and apply stemming
        tokens = text.split()
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Filter tokens (ensure token length > 2 or special tokens)
        tokens = [t for t in tokens if len(t) > 2 or t in {"<url>", "<mention>", "<hashtag>"}]
        
        # Apply subsampling logic (as in your original _preprocess)
        if self.subsample_t:
            total = len(tokens)
            if total > 0:
                word_counts = Counter(tokens)
                new_tokens = []
                for token in tokens:
                    freq = word_counts[token] / total
                    discard_prob = 1.0 - np.sqrt(self.subsample_t / freq)
                    if discard_prob > 1.0:
                        continue
                    if np.random.random() < discard_prob:
                        continue
                    new_tokens.append(token)
                tokens = new_tokens
        return tokens

    def tokenize(self, text):
        """
        Tokenizes a single string using the cleaning and subsampling logic.
        """
        return self.clean_text(text)

    def _preprocess(self, text, output_file):
        """
        Processes the full text, writes the tokens to file, and returns the tokens.
        """
        tokens = self.clean_text(text)
        output_path = os.path.join("data", output_file)
        os.makedirs("data", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(" ".join(tokens))
        return tokens