from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def embed_text(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True, show_progress_bar=False)

def embed_texts(texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def rank_urls(url_texts: List[Tuple[str, str]], query: str) -> List[Tuple[str, str, float]]:
    query_emb = embed_text(query)
    texts = [text for _, text in url_texts]
    doc_embs = embed_texts(texts)

    scored = []
    for (url, text), doc_emb in zip(url_texts, doc_embs):
        # Cosine similarity
        score = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scored.append((url, text, float(score)))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored