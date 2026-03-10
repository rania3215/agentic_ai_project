# main.py
from utils import chunk_text
from url_fetch_agent import search_urls, fetch_content
from summarizer import summarize_text
from ranking_tf import rank_urls
from generator import generate_answer
from vector_store import create_vector_store, retrieve_chunks


def agentic_rag(query: str):
    # 1️⃣ Search URLs
    urls = search_urls(query)
    
    # 2️⃣ Fetch content
    url_texts = fetch_content(urls)
    
    # 3️⃣ Summarize content
    summarized = [(url, summarize_text(text, sentences_count=3)) for url, text in url_texts]
    
    # 4️⃣ Rank URLs
    ranked = rank_urls(summarized, query)  # each item is (url, summary, score)
    
    # 5️⃣ Extract summaries (or chunks)
    chunks = [summary for _, summary, _ in ranked]
    
    # 6️⃣ Generate final answer
    answer = generate_answer(query, " ".join(chunks))
    
    return answer, ranked

if __name__ == "__main__":
    query = input("Enter your question: ")
    answer, ranked_results = agentic_rag(query)
    print("\n=== Agentic RAG Answer ===\n")
    print(answer)
    print("\n=== Ranked URLs ===\n")
    for url, summary, score in ranked_results:
        print(f"URL: {url}\nSummary: {summary}\nScore: {score:.4f}\n")