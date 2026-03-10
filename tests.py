# tests.py
import time
import pytest
from utils import clean_html
from summarizer import summarize_text
from ranking_tf import rank_urls, embed_text
from main import agentic_rag
from sklearn.metrics.pairwise import cosine_similarity
import requests

# --------------------------
# Unit Tests
# --------------------------
def test_clean_html():
    html = "<p>Hello <a href='url'>World</a></p>"
    assert clean_html(html) == "Hello World"

def test_summarize_text():
    text = "AI is amazing. It can help humans. It learns fast."
    summary = summarize_text(text, sentences_count=2)
    assert isinstance(summary, str)
    assert "AI" in summary

def test_rank_urls():
    url_texts = [("url1", "AI is great."), ("url2", "Humans are amazing.")]
    query = "AI"
    ranked = rank_urls(url_texts, query)
    # The most relevant URL should be first
    assert ranked[0][0] == "url1"
    # Scores should be floats
    assert isinstance(ranked[0][2], float)

# --------------------------
# Integration Test: Agentic RAG
# --------------------------
def test_agentic_rag():
    query = "latest AI trends"
    answer, ranked = agentic_rag(query)
    assert isinstance(answer, str)
    # Should return a non-empty list
    assert len(ranked) > 0
    # Each item should have URL, summary, score
    for url, summary, score in ranked:
        assert isinstance(url, str)
        assert isinstance(summary, str)
        assert isinstance(score, float)

# --------------------------
# Benchmark Test
# --------------------------
def test_pipeline_performance():
    query = "AI in healthcare"
    start_time = time.time()
    answer, ranked = agentic_rag(query)
    duration = time.time() - start_time
    print(f"\nPipeline executed in {duration:.2f} seconds")
    # Check pipeline finishes in <60s for test data
    assert duration < 60

# --------------------------
# Semantic similarity
# --------------------------
def test_semantic_similarity():
    text1 = "AI helps humans in multiple industries."
    text2 = "AI is assisting humans in healthcare and finance."
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    sim = cosine_similarity([emb1], [emb2])[0][0]
    print(f"\nSemantic similarity: {sim:.3f}")
    # Should be moderately similar
    assert sim > 0.5

# --------------------------
# Test POST API integration
# --------------------------
def test_agentic_rag_post(monkeypatch):
    class MockPostResponse:
        def __init__(self, json_data):
            self._json = json_data
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def json(self):
            return self._json

        @property
        def text(self):
            return str(self._json)

    def mock_post(*args, **kwargs):
        return MockPostResponse({"result": "AI can assist doctors in diagnostics."})

    monkeypatch.setattr(requests, "post", mock_post)

    query = "AI in medicine"
    answer, ranked = agentic_rag(query)
    assert isinstance(answer, str)
    assert len(ranked) > 0
    assert "AI" in answer

# --------------------------
# Test multiple API responses (GET)
# --------------------------
def test_agentic_rag_multiple_apis(monkeypatch):
    responses = [
        {"data": "AI improves energy efficiency."},
        {"data": "AI optimizes logistics."},
    ]

    class MockGetResponse:
        def __init__(self, data):
            self._data = data
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def json(self):
            return self._data

        @property
        def text(self):
            return str(self._data)

    def mock_get(*args, **kwargs):
        return MockGetResponse(responses.pop(0))

    monkeypatch.setattr(requests, "get", mock_get)

    query = "AI in industry"
    answer, ranked = agentic_rag(query)
    assert isinstance(answer, str)
    assert len(ranked) == 2
    for url, summary, score in ranked:
        assert isinstance(summary, str)
        assert isinstance(score, float)

# --------------------------
# Run all tests if executed directly
# --------------------------
if __name__ == "__main__":
    print("Running all tests...\n")
    pytest.main(["-v", "tests.py"])