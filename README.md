Agentic AI Project with RAG
Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) system capable of:

Searching the web for relevant URLs using DuckDuckGo.

Fetching and parsing HTML content from the URLs.

Summarizing content using LexRank.

Ranking content by semantic similarity to a user query using sentence embeddings.

Persisting content into a vector store (FAISS) for retrieval.

Generating a grounded answer using a large language model (Flan-T5) based on retrieved content.

This design allows the agent to answer user queries with contextually accurate information and optional citations.
Project Architecture
┌───────────────────────────────┐
│          User Query           │
└──────────────┬────────────────┘
               │
               ▼
      ┌───────────────────┐
      │  URL Search Agent │
      │ (DuckDuckGo API)  │
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │ Content Fetcher   │
      │ (requests + HTML) │
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │   Summarizer      │
      │   (LexRank)       │
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │  Ranker (TF)      │
      │ Sentence Embeds   │
      │ Cosine Similarity │
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │ Vector Store (FAISS) │
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │  LLM Generator    │
      │ (Flan-T5)         │
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │   Answer Output   │
      └───────────────────┘
1- Setup Python environment
       - python -m venv agent_env
-.\agent_env\Scripts\activate  # Windows
-source agent_env/bin/activate # macOS/Linux
2- Install dependencies
pip install -r requirements.txt
3- Run the main agent:
python main.py
Enter your question: latest AI trends in 2026

=== Agentic AI Answer ===
AI agents are evolving from personal assistants to orchestrators of complex workflows. 
Research labs increasingly use AI to assist experiments, and industries adopt retrieval-based generation for accurate, context-aware responses.

=== Detailed Scores ===
URL: https://news.microsoft.com/source/features/ai/whats-next-in-ai-7-trends-to-watch-in-2026/
Summary: Microsoft AI’s Diagnostic Orchestrator solved complex medical cases with 85.5% accuracy...
Score: 0.6013

URL: https://www.ibm.com/think/insights/artificial-intelligence-future
Summary: LLMs are trained on specific domains and provide always-on assistance for enterprises...
Score: 0.4046
4-Run all tests:
pytest -v tests.py
Includes unit tests for:

HTML cleaning

Summarization

URL ranking

Semantic similarity

Pipeline performance. 


Key Features

RAG-enabled: Uses vector store to retrieve relevant chunks before generation.

Dynamic URL fetching: Works on live web content.

Summarization: Condenses long pages for efficiency.

Semantic ranking: Ensures top results are most relevant.

LLM-grounded generation: Produces accurate answers with context.
