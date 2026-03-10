from ddgs import DDGS
from typing import List

def search_urls(query: str, max_results: int = 10) -> List[str]:
    """Search URLs using DuckDuckGo and return a list of links."""
    urls: List[str] = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            urls.append(result['href'])
    return urls