import requests
from ddgs import DDGS
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional

def is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc != ""

def search_urls(query: str, max_results: int = 5) -> List[str]:
    """Search DuckDuckGo and return a list of safe URLs."""
    urls: List[str] = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            href = result.get('href')
            if href and is_safe_url(href):
                urls.append(href)
            if len(urls) >= max_results:
                break
    return urls

def fetch_content(urls: List[str], method: str = "GET", data: Optional[dict] = None) -> List[Tuple[str, str]]:
    """Fetch text content from URLs (HTML or JSON). Supports GET or POST."""
    url_texts = []
    for url in urls:
        try:
            if method.upper() == "POST":
                r = requests.post(url, json=data, timeout=5)
            else:
                r = requests.get(url, timeout=5)

            if r.status_code == 200:
                content_type = r.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    text = str(r.json())  # Convert JSON to string for LLM
                else:
                    soup = BeautifulSoup(r.text, "html.parser")
                    text = " ".join(p.get_text() for p in soup.find_all("p"))
                url_texts.append((url, text))
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return url_texts