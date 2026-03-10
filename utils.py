import re

def clean_html(html: str) -> str:
    text = re.sub(r"<.*?>", "", html)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]