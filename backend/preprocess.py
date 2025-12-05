import re
import unicodedata
from typing import List

CONTROL_CHARS_RE = re.compile(r"[\r\t]+")
MULTI_SPACE_RE = re.compile(r"\s+")

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")

def strip_control_chars(text: str) -> str:
    return CONTROL_CHARS_RE.sub(" ", text)

def strip_unwanted_chars(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.,;:?!'\"\-()\[\]/\\\s]", " ", text)

def collapse_spaces(text: str) -> str:
    return MULTI_SPACE_RE.sub(" ", text).strip()

def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = strip_control_chars(text)
    text = strip_unwanted_chars(text)
    text = collapse_spaces(text)
    return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks
