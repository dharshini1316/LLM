from __future__ import annotations

import re
from typing import Iterable, List

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_WS_RE = re.compile(r"\s+")

# Simple patterns for obvious secret-like values (API keys, tokens, etc.).
SECRET_PATTERNS = [
    re.compile(r"(api_?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{24,}"),  # JWT-like
]


def strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return MULTI_WS_RE.sub(" ", text).strip()


def remove_secrets(text: str) -> str:
    cleaned = text
    for pat in SECRET_PATTERNS:
        cleaned = pat.sub("[REDACTED_SECRET]", cleaned)
    return cleaned


def is_log_like(line: str) -> bool:
    """Heuristic: keep log-style lines even if they're noisy."""
    if any(tok in line for tok in ("GET ", "POST ", "ssh", "login", "failed password", "403", "404", "500")):
        return True
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", line):
        return True
    if re.search(r"\b\d{2}:\d{2}:\d{2}\b", line):
        return True
    return False


def clean_line(line: str) -> str:
    line = strip_html(line)
    line = remove_secrets(line)
    line = normalize_whitespace(line)
    return line


def clean_corpus(lines: Iterable[str]) -> List[str]:
    """Clean, deduplicate, and lightly filter corpus lines."""
    seen = set()
    out: List[str] = []
    for raw in lines:
        line = clean_line(raw)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        # Keep all log-like lines; otherwise require a bit of length.
        if len(line) < 10 and not is_log_like(line):
            continue
        out.append(line)
    return out

