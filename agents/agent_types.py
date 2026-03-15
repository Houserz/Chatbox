from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RetrievedDoc:
    text: str
    metadata: Dict[str, Any]


def safe_text(x: Any) -> str:
    return "" if x is None else str(x)


def truncate(s: str, max_chars: int = 1400) -> str:
    s = safe_text(s)
    return s if len(s) <= max_chars else s[:max_chars] + "..."
