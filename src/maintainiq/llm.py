"""Generation backends. The system stays usable without a local model."""

from __future__ import annotations

import logging
import re

import requests

from maintainiq.config import Settings
from maintainiq.retrieval import tokenize
from maintainiq.schemas import RetrievalResult

logger = logging.getLogger(__name__)


class LLMBackend:
    name: str

    def generate(self, prompt: str, evidence: list[RetrievalResult]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_s
        self.name = f"ollama:{self.model}"

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.ok and any(m.get("name") == self.model for m in resp.json().get("models", []))
        except (requests.RequestException, ValueError):
            return False

    def generate(self, prompt: str, evidence: list[RetrievalResult]) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


_CAUSAL_QUESTION = re.compile(r"\b(why|cause|caused|reason|root)\b", re.I)
_CAUSE_CUES = frozenset(tokenize("caused cause indicates point due from result lead root diagnosis"))


class ExtractiveBackend(LLMBackend):
    """Deterministic fallback with no generative step. For each top evidence document it quotes
    the sentence that best matches the question and cites it. For causal questions it prefers
    root-cause and diagnosis sentences over symptom lists. It never synthesizes across sources,
    so it is conservative by design."""

    name = "extractive"

    def generate(self, prompt: str, evidence: list[RetrievalResult]) -> str:
        question = _question_from_prompt(prompt)
        terms = set(tokenize(question))
        causal = bool(_CAUSAL_QUESTION.search(question))
        lines, used = [], set()
        for n, result in enumerate(evidence[:3], start=1):
            sentence = _best_sentence(result.document.text, terms, causal)
            if sentence and sentence not in used:
                used.add(sentence)
                lines.append(f"{sentence} [E{n}]")
        return " ".join(lines)


def _question_from_prompt(prompt: str) -> str:
    match = re.search(r"Question:\s*(.+?)\s*Answer:\s*$", prompt, re.DOTALL)
    return match.group(1) if match else ""


def _best_sentence(text: str, terms: set[str], causal: bool) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    candidates = []
    for sentence in sentences:
        if sentence.lower().startswith("work order"):
            continue
        tokens = set(tokenize(sentence))
        score = len(terms & tokens)
        if causal:
            score += 1.5 * len(_CAUSE_CUES & tokens)
            if sentence.startswith("Symptoms:"):
                score -= 2
        candidates.append((score, -len(sentence), sentence))
    if not candidates:
        return ""
    best = max(candidates)
    if best[0] <= 0:
        return ""
    return best[2].rstrip(".") + "."


def build_backend(settings: Settings) -> LLMBackend:
    """Use Ollama when configured and the model is installed. Otherwise fall back and say so."""
    if settings.llm_provider == "ollama":
        backend = OllamaBackend(settings)
        if backend.is_available():
            return backend
        logger.warning("Ollama model %s not available; using extractive fallback", settings.llm_model)
    return ExtractiveBackend()
