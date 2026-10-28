import requests

from maintainiq.config import Settings
from maintainiq.llm import ExtractiveBackend, OllamaBackend, build_backend


def test_falls_back_to_extractive_when_ollama_is_unreachable():
    settings = Settings(llm_provider="ollama", ollama_base_url="http://127.0.0.1:9", llm_model="llama3.2:latest")
    assert isinstance(build_backend(settings), ExtractiveBackend)


def test_uses_ollama_when_model_is_installed(monkeypatch):
    class FakeResp:
        ok = True

        def json(self):
            return {"models": [{"name": "llama3.2:latest"}]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    settings = Settings(llm_provider="ollama", llm_model="llama3.2:latest")
    backend = build_backend(settings)
    assert isinstance(backend, OllamaBackend)
    assert backend.name == "ollama:llama3.2:latest"


def test_extractive_provider_never_calls_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network should not be used")

    monkeypatch.setattr(requests, "get", boom)
    assert isinstance(build_backend(Settings(llm_provider="extractive")), ExtractiveBackend)


def test_ollama_generate_posts_deterministic_options(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": " Answer [E1]. "}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json)
        return FakeResp()

    monkeypatch.setattr(requests, "post", fake_post)
    backend = OllamaBackend(Settings(llm_model="llama3.2:latest"))
    assert backend.generate("prompt", []) == "Answer [E1]."
    assert captured["json"]["options"]["temperature"] == 0.0
    assert captured["json"]["stream"] is False
