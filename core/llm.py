"""LLM factory — provider-agnostic.

Started Groq-only; now a small registry so the same `get_llm()` call
works for Groq, OpenAI, Anthropic, or a local Ollama model. Each provider
is imported lazily inside its builder so a missing optional dependency
for a provider you don't use never breaks the ones you do.
"""

from typing import Optional

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.1",
}

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _build_groq(api_key: str, model_name: str, streaming: bool, **_):
    from langchain_groq import ChatGroq

    return ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=streaming)


def _build_openai(api_key: str, model_name: str, streaming: bool, **_):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(api_key=api_key, model=model_name, streaming=streaming)


def _build_anthropic(api_key: str, model_name: str, streaming: bool, **_):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(api_key=api_key, model=model_name, streaming=streaming)


def _build_ollama(model_name: str, base_url: str, streaming: bool, **_):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model_name, base_url=base_url or DEFAULT_OLLAMA_BASE_URL)


# provider name -> (builder, needs_api_key)
_PROVIDERS = {
    "groq": (_build_groq, True),
    "openai": (_build_openai, True),
    "anthropic": (_build_anthropic, True),
    "ollama": (_build_ollama, False),
}

# Kept for the legacy single-provider call sites and as the implicit
# default when `provider` isn't specified.
DEFAULT_MODEL = DEFAULT_MODELS["groq"]


def get_llm(
    api_key: str = "",
    provider: str = "groq",
    model_name: Optional[str] = None,
    streaming: bool = True,
    base_url: Optional[str] = None,
):
    """Return a LangChain chat model for the given provider.

    Args:
        api_key: required for groq/openai/anthropic; ignored for ollama.
        provider: one of "groq", "openai", "anthropic", "ollama".
        model_name: defaults to a sane per-provider model if omitted.
        base_url: only used by ollama (defaults to localhost:11434).

    Raises:
        ValueError: unknown provider, or a required API key is missing.
    """
    provider = (provider or "groq").strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose one of: {', '.join(_PROVIDERS)}"
        )

    builder, needs_key = _PROVIDERS[provider]
    if needs_key and not api_key:
        raise ValueError(f"An API key is required for provider '{provider}'.")

    resolved_model = model_name or DEFAULT_MODELS[provider]
    return builder(
        api_key=api_key,
        model_name=resolved_model,
        streaming=streaming,
        base_url=base_url,
    )
