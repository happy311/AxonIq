"""
AxonIQ — LLM Provider

Strategy (priority order):
  1. Anthropic Claude  — best format adherence, no parsing fragility
                         requires ANTHROPIC_API_KEY env var
                         Runtime fallback: tries all CLAUDE_FALLBACK_MODELS
                         before switching to HuggingFace
  2. HuggingFace       — free fallback; auto-probes working chat model
                         Runtime fallback: cycles through _FREE_FALLBACK_MODELS
                         when the current model's credits/quota run out
  3. Ollama            — local dev fallback when USE_HF is false

All providers expose the same interface:
  llm.invoke(lc_messages) → _FakeResponse(content=str)

This module is the ONLY place that knows about LLM providers.
All other modules call llm.invoke() without caring which backend is live.

BUG FIX (credits exhaustion):
  _ClaudeLLM now carries a list of models and cycles to the next on quota/rate
  errors at invoke time.  _HFChatLLM.invoke() rebuilds with the next model from
  _FREE_FALLBACK_MODELS when the current one fails.  If all Claude models are
  exhausted, _FallbackLLM automatically falls through to the HF backend.
"""
from __future__ import annotations
import json
from loguru import logger
from api.core.config import (
    USE_CLAUDE, ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    USE_HF, HF_TOKEN, HF_MODEL,
)


# ── Shared response wrapper (mimics LangChain AIMessage) ─────────────────────

class _FakeResponse:
    """Uniform response wrapper so every backend looks identical upstream."""
    def __init__(self, content: str):
        self.content = content


def _lc_role(msg) -> str:
    cls = type(msg).__name__
    if cls == "SystemMessage": return "system"
    if cls == "AIMessage":     return "assistant"
    return "user"


# ── Credit / quota error detection ───────────────────────────────────────────

def _is_quota_error(exc: Exception) -> bool:
    """
    Return True when the exception signals credit exhaustion or a hard
    rate-limit that won't recover within a single request cycle.
    We want to fall through to the next model; transient 5xx errors
    should still propagate so the retry logic in node_llm handles them.
    """
    msg = str(exc).lower()
    # Anthropic: 429 OverloadedError / credit_balance_too_low / rate_limit
    anthropic_quota = any(k in msg for k in (
        "credit", "quota", "overloaded", "rate_limit", "too many requests",
        "insufficient_quota", "billing",
    ))
    # HuggingFace: 429 / model too busy / loading
    hf_quota = any(k in msg for k in (
        "rate limit", "too many requests", "model is currently loading",
        "model is overloaded", "service unavailable", "quota",
    ))
    # HTTP 429 in the exception type name
    is_429 = "429" in msg or "ratelimit" in msg.replace("_", "").replace("-", "")
    return anthropic_quota or hf_quota or is_429


# ── Backend 1: Anthropic Claude ───────────────────────────────────────────────

# Ordered fallback list.  Primary model comes first (from ANTHROPIC_MODEL env).
# Subsequent entries are cheaper/lighter models that are less likely to hit
# credit limits and can still produce valid clinical JSON output.
_CLAUDE_FALLBACK_MODELS = [
    # Primary is injected at runtime from ANTHROPIC_MODEL env var
    "claude-haiku-4-5-20251001",    # fast, cheap, good JSON adherence
    "claude-haiku-3-5",             # older Haiku, very low cost
]


class _ClaudeLLM:
    """
    Anthropic API client with per-invoke model fallback.

    At startup: probes the configured primary model only (fast).
    At invoke time: if the primary model returns a quota/credits error,
    automatically tries the next model in _CLAUDE_FALLBACK_MODELS.
    """

    def __init__(self, api_key: str, primary_model: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

        # Build ordered model list: primary first, then fallbacks (deduplicated)
        models: list[str] = [primary_model]
        for m in _CLAUDE_FALLBACK_MODELS:
            if m not in models:
                models.append(m)
        self._models      = models
        self._model_idx   = 0            # index of currently active model
        self._model       = models[0]
        self._probe()

    def _probe(self):
        """Fail fast at startup only on genuine auth errors (invalid key).
        Transient errors (overload, 503, timeout) are ignored so a busy
        startup moment does not permanently disable Claude for this worker.
        """
        try:
            self._client.messages.create(
                model=self._model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
        except Exception as e:
            msg = str(e).lower()
            auth_error = any(k in msg for k in (
                "invalid_api_key", "authentication_error",
                "invalid x-api-key", "401", "403",
            ))
            if auth_error:
                raise   # key is genuinely wrong — fail fast
            # Transient (overload, 503, timeout) — log and continue;
            # the real invoke() call will retry or fall back as needed.
            logger.warning(
                "[LLM] Claude probe returned non-auth error (treated as transient): {}",
                str(e)[:120],
            )

    def invoke(self, lc_messages) -> _FakeResponse:
        system_parts: list[str] = []
        turns: list[dict]       = []

        for m in lc_messages:
            role = _lc_role(m)
            if role == "system":
                system_parts.append(m.content)
            else:
                turns.append({"role": role, "content": m.content})

        if not turns:
            turns = [{"role": "user", "content": "Continue."}]

        kwargs: dict = {
            "max_tokens":  1024,
            "temperature": 0.1,
            "messages":    turns,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        # Try from current model index; on quota error advance to next model
        start_idx = self._model_idx
        for offset in range(len(self._models)):
            idx   = (start_idx + offset) % len(self._models)
            model = self._models[idx]
            try:
                resp = self._client.messages.create(model=model, **kwargs)
                if idx != self._model_idx:
                    logger.info("[LLM] Claude: switched to model '{}' (idx={})", model, idx)
                    self._model_idx = idx
                    self._model     = model
                return _FakeResponse(resp.content[0].text)
            except Exception as e:
                if _is_quota_error(e) and offset < len(self._models) - 1:
                    logger.warning(
                        "[LLM] Claude model '{}' quota/credits exhausted — trying next model. err={}",
                        model, str(e)[:120],
                    )
                    continue
                # Non-quota error or last model → re-raise so caller can decide
                raise

        raise RuntimeError(f"[LLM] All {len(self._models)} Claude models exhausted their credits.")


# ── Backend 2: HuggingFace (free direct API — same as old code) ──────────────

# Models that are text-generation only (no chat template) — skip them
_BROKEN_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "gpt2",
    "facebook/opt-1.3b",
}

# Ordered fallback list for the free direct HF API.
# Preferred model (from HF_MODEL env var) is inserted first at runtime.
_FREE_FALLBACK_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",       # primary — requires Meta gated access
    "meta-llama/Llama-3.2-3B-Instruct",             # smaller Llama, same gated access
    "Qwen/Qwen2.5-7B-Instruct",                     # open access, no gating needed
    "HuggingFaceH4/zephyr-7b-beta",                 # open access
    "mistralai/Mistral-7B-Instruct-v0.2",           # open access (v0.2 works, v0.3 doesn't)
    "microsoft/Phi-3-mini-4k-instruct",              # open access
    "google/gemma-2-2b-it",                         # open access
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",          # always available — last resort
]


class _HFChatLLM:
    """
    HuggingFace LLM via the free direct Inference API.

    Runtime fallback: if invoke() fails with a quota/overload error the model
    list is advanced and the next model is built and retried immediately.
    No restart required — the switch happens transparently mid-conversation.
    """

    def __init__(self, token: str, preferred: str | None = None):
        self._token      = token
        self._preferred  = preferred
        self._inner      = None
        self._model      = None
        self._model_idx  = 0          # index into self._ordered_models
        self._ordered_models: list[str] = self._make_model_list()
        self._inner, self._model, self._model_idx = self._build_from(0)

    def _make_model_list(self) -> list[str]:
        models: list[str] = []
        if self._preferred and self._preferred not in _BROKEN_MODELS:
            models.append(self._preferred)
        for m in _FREE_FALLBACK_MODELS:
            if m not in models:
                models.append(m)
        return models

    def _build_from(self, start_idx: int):
        """Try to build a working ChatHuggingFace starting from start_idx."""
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        from langchain_core.messages import HumanMessage

        for idx in range(start_idx, len(self._ordered_models)):
            model = self._ordered_models[idx]
            try:
                logger.info("[LLM] Probing HF direct API (free): {}", model)
                endpoint = HuggingFaceEndpoint(
                    repo_id=model,
                    huggingfacehub_api_token=self._token,
                    temperature=0.15,
                    max_new_tokens=1024,
                    task="text-generation",
                )
                inner = ChatHuggingFace(llm=endpoint, verbose=False)
                inner.invoke([HumanMessage(content="Hi")])
                logger.info("[LLM] ✓ HF direct API works (free, no credits): {}", model)
                return inner, model, idx
            except Exception as e:
                logger.warning("[LLM] ✗ {} — {}", model, str(e)[:120])

        raise RuntimeError(
            "[LLM] No working HuggingFace model found via direct API.\n"
            "Check: (1) HF_TOKEN is valid, (2) meta-llama gated access approved on HF."
        )

    def invoke(self, lc_messages) -> _FakeResponse:
        start_idx = self._model_idx
        for attempt in range(len(self._ordered_models) - start_idx):
            try:
                result  = self._inner.invoke(lc_messages)
                content = result.content if hasattr(result, "content") else str(result)
                return _FakeResponse(content)
            except Exception as e:
                if _is_quota_error(e) and self._model_idx < len(self._ordered_models) - 1:
                    logger.warning(
                        "[LLM] HF model '{}' quota/overload — switching to next. err={}",
                        self._model, str(e)[:120],
                    )
                    next_idx = self._model_idx + 1
                    try:
                        self._inner, self._model, self._model_idx = self._build_from(next_idx)
                        logger.info("[LLM] HF switched to '{}'", self._model)
                        continue   # retry the invoke with the new model
                    except RuntimeError:
                        # All remaining HF models failed — let the error propagate
                        raise e
                # Non-quota error or last model
                logger.error("[LLM] HF invoke failed on {}: {}", self._model, e)
                raise


# ── Backend 3 (runtime wrapper): chains Claude → HF transparently ────────────

class _FallbackLLM:
    """
    Wraps two providers (primary + secondary) and falls through to secondary
    when the primary exhausts all its own per-model fallbacks.

    This is the top-level llm object when both Claude and HF are configured.
    It means: if every Claude model is out of credits this turn, we still get
    an answer from HuggingFace without an unhandled exception.
    """

    def __init__(self, primary, secondary):
        self._primary   = primary
        self._secondary = secondary

    def invoke(self, lc_messages) -> _FakeResponse:
        try:
            return self._primary.invoke(lc_messages)
        except Exception as e:
            if _is_quota_error(e):
                logger.warning(
                    "[LLM] Primary provider fully exhausted — falling back to secondary. err={}",
                    str(e)[:120],
                )
                return self._secondary.invoke(lc_messages)
            raise   # non-quota errors still propagate


# ── Factory: Claude → HuggingFace → Ollama ───────────────────────────────────

def _build_llm():
    """
    Build the best available LLM provider.
    Tries Claude first (most reliable), then HuggingFace, then local Ollama.
    When both Claude and HF are configured, wraps them in _FallbackLLM so
    mid-conversation credit exhaustion on Claude falls through to HF
    automatically.
    """
    claude_llm = None
    hf_llm     = None

    # 1. Try Anthropic Claude
    if USE_CLAUDE:
        try:
            claude_llm = _ClaudeLLM(ANTHROPIC_API_KEY, ANTHROPIC_MODEL)
            logger.info("[LLM] ✓ Using Anthropic Claude (primary={})", ANTHROPIC_MODEL)
        except Exception as e:
            logger.warning("[LLM] Claude unavailable ({}) — falling back to HuggingFace", str(e)[:100])

    # 2. Try HuggingFace Serverless Inference
    if USE_HF:
        preferred = HF_MODEL.strip() if HF_MODEL else None
        if preferred in _BROKEN_MODELS:
            logger.warning("[LLM] HF_MODEL '{}' is a text model — auto-selecting", preferred)
            preferred = None
        try:
            logger.info("[LLM] Building HuggingFace fallback (preferred={})", preferred)
            hf_llm = _HFChatLLM(token=HF_TOKEN, preferred=preferred)
        except Exception as e:
            logger.warning("[LLM] HuggingFace unavailable: {}", str(e)[:100])

    # Return the best chain available
    if claude_llm and hf_llm:
        logger.info("[LLM] Both Claude and HF available — using _FallbackLLM chain (Claude → HF)")
        return _FallbackLLM(primary=claude_llm, secondary=hf_llm)

    if claude_llm:
        return claude_llm

    if hf_llm:
        return hf_llm

    # 3. Local / remote Ollama (dev mode or OLLAMA_BASE_URL is set)
    from langchain_ollama import ChatOllama
    from api.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
    logger.info("[LLM] Using Ollama at {} (model={})", OLLAMA_BASE_URL, OLLAMA_MODEL)
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.15,
        num_predict=1024,
    )


llm = _build_llm()

