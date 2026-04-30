import asyncio
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

from src.models.registry import ModelConfig, Provider
from src.models.response import LLMResponse


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

async def _call_openai(prompt: str, config: ModelConfig, system: Optional[str]) -> LLMResponse:
    import openai  # lazy import — only needed if using this provider

    client = openai.AsyncOpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=config.model_id,
        messages=messages,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = response.choices[0]
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    return LLMResponse(
        text=choice.message.content,
        model_id=config.model_id,
        provider=config.provider.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=config.estimate_cost(input_tokens, output_tokens),
        raw_response=response.model_dump(),
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

async def _call_anthropic(prompt: str, config: ModelConfig, system: Optional[str]) -> LLMResponse:
    import anthropic  # lazy import

    client = anthropic.AsyncAnthropic()
    kwargs = dict(
        model=config.model_id,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system

    t0 = time.perf_counter()
    response = await client.messages.create(**kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return LLMResponse(
        text=response.content[0].text,
        model_id=config.model_id,
        provider=config.provider.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=config.estimate_cost(input_tokens, output_tokens),
        raw_response=response.model_dump(),
    )


# ---------------------------------------------------------------------------
# Ollama (local REST API)
# ---------------------------------------------------------------------------

async def _call_ollama(
    prompt: str,
    config: ModelConfig,
    system: Optional[str],
    base_url: str = "http://localhost:11434",
) -> LLMResponse:
    import tiktoken  # use cl100k as a rough token counter for local models

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": config.model_id,
        "messages": messages,
        "stream": False,
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(f"{base_url}/api/chat", json=payload)
        resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000

    data = resp.json()
    text = data["message"]["content"]

    # Ollama reports eval_count (output) and prompt_eval_count (input)
    input_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)

    # Fall back to tiktoken estimate if Ollama didn't report counts
    if input_tokens == 0:
        enc = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(enc.encode(prompt))
        output_tokens = len(enc.encode(text))

    return LLMResponse(
        text=text,
        model_id=config.model_id,
        provider=config.provider.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=0.0,   # local model — no cloud cost
        raw_response=data,
    )


# ---------------------------------------------------------------------------
# Groq (OpenAI-compatible — uses openai SDK with a custom base_url)
# ---------------------------------------------------------------------------

async def _call_groq(prompt: str, config: ModelConfig, system: Optional[str]) -> LLMResponse:
    import os
    import openai

    client = openai.AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=config.model_id,
        messages=messages,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = response.choices[0]
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    return LLMResponse(
        text=choice.message.content,
        model_id=config.model_id,
        provider=config.provider.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=0.0,   # free tier
        raw_response=response.model_dump(),
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_DISPATCH = {
    Provider.OPENAI: _call_openai,
    Provider.ANTHROPIC: _call_anthropic,
    Provider.OLLAMA: _call_ollama,
    Provider.GROQ: _call_groq,
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def send_request(
    prompt: str,
    config: ModelConfig,
    system: Optional[str] = None,
) -> LLMResponse:
    handler = _DISPATCH.get(config.provider)
    if handler is None:
        raise ValueError(f"No handler registered for provider: {config.provider}")
    return await handler(prompt, config, system)


def send_request_sync(
    prompt: str,
    config: ModelConfig,
    system: Optional[str] = None,
) -> LLMResponse:
    """Synchronous wrapper for use outside async contexts (e.g. scripts, tests)."""
    return asyncio.run(send_request(prompt, config, system))
