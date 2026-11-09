from __future__ import annotations

import asyncio
import json
import logging
import random
import time

from openai import APIStatusError, AsyncOpenAI

from corpus_agent.config import RateLimits

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 503, 504}


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self._requests_per_minute = max(requests_per_minute, 1)
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self._requests_per_minute:
            wait = self._timestamps[0] - cutoff
            if wait > 0:
                await asyncio.sleep(wait)
        self._timestamps.append(time.monotonic())


class LLMClient:
    def __init__(
        self,
        async_client: AsyncOpenAI,
        rate_limits: RateLimits,
        max_tokens: int | None = None,
    ):
        self._client = async_client
        self._rate_limits = rate_limits
        self._limiter = RateLimiter(rate_limits.requests_per_minute)
        self._max_tokens = max_tokens

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
    ) -> str:
        return await self._call(model, system, user, temperature, json_mode=False)

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> dict:
        for attempt in range(self._rate_limits.retry_max_attempts + 1):
            text = await self._call(model, system, user, temperature, json_mode=True, max_tokens=max_tokens)
            candidate = text.strip()
            if candidate.startswith("```"):
                lines = candidate.splitlines()
                if len(lines) >= 3:
                    candidate = "\n".join(lines[1:-1]).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                start = candidate.find("{")
                end = candidate.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(candidate[start : end + 1])
                    except json.JSONDecodeError:
                        pass
                if attempt < self._rate_limits.retry_max_attempts:
                    logger.warning(
                        "Invalid JSON response (attempt %d/%d); retrying",
                        attempt + 1,
                        self._rate_limits.retry_max_attempts,
                    )
                    continue
                raise

    async def _call(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        json_mode: bool,
        max_tokens: int | None = None,
    ) -> str:
        delay = self._rate_limits.retry_initial_delay_seconds
        for attempt in range(self._rate_limits.retry_max_attempts + 1):
            await self._limiter.acquire()
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                }
                effective_max_tokens = max_tokens if max_tokens is not None else self._max_tokens
                if effective_max_tokens is not None:
                    kwargs["max_tokens"] = effective_max_tokens
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = await self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content is not None:
                    return content

                finish_reason = response.choices[0].finish_reason
                if attempt < self._rate_limits.retry_max_attempts:
                    jitter = random.uniform(0, delay * 0.5)
                    logger.warning(
                        "Empty model content (finish_reason=%s, attempt %d/%d), sleeping %.1fs",
                        finish_reason,
                        attempt + 1,
                        self._rate_limits.retry_max_attempts,
                        delay + jitter,
                    )
                    await asyncio.sleep(delay + jitter)
                    delay *= 2
                    continue
                raise ValueError(
                    f"Model returned empty content (finish_reason={finish_reason})"
                )
            except APIStatusError as e:
                if (
                    e.status_code in RETRYABLE_STATUS_CODES
                    and attempt < self._rate_limits.retry_max_attempts
                ):
                    jitter = random.uniform(0, delay * 0.5)
                    logger.warning(
                        "Retryable error %d (attempt %d/%d), sleeping %.1fs",
                        e.status_code,
                        attempt + 1,
                        self._rate_limits.retry_max_attempts,
                        delay + jitter,
                    )
                    await asyncio.sleep(delay + jitter)
                    delay *= 2
                else:
                    raise
        raise RuntimeError("Unexpected retry loop exit")
