import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from corpus_agent.config import RateLimits
from corpus_agent.llm import LLMClient, RateLimiter


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_burst_up_to_limit(self):
        limiter = RateLimiter(requests_per_minute=60)
        start = time.monotonic()
        for _ in range(10):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_rate_limiter_throttles_past_limit(self):
        limiter = RateLimiter(requests_per_minute=120)
        for _ in range(120):
            await limiter.acquire()
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.4


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_returns_text(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello world"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        rate_limits = RateLimits(requests_per_minute=1000)
        client = LLMClient(async_client=mock_client, rate_limits=rate_limits)

        result = await client.complete(
            model="test-model",
            system="You are helpful.",
            user="Say hello.",
        )
        assert result == "hello world"
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_json_parses_response(self):
        payload = {"emails": [{"sender": "a@b.com"}]}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(payload)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        rate_limits = RateLimits(requests_per_minute=1000)
        client = LLMClient(async_client=mock_client, rate_limits=rate_limits)

        result = await client.complete_json(
            model="test-model",
            system="Return JSON.",
            user="Give me emails.",
        )
        assert result == payload

