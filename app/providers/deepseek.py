from __future__ import annotations

import json
import os

import httpx

from app.providers.base import Provider, ProviderError
from app.schemas import output_schema


class DeepSeek(Provider):
    name = "deepseek"

    def __init__(self, settings, api_key=""):
        self.key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = settings["deepseek_model"]
        self.timeout = settings["deepseek_timeout_seconds"]

    async def request(self, system, prompt, schema):
        if not self.key:
            raise ProviderError("请填写本次 DeepSeek 密钥或设置进程环境 DEEPSEEK_API_KEY", "credentials_missing", False)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                response = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system
                                + "\n仅输出 JSON，遵循schema："
                                + json.dumps(output_schema(schema), ensure_ascii=False),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "max_tokens": 16384,
                    },
                )
            if response.status_code != 200:
                code = {401: "credentials_invalid", 402: "quota", 429: "rate_limit"}.get(
                    response.status_code, "http_error"
                )
                raise ProviderError(
                    f"DeepSeek 请求失败（HTTP {response.status_code}），请检查凭据、余额或稍后重试", code
                )
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ProviderError("DeepSeek 输出截断或未正常结束", "truncated")
            return json.loads(choice["message"]["content"]), response.json().get("usage")
        except ProviderError:
            raise
        except httpx.TimeoutException:
            raise ProviderError("DeepSeek 调用超时；远端请求可能仍在处理", "timeout") from None
        except httpx.HTTPError:
            raise ProviderError("DeepSeek 网络连接失败", "network") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("DeepSeek 响应不是有效 JSON 结果", "invalid_output") from None
