"""OpenAI 兼容 HTTP 客户端。

纯标准库实现（urllib + base64），零强制新依赖。该模块为叶子模块，
不含 agent/工具调用概念，同时服务 Agent 与 VLM 标注两个子系统。

- ``base_url`` 与 ``model`` 由调用方（CLI 参数）强制传入，不设内置默认；
  环境变量 ``OPENAI_BASE_URL`` / ``OPENAI_MODEL`` 可提供默认值。
- ``api_key`` 来自参数或环境变量 ``OPENAI_API_KEY``，缺失时抛 ValueError。

@FileName:   client.py
@Function:   OpenAI 兼容 chat/completions 调用与图片多模态编码
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

#: 触发重试的 HTTP 状态码（网络错误/限流/服务端错误）。
_RETRY_STATUSES: tuple[int, ...] = (408, 429, 500, 502, 503, 504)


class APIError(Exception):
    """API 调用错误。

    Attributes:
        status: HTTP 状态码；0 表示网络/解析错误。
        message: 人类可读错误描述。
        body: 服务端返回的原始响应体片段。
    """

    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.body = body


class OpenAIClient:
    """OpenAI 兼容 chat/completions 客户端。

    Args:
        api_key:     API 密钥；缺省读环境变量 ``OPENAI_API_KEY``。
        base_url:    API 基地址，如 ``https://api.openai.com/v1``；
                     缺省读环境变量 ``OPENAI_BASE_URL``。
        timeout:     单次请求超时（秒）。
        max_retries: 可重试错误的最大重试次数（指数退避）。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError("缺少 API 密钥：请通过参数或环境变量 OPENAI_API_KEY 提供")
        resolved_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        if not resolved_url:
            raise ValueError("缺少 API 基地址：请通过参数或环境变量 OPENAI_BASE_URL 提供")
        self.api_key = resolved_key
        self.base_url = resolved_url
        self.timeout = timeout
        self.max_retries = max_retries

    # ---- 对外接口 ----

    def chat(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict:
        """非流式调用 chat/completions。

        Args:
            messages: OpenAI 消息列表（role/content 或多模态 content 数组）。
            model:    模型名（调用方强制传入）。
            tools:    可选的 function-calling 工具定义列表。
            temperature: 采样温度。
            max_tokens: 最大输出 token 数。

        Returns:
            完整响应 JSON（dict）。

        Raises:
            APIError: 请求失败或服务端返回错误。
        """
        payload: dict = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self._post_with_retry("/chat/completions", payload)
        return response

    def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """流式调用 chat/completions，逐段产出 content 增量。

        仅支持纯文本增量；含 tool_calls 的响应应改用 :meth:`chat`。

        Yields:
            每段 content delta 文本。
        """
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        response = self._post_with_retry("/chat/completions", payload)
        for line in response:
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content

    # ---- 内部实现 ----

    def _post_with_retry(self, path: str, payload: dict) -> object:
        """POST JSON 并处理重试，返回响应体（dict 或原始字节迭代器）。"""
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        attempt = 0
        while True:
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if payload.get("stream"):
                        return response  # 流式：调用方逐行读取
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in _RETRY_STATUSES and attempt < self.max_retries:
                    attempt += 1
                    _sleep_backoff(attempt)
                    logger.warning("API 返回 %d，第 %d 次重试", exc.code, attempt)
                    continue
                raise APIError(exc.code, f"API 请求失败: HTTP {exc.code}", error_body) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    _sleep_backoff(attempt)
                    logger.warning("网络错误 (%s)，第 %d 次重试", type(exc).__name__, attempt)
                    continue
                raise APIError(0, f"网络请求失败: {type(exc).__name__}: {exc}") from exc


def _sleep_backoff(attempt: int) -> None:
    """指数退避：0.5s / 1s / 2s ..."""
    time.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """将图片编码为 (mime, base64)。

    使用 ``Path.read_bytes()`` 读取（Windows 中文路径安全，不受
    ``cv2.imread`` 非 ASCII 路径限制）。

    Args:
        image_path: 图片路径。

    Returns:
        (mime 类型, base64 字符串)。
    """
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return mime, encoded


def image_url_content_part(image_path: Path, *, detail: str = "auto") -> dict:
    """构造多模态消息中的图片 content 元素。

    Args:
        image_path: 图片路径。
        detail:     OpenAI 图片细节级别（low/high/auto）。

    Returns:
        ``{"type": "image_url", "image_url": {"url": "data:...;base64,..."}}``。
    """
    mime, encoded = encode_image_base64(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": detail},
    }
