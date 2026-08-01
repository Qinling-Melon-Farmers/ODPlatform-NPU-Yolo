import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from od_platform.agent.client import (
    APIError,
    OpenAIClient,
    encode_image_base64,
    image_url_content_part,
)


class FakeHTTPResponse:
    """模拟 urllib 响应对象。"""

    def __init__(self, payload: bytes | str, *, code: int = 200) -> None:
        self._payload = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        self._code = code

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload

    def __iter__(self):
        return iter(self._payload.splitlines(keepends=True))


class ShortLivedResponse:
    """模拟真实 HTTPResponse：退出 context 后迭代抛错。

    用于回归验证：流式客户端必须在 with 生命周期内迭代响应，
    不能在退出 context 后再读取。
    """

    def __init__(self, payload: bytes | str) -> None:
        self._payload = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        self._closed = False

    def __enter__(self) -> "ShortLivedResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self._closed = True
        return None

    def __iter__(self):
        if self._closed:
            raise RuntimeError("response already closed")
        return iter(self._payload.splitlines(keepends=True))


def _find_header(headers: dict, name: str) -> str | None:
    """大小写不敏感地从 headers dict 查找（urllib 将键规范化为 Content-type）。"""
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def _capture_request(request) -> dict:
    """从 urllib Request 提取请求信息。"""
    return {
        "url": request.full_url,
        "auth": _find_header(request.headers, "Authorization"),
        "content_type": _find_header(request.headers, "Content-Type"),
        "body": json.loads(request.data.decode("utf-8")) if request.data else None,
    }


class TestOpenAIClient(unittest.TestCase):
    def _make_client(self) -> OpenAIClient:
        return OpenAIClient(api_key="test-key", base_url="https://example.com/v1")

    def test_chat_builds_request_and_parses_response(self) -> None:
        client = self._make_client()
        response_payload = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        captured: dict = {}

        def fake_urlopen(request, **kwargs):
            captured.update(_capture_request(request))
            return FakeHTTPResponse(json.dumps(response_payload))

        with patch("od_platform.agent.client.urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.chat(
                [{"role": "user", "content": "你好"}],
                model="test-model",
                tools=[{"type": "function", "function": {"name": "tool"}}],
            )

        self.assertEqual(result["choices"][0]["message"]["content"], "hi")
        self.assertTrue(captured["url"].endswith("/v1/chat/completions"))
        self.assertEqual(captured["auth"], "Bearer test-key")
        self.assertEqual(captured["content_type"], "application/json")
        self.assertEqual(captured["body"]["model"], "test-model")
        self.assertEqual(captured["body"]["messages"][0]["role"], "user")
        self.assertEqual(len(captured["body"]["tools"]), 1)
        self.assertEqual(captured["body"]["temperature"], 0.0)

    def test_chat_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAIClient(base_url="https://example.com/v1")

    def test_chat_missing_base_url_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAIClient(api_key="k")

    def test_http_error_maps_to_api_error(self) -> None:
        client = self._make_client()

        def fake_urlopen(request, **kwargs):
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"rate limited")
            )

        with patch("od_platform.agent.client.urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(APIError) as ctx:
                client.chat([{"role": "user", "content": "x"}], model="m")
        self.assertEqual(ctx.exception.status, 429)
        self.assertIn("rate limited", ctx.exception.body)

    def test_retry_on_5xx_then_success(self) -> None:
        client = OpenAIClient(api_key="k", base_url="https://example.com/v1", max_retries=2)
        calls = {"count": 0}

        def fake_urlopen(request, **kwargs):
            calls["count"] += 1
            if calls["count"] <= 2:
                raise urllib.error.HTTPError(
                    request.full_url, 503, "Unavailable", {}, io.BytesIO(b"down")
                )
            return FakeHTTPResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}))

        with patch("od_platform.agent.client.urllib.request.urlopen", side_effect=fake_urlopen), patch(
            "od_platform.agent.client._sleep_backoff"
        ):
            result = client.chat([{"role": "user", "content": "x"}], model="m")
        self.assertEqual(calls["count"], 3)
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")

    def test_retry_exhausted_raises_api_error(self) -> None:
        client = OpenAIClient(api_key="k", base_url="https://example.com/v1", max_retries=1)

        def fake_urlopen(request, **kwargs):
            raise urllib.error.HTTPError(request.full_url, 500, "Error", {}, io.BytesIO(b"boom"))

        with patch("od_platform.agent.client.urllib.request.urlopen", side_effect=fake_urlopen), patch(
            "od_platform.agent.client._sleep_backoff"
        ):
            with self.assertRaises(APIError) as ctx:
                client.chat([{"role": "user", "content": "x"}], model="m")
        self.assertEqual(ctx.exception.status, 500)

    def test_chat_stream_parses_sse_deltas(self) -> None:
        client = self._make_client()
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "你"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "好"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {}}]}),
            "data: [DONE]",
        ]
        payload = "\n".join(sse_lines) + "\n"

        def fake_urlopen(request, **kwargs):
            return FakeHTTPResponse(payload)

        with patch("od_platform.agent.client.urllib.request.urlopen", side_effect=fake_urlopen):
            chunks = list(client.chat_stream([{"role": "user", "content": "x"}], model="m"))
        self.assertEqual(chunks, ["你", "好"])

    def test_stream_holds_response_lifetime(self) -> None:
        """流式客户端必须在 with 生命周期内迭代（退出 context 后读取失败场景）。"""
        client = self._make_client()
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "流"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "式"}}]}),
            "data: [DONE]",
        ]
        payload = "\n".join(sse_lines) + "\n"

        def fake_urlopen(request, **kwargs):
            return ShortLivedResponse(payload)

        with patch("od_platform.agent.client.urllib.request.urlopen", side_effect=fake_urlopen):
            chunks = list(client.chat_stream([{"role": "user", "content": "x"}], model="m"))
        self.assertEqual(chunks, ["流", "式"])

    def test_network_error_retries_then_raises(self) -> None:
        client = OpenAIClient(api_key="k", base_url="https://example.com/v1", max_retries=1)

        def fake_urlopen(request, **kwargs):
            raise TimeoutError("timeout")

        with patch("od_platform.agent.client.urllib.request.urlopen", side_effect=fake_urlopen), patch(
            "od_platform.agent.client._sleep_backoff"
        ):
            with self.assertRaises(APIError) as ctx:
                client.chat([{"role": "user", "content": "x"}], model="m")
        self.assertEqual(ctx.exception.status, 0)


class TestImageEncoding(unittest.TestCase):
    def test_encode_image_base64_unicode_path(self) -> None:
        """中文路径下的 base64 编码（回归：不走 cv2.imread）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "中文图片" / "样例.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-data")
            mime, encoded = encode_image_base64(image_path)
            self.assertEqual(mime, "image/png")
            self.assertTrue(encoded.startswith("iVBOR"))  # \x89PNG 的 base64

    def test_image_url_content_part(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "img.jpg"
            image_path.write_bytes(b"jpeg-data")
            part = image_url_content_part(image_path)
            self.assertEqual(part["type"], "image_url")
            self.assertTrue(part["image_url"]["url"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
