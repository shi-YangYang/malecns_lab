"""Loopback-only API for bounded, independent full-brain experiments."""

from __future__ import annotations

import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from .jobs import JobError, JobManager

MAX_BODY_BYTES = 4096


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("重复的参数名。")
        result[key] = value
    return result


class AtlasHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    manager: JobManager


class Handler(BaseHTTPRequestHandler):
    server_version = "MaleCNSAtlas/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def _local_request(self, mutate=False):
        ports = (8765, 8766, self.server.server_address[1])
        hosts = {f"{host}:{port}" for host in ("127.0.0.1", "localhost") for port in ports}
        if self.headers.get("Host") not in hosts:
            raise JobError("只允许本地页面访问。", 403)
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{host}" for host in hosts}:
            raise JobError("不允许来自其他网站的请求。", 403)
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise JobError("不允许来自其他网站的请求。", 403)
        if mutate and self.headers.get("X-MaleCNS-Request") != "1":
            raise JobError("请求缺少本地校验标记，请刷新页面后重试。", 403)

    def _parts(self):
        url = urlsplit(self.path)
        if url.scheme or url.netloc or url.query or url.fragment:
            raise JobError("接口路径无效。", 404)
        parts = unquote(url.path).split("/")
        if parts[:2] != ["", "api"] or any(part in (".", "..") for part in parts):
            raise JobError("接口不存在。", 404)
        return parts[2:]

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise JobError("不支持此请求格式。", 400)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise JobError("请求长度无效。", 400) from error
        if not 0 < length <= MAX_BODY_BYTES:
            raise JobError("请求内容为空或过长。", 413)
        if self.headers.get_content_type() != "application/json":
            raise JobError("请使用 JSON 请求。", 415)
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete body")
            return json.loads(body.decode("utf-8"), object_pairs_hook=strict_object)
        except (ValueError, UnicodeError) as error:
            raise JobError("请求参数无法解析。", 400) from error

    def _send(self, status: int, payload, content_type="application/json; charset=utf-8"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, mutate=False):
        try:
            self._local_request(mutate)
            parts = self._parts()
            manager = self.server.manager
            if mutate:
                if parts == ["experiments"]:
                    self._send(202, manager.create(self._body()))
                elif len(parts) == 3 and parts[0] == "experiments" and parts[2] == "cancel":
                    self._send(200, manager.cancel(parts[1]))
                else:
                    raise JobError("接口不存在。", 404)
            elif parts == ["health"]:
                self._send(200, manager.health())
            elif parts == ["experiments"]:
                self._send(200, {"experiments": manager.list()})
            elif len(parts) == 2 and parts[0] == "experiments":
                self._send(200, manager.get(parts[1]))
            elif len(parts) == 4 and parts[0] == "experiments" and parts[2] == "assets":
                path = manager.asset(parts[1], parts[3])
                self._send(200, path.read_bytes(), "image/png" if path.suffix == ".png" else "application/json; charset=utf-8")
            else:
                raise JobError("接口不存在。", 404)
        except JobError as error:
            self._send(error.status, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._send(500, {"error": "本地服务暂时无法处理请求，请重试。"})

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch(mutate=True)

    def do_OPTIONS(self):
        self._send(405, {"error": "不支持跨站请求。"})


def main() -> int:
    # Bind before loading persisted jobs: a duplicate server cannot disturb jobs
    # owned by the already-running server.
    server = AtlasHTTPServer(("127.0.0.1", 8766), Handler)
    manager = None
    try:
        manager = server.manager = JobManager()
        def stop(_signum, _frame):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        print("MaleCNS 实验服务：http://127.0.0.1:8766", flush=True)
        server.serve_forever(poll_interval=0.2)
    finally:
        if manager:
            manager.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
