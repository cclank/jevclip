"""The single network boundary: POST /v1/systemone.

Everything above this module is code — binding sources, writing questions,
deciding what happens next. This module only carries a state and typed
questions out and typed answers back.
"""

import base64
import hashlib
import http.client
import json
import os
import queue
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote, urlsplit

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
TIMEOUT = 30.0
WORKERS = 8
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
RETRY_AFTER = 1.5


class JudgeError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def request_hash(payload):
    """Identical request, identical key: the exact thing a cached answer can
    stand in for. Any change to model, state or question wording misses."""
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JevClient:
    def __init__(self, api_key=None, model=MODEL, timeout=TIMEOUT, transport=None, workers=WORKERS):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.workers = workers
        self.transport = transport or self._post
        self.usage = {"requests": 0, "input_tokens": 0, "usd": 0.0}
        self._lock = threading.Lock()
        self._idle = queue.LifoQueue()
        url = urlsplit(ENDPOINT)
        self._host, self._path = url.hostname, url.path

    def payload(self, state, questions):
        return {"model": self.model, "state": state, "questions": questions}

    def ask(self, state, questions):
        raw = self.transport(self.payload(state, questions), self.api_key, self.timeout)
        tokens = (raw.get("usage") or {}).get("input_tokens") or 0
        with self._lock:
            self.usage["requests"] += 1
            self.usage["input_tokens"] += tokens
            self.usage["usd"] += tokens * USD_PER_INPUT_TOKEN
        return raw

    def map(self, fn, items):
        """`fn` over items, concurrently, in order. Latency is round-trip
        bound, so requests go out in parallel."""
        items = list(items)
        if len(items) <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(items))) as pool:
            return list(pool.map(fn, items))

    def close(self):
        while True:
            try:
                self._idle.get_nowait().close()
            except queue.Empty:
                return

    def _post(self, payload, api_key, timeout):
        if not api_key:
            raise JudgeError("no_api_key")
        body = json.dumps(payload).encode("utf-8")
        headers = {"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"}
        for attempt in (0, 1):
            try:
                status, data = self._send(body, headers, timeout)
            except JudgeError as exc:
                if attempt == 0 and exc.code in ("timeout", "dropped"):
                    continue
                raise
            if status == 429 and attempt == 0:
                time.sleep(RETRY_AFTER)
                continue
            if status != 200:
                raise JudgeError({401: "unauthorized", 403: "unauthorized", 429: "rate_limited"}
                                 .get(status, "http_%d" % status))
            try:
                return json.loads(data)
            except ValueError:
                raise JudgeError("bad_response")

    def _send(self, body, headers, timeout):
        # Connections are pooled and kept alive. Measured from mainland China,
        # a fresh TLS handshake per request doubled latency (1.06 s -> 0.52 s).
        try:
            conn = self._idle.get_nowait()
        except queue.Empty:
            conn = self._connect(timeout)
        try:
            conn.request("POST", self._path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
        except TimeoutError:
            conn.close()
            raise JudgeError("timeout")
        except (http.client.HTTPException, ConnectionError):
            conn.close()
            raise JudgeError("dropped")
        except OSError:
            conn.close()
            raise JudgeError("unreachable")
        if response.will_close:
            conn.close()
        else:
            self._idle.put(conn)
        return response.status, data

    def _connect(self, timeout):
        # http.client ignores HTTPS_PROXY on its own; honour it (and the macOS
        # system proxy, via getproxies) the way urllib would.
        proxy = urllib.request.getproxies().get("https")
        if not proxy or urllib.request.proxy_bypass(self._host):
            return http.client.HTTPSConnection(self._host, timeout=timeout)
        p = urlsplit(proxy if "://" in proxy else "http://" + proxy)
        conn = http.client.HTTPSConnection(p.hostname, p.port or 8080, timeout=timeout)
        headers = {}
        if p.username:
            creds = "%s:%s" % (unquote(p.username), unquote(p.password or ""))
            headers["Proxy-Authorization"] = "Basic " + base64.b64encode(creds.encode()).decode()
        conn.set_tunnel(self._host, 443, headers=headers)
        return conn
