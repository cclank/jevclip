"""The summary writer: any OpenAI-compatible /chat/completions endpoint.

Jev judges and never writes; this is the one place text gets generated, and
it only ever sees material that code already selected.
"""

import json
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit

_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
# A scratchpad cut off by max_tokens never closes; drop everything before the close tag.
_OPEN_THINK = re.compile(r"^.*?</(?:think|thinking|reasoning)>", re.S | re.I)


class LLMError(Exception):
    pass


class ChatLLM:
    def __init__(self, base_url, api_key, model, max_tokens=4000, timeout=180.0, transport=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.transport = transport or _post

    @classmethod
    def from_env(cls):
        """JEVCLIP_LLM_* first; otherwise MiniMax's variables with MiniMax-M2.
        Returns None when nothing usable is configured."""
        if os.environ.get("JEVCLIP_LLM_BASE_URL"):
            base = os.environ["JEVCLIP_LLM_BASE_URL"]
            key = os.environ.get("JEVCLIP_LLM_API_KEY", "")
            model = os.environ.get("JEVCLIP_LLM_MODEL", "")
        else:
            base = os.environ.get("MINIMAX_BASE_URL", "")
            key = os.environ.get("MINIMAX_API_KEY", "")
            model = os.environ.get("JEVCLIP_LLM_MODEL", "MiniMax-M2")
        if not (base and key and model):
            return None
        if urlsplit(base).path in ("", "/"):  # a bare host, e.g. https://api.minimax.io
            base = base.rstrip("/") + "/v1"
        return cls(base, key, model, max_tokens=int(os.environ.get("JEVCLIP_LLM_MAX_TOKENS", "4000")))

    def chat(self, system, user):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": 0.3,
        }
        raw = self.transport(self.base_url + "/chat/completions", body, self.api_key, self.timeout)
        try:
            text = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError("unexpected response shape")
        return _OPEN_THINK.sub("", _THINK.sub("", text)).strip()


def _post(url, body, api_key, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LLMError("http_%d" % exc.code)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError("unreachable: %s" % exc)
    except ValueError:
        raise LLMError("bad_response")
