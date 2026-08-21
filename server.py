#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp[cli]>=1.0.0,<2.0.0",
#   "httpx>=0.27.0",
#   "tree-sitter>=0.21",
#   "tree-sitter-java>=0.21",
# ]
# ///
"""
Local MCP Server: routes implementation work to local Ollama models.

The goal is to save Claude tokens: file contents are read, edited, and written
back entirely on the server side, so they almost never round-trip through
Claude's context. Claude only sees a short summary or a guard-rail rejection.

Tools (in order of token-efficiency):
  local_outline(files)             -> deterministic API skeleton, NO model call
  local_edit(files, instruction)   -> modifies existing files in place
  local_write(path, instruction)   -> creates a new file from scratch
  local_read(files, instruction)   -> analyzes files, returns text (map-reduce if large)
  local_snippet(prompt)            -> returns short text (round-trip; fallback)

Single-model architecture:
  All Ollama-backed tools call one configured model (default gemma4:12b). The
  model is kept resident with keep_alive=-1, but that only holds while num_ctx
  stays constant: a different num_ctx makes Ollama reload the model (~5 s). For
  that reason every call uses one shared num_ctx (see NUM_CTX); num_predict and
  temperature are runtime params and can vary per call without a reload.

Notes:
  - Deletion and rename are intentionally NOT tools: they involve no file body,
    so they save no Claude tokens and would only add tool-schema overhead. Use
    the built-in tools or Bash for those.
  - Non-English instructions are translated to English at the boundary by
    _normalize_instruction(). All guard-rails and the system prompt are
    English-only.

Target OS: Windows 10. All file I/O is Windows-correct: CRLF preservation,
case-insensitive path normalization, locked-file detection, atomic rename via
same-directory temp file + os.replace.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import tempfile
import threading
import time
import concurrent.futures
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# tree-sitter is optional: when present it lets local_read / local_outline
# extract Java structure deterministically with no model call (Task-023). If it
# is missing the server still runs; the structural paths fall back gracefully.
try:
    import tree_sitter_java as _tsjava
    from tree_sitter import Language as _TSLanguage, Parser as _TSParser

    _JAVA_LANGUAGE = _TSLanguage(_tsjava.language())
    try:
        _JAVA_PARSER = _TSParser(_JAVA_LANGUAGE)   # tree-sitter >= 0.22
    except TypeError:                              # older API
        _JAVA_PARSER = _TSParser()
        _JAVA_PARSER.set_language(_JAVA_LANGUAGE)
except Exception:
    _JAVA_PARSER = None

mcp = FastMCP("local-mcp")

# ---------------------------------------------------------------------------
# Opt-in debug logging (Task-008). Off by default and fully silent: a single
# NullHandler, nothing written to disk, nothing on stdout/stderr (stdout carries
# the MCP protocol and must never be touched). Enable with LOCAL_MCP_DEBUG=1 (or
# "log_level": "DEBUG" in model-config.json); output then goes to a rotating
# local-mcp.log next to this file. File bodies are only ever logged here, so
# normal operation stays quiet and private.
# ---------------------------------------------------------------------------
_log = logging.getLogger("local-mcp")
_log.propagate = False
_log.addHandler(logging.NullHandler())
_log.setLevel(logging.WARNING)


def _enable_debug_logging() -> None:
    """Attach a rotating file handler at DEBUG. Idempotent."""
    if any(isinstance(h, RotatingFileHandler) for h in _log.handlers):
        return
    logfile = Path(__file__).resolve().parent / "local-mcp.log"
    handler = RotatingFileHandler(
        logfile, maxBytes=1_000_000, backupCount=2, encoding="utf-8", delay=True
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _log.addHandler(handler)
    _log.setLevel(logging.DEBUG)


def _bound(text: str, limit: int = 2000) -> str:
    """Bound a value for logging so the log file cannot blow up."""
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit]}... [+{len(text) - limit} chars]"


# Env var can enable debug BEFORE the config is read, so a malformed config file
# is itself logged. The config's log_level is honored too, just after load.
if os.environ.get("LOCAL_MCP_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
    _enable_debug_logging()

# ---------------------------------------------------------------------------
# Model configuration; loaded from model-config.json (optional).
# See configs/ for ready-to-use templates.
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS: dict = {
    # Provider: "ollama" (default, local) or "openrouter" (remote, OpenAI-compat).
    "provider": "ollama",
    "model": "gemma4:e4b",
    # Ollama-specific
    "ollama_url": "http://localhost:11434/api/chat",
    # OpenRouter-specific (ignored when provider="ollama")
    "openrouter_url": "https://openrouter.ai/api/v1/chat/completions",
    "openrouter_referer": "https://claude.ai/code",
    "openrouter_title": "local-mcp",
    "openrouter_extra_body": {},
    # Context window. ONE value for every Ollama call on purpose: changing
    # num_ctx between calls forces Ollama to reload the model (~5 s) and defeats
    # keep_alive, so snippet/translate no longer use smaller contexts. (Task-022)
    "num_ctx": 32768,
    # Max output tokens per call type. num_predict is a runtime parameter; it
    # does NOT trigger a reload, so it can vary freely. read_num_predict bounds
    # local_read: its output is the only cost that tool pays back into Claude's
    # context, so it must not run away. (Task-014)
    "read_num_predict": 1024,
    "snippet_num_predict": 1024,
    "translate_num_predict": 512,
    # Sampling. temperature 0 keeps full-file regeneration deterministic and
    # format-clean; read_temperature adds a little fluency for analysis output.
    # repeat_penalty 1.0 disables the penalty (code legitimately repeats tokens,
    # which matters for verbose languages like Java). top_p/top_k/seed/num_gpu
    # are optional: when null the model's own default applies. (Task-018)
    "temperature": 0,
    "read_temperature": 0.2,
    "repeat_penalty": 1.0,
    "top_p": None,
    "top_k": None,
    "seed": None,
    "num_gpu": None,
    # Reasoning ("thinking") toggle for thinking-capable models (e.g. gemma4,
    # qwen3). These tools want the answer, not the chain-of-thought: when thinking
    # is on, the model spends its num_predict budget emitting a `thinking` stream
    # and can hit the token cap (done_reason="length") before producing ANY
    # `message.content`, which _stream_ollama then returns as "". Forcing it off
    # keeps the whole budget for the answer. Ollama ignores this key for models
    # that do not support thinking, so it is safe to always send. (Task-025)
    "think": False,
    "timeout": 1200,
    # Seconds to wait in the single-worker FIFO queue before giving up. Bounds
    # ONLY the wait to start, not generation; once a call begins streaming,
    # the per-chunk `timeout` above governs it. (Task-004)
    #
    # null (default) => derive it from `timeout`: a queued call must outwait the
    # calls ahead of it, so the queue wait is QUEUE_TIMEOUT_FACTOR x `timeout`
    # (room for several parallel calls to drain one-by-one). A standalone number
    # could sit BELOW `timeout`, letting a queued call give up before a single
    # call ahead can even finish; an explicit value below `timeout` is raised to
    # `timeout` for the same reason. The per-chunk `timeout` is the real stall
    # guard, so a generous queue wait is safe. (Task-026)
    "queue_timeout": None,
    # "WARNING" (default, silent) or "DEBUG" to trace to local-mcp.log. The
    # LOCAL_MCP_DEBUG=1 env var does the same and applies even before this file
    # is read. (Task-008)
    "log_level": "WARNING",
}

# A queued call must be willing to wait for the calls ahead of it to drain on the
# single GPU worker; one call is bounded by `timeout`, so the queue wait is a
# multiple of it. 4x leaves room for a handful of parallel calls (Claude Code
# fires tool calls in parallel) without rejecting work that would still be served
# shortly. (Task-026)
_QUEUE_TIMEOUT_FACTOR = 4


def _load_model_config() -> dict:
    """Load model-config.json from the same directory as server.py.

    Missing file or missing keys fall back to _CONFIG_DEFAULTS.
    """
    config_path = Path(__file__).resolve().parent / "model-config.json"
    cfg = dict(_CONFIG_DEFAULTS)
    user_cfg: dict = {}
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                user_cfg = loaded
                for key in _CONFIG_DEFAULTS:
                    if key in user_cfg:
                        cfg[key] = user_cfg[key]
                # Backward compat: older configs used edit_ctx (plus
                # snippet_ctx/translate_ctx). Map edit_ctx onto the unified
                # num_ctx; the smaller per-task contexts are intentionally
                # dropped because varying num_ctx reloads the model (Task-022).
                if "num_ctx" not in user_cfg and "edit_ctx" in user_cfg:
                    cfg["num_ctx"] = user_cfg["edit_ctx"]
        except (json.JSONDecodeError, OSError) as e:
            # Use defaults, but leave a trace (visible only when debug is on).
            _log.warning("malformed model-config.json (%s); using defaults", e)
    # Keep the queue wait aligned with the per-call timeout (Task-026). Derive it
    # when unset; if set explicitly but below one full call, raise it so a queued
    # call never gives up before a single call ahead of it can finish.
    if user_cfg.get("queue_timeout") is None:
        cfg["queue_timeout"] = cfg["timeout"] * _QUEUE_TIMEOUT_FACTOR
    elif cfg["queue_timeout"] < cfg["timeout"]:
        _log.warning(
            "queue_timeout (%s) is below timeout (%s); raising it to timeout so a "
            "queued call is not rejected before a single call ahead can finish.",
            cfg["queue_timeout"], cfg["timeout"],
        )
        cfg["queue_timeout"] = cfg["timeout"]
    return cfg


_cfg = _load_model_config()

# The config can enable debug too (env var already handled above).
if str(_cfg.get("log_level", "")).upper() == "DEBUG":
    _enable_debug_logging()

PROVIDER: str              = _cfg["provider"]
MODEL: str                 = _cfg["model"]
OLLAMA_URL: str            = _cfg["ollama_url"]
OPENROUTER_URL: str        = _cfg["openrouter_url"]
OPENROUTER_REFERER: str    = _cfg["openrouter_referer"]
OPENROUTER_TITLE: str      = _cfg["openrouter_title"]
OPENROUTER_EXTRA_BODY: dict = _cfg["openrouter_extra_body"]
NUM_CTX: int               = _cfg["num_ctx"]
READ_NUM_PREDICT: int      = _cfg["read_num_predict"]
SNIPPET_NUM_PREDICT: int   = _cfg["snippet_num_predict"]
TRANSLATE_NUM_PREDICT: int = _cfg["translate_num_predict"]
TEMPERATURE                = _cfg["temperature"]
READ_TEMPERATURE           = _cfg["read_temperature"]
REPEAT_PENALTY             = _cfg["repeat_penalty"]
TOP_P                      = _cfg["top_p"]
TOP_K                      = _cfg["top_k"]
SEED                       = _cfg["seed"]
NUM_GPU                    = _cfg["num_gpu"]
THINK                      = _cfg["think"]
TIMEOUT: int               = _cfg["timeout"]
QUEUE_TIMEOUT: int         = _cfg["queue_timeout"]

if PROVIDER == "openrouter":
    _OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
    if not _OPENROUTER_API_KEY:
        raise RuntimeError(
            "provider=openrouter requires the OPENROUTER_API_KEY environment variable. "
            'Set it before starting the server: $env:OPENROUTER_API_KEY = "sk-or-..."'
        )
else:
    _OPENROUTER_API_KEY = ""

# One GPU, one request at a time. Requests are queued FIFO by a single-worker
# executor so parallel agents wait rather than fail. _call_openrouter does not
# need this because it is a remote endpoint with no GPU contention.
_OLLAMA_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Qwen3-specific: /no_think suppresses the reasoning chain, and a defensive
# stripper catches any <think> tags that leak through.
_IS_QWEN3: bool = "qwen3" in MODEL.lower()

# ---------------------------------------------------------------------------
# Guard-rail constants (tune freely)
# ---------------------------------------------------------------------------
# Reject a file edit if new size < SHRINK_RATIO * old size AND the instruction
# does not contain a removal keyword.
SHRINK_RATIO = 0.5

# English only. Non-English instructions are translated to English at the
# boundary by _normalize_instruction(), so this list never needs to grow
# language-by-language.
REMOVAL_KEYWORDS = (
    "delete", "remove", "strip", "drop", "clear", "empty", "shrink",
    "erase", "purge", "discard",
)

# Patterns that indicate the caller wants the file echoed verbatim rather than
# analyzed. local_read is an analysis tool; verbatim retrieval belongs to the
# built-in Read tool. Checked after language normalization so translation is
# already in English.
_RETRIEVAL_RE = re.compile(
    r"\b(verbatim|word[\s\-]for[\s\-]word|every\s+line"
    r"|full\s+content|entire\s+content|complete\s+content"
    r"|exact\s+content|in\s+its\s+entirety)\b",
    re.IGNORECASE,
)

# Lazy-output markers; matched as WHOLE TRIMMED LINES, and only flagged when
# the same line was NOT already present in the original file. Whole-line +
# delta-against-original keeps false positives near zero.
TRUNCATION_MARKERS = (
    "... rest of file unchanged",
    "... rest of the file unchanged",
    "... rest of code unchanged",
    "// ... existing code ...",
    "// ... rest of code ...",
    "// ... rest of file ...",
    "/* ... existing code ... */",
    "# ... existing code ...",
    "# (unchanged)",
    "# rest of file unchanged",
    "<TRUNCATED>",
    "[TRUNCATED]",
)

# Extensions for which the bracket-delta guard runs.
BRACKET_CHECK_EXTS = {
    ".py", ".java", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".json",
}

# Max chars of raw model output to echo back when parsing fails even after
# retry. Kept small: Claude rarely reconstructs the fix from the raw dump, it
# just retries, so the extra context is mostly wasted. (Task-015)
PARSE_FAIL_ECHO_LIMIT = 200

# ---------------------------------------------------------------------------
# Java/Kotlin guard data (Task-020). Java and Spring are a top use-case and are
# exactly where a 12B model omits boilerplate or breaks the package/filename
# contract; these are catchable deterministically with no model call.
# ---------------------------------------------------------------------------
_JAVA_EXTS = {".java", ".kt"}

# Canonical "I omitted real code here" comment phrases. A new line is flagged
# only when, after stripping comment delimiters and filler (dots, parens, the
# article "the"), it reduces to one of these AND it was not already present in
# the original. The normalization keeps this conservative: only pure placeholder
# comments match, not comments that merely contain one of these words.
_JAVA_PLACEHOLDER_PHRASES = frozenset({
    "getters and setters", "getter and setter", "getters setters",
    "rest of class", "rest of file", "rest of code", "rest of method",
    "other methods", "other methods unchanged", "other fields",
    "other members", "other code",
    "existing methods", "existing code", "existing fields", "existing members",
    "remaining methods", "remaining fields", "remaining code",
    "constructors omitted", "constructor omitted", "methods omitted",
    "fields omitted", "imports omitted", "body omitted",
    "unchanged", "no change", "no changes",
})

_JAVA_COMMENT_PREFIX_RE = re.compile(r"^\s*(?://+|/\*+|\*+|#)")
_JAVA_COMMENT_OPEN_RE = re.compile(r"^\s*(?://+|/\*+|\*+|#)\s*")
_JAVA_COMMENT_CLOSE_RE = re.compile(r"\s*\*/\s*$")

# Source roots under which a file's package must mirror its directory layout.
_JAVA_SOURCE_ROOTS = (
    ("src", "main", "java"), ("src", "test", "java"),
    ("src", "main", "kotlin"), ("src", "test", "kotlin"),
)

# Kotlin omits the trailing semicolon, so it is optional here.
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;?", re.MULTILINE)
_PUBLIC_TYPE_RE = re.compile(
    r"\bpublic\s+(?:final\s+|abstract\s+|sealed\s+|non-sealed\s+|strictfp\s+)*"
    r"(?:class|interface|enum|record|@interface)\s+(\w+)"
)
_IMPORT_LINE_RE = re.compile(r"^\s*import\s+", re.MULTILINE)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------
# Ollama sometimes 500s briefly while loading a cold model; retry once. (Task-009)
_OLLAMA_LOAD_RETRY_BACKOFF = 1.5


def _map_ollama_error(model: str, exc: Exception) -> str:
    """Turn a known Ollama failure into a one-line, actionable message. (Task-009)

    The not-set-up-correctly cases (Ollama down, model not pulled) are the most
    common first-run failures for this server, so they get a clear fix instead of
    a raw httpx string.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return (
            f"Ollama is not reachable at {OLLAMA_URL}. Is it running? "
            "Start Ollama, or fix 'ollama_url' in model-config.json."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        resp = exc.response
        try:
            resp.read()
            body = resp.text
        except Exception:
            body = ""
        code = resp.status_code
        if code == 404:
            return (
                f"Model '{model}' is not available in Ollama (HTTP 404). "
                f"Pull it first: ollama pull {model}"
            )
        return f"Ollama returned HTTP {code}: {body[:300] or exc}"
    return f"Ollama call failed: {exc}"


def _stream_ollama(payload: dict) -> str:
    """One streaming request to Ollama. Raises httpx errors; caller maps them."""
    chunks: list[str] = []
    saw_thinking = False
    done_reason: str | None = None
    with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message", {})
            content = message.get("content", "")
            if content:
                chunks.append(content)
            if message.get("thinking"):
                saw_thinking = True
            if data.get("done", False):
                done_reason = data.get("done_reason")
                break
    out = "".join(chunks)
    # A thinking-capable model can burn the whole num_predict budget on its
    # `thinking` stream and stop (done_reason="length") before emitting any
    # answer, leaving content empty. Surface that instead of returning "".
    # The fix is to keep thinking off (THINK=False) or raise num_predict. (Task-025)
    if not out and saw_thinking:
        raise httpx.HTTPError(
            "Model produced only reasoning ('thinking') and no answer before the "
            f"output budget ran out (done_reason={done_reason!r}). Set \"think\": "
            "false in model-config.json (or raise *_num_predict)."
        )
    return out


def _call_ollama_impl(
    model: str,
    messages: list[dict],
    system: str | None,
    num_ctx: int | None,
    num_predict: int | None,
    temperature: float | None,
) -> str:
    """HTTP call to Ollama. Runs inside the single-worker executor."""
    options: dict = {"num_ctx": num_ctx if num_ctx is not None else NUM_CTX}
    if num_predict is not None:
        options["num_predict"] = num_predict
    # Sampling knobs (Task-018). temperature is per-call; the rest are global.
    # Only emit a key when it is set so the model's own default applies when null.
    if temperature is not None:
        options["temperature"] = temperature
    if REPEAT_PENALTY is not None:
        options["repeat_penalty"] = REPEAT_PENALTY
    if TOP_P is not None:
        options["top_p"] = TOP_P
    if TOP_K is not None:
        options["top_k"] = TOP_K
    if SEED is not None:
        options["seed"] = SEED
    if NUM_GPU is not None:
        options["num_gpu"] = NUM_GPU
    # Ollama's /api/chat IGNORES a top-level "system" field; the system prompt
    # must be a role:"system" message. Prepend it here. (Task-001)
    chat_messages = messages
    if system:
        chat_messages = [{"role": "system", "content": system}] + list(messages)
    payload: dict = {
        "model": model,
        "messages": chat_messages,
        "stream": True,
        # Keep the model resident. This only holds while num_ctx is constant
        # across calls; a different num_ctx forces a reload (~5 s), so every
        # call uses one shared NUM_CTX. (Task-022)
        "keep_alive": -1,
        # Disable the model's reasoning phase so num_predict is spent on the
        # answer, not a `thinking` stream (see THINK config). (Task-025)
        "think": THINK,
        "options": options,
    }
    # Streaming: timeout applies per-chunk, not to the whole response (avoids
    # false timeouts on large outputs). Connection/HTTP failures are mapped to
    # actionable messages; a transient 500 (cold-model load) is retried once. The
    # happy path runs the request once with no added latency. (Task-009)
    for attempt in range(2):
        try:
            return _stream_ollama(payload)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 500 and attempt == 0:
                time.sleep(_OLLAMA_LOAD_RETRY_BACKOFF)
                continue
            raise httpx.HTTPError(_map_ollama_error(model, e)) from e
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise httpx.HTTPError(_map_ollama_error(model, e)) from e
    # Unreachable: the loop either returns or raises.
    raise httpx.HTTPError(f"Ollama call to '{model}' failed after a load retry.")


def _call_ollama(
    model: str,
    messages: list[dict],
    system: str | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
) -> str:
    """Submit an Ollama call to the single-worker queue and wait for the result.

    Parallel callers (e.g. agents) are queued FIFO and processed sequentially.
    QUEUE_TIMEOUT bounds ONLY the time spent waiting to start, not generation:
    once a call begins streaming, execution is governed by the per-chunk httpx
    timeout (TIMEOUT) inside _call_ollama_impl, so a long but healthy generation
    is never killed by a wall-clock cap. (Task-004)
    """
    started = threading.Event()

    def _job() -> str:
        started.set()  # set the moment the worker picks this up, before the HTTP call
        return _call_ollama_impl(
            model, messages, system, num_ctx, num_predict, temperature
        )

    future = _OLLAMA_EXECUTOR.submit(_job)
    if not started.wait(timeout=QUEUE_TIMEOUT):
        # Still queued behind other work after QUEUE_TIMEOUT; it has not begun.
        future.cancel()
        raise httpx.HTTPError(
            f"Ollama busy: still queued after {QUEUE_TIMEOUT} s behind other "
            "local-model calls. Try again shortly."
        )
    # Started: wait for completion with no wall-clock cap. A genuinely stalled
    # GPU is caught by the per-chunk streaming timeout in _call_ollama_impl.
    return future.result()


def _call_openrouter(
    model: str,
    messages: list[dict],
    system: str | None = None,
    num_ctx: int | None = None,       # accepted but ignored; OpenRouter decides context
    num_predict: int | None = None,   # -> max_tokens
    temperature: float | None = None,
) -> str:
    """Call an OpenAI-compatible remote endpoint (OpenRouter), non-streaming."""
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload: dict = {
        "model": model,
        "messages": full_messages,
        "stream": False,
    }
    if num_predict is not None:
        payload["max_tokens"] = num_predict
    if temperature is not None:
        payload["temperature"] = temperature
    if OPENROUTER_EXTRA_BODY:
        payload.update(OPENROUTER_EXTRA_BODY)

    headers = {
        "Authorization": f"Bearer {_OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }

    try:
        resp = httpx.post(
            OPENROUTER_URL, json=payload, headers=headers, timeout=TIMEOUT
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:400]
        hint = ""
        if e.response.status_code in (401, 403):
            hint = " Check the OPENROUTER_API_KEY environment variable."
        elif e.response.status_code == 404:
            hint = f" Is the model slug '{model}' correct?"
        raise httpx.HTTPError(
            f"OpenRouter returned {e.response.status_code}: {body}{hint}"
        ) from e
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise httpx.HTTPError(
            f"OpenRouter is not reachable at {OPENROUTER_URL} ({e})."
        ) from e

    try:
        data = resp.json()
    except Exception as e:
        raise httpx.HTTPError(f"OpenRouter response is not valid JSON: {e}") from e

    choices = data.get("choices") or []
    if not choices:
        raise httpx.HTTPError(
            f"OpenRouter returned no choices. Response: {resp.text[:400]}"
        )
    return choices[0].get("message", {}).get("content") or ""


def _call_model(
    model: str,
    messages: list[dict],
    system: str | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
) -> str:
    """Dispatch to the configured provider (Ollama or OpenRouter)."""
    if PROVIDER == "openrouter":
        return _call_openrouter(model, messages, system, num_ctx, num_predict, temperature)
    return _call_ollama(model, messages, system, num_ctx, num_predict, temperature)


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> if a qwen3 model emits any. No-op for other models."""
    if not _IS_QWEN3:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _maybe_no_think(prompt: str) -> str:
    """Append /no_think only for Qwen3 models that support it."""
    return f"{prompt}\n\n/no_think" if _IS_QWEN3 else prompt


# ---------------------------------------------------------------------------
# Language detection + translation pre-pass
# ---------------------------------------------------------------------------
# Tokens that strongly indicate the instruction is NOT English. Each entry
# would be unusual in an English code-edit instruction. The list is biased
# toward "false negative" (treat as English): if no marker fires, we skip the
# translation pre-pass entirely. The cost of a missed translation is just a
# slightly worse model output; the cost of a false positive is a wasted
# Ollama call.
_NON_ENGLISH_MARKERS = frozenset({
    # Italian
    "il", "lo", "gli", "che", "non", "del", "della", "dello", "delle", "degli",
    "nel", "nella", "alla", "agli", "questo", "questa", "quello", "quella",
    "rimuovi", "rimuovere", "elimina", "eliminare", "cancella", "cancellare",
    "aggiungi", "aggiungere", "modifica", "modificare", "togli", "togliere",
    "svuota", "scrivi", "scrivere", "crea", "creare", "leggi", "leggere",
    "metodo", "classe", "funzione", "rinomina", "rinominare",
    # French
    "supprime", "supprimer", "ajoute", "ajouter", "modifie", "modifier",
    "retire", "retirer", "fichier", "méthode", "renomme", "renommer",
    "écris", "écrire", "crée", "créer", "lis", "lire", "fonction",
    # Spanish
    "borra", "borrar", "añade", "añadir", "quita", "quitar", "archivo",
    "método", "renombra", "renombrar", "escribe", "escribir", "lee", "leer",
    "clase", "función",
    # German
    "lösche", "löschen", "entferne", "entfernen", "datei", "methode",
    "hinzufügen", "umbenennen", "schreibe", "schreiben", "erstelle",
    "erstellen", "lese", "lesen", "klasse", "funktion",
})

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")


def _is_probably_english(text: str) -> bool:
    """
    Cheap, conservative English detector. Returns True if the text looks like
    English. Errs toward True (no translation) so the common case is free.
    Returns False on:
      - any non-ASCII Latin letter (à, é, ñ, ö, ß, ç, ...)
      - any token in _NON_ENGLISH_MARKERS
    """
    for ch in text:
        if ord(ch) > 127 and ch.isalpha():
            return False
    lowered = text.lower()
    for tok in _WORD_RE.findall(lowered):
        if tok in _NON_ENGLISH_MARKERS:
            return False
    return True


def _normalize_instruction(instruction: str) -> str:
    """
    If the instruction is not English, translate it to English using the same
    local model. This collapses every guard-rail into an English-only problem
    and improves model output quality (qwen3-coder is stronger on English
    instructions). On translation failure, return the original; better to
    attempt the edit than to hard-fail on a translation glitch.
    """
    if _is_probably_english(instruction):
        return instruction
    try:
        translated = _call_model(
            MODEL,
            [{"role": "user", "content": (
                "Translate the following instruction to English. Output ONLY "
                "the translation as plain text; no preamble, no quotes, no "
                "explanation, no markdown.\n\n"
                f"{_maybe_no_think(instruction)}"
            )}],
            num_ctx=NUM_CTX,
            num_predict=TRANSLATE_NUM_PREDICT,
            temperature=TEMPERATURE,
        )
    except httpx.HTTPError as e:
        _log.warning("translation failed (%s); using original instruction", e)
        return instruction
    cleaned = _strip_think_tags(translated).strip()
    _log.debug("translated instruction: %r -> %r", instruction, cleaned)
    return cleaned or instruction


# ---------------------------------------------------------------------------
# System prompt for edit/write tools
# ---------------------------------------------------------------------------
EDIT_SYSTEM = """\
Code editing assistant. Output ONLY «file» blocks with the COMPLETE new content
of each modified file. No prose, no markdown fences, no other tags. Use the
exact absolute path from the input. Omit unchanged files. Never truncate,
never use "... rest unchanged" placeholders. Never emit any tag other than
«file»; these tools only read, edit, and create file contents.

Example A: add an int age field to Foo

INPUT:
«file path="/project/src/Foo.java"»
public record Foo(String name) {}
«/file»

OUTPUT:
«file path="/project/src/Foo.java"»
public record Foo(String name, int age) {}
«/file»

Example B: remove the unused method b

INPUT:
«file path="/project/src/Util.java"»
public class Util {
    public static int a() { return 1; }
    public static int b() { return 2; }
}
«/file»

OUTPUT:
«file path="/project/src/Util.java"»
public class Util {
    public static int a() { return 1; }
}
«/file»
"""

READ_SYSTEM = """\
File analysis assistant. You receive one or more files and an analysis instruction.
Answer in the fewest words that fully address the instruction. No preamble, no
restating the question, no closing summary, no markdown headings unless asked.
Do NOT output «file» blocks or code fences unless the instruction asks for code.
If the answer is a list, return a plain bulleted list and nothing else.
"""

# Appended to EDIT_SYSTEM only when a Java/Kotlin file is the target (Task-021).
# It costs nothing on non-Java calls and steers the model away from the exact
# mistakes the Task-020 guards reject, so they fire less often.
JAVA_RULES = """\
Java/Kotlin file in play, so also:
- Emit the package declaration that matches the file's directory; do not alter it.
- Preserve every import and annotation. Never replace getters, setters,
  constructors, or any member with a comment like "// getters and setters".
- The public top-level type must match the filename (Foo.java -> public ... Foo).
- Use idiomatic Spring: constructor injection; @Service/@Repository/@RestController
  where implied; JPA annotations (@Entity, @Id, @GeneratedValue, @Column) on entities.
"""


def _edit_system_for(paths: list[str]) -> str:
    """EDIT_SYSTEM, plus the Java rules block when any target is Java/Kotlin."""
    if any(Path(p).suffix.lower() in _JAVA_EXTS for p in paths):
        return f"{EDIT_SYSTEM}\n{JAVA_RULES}"
    return EDIT_SYSTEM


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
_FILE_BLOCK_RE = re.compile(r'«file path="([^"]+)"»\n?(.*?)\n?«/file»', re.DOTALL)


def _parse_file_blocks(text: str) -> dict[str, str]:
    return {path: content for path, content in _FILE_BLOCK_RE.findall(text)}


def _fallback_markdown_extract(text: str, files: list[str]) -> dict[str, str]:
    """If the model returned a fenced code block instead of a «file» block,
    map it to the only input file. Single-file only."""
    if len(files) != 1:
        return {}
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if not match:
        return {}
    return {files[0]: match.group(1)}


def _extract_file_changes(raw: str, fallback_files: list[str]) -> dict[str, str]:
    """Try «file» block parsing, then fall back to fenced markdown."""
    changes = _parse_file_blocks(raw)
    if changes:
        return changes
    return _fallback_markdown_extract(raw, fallback_files)


def _call_with_parse_retry(
    first_msg: str,
    fallback_files: list[str],
    num_predict: int | None = None,
    system: str = EDIT_SYSTEM,
) -> tuple[dict[str, str] | None, str]:
    """
    Call the model with first_msg. If the output cannot be parsed into any
    «file» block (nor a markdown-fenced fallback), retry ONCE with a stricter
    user message that tells the model its previous output was malformed. If
    the retry also fails, return a bounded error; the raw output is truncated
    to PARSE_FAIL_ECHO_LIMIT chars so the caller's context is not blown up.

    Returns (changes, error). On success, error is ''. On failure, changes is
    None and error is a human-readable diagnostic.
    """
    try:
        raw = _call_model(
            MODEL, [{"role": "user", "content": first_msg}], system=system,
            num_ctx=NUM_CTX, num_predict=num_predict, temperature=TEMPERATURE,
        )
    except (httpx.HTTPError, IndexError, KeyError, ValueError) as e:
        return None, f"Model call failed: {e}"
    raw = _strip_think_tags(raw)
    _log.debug("model raw output (%d chars): %s", len(raw), _bound(raw))

    changes = _extract_file_changes(raw, fallback_files)
    if changes:
        return changes, ""

    # Retry with a stricter prompt. We keep the same system prompt and resend
    # the original task, but prepend a hard instruction about format.
    _log.warning("parse-retry: no «file» blocks in first output; retrying once")
    retry_msg = (
        "Previous output was MALFORMED and unparseable. "
        "Output ONLY «file» blocks. Try again.\n\n"
        f"{first_msg}"
    )
    try:
        raw = _call_model(
            MODEL, [{"role": "user", "content": retry_msg}], system=system,
            num_ctx=NUM_CTX, num_predict=num_predict, temperature=TEMPERATURE,
        )
    except (httpx.HTTPError, IndexError, KeyError, ValueError) as e:
        return None, f"Model call failed on retry: {e}"
    raw = _strip_think_tags(raw)
    _log.debug("model raw output on retry (%d chars): %s", len(raw), _bound(raw))

    changes = _extract_file_changes(raw, fallback_files)
    if changes:
        return changes, ""

    _log.warning("parse failure persisted after retry; surfacing bounded error")
    truncated = raw[:PARSE_FAIL_ECHO_LIMIT]
    if len(raw) > PARSE_FAIL_ECHO_LIMIT:
        truncated += f"\n... [{len(raw) - PARSE_FAIL_ECHO_LIMIT} more chars truncated]"
    return None, (
        "No «file» blocks found after retry. Raw output (truncated):\n"
        f"{truncated}"
    )


# ---------------------------------------------------------------------------
# Path normalization (Windows-aware: case-insensitive, slash-agnostic)
# ---------------------------------------------------------------------------
def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


# ---------------------------------------------------------------------------
# File I/O; preserves original line endings (CRLF on Windows must NOT be
# silently rewritten to LF on every edit).
# ---------------------------------------------------------------------------
def _read_file(path: Path) -> tuple[str, bytes, bytes]:
    """
    Returns (lf_text, original_eol, original_bytes).
    The text is normalized to LF for the model; eol is captured to re-apply on write.

    Raises UnicodeDecodeError for binary content (NUL byte sniff) or non-UTF-8
    encodings; callers turn that into a clean diagnostic instead of crashing the
    whole tool call. (Task-003)
    """
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise UnicodeDecodeError("utf-8", raw[:1], 0, 1, "NUL byte: file looks binary")
    eol = b"\r\n" if raw.count(b"\r\n") > 0 else b"\n"
    text = raw.decode("utf-8")
    lf_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return lf_text, eol, raw


def _encode_with_eol(lf_text: str, eol: bytes) -> bytes:
    """Encode LF text back to bytes using the requested line ending."""
    normalized = lf_text.replace("\r\n", "\n").replace("\r", "\n")
    if eol == b"\r\n":
        normalized = normalized.replace("\n", "\r\n")
    return normalized.encode("utf-8")


def _atomic_write(target: Path, content: bytes) -> None:
    """
    Atomic write on Windows: temp file in the SAME directory, then os.replace.
    Raises PermissionError/OSError on locked or unwritable files; caller handles those.
    """
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(parent), prefix=".local-mcp-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp, str(target))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_context_blocks(
    context_files: list[str], edit_tool: str
) -> tuple[list[str] | None, str]:
    """Read read-only reference files into «context» blocks for the prompt.

    These are embedded for the model to consult but are NOT added to the
    editable allowlist, so any block the model emits for them is rejected like
    any other out-of-allowlist path. Same UTF-8/binary handling as the editable
    files (Task-003). Returns (blocks, "") on success, or (None, error). (Task-012)
    """
    blocks: list[str] = []
    for raw_path in context_files:
        p = Path(raw_path)
        if not p.exists():
            return None, f"Error: context file not found: {raw_path}"
        if not p.is_file():
            return None, f"Error: context path is not a regular file: {raw_path}"
        try:
            lf, _eol, _raw = _read_file(p)
        except UnicodeDecodeError:
            return None, (
                f"Error: context file {raw_path} is not UTF-8 text (binary or "
                f"unknown encoding); use the built-in {edit_tool} tool for this file."
            )
        except OSError as e:
            return None, f"Error reading context file {raw_path}: {e}"
        blocks.append(f'«context path="{raw_path}"»\n{lf}\n«/context»')
    return blocks, ""


# Header that precedes «context» blocks so the model knows they are reference
# material it must not rewrite. Only added when context_files are present, so it
# costs nothing on the common no-context call. (Task-012)
_CONTEXT_PREAMBLE = (
    "The «context» blocks below are READ-ONLY references. Do NOT output them; "
    "only output «file» blocks for files you actually changed.\n\n"
)


# ---------------------------------------------------------------------------
# Input-size guard; pre-model-call validation
# ---------------------------------------------------------------------------
# Conservative character-per-token ratio. Code typically runs 3-4 chars/token;
# using 3 over-estimates token usage, providing a safety margin.
_CHARS_PER_TOKEN = 3

# Budget reserved for the system prompt, instruction text, and framing.
# EDIT_SYSTEM ~250 tokens, READ_SYSTEM ~80 tokens, instruction up to 200 tokens;
# 512 covers all cases with headroom.
_PROMPT_OVERHEAD_TOKENS = 512


def _check_input_size(content: str, ctx_limit: int, label: str) -> str | None:
    """
    Estimate the token count of *content* and reject if it will not fit as input.

    In Ollama, num_ctx covers prompt PLUS completion. This checks only that the
    input fits; reserving room for the OUTPUT (the regenerated editable files) is
    done by the caller, which knows how much of the input will be re-emitted
    versus consulted as read-only context. (Task-006, Task-012)

    Returns None if acceptable, or a human-readable error string.
    """
    available = ctx_limit - _PROMPT_OVERHEAD_TOKENS
    estimated_tokens = len(content) // _CHARS_PER_TOKEN
    if estimated_tokens > available:
        return (
            f"Error: {label} is too large to send to the model "
            f"(estimated ~{estimated_tokens} tokens; limit after overhead is "
            f"{available} of num_ctx={ctx_limit}). Use the built-in tools for this file."
        )
    return None


# ---------------------------------------------------------------------------
# Guard-rails; pre-write validation
# ---------------------------------------------------------------------------
def _instruction_allows_shrink(instruction: str) -> bool:
    low = instruction.lower()
    return any(kw in low for kw in REMOVAL_KEYWORDS)


def _check_non_empty(content: str) -> str | None:
    if not content.strip():
        return "empty content (use the built-in tools to remove a file)"
    return None


def _check_truncation_markers(new: str, original: str | None) -> str | None:
    original_lines: set[str] = set()
    if original is not None:
        original_lines = {line.strip() for line in original.splitlines()}
    for line in new.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for marker in TRUNCATION_MARKERS:
            if stripped == marker and stripped not in original_lines:
                return f"truncation marker on its own line: {marker!r}"
    return None


def _check_shrink(new: str, original: str, instruction: str) -> str | None:
    if len(original) == 0:
        return None
    if len(new) >= SHRINK_RATIO * len(original):
        return None
    if _instruction_allows_shrink(instruction):
        return None
    return (
        f"suspicious shrink ({len(original)} -> {len(new)} chars, "
        f"instruction had no removal keyword)"
    )


def _bracket_delta(text: str) -> tuple[int, int, int]:
    return (
        text.count("{") - text.count("}"),
        text.count("(") - text.count(")"),
        text.count("[") - text.count("]"),
    )


def _check_parses(content: str, ext: str) -> str | None:
    """
    Semantic guard: for languages with a free stdlib parser, actually parse
    the new content and reject on syntax errors. This catches mid-stream
    truncation and malformed edits that the bracket-delta heuristic misses
    (e.g. unterminated strings, stray indentation, missing commas in JSON).
    Only runs for .py and .json; adding JS/TS would require shelling out.
    """
    ext = ext.lower()
    if ext == ".py":
        try:
            ast.parse(content)
        except SyntaxError as e:
            line = e.lineno if e.lineno is not None else "?"
            return f"python syntax error at line {line}: {e.msg}"
    elif ext == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return f"json parse error at line {e.lineno}: {e.msg}"
    return None


def _check_bracket_delta(new: str, original: str | None, ext: str) -> str | None:
    if ext.lower() not in BRACKET_CHECK_EXTS:
        return None
    new_d = _bracket_delta(new)
    if original is None:
        # Absolute balance check (used by local_write; no original to diff against)
        if new_d != (0, 0, 0):
            return (
                f"unbalanced brackets in new file: "
                f"{{}}={new_d[0]} ()={new_d[1]} []={new_d[2]}"
            )
        return None
    old_d = _bracket_delta(original)
    if new_d != (0, 0, 0) and new_d != old_d:
        return (
            f"bracket delta changed: old={old_d} new={new_d} "
            "(possible mid-stream truncation)"
        )
    return None


def _java_placeholder_phrase(line: str) -> str | None:
    """Reduce a comment-only line to its canonical placeholder phrase, or None.

    Strips comment delimiters and filler (dots, parens/brackets, the article
    "the") so that "// ... rest of the class ..." normalizes to "rest of class".
    Returns None for any line that is not a pure comment.
    """
    if not _JAVA_COMMENT_PREFIX_RE.match(line):
        return None
    s = _JAVA_COMMENT_OPEN_RE.sub("", line.strip())
    s = _JAVA_COMMENT_CLOSE_RE.sub("", s)
    s = re.sub(r"[.()\[\]*/]", " ", s)
    s = re.sub(r"\bthe\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s or None


def _expected_java_package(path: str) -> str | None:
    """Package a file should declare, derived from its path under a source root.

    Returns the dotted package (possibly "" for a file directly in the root), or
    None when no `src/{main,test}/{java,kotlin}` root is present in the path (so
    the guard simply does not run for files outside a standard layout).
    """
    parts = Path(path).parts
    lowered = [p.lower() for p in parts]
    for root in _JAVA_SOURCE_ROOTS:
        n = len(root)
        for i in range(len(lowered) - n + 1):
            if tuple(lowered[i:i + n]) == root:
                return ".".join(parts[i + n:-1])
    return None


def _check_java(
    new: str, original: str | None, path: str, instruction: str
) -> list[str]:
    """Java/Kotlin-specific guards (Task-020). No-op for other extensions.

    Catches the dominant small-model failures on this verbose, top use-case:
    omitted boilerplate, a package that contradicts the file location, a public
    type that does not match the filename, and silently dropped imports.
    """
    ext = Path(path).suffix.lower()
    if ext not in _JAVA_EXTS:
        return []
    failures: list[str] = []

    # 1. Omission placeholders (delta against the original, like truncation markers)
    original_phrases: set[str] = set()
    if original is not None:
        for line in original.splitlines():
            ph = _java_placeholder_phrase(line)
            if ph:
                original_phrases.add(ph)
    for line in new.splitlines():
        ph = _java_placeholder_phrase(line)
        if ph and ph in _JAVA_PLACEHOLDER_PHRASES and ph not in original_phrases:
            failures.append(f"java omission placeholder on its own line: {line.strip()!r}")
            break

    # 2. Package must match the directory under a detected source root
    expected = _expected_java_package(path)
    if expected:
        m = _PACKAGE_RE.search(new)
        declared = m.group(1) if m else ""
        if declared != expected:
            failures.append(
                f"package {declared!r} does not match location (expected {expected!r})"
                if declared else f"missing package declaration (expected {expected!r})"
            )

    # 3. Public top-level type must match the filename (.java only; Kotlin allows
    #    multiple top-level types and does not tie them to the filename)
    if ext == ".java":
        stem = Path(path).stem
        if stem not in ("package-info", "module-info"):
            m = _PUBLIC_TYPE_RE.search(new)
            if m and m.group(1) != stem:
                failures.append(
                    f"public type {m.group(1)!r} does not match filename {stem!r}"
                )

    # 4. Import-loss heuristic (edit only; mirrors the shrink guard, scoped to
    #    imports). Only fires on a material drop so removing one unused import is
    #    not flagged.
    if original is not None and not _instruction_allows_shrink(instruction):
        old_imp = len(_IMPORT_LINE_RE.findall(original))
        new_imp = len(_IMPORT_LINE_RE.findall(new))
        if old_imp >= 4 and new_imp < old_imp * 0.7:
            failures.append(
                f"import count dropped {old_imp} -> {new_imp} with no removal keyword"
            )

    return failures


# ---------------------------------------------------------------------------
# Deterministic structural extraction (Task-023)
# Turns a code file into a compact API skeleton (package, types, annotations,
# fields, method signatures) with NO model call. Used by local_outline to answer
# structural questions exactly, and by local_read to feed condensed context to
# the model instead of full file bodies.
# ---------------------------------------------------------------------------
_OUTLINE_EXTS = {".java", ".py"}


def _ts_text(code: bytes, node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _ts_annotations(code: bytes, node) -> list[str]:
    out: list[str] = []
    for child in node.children:
        if child.type == "modifiers":
            for m in child.children:
                if m.type in ("annotation", "marker_annotation"):
                    out.append(_ts_text(code, m))
    return out


def _java_outline(lf_text: str) -> str | None:
    """Compact Java API skeleton via tree-sitter, or None if unavailable/empty.

    Emits the package, each type with its annotations, field declarations, and
    method/constructor signatures (return type, name, parameters, annotations).
    Method bodies are dropped, which is where almost all the bytes are.
    """
    if _JAVA_PARSER is None:
        return None
    code = lf_text.encode("utf-8")
    tree = _JAVA_PARSER.parse(code)
    lines: list[str] = []

    pkg = re.search(r"^\s*package\s+([\w.]+)", lf_text, re.MULTILINE)
    if pkg:
        lines.append(f"package {pkg.group(1)}")

    def emit(node, depth: int) -> None:
        pad = "  " * depth
        for child in node.children:
            if child.type in ("class_declaration", "interface_declaration",
                               "record_declaration", "enum_declaration"):
                name = child.child_by_field_name("name")
                anns = _ts_annotations(code, child)
                ann_s = (" ".join(anns) + " ") if anns else ""
                kind = child.type.split("_")[0]
                nm = _ts_text(code, name) if name else "?"
                # record_declaration carries its components like a param list
                params = child.child_by_field_name("parameters")
                psig = _ts_text(code, params) if params else ""
                lines.append(f"{pad}{ann_s}{kind} {nm}{psig}")
                body = child.child_by_field_name("body")
                if body:
                    emit(body, depth + 1)
            elif child.type == "field_declaration":
                anns = _ts_annotations(code, child)
                ann_s = (" ".join(anns) + " ") if anns else ""
                ftype = child.child_by_field_name("type")
                decl = child.child_by_field_name("declarator")
                fname = decl.child_by_field_name("name") if decl else None
                if ftype and fname:
                    lines.append(
                        f"{pad}{ann_s}field {_ts_text(code, ftype)} {_ts_text(code, fname)}"
                    )
            elif child.type in ("method_declaration", "constructor_declaration"):
                name = child.child_by_field_name("name")
                params = child.child_by_field_name("parameters")
                rtype = child.child_by_field_name("type")
                anns = _ts_annotations(code, child)
                ann_s = (" ".join(anns) + " ") if anns else ""
                rt = (_ts_text(code, rtype) + " ") if rtype else ""
                nm = _ts_text(code, name) if name else "?"
                ps = _ts_text(code, params) if params else "()"
                lines.append(f"{pad}{ann_s}{rt}{nm}{ps}")
            else:
                emit(child, depth)

    emit(tree.root_node, 0)
    body_lines = [ln for ln in lines if not ln.startswith("package ")]
    return "\n".join(lines) if body_lines else None


def _python_outline(lf_text: str) -> str | None:
    """Compact Python API skeleton via the stdlib ast, or None on syntax error."""
    try:
        tree = ast.parse(lf_text)
    except SyntaxError:
        return None
    lines: list[str] = []

    def sig(fn: ast.AST) -> str:
        try:
            args = ast.unparse(fn.args)
        except Exception:
            args = "..."
        ret = ""
        if getattr(fn, "returns", None) is not None:
            try:
                ret = f" -> {ast.unparse(fn.returns)}"
            except Exception:
                ret = ""
        kw = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
        return f"{kw} {fn.name}({args}){ret}"

    def decos(node: ast.AST, pad: str) -> None:
        for d in getattr(node, "decorator_list", []):
            try:
                lines.append(f"{pad}@{ast.unparse(d)}")
            except Exception:
                pass

    def emit(body: list[ast.stmt], depth: int) -> None:
        pad = "  " * depth
        for node in body:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(
                    b for b in (_safe_unparse(x) for x in node.bases) if b
                )
                decos(node, pad)
                lines.append(f"{pad}class {node.name}({bases})" if bases
                             else f"{pad}class {node.name}")
                emit(node.body, depth + 1)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decos(node, pad)
                lines.append(f"{pad}{sig(node)}")
    emit(tree.body, 0)
    return "\n".join(lines) if lines else None


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _outline_text(path: str, lf_text: str) -> str | None:
    """Dispatch to a language-specific outliner, or None if unsupported/empty."""
    ext = Path(path).suffix.lower()
    if ext == ".java":
        return _java_outline(lf_text)
    if ext == ".py":
        return _python_outline(lf_text)
    return None


# ---------------------------------------------------------------------------
# Map-reduce for oversized local_read inputs (Task-023)
# The local model is free, so an input larger than one context window is handled
# by summarizing each file/chunk in its own call and synthesizing the partials,
# rather than refusing. Each partial is bounded so the synthesis stays small.
# ---------------------------------------------------------------------------
# Per-file/per-chunk partials are bounded tighter than the final answer; the
# synthesis is bounded by READ_NUM_PREDICT, the same cap as a single-pass read.
_MAPREDUCE_PARTIAL_NUM_PREDICT = 768


def _read_model_call(content: str, num_predict: int | None) -> str:
    return _strip_think_tags(_call_model(
        MODEL, [{"role": "user", "content": _maybe_no_think(content)}],
        system=READ_SYSTEM, num_ctx=NUM_CTX, num_predict=num_predict,
        temperature=READ_TEMPERATURE,
    ))


def _chunk_by_lines(lf: str, char_budget: int) -> list[str]:
    """Split text into chunks of at most char_budget chars, breaking on lines."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for ln in lf.splitlines(keepends=True):
        if cur and cur_len + len(ln) > char_budget:
            chunks.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += len(ln)
    if cur:
        chunks.append("".join(cur))
    return chunks


def _read_mapreduce(files_data: list[tuple[str, str]], instruction: str) -> str:
    """Summarize each file/chunk in its own free local call, then synthesize.

    Oversized files are first reduced to a deterministic skeleton when possible
    (no extra model cost); only when even the skeleton will not fit is the raw
    file chunked. Returns the final answer text, or an error string.
    """
    available = NUM_CTX - _PROMPT_OVERHEAD_TOKENS
    map_char_budget = max(
        1000, (available - _MAPREDUCE_PARTIAL_NUM_PREDICT) * _CHARS_PER_TOKEN
    )
    suffix = f"\n\nInstruction: {instruction}\n\nAnswer for THIS FILE ONLY, concisely."
    partials: list[str] = []

    for path, lf in files_data:
        block = f'«file path="{path}"»\n{lf}\n«/file»'
        try:
            if len(block) + len(suffix) <= map_char_budget:
                partials.append(
                    f"=== {path} ===\n"
                    + _read_model_call(block + suffix, _MAPREDUCE_PARTIAL_NUM_PREDICT)
                )
                continue
            skel = _outline_text(path, lf)
            if skel and len(skel) + len(suffix) <= map_char_budget:
                content = (
                    f'«outline path="{path}"»\n{skel}\n«/outline»'
                    f"\n\nInstruction: {instruction}\n\nThis is a signatures-only "
                    "skeleton (method bodies omitted). Answer for THIS FILE ONLY, concisely."
                )
                partials.append(
                    f"=== {path} (outline) ===\n"
                    + _read_model_call(content, _MAPREDUCE_PARTIAL_NUM_PREDICT)
                )
                continue
            chunks = _chunk_by_lines(lf, map_char_budget - len(suffix))
            for i, ch in enumerate(chunks):
                content = f'«file path="{path}" part {i+1}/{len(chunks)}»\n{ch}\n«/file»' + suffix
                partials.append(
                    f"=== {path} part {i+1}/{len(chunks)} ===\n"
                    + _read_model_call(content, _MAPREDUCE_PARTIAL_NUM_PREDICT)
                )
        except (httpx.HTTPError, IndexError, KeyError, ValueError) as e:
            return f"Model call failed during map pass on {path}: {e}"

    if len(partials) == 1:
        head, _, body = partials[0].partition("\n")
        return body or head

    joined = "\n\n".join(partials)
    reduce_msg = (
        f"These are partial analyses of separate files/chunks:\n\n{joined}"
        f"\n\nInstruction: {instruction}\n\nSynthesize ONE coherent answer to the "
        "instruction. Do not repeat the per-file headers."
    )
    if _check_input_size(reduce_msg, NUM_CTX, "partial summaries") is None:
        try:
            return _read_model_call(reduce_msg, READ_NUM_PREDICT)
        except (httpx.HTTPError, IndexError, KeyError, ValueError) as e:
            return f"Model call failed during reduce pass: {e}"
    # Too many partials to synthesize in one pass: return them as-is.
    return joined


# ---------------------------------------------------------------------------
# local_outline
# ---------------------------------------------------------------------------
@mcp.tool()
def local_outline(files: list[str]) -> str:
    """Return a compact API skeleton (package, types, annotations, fields, method
    signatures; no bodies) for .java/.py files. DETERMINISTIC, no model call.
    Use instead of local_read/Read when you need an interface or shape, not
    implementations: a few hundred tokens instead of the whole file.

    Args:
        files: absolute paths to outline.
    """
    sections: list[str] = []
    for raw_path in files:
        p = Path(raw_path)
        if not p.exists():
            return f"Error: file not found: {raw_path}"
        if not p.is_file():
            return f"Error: not a regular file: {raw_path}"
        if p.suffix.lower() not in _OUTLINE_EXTS:
            return (
                f"Error: local_outline supports {sorted(_OUTLINE_EXTS)} only; "
                f"{raw_path} is unsupported. Use local_read for prose analysis."
            )
        try:
            lf, _eol, _raw = _read_file(p)
        except UnicodeDecodeError:
            return f"Error: {raw_path} is not UTF-8 text."
        except OSError as e:
            return f"Error reading {raw_path}: {e}"
        skel = _outline_text(raw_path, lf)
        if skel is None:
            if p.suffix.lower() == ".java" and _JAVA_PARSER is None:
                return (
                    "Error: Java outline needs tree-sitter, which is not installed. "
                    "Run via 'uv run server.py' (it declares the dependency) or use local_read."
                )
            return f"Error: could not extract an outline from {raw_path} (syntax error or empty)."
        sections.append(f"{raw_path}\n{skel}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# local_read
# ---------------------------------------------------------------------------
@mcp.tool()
def local_read(files: list[str], instruction: str) -> str:
    """Analyze files via the local model; returns analysis text, never modifies
    files. Use for summaries, review, pattern-finding. Output enters Claude's
    context and is capped, so ask a focused question; a broad instruction yields
    a broad, token-costly answer. NOT for verbatim retrieval (use the built-in
    Read). Call sequentially (single GPU).

    Args:
        files: absolute paths to analyze.
        instruction: the analysis task (any language; translated server-side).
    """
    instruction = _normalize_instruction(instruction)

    if _RETRIEVAL_RE.search(instruction):
        return (
            "Error: local_read cannot be used for verbatim file retrieval. "
            "Use the built-in Read tool instead; it reads the file directly "
            "into context without a local-model round-trip. "
            "local_read is for analysis tasks (summarize, review, find patterns)."
        )

    files_data: list[tuple[str, str]] = []
    for raw_path in files:
        p = Path(raw_path)
        if not p.exists():
            return f"Error: file not found: {raw_path}"
        if not p.is_file():
            return f"Error: not a regular file: {raw_path}"
        try:
            lf, _eol, _raw = _read_file(p)
        except UnicodeDecodeError:
            return (
                f"Error: {raw_path} is not UTF-8 text (binary or unknown "
                "encoding); use the built-in Read tool for this file."
            )
        except OSError as e:
            return f"Error reading {raw_path}: {e}"
        files_data.append((raw_path, lf))

    file_blocks = [f'«file path="{path}"»\n{lf}\n«/file»' for path, lf in files_data]
    user_msg = "\n\n".join(file_blocks) + f"\n\nInstruction: {instruction}"
    user_msg = _maybe_no_think(user_msg)

    # When everything fits in one context, a single pass is cheapest. Otherwise
    # map-reduce over free local calls instead of refusing the input. (Task-023)
    if _check_input_size(user_msg, NUM_CTX, f"{len(files)} file(s)") is not None:
        return _read_mapreduce(files_data, instruction)

    try:
        raw = _call_model(
            MODEL,
            [{"role": "user", "content": user_msg}],
            system=READ_SYSTEM,
            num_ctx=NUM_CTX,
            num_predict=READ_NUM_PREDICT,
            temperature=READ_TEMPERATURE,
        )
    except (httpx.HTTPError, IndexError, KeyError, ValueError) as e:
        return f"[{MODEL}] Model call failed: {e}"
    return _strip_think_tags(raw)


# ---------------------------------------------------------------------------
# local_edit
# ---------------------------------------------------------------------------
@mcp.tool()
def local_edit(
    files: list[str], instruction: str, context_files: list[str] | None = None
) -> str:
    """Edit existing files locally instead of the built-in Edit tool: file
    contents never enter Claude's context. Validates each change server-side and
    applies atomically. Call sequentially (single GPU).

    Args:
        files: absolute paths to edit in place (cannot create/delete/rename).
        instruction: the change (any language; translated). Include a removal
            verb (delete/remove/strip/...) for a large size reduction, or the
            shrink guard rejects.
        context_files: optional absolute paths the model may READ for reference
            (interfaces, callers, constants) but must NOT edit. Lets a cross-file
            edit succeed without those files entering Claude's context.
    """
    # 0. Normalize instruction to English (no-op if already English)
    instruction = _normalize_instruction(instruction)

    # 1. Read inputs (preserve line endings, build allowlist of normalized paths)
    originals: dict[str, tuple[str, bytes, bytes]] = {}
    canonical: dict[str, str] = {}  # normalized -> original input string
    for raw_path in files:
        p = Path(raw_path)
        if not p.exists():
            return f"Error: file not found: {raw_path}"
        if not p.is_file():
            return f"Error: not a regular file: {raw_path}"
        try:
            lf, eol, raw = _read_file(p)
        except UnicodeDecodeError:
            return (
                f"Error: {raw_path} is not UTF-8 text (binary or unknown "
                "encoding); use the built-in Edit tool for this file."
            )
        except OSError as e:
            return f"Error reading {raw_path}: {e}"
        originals[raw_path] = (lf, eol, raw)
        canonical[_norm_path(raw_path)] = raw_path

    # 1b. Read read-only context files (consulted, never editable). (Task-012)
    context_block = ""
    if context_files:
        ctx_blocks, ctx_err = _read_context_blocks(context_files, "Edit")
        if ctx_blocks is None:
            return ctx_err
        context_block = _CONTEXT_PREAMBLE + "\n\n".join(ctx_blocks) + "\n\n"

    # 2. Build prompt; embed LF-normalized contents
    files_block = "\n\n".join(
        f'«file path="{path}"»\n{originals[path][0]}\n«/file»'
        for path in files
    )
    user_msg = f"{context_block}{files_block}\n\nInstruction: {instruction}"
    user_msg = _maybe_no_think(user_msg)

    label = f"{len(files)} file(s)"
    if context_files:
        label += f" + {len(context_files)} context file(s)"

    _log.debug(
        "local_edit: model=%s files=%s context=%s instruction=%r",
        MODEL, files, context_files or [], _bound(instruction, 300),
    )

    # The whole prompt (editable + context) must fit as input.
    size_err = _check_input_size(user_msg, NUM_CTX, label)
    if size_err:
        return size_err

    # Reserve room for the OUTPUT, which is only the editable files regenerated
    # in full (context files are never re-emitted). cap num_predict at the
    # context left after the whole input; reject if that is not enough to
    # re-emit the editable files. (Task-006, generalized for context in Task-012)
    estimated_input = len(user_msg) // _CHARS_PER_TOKEN
    editable_output_tokens = sum(len(originals[p][0]) for p in files) // _CHARS_PER_TOKEN
    edit_num_predict = max(256, (NUM_CTX - _PROMPT_OVERHEAD_TOKENS) - estimated_input)
    if edit_num_predict < editable_output_tokens:
        return (
            f"Error: {label} too large to regenerate in one pass "
            f"(input ~{estimated_input} tokens leaves room for ~{edit_num_predict} "
            f"output tokens, but the editable files need ~{editable_output_tokens} "
            f"within num_ctx={NUM_CTX}). Use the built-in Edit tool, or pass fewer "
            "context files."
        )

    # 3. Call model (with one automatic retry on parse failure). Java/Kotlin
    #    targets get the Java rules appended to the system prompt. (Task-021)
    file_changes_raw, err = _call_with_parse_retry(
        user_msg, files, num_predict=edit_num_predict,
        system=_edit_system_for(files),
    )
    if file_changes_raw is None:
        return f"[{MODEL}] {err}"

    # 4. Resolve emitted paths against the allowlist (Windows-aware normalize)
    file_changes: dict[str, str] = {}
    unknown: list[str] = []
    for emitted_path, content in file_changes_raw.items():
        norm = _norm_path(emitted_path)
        if norm not in canonical:
            unknown.append(emitted_path)
            continue
        file_changes[canonical[norm]] = content

    if unknown:
        _log.debug("rejected out-of-allowlist paths: %s", unknown)
        return (
            f"[{MODEL}] REJECTED; model emitted paths not in the input allowlist:\n"
            + "\n".join(f"  - {p}" for p in unknown)
            + "\nNo files were modified."
        )

    # 5. Identity no-op: silently drop unchanged files
    no_ops: list[str] = []
    for path in list(file_changes.keys()):
        if file_changes[path] == originals[path][0]:
            no_ops.append(path)
            del file_changes[path]

    # 6. Run guards on remaining file changes
    failures: list[str] = []
    for path, new_content in file_changes.items():
        original_lf = originals[path][0]
        ext = Path(path).suffix
        for check in (
            _check_non_empty(new_content),
            _check_truncation_markers(new_content, original_lf),
            _check_shrink(new_content, original_lf, instruction),
            _check_bracket_delta(new_content, original_lf, ext),
            _check_parses(new_content, ext),
        ):
            if check:
                failures.append(f"{path}: {check}")
        for jcheck in _check_java(new_content, original_lf, path, instruction):
            failures.append(f"{path}: {jcheck}")

    if failures:
        _log.debug(
            "guard rejections: %s; rejected content: %s",
            failures, {p: _bound(c, 500) for p, c in file_changes.items()},
        )
        return (
            f"[{MODEL}] REJECTED; guard-rail failures:\n"
            + "\n".join(f"  - {f}" for f in failures)
            + "\nNo files were modified."
        )

    if not file_changes:
        _log.debug(
            "no-op: all emitted content matched originals (unchanged=%s)",
            [Path(p).name for p in no_ops],
        )
        return "No changes proposed (model output matched originals)."

    # 7. Atomic apply with revert on failure
    written: list[str] = []
    try:
        for path, new_content in file_changes.items():
            eol = originals[path][1]
            _atomic_write(Path(path), _encode_with_eol(new_content, eol))
            written.append(path)
    except OSError as e:
        # Revert any successful writes by restoring original bytes
        for path in written:
            try:
                Path(path).write_bytes(originals[path][2])
            except OSError:
                pass
        msg = str(e)
        if isinstance(e, PermissionError):
            msg = f"file is locked or not writable ({e})"
        _log.warning("write failed (%s); reverted %d file(s)", e, len(written))
        return (
            f"[{MODEL}] REJECTED during apply; {msg}\n"
            f"All successful writes have been reverted. No files modified."
        )

    # Report by exception: Claude already knows the paths it passed, so on
    # all-success just give the count; only name the surprising no-op minority.
    summary = f"Modified {len(file_changes)} file(s)."
    if no_ops:
        summary += " Unchanged: " + ", ".join(Path(p).name for p in no_ops)
    _log.debug("local_edit applied: %s (written=%s)", summary, written)
    return summary


# ---------------------------------------------------------------------------
# local_write
# ---------------------------------------------------------------------------
@mcp.tool()
def local_write(
    path: str, instruction: str, context_files: list[str] | None = None
) -> str:
    """Create a NEW file locally instead of the built-in Write tool. Worth it
    only when the instruction is much shorter than the output (stubs,
    boilerplate, scaffolds). Refuses to overwrite (use local_edit). Call
    sequentially (single GPU).

    Args:
        path: absolute path to create.
        instruction: concise spec (any language; translated server-side).
        context_files: optional absolute paths the model may READ for reference
            (an interface to implement, a sibling to match) but must NOT write,
            without those files entering Claude's context.
    """
    target = Path(path)
    if target.exists():
        return f"Error: path already exists (use local_edit instead): {path}"

    instruction = _normalize_instruction(instruction)

    context_block = ""
    if context_files:
        ctx_blocks, ctx_err = _read_context_blocks(context_files, "Write")
        if ctx_blocks is None:
            return ctx_err
        context_block = _CONTEXT_PREAMBLE + "\n\n".join(ctx_blocks) + "\n\n"

    user_msg = (
        f"{context_block}"
        f'Create a new file at the absolute path "{path}".\n\n'
        f"Instruction: {instruction}"
    )
    user_msg = _maybe_no_think(user_msg)

    if context_files:
        size_err = _check_input_size(
            user_msg, NUM_CTX, f"{len(context_files)} context file(s)"
        )
        if size_err:
            return size_err

    file_changes_raw, err = _call_with_parse_retry(
        user_msg, [path], system=_edit_system_for([path])
    )
    if file_changes_raw is None:
        return f"[{MODEL}] {err}"

    if len(file_changes_raw) != 1:
        return (
            f"[{MODEL}] REJECTED; local_write expects exactly 1 «file» block, "
            f"got {len(file_changes_raw)}."
        )

    emitted_path, content = next(iter(file_changes_raw.items()))
    if _norm_path(emitted_path) != _norm_path(path):
        return (
            f"[{MODEL}] REJECTED; model wrote to a different path than requested.\n"
            f"  requested: {path}\n  emitted:   {emitted_path}"
        )

    # Guards (no original; use absolute bracket balance)
    failures: list[str] = []
    for check in (
        _check_non_empty(content),
        _check_truncation_markers(content, None),
        _check_bracket_delta(content, None, target.suffix),
        _check_parses(content, target.suffix),
    ):
        if check:
            failures.append(f"{path}: {check}")
    for jcheck in _check_java(content, None, path, instruction):
        failures.append(f"{path}: {jcheck}")
    if failures:
        return (
            f"[{MODEL}] REJECTED; guard-rail failures:\n"
            + "\n".join(f"  - {f}" for f in failures)
            + "\nFile was not created."
        )

    # Write (new file -> default to LF)
    try:
        _atomic_write(target, _encode_with_eol(content, b"\n"))
    except OSError as e:
        msg = f"file is locked or not writable ({e})" if isinstance(e, PermissionError) else str(e)
        return f"[{MODEL}] REJECTED during apply; {msg}\nFile was not created."

    line_count = content.count("\n") + (0 if content.endswith("\n") else 1)
    return f"Created {path} ({line_count} lines)"


# ---------------------------------------------------------------------------
# local_snippet
# ---------------------------------------------------------------------------
@mcp.tool()
def local_snippet(prompt: str) -> str:
    """Fallback for short text with no file destination (regex, SQL, one-liners).
    Output enters Claude's context, so prefer local_edit/local_write when the
    result lands in a file. Call sequentially (single GPU).

    Args:
        prompt: the task (any language; translated server-side).
    """
    prompt = _normalize_instruction(prompt)
    snippet_system = (
        "Terse code/snippet generator. Output ONLY the bare artifact (the code, "
        "regex, query, or text) and nothing else: no prose, no explanation, no "
        "examples, no summary, no surrounding markdown fences or backticks."
    )
    full_prompt = _maybe_no_think(prompt)
    try:
        raw = _call_model(
            MODEL,
            [{"role": "user", "content": full_prompt}],
            system=snippet_system,
            num_ctx=NUM_CTX,
            num_predict=SNIPPET_NUM_PREDICT,
            temperature=READ_TEMPERATURE,
        )
    except httpx.HTTPError as e:
        return f"[{MODEL}] Model call failed: {e}"
    return _strip_think_tags(raw)


if __name__ == "__main__":
    mcp.run()
