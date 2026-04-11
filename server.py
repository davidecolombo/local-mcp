#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp[cli]>=1.0.0",
#   "httpx>=0.27.0",
# ]
# ///
"""
Local MCP Server — routes implementation work to local Ollama models.

The goal is to save Claude tokens: file contents are read, edited, and written
back entirely on the server side, so they almost never round-trip through
Claude's context. Claude only sees a short summary or a guard-rail rejection.

Tools (in order of token-efficiency):
  local_edit(files, instruction)    -> modifies existing files in place
  local_write(path,  instruction)   -> creates a new file from scratch
  local_delete(paths)               -> deletes files (no LLM call)
  local_rename(src, dst)            -> renames/moves a file (no LLM call)
  local_snippet(prompt)             -> returns text (round-trip; fallback)

Single-model architecture (iteration 2):
  All Ollama-backed tools call qwen3-coder:30b (MoE, 3B active params per
  token). 16 GB VRAM cannot host two models simultaneously, so trying to
  alternate between a small and large tier just thrashes. The coder model is
  pinned in VRAM with keep_alive=-1 so it never gets evicted between calls.
  Some expert layers spill to CPU (~27%) — acceptable because MoE only
  activates ~3B of the 30B total params per token. Measured ~48 tok/s on
  RTX 5070 Ti.

Iteration 3:
  - Delete and rename are split into dedicated no-LLM tools (local_delete,
    local_rename). local_edit no longer parses or accepts <delete/> blocks
    of any form. The model is no longer trusted with the decision to remove
    or move files; the caller already knows what it wants and a syscall
    is cheaper and more reliable than a parsed tag.
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
import os
import re
import tempfile
import threading
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-mcp")

# ---------------------------------------------------------------------------
# Model configuration — loaded from model-config.json (optional).
# See configs/ for ready-to-use templates.
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS: dict = {
    "model": "qwen3-coder:30b",
    "ollama_url": "http://localhost:11434/api/chat",
    "edit_ctx": 32768,
    "snippet_ctx": 4096,
    "snippet_num_predict": 1024,
    "translate_ctx": 2048,
    "translate_num_predict": 512,
    "timeout": 1200,
}


def _load_model_config() -> dict:
    """Load model-config.json from the same directory as server.py.

    Missing file or missing keys fall back to _CONFIG_DEFAULTS.
    """
    config_path = Path(__file__).resolve().parent / "model-config.json"
    cfg = dict(_CONFIG_DEFAULTS)
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                for key in _CONFIG_DEFAULTS:
                    if key in user_cfg:
                        cfg[key] = user_cfg[key]
        except (json.JSONDecodeError, OSError):
            pass  # Malformed or unreadable — use defaults silently
    return cfg


_cfg = _load_model_config()

MODEL: str               = _cfg["model"]
OLLAMA_URL: str           = _cfg["ollama_url"]
EDIT_CTX: int             = _cfg["edit_ctx"]
SNIPPET_CTX: int          = _cfg["snippet_ctx"]
SNIPPET_NUM_PREDICT: int  = _cfg["snippet_num_predict"]
TRANSLATE_CTX: int        = _cfg["translate_ctx"]
TRANSLATE_NUM_PREDICT: int = _cfg["translate_num_predict"]
TIMEOUT: int              = _cfg["timeout"]

# One GPU, one request at a time. Non-blocking: callers get an immediate
# error rather than queueing behind a long-running generation.
_OLLAMA_LOCK = threading.Lock()

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

# Lazy-output markers — matched as WHOLE TRIMMED LINES, and only flagged when
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
# retry. Bounded so a rambling model response can't blow up Claude's context.
PARSE_FAIL_ECHO_LIMIT = 600


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------
def _call_ollama(
    model: str,
    messages: list[dict],
    system: str | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
) -> str:
    options: dict = {"num_ctx": num_ctx if num_ctx is not None else EDIT_CTX}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        # Pin the model in VRAM forever — single-model architecture, no eviction.
        "keep_alive": -1,
        "options": options,
    }
    if system:
        payload["system"] = system
    # Streaming: timeout applies per-chunk, not to the whole response.
    # This avoids false timeouts when the model outputs a large file; as long
    # as the model keeps producing tokens the connection stays alive.
    # Non-blocking lock: if another call is already running, fail immediately
    # rather than queueing (a queued call would time out behind a long generation).
    if not _OLLAMA_LOCK.acquire(blocking=False):
        raise httpx.HTTPError(
            "Ollama busy: another local_* call is in progress. "
            "Retry this call sequentially after the current one completes."
        )
    try:
        chunks: list[str] = []
        with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=TIMEOUT) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content", "")
                if content:
                    chunks.append(content)
                if data.get("done", False):
                    break
        return "".join(chunks)
    finally:
        _OLLAMA_LOCK.release()


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
      - any non-ASCII Latin letter (à, é, ñ, ö, ß, ç, …)
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
    instructions). On translation failure, return the original — better to
    attempt the edit than to hard-fail on a translation glitch.
    """
    if _is_probably_english(instruction):
        return instruction
    try:
        translated = _call_ollama(
            MODEL,
            [{"role": "user", "content": (
                "Translate the following instruction to English. Output ONLY "
                "the translation as plain text — no preamble, no quotes, no "
                "explanation, no markdown.\n\n"
                f"{_maybe_no_think(instruction)}"
            )}],
            num_ctx=TRANSLATE_CTX,
            num_predict=TRANSLATE_NUM_PREDICT,
        )
    except httpx.HTTPError:
        return instruction
    cleaned = _strip_think_tags(translated).strip()
    return cleaned or instruction


# ---------------------------------------------------------------------------
# System prompt for edit/write tools
# ---------------------------------------------------------------------------
EDIT_SYSTEM = """\
Code editing assistant. Output ONLY «file» blocks with the COMPLETE new content
of each modified file. No prose, no markdown fences, no other tags. Use the
exact absolute path from the input. Omit unchanged files. Never truncate,
never use "... rest unchanged" placeholders. Never emit <delete> or any tag
other than «file» (deletion and rename are handled by separate tools).

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
) -> tuple[dict[str, str] | None, str]:
    """
    Call the model with first_msg. If the output cannot be parsed into any
    «file» block (nor a markdown-fenced fallback), retry ONCE with a stricter
    user message that tells the model its previous output was malformed. If
    the retry also fails, return a bounded error — the raw output is truncated
    to PARSE_FAIL_ECHO_LIMIT chars so the caller's context is not blown up.

    Returns (changes, error). On success, error is ''. On failure, changes is
    None and error is a human-readable diagnostic.
    """
    try:
        raw = _call_ollama(
            MODEL, [{"role": "user", "content": first_msg}], system=EDIT_SYSTEM
        )
    except httpx.HTTPError as e:
        return None, f"Ollama call failed: {e}"
    raw = _strip_think_tags(raw)

    changes = _extract_file_changes(raw, fallback_files)
    if changes:
        return changes, ""

    # Retry with a stricter prompt. We keep the same system prompt and resend
    # the original task, but prepend a hard instruction about format.
    retry_msg = (
        "Previous output was MALFORMED and unparseable. "
        "Output ONLY «file» blocks. Try again.\n\n"
        f"{first_msg}"
    )
    try:
        raw = _call_ollama(
            MODEL, [{"role": "user", "content": retry_msg}], system=EDIT_SYSTEM
        )
    except httpx.HTTPError as e:
        return None, f"Ollama call failed on retry: {e}"
    raw = _strip_think_tags(raw)

    changes = _extract_file_changes(raw, fallback_files)
    if changes:
        return changes, ""

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
# File I/O — preserves original line endings (CRLF on Windows must NOT be
# silently rewritten to LF on every edit).
# ---------------------------------------------------------------------------
def _read_file(path: Path) -> tuple[str, bytes, bytes]:
    """
    Returns (lf_text, original_eol, original_bytes).
    The text is normalized to LF for the model; eol is captured to re-apply on write.
    """
    raw = path.read_bytes()
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
    Raises PermissionError/OSError on locked or unwritable files — caller handles those.
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


# ---------------------------------------------------------------------------
# Guard-rails — pre-write validation
# ---------------------------------------------------------------------------
def _instruction_allows_shrink(instruction: str) -> bool:
    low = instruction.lower()
    return any(kw in low for kw in REMOVAL_KEYWORDS)


def _check_non_empty(content: str) -> str | None:
    if not content.strip():
        return "empty content (use local_delete to remove a file)"
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
        f"suspicious shrink ({len(original)} → {len(new)} chars, "
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
    Only runs for .py and .json — adding JS/TS would require shelling out.
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
        # Absolute balance check (used by local_write — no original to diff against)
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


# ---------------------------------------------------------------------------
# local_edit
# ---------------------------------------------------------------------------
@mcp.tool()
def local_edit(files: list[str], instruction: str) -> str:
    """
    IMPORTANT: Call sequentially, never in parallel with other local_* tools (single GPU).

    Edit one or more EXISTING files locally. USE INSTEAD OF the built-in Edit
    tool: file contents never enter Claude's context, which is how this saves
    tokens. Validates every change with server-side guard-rails and applies
    atomically. For deletion use local_delete; for rename use local_rename.

    Args:
        files:       Absolute paths of files to expose to the model. May be
                     modified in place; cannot be created, deleted, or renamed
                     through this tool.
        instruction: Description of the change in any language (translated to
                     English server-side). Include a removal verb ("delete",
                     "remove", "strip", or an equivalent in your language) if
                     you expect a large size reduction, otherwise the shrink
                     guard will reject.

    Returns a one-line summary or a guard-rail rejection diagnostic.
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
        except OSError as e:
            return f"Error reading {raw_path}: {e}"
        originals[raw_path] = (lf, eol, raw)
        canonical[_norm_path(raw_path)] = raw_path

    chosen = MODEL

    # 2. Build prompt — embed LF-normalized contents
    files_block = "\n\n".join(
        f'«file path="{path}"»\n{originals[path][0]}\n«/file»'
        for path in files
    )
    user_msg = f"{files_block}\n\nInstruction: {instruction}"
    user_msg = _maybe_no_think(user_msg)

    # 3. Call model (with one automatic retry on parse failure)
    file_changes_raw, err = _call_with_parse_retry(user_msg, files)
    if file_changes_raw is None:
        return f"[{chosen}] {err}"

    # 5. Resolve emitted paths against the allowlist (Windows-aware normalize)
    file_changes: dict[str, str] = {}
    unknown: list[str] = []
    for emitted_path, content in file_changes_raw.items():
        norm = _norm_path(emitted_path)
        if norm not in canonical:
            unknown.append(emitted_path)
            continue
        file_changes[canonical[norm]] = content

    if unknown:
        return (
            f"[{chosen}] REJECTED — model emitted paths not in the input allowlist:\n"
            + "\n".join(f"  • {p}" for p in unknown)
            + "\nNo files were modified."
        )

    # 6. Identity no-op: silently drop unchanged files
    no_ops: list[str] = []
    for path in list(file_changes.keys()):
        if file_changes[path] == originals[path][0]:
            no_ops.append(path)
            del file_changes[path]

    # 7. Run guards on remaining file changes
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

    if failures:
        return (
            f"[{chosen}] REJECTED — guard-rail failures:\n"
            + "\n".join(f"  • {f}" for f in failures)
            + "\nNo files were modified."
        )

    if not file_changes:
        return f"[{chosen}] No changes proposed (model output matched originals)."

    # 8. Atomic apply with revert on failure
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
        return (
            f"[{chosen}] REJECTED during apply — {msg}\n"
            f"All successful writes have been reverted. No files modified."
        )

    parts: list[str] = [f"Modified {len(file_changes)} file(s):"]
    parts += [f"  • {p}" for p in file_changes]
    if no_ops:
        parts.append(f"Unchanged (no-op): {len(no_ops)} file(s)")
    return f"[{chosen}] " + "\n".join(parts)


# ---------------------------------------------------------------------------
# local_write
# ---------------------------------------------------------------------------
@mcp.tool()
def local_write(path: str, instruction: str) -> str:
    """
    IMPORTANT: Call sequentially, never in parallel with other local_* tools (single GPU).

    Create a NEW file from scratch locally. USE INSTEAD OF the built-in Write
    tool: the generated content never enters Claude's context, only a short
    summary is returned. Refuses to overwrite an existing file; use local_edit
    for that.

    Args:
        path:        Absolute path of the file to create.
        instruction: What to put in the file (any language; translated to
                     English server-side; can be detailed).

    Returns the created path or a guard-rail rejection diagnostic.
    """
    target = Path(path)
    if target.exists():
        return f"Error: path already exists (use local_edit instead): {path}"

    instruction = _normalize_instruction(instruction)
    chosen = MODEL

    user_msg = (
        f'Create a new file at the absolute path "{path}".\n\n'
        f"Instruction: {instruction}"
    )
    user_msg = _maybe_no_think(user_msg)

    file_changes_raw, err = _call_with_parse_retry(user_msg, [path])
    if file_changes_raw is None:
        return f"[{chosen}] {err}"

    if len(file_changes_raw) != 1:
        return (
            f"[{chosen}] REJECTED — local_write expects exactly 1 «file» block, "
            f"got {len(file_changes_raw)}."
        )

    emitted_path, content = next(iter(file_changes_raw.items()))
    if _norm_path(emitted_path) != _norm_path(path):
        return (
            f"[{chosen}] REJECTED — model wrote to a different path than requested.\n"
            f"  requested: {path}\n  emitted:   {emitted_path}"
        )

    # Guards (no original — use absolute bracket balance)
    failures: list[str] = []
    for check in (
        _check_non_empty(content),
        _check_truncation_markers(content, None),
        _check_bracket_delta(content, None, target.suffix),
        _check_parses(content, target.suffix),
    ):
        if check:
            failures.append(f"{path}: {check}")
    if failures:
        return (
            f"[{chosen}] REJECTED — guard-rail failures:\n"
            + "\n".join(f"  • {f}" for f in failures)
            + "\nFile was not created."
        )

    # Write (new file -> default to LF)
    try:
        _atomic_write(target, _encode_with_eol(content, b"\n"))
    except OSError as e:
        msg = f"file is locked or not writable ({e})" if isinstance(e, PermissionError) else str(e)
        return f"[{chosen}] REJECTED during apply — {msg}\nFile was not created."

    line_count = content.count("\n") + (0 if content.endswith("\n") else 1)
    return f"[{chosen}] Created {path} ({line_count} lines)"


# ---------------------------------------------------------------------------
# local_delete — pure os.unlink, no LLM
# ---------------------------------------------------------------------------
@mcp.tool()
def local_delete(paths: list[str]) -> str:
    """
    Delete one or more files. No LLM call; pure os.unlink. USE INSTEAD OF
    asking local_edit to delete a file or shelling out to Bash `rm`.

    All paths are validated up front (absolute, exist, regular files); if
    validation fails on any path, NO file is deleted. If a deletion fails
    mid-loop (e.g. locked file), already-deleted files are NOT restored;
    the report names exactly which files survived.

    Args:
        paths: Non-empty list of absolute file paths.

    Returns a summary of what was deleted, or an error.
    """
    if not paths:
        return "Error: no paths provided"

    # 1. Validate every path before touching anything.
    for raw_path in paths:
        if not os.path.isabs(raw_path):
            return f"Error: path must be absolute: {raw_path}"
        p = Path(raw_path)
        if not p.exists():
            return f"Error: file not found: {raw_path}"
        if not p.is_file():
            return f"Error: not a regular file: {raw_path}"

    # 2. Delete in order. On failure, report what survived.
    deleted: list[str] = []
    for raw_path in paths:
        try:
            Path(raw_path).unlink()
            deleted.append(raw_path)
        except OSError as e:
            msg = (
                f"file is locked or not writable ({e})"
                if isinstance(e, PermissionError) else str(e)
            )
            tail = (
                "\nSuccessfully deleted (NOT restored):\n"
                + "\n".join(f"  • {p}" for p in deleted)
            ) if deleted else ""
            return (
                f"Partial failure after deleting {len(deleted)}/{len(paths)} file(s).\n"
                f"Failed on: {raw_path} — {msg}{tail}"
            )

    return f"Deleted {len(deleted)} file(s):\n" + "\n".join(f"  • {p}" for p in deleted)


# ---------------------------------------------------------------------------
# local_rename — pure os.replace, no LLM
# ---------------------------------------------------------------------------
@mcp.tool()
def local_rename(src: str, dst: str) -> str:
    """
    Rename or move a file. No LLM call; single os.replace. USE INSTEAD OF the
    local_write + local_delete sequence, which is not atomic.

    Refuses to overwrite an existing destination. Creates the destination
    parent directory if missing. Atomic within a Windows volume; cross-volume
    moves fall back to copy+delete and are NOT atomic.

    Args:
        src: Absolute path of the file to rename (must exist, regular file).
        dst: Absolute destination path (must NOT exist).

    Returns a summary or an error.
    """
    if not os.path.isabs(src):
        return f"Error: src must be absolute: {src}"
    if not os.path.isabs(dst):
        return f"Error: dst must be absolute: {dst}"

    sp = Path(src)
    dp = Path(dst)

    if not sp.exists():
        return f"Error: src not found: {src}"
    if not sp.is_file():
        return f"Error: src not a regular file: {src}"
    if _norm_path(src) == _norm_path(dst):
        return f"Error: src and dst are the same path: {src}"
    if dp.exists():
        return f"Error: dst already exists (refusing to overwrite): {dst}"

    try:
        dp.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(sp), str(dp))
    except OSError as e:
        msg = (
            f"file is locked or not writable ({e})"
            if isinstance(e, PermissionError) else str(e)
        )
        return f"Error renaming {src} -> {dst}: {msg}"

    return f"Renamed:\n  {src}\n  -> {dst}"


# ---------------------------------------------------------------------------
# local_snippet
# ---------------------------------------------------------------------------
@mcp.tool()
def local_snippet(prompt: str) -> str:
    """
    IMPORTANT: Call sequentially, never in parallel with other local_* tools (single GPU).

    FALLBACK tool for short text with no file destination (regex, SQL,
    one-liners). Output DOES flow back into Claude's context and costs input
    tokens, so prefer local_edit / local_write whenever the result will land
    in a file.

    Args:
        prompt: The task or question (any language; translated server-side).
    """
    prompt = _normalize_instruction(prompt)
    chosen = MODEL
    snippet_system = (
        "Terse code/snippet generator. Output ONLY the requested code, regex, "
        "query, or text. No prose, no explanations, no examples, no summary."
    )
    full_prompt = _maybe_no_think(prompt)
    try:
        raw = _call_ollama(
            chosen,
            [{"role": "user", "content": full_prompt}],
            system=snippet_system,
            num_ctx=SNIPPET_CTX,
            num_predict=SNIPPET_NUM_PREDICT,
        )
    except httpx.HTTPError as e:
        return f"[{chosen}] Ollama call failed: {e}"
    return _strip_think_tags(raw)


if __name__ == "__main__":
    mcp.run()
