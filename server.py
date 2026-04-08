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
  local_edit(files, instruction, model)    -> modifies existing files in place
  local_write(path,  instruction, model)   -> creates a new file from scratch
  local_snippet(prompt, model)             -> returns text (round-trip; fallback)

Single-model architecture (iteration 2):
  All three tools call qwen3-coder:30b (MoE, 3B active params per token).
  16 GB VRAM cannot host two models simultaneously, so trying to alternate
  between a small and large tier just thrashes. The coder model is pinned in
  VRAM with keep_alive=-1 so it never gets evicted between calls. Some expert
  layers spill to CPU (~27%) — acceptable because MoE only activates ~3B of
  the 30B total params per token. Measured ~48 tok/s on RTX 5070 Ti.

  The `model` parameter on each tool is preserved for backward compatibility
  but is currently a no-op.

Target OS: Windows 10. All file I/O is Windows-correct: CRLF preservation,
case-insensitive path normalization, locked-file detection, atomic rename via
same-directory temp file + os.replace.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-mcp")

# ---------------------------------------------------------------------------
# Model — single-model architecture, see iteration 2 in plans/.
# ---------------------------------------------------------------------------
MODEL = "qwen3-coder:30b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Edit/write context window. 32k is plenty for multi-file edits and keeps the
# KV-cache VRAM footprint manageable on 16 GB GPUs (the 30b weights already
# use most of it).
EDIT_CTX = 32768

# Snippet tool uses a much smaller context (no files in the prompt) and a hard
# cap on generated tokens to prevent qwen3 from rambling in markdown.
SNIPPET_CTX = 4096
SNIPPET_NUM_PREDICT = 1024

# httpx timeout in seconds. qwen3-coder:30b on a multi-file edit with partial
# CPU offload can take many minutes under load.
TIMEOUT = 1200

# ---------------------------------------------------------------------------
# Guard-rail constants (tune freely)
# ---------------------------------------------------------------------------
# Reject a file edit if new size < SHRINK_RATIO * old size AND the instruction
# does not contain a removal keyword.
SHRINK_RATIO = 0.5

REMOVAL_KEYWORDS = (
    "delete", "remove", "strip", "drop", "clear", "empty", "shrink",
    "cancella", "rimuovi", "elimina", "svuota", "togli",
)

# Stricter than REMOVAL_KEYWORDS: these phrases clearly signal WHOLE-FILE
# deletion, not in-file code removal. Used by the delete-intent guard to
# veto <delete/> blocks when the instruction doesn't clearly ask for one.
WHOLE_FILE_DELETE_PHRASES = (
    "delete the file", "delete file", "delete this file", "delete these files",
    "remove the file", "remove file", "remove this file", "remove these files",
    "rimuovi il file", "rimuovi file", "rimuovi questo file",
    "elimina il file", "elimina file", "elimina questo file",
    "cancella il file", "cancella file", "cancella questo file",
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

# ---------------------------------------------------------------------------
# System prompt for edit/write tools
# ---------------------------------------------------------------------------
EDIT_SYSTEM = """\
You are a code editing assistant. Output ONLY the complete new content of each
modified file using <file> blocks, OR <delete/> blocks for files that should
be removed entirely.

Format for modifying or creating a file:

<file path="/absolute/path/to/file">
[complete file content — never truncate, never use placeholders]
</file>

Format for deleting a file (the path="..." attribute is REQUIRED and the tag
MUST be self-closing — never emit a bare <delete/> with no path):

<delete path="/absolute/path/to/file"/>

Example A — modify Foo.java to add a field, AND remove the obsolete Bar.java:

INPUT:
<file path="/project/src/Foo.java">
public record Foo(String name) {}
</file>
<file path="/project/src/Bar.java">
public class Bar { /* obsolete */ }
</file>
Instruction: add an int age field to Foo, and delete Bar entirely

OUTPUT:
<file path="/project/src/Foo.java">
public record Foo(String name, int age) {}
</file>
<delete path="/project/src/Bar.java"/>

Example B — delete a single file (delete-only, no <file> blocks at all):

INPUT:
<file path="/project/scripts/old_script.py">
print("obsolete")
</file>
Instruction: delete this file

OUTPUT:
<delete path="/project/scripts/old_script.py"/>

Example C — remove a method from a file (NOT a file deletion):

INPUT:
<file path="/project/src/Util.java">
public class Util {
    public static int a() { return 1; }
    public static int b() { return 2; }
}
</file>
Instruction: remove the unused method b

OUTPUT:
<file path="/project/src/Util.java">
public class Util {
    public static int a() { return 1; }
}
</file>

CRITICAL — <delete/> is for WHOLE-FILE deletion ONLY:
- Emit <delete path="..."/> ONLY when the instruction explicitly asks to
  delete the entire file (e.g. "delete the file X", "delete file Foo.java",
  "rimuovi il file Bar.py", "elimina old_script.py", "cancella il file Y").
- For every other removal request — remove a method, remove a field,
  remove a class, remove unused imports, remove a block, strip comments —
  you MUST output a <file> block containing the edited content with the
  target portion excised. Do NOT emit <delete/>.
- When the instruction is ambiguous, prefer <file>. A <delete/> that was
  not explicitly asked for is the worst possible outcome and will be
  rejected by the server.

Rules:
- Output ONLY <file> and <delete/> blocks. No explanations, no markdown fences,
  no preamble, no commentary.
- <file> blocks must contain the COMPLETE file content. Never write
  "... rest of file unchanged" or any other placeholder — this will be rejected.
- If a file should be removed entirely, emit <delete path="..."/>. NEVER emit
  a bare <delete/> with no path attribute, NEVER emit a <file> block to "name"
  the file you want to delete, and NEVER emit an empty or near-empty <file>
  block as a way to "remove" code — all three will be rejected.
- Use the exact same absolute path that was given in the input.
- If a file does not need any change, omit it entirely from the output.
- Do NOT wrap code in ```java or any markdown fences.
"""


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
        "stream": False,
        # Pin the model in VRAM forever — single-model architecture, no eviction.
        "keep_alive": -1,
        "options": options,
    }
    if system:
        payload["system"] = system
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _strip_think_tags(text: str) -> str:
    """Defensive: strip <think>...</think> if a qwen3 model emits any despite /no_think."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
_FILE_BLOCK_RE = re.compile(r'<file path="([^"]+)">\n?(.*?)\n?</file>', re.DOTALL)
_DELETE_BLOCK_RE = re.compile(r'<delete\s+path="([^"]+)"\s*/>')


def _parse_file_blocks(text: str) -> dict[str, str]:
    return {path: content for path, content in _FILE_BLOCK_RE.findall(text)}


def _parse_delete_blocks(text: str) -> list[str]:
    """Parse <delete/> blocks AFTER stripping <file> blocks, so the <delete>
    regex cannot accidentally match inside file content."""
    text_no_files = _FILE_BLOCK_RE.sub("", text)
    return _DELETE_BLOCK_RE.findall(text_no_files)


def _fallback_markdown_extract(text: str, files: list[str]) -> dict[str, str]:
    """If the model returned a fenced code block instead of a <file> block,
    map it to the only input file. Single-file only."""
    if len(files) != 1:
        return {}
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if not match:
        return {}
    return {files[0]: match.group(1)}


# Loose match: any <delete...> tag, regardless of attributes or self-closing form.
_LOOSE_DELETE_RE = re.compile(r"<delete\b[^>]*>", re.IGNORECASE)


def _infer_single_file_delete(
    raw: str, files: list[str], instruction: str
) -> list[str]:
    """
    Fallback for the case where the model emits a malformed delete (e.g.
    bare <delete/> with no path attribute, or <file>PATH</file><delete/>) for a
    single-file input. Safe because it requires ALL of:
      - exactly one file in the input,
      - an EXPLICIT whole-file deletion phrase in the instruction,
      - some form of <delete> tag in the raw output.
    Without all three, returns []. The strict phrase check ensures the user
    clearly asked for file-level deletion, not in-file code removal.
    """
    if len(files) != 1:
        return []
    if not _instruction_requests_whole_file_delete(instruction):
        return []
    if not _LOOSE_DELETE_RE.search(raw):
        return []
    return [files[0]]


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


def _instruction_requests_whole_file_delete(instruction: str) -> bool:
    """
    Stricter than _instruction_allows_shrink: returns True only if the
    instruction explicitly asks to delete a WHOLE FILE (not just remove some
    code from inside it). Used by the delete-intent guard to veto <delete/>
    blocks unless the user clearly asked for file-level deletion.
    """
    low = instruction.lower()
    return any(phrase in low for phrase in WHOLE_FILE_DELETE_PHRASES)


def _check_non_empty(content: str) -> str | None:
    if not content.strip():
        return "empty content (use <delete/> instead?)"
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
def local_edit(
    files: list[str],
    instruction: str,
    model: str = "auto",
) -> str:
    """
    USE THIS INSTEAD OF the built-in Edit tool for any file modification.
    Prefer this over Edit even for small changes — the whole point of this tool
    is that file contents never enter Claude's context, which is the only way
    to actually save tokens. Fall back to Edit only if local_edit refuses or
    the MCP server is unavailable.

    Edit one or more EXISTING files locally without round-tripping their contents
    through Claude. Reads the files, sends them to qwen3-coder:30b with the
    instruction, parses <file> and <delete/> blocks from the response, validates
    every change with server-side guard-rails, and atomically applies the result.

    This is the most token-efficient tool — Claude only sees a short summary or
    a structured rejection diagnostic, never the file contents.

    Args:
        files:       Absolute paths of files to expose to the model. The model
                     may modify any of them, or emit <delete path="..."/> to
                     remove one. Deletes are restricted to this exact set.
        instruction: Plain-English description of the change. Include words like
                     "delete", "remove", "rimuovi" if you expect a large reduction
                     in file size, otherwise the shrink guard will reject.
        model:       Kept for backward compatibility but currently ignored —
                     iteration 2 of the server uses a single model
                     (qwen3-coder:30b) for all tools, pinned in VRAM.

    Returns a one-paragraph summary of what was modified/deleted, OR a guard-rail
    rejection diagnostic explaining what to fix on the next attempt.
    """
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

    # 2. Single-model architecture — `model` arg is preserved for backward compat
    #    but no longer routes. See module docstring.
    chosen = MODEL

    # 3. Build prompt — embed LF-normalized contents
    files_block = "\n\n".join(
        f'<file path="{path}">\n{originals[path][0]}\n</file>'
        for path in files
    )
    user_msg = (
        f"{files_block}\n\n"
        f"Instruction: {instruction}\n\n"
        f"IMPORTANT: Output ONLY <file> and <delete/> blocks. "
        f"No markdown fences, no commentary. /no_think"
    )

    # 4. Call model
    try:
        raw = _call_ollama(
            chosen, [{"role": "user", "content": user_msg}], system=EDIT_SYSTEM
        )
    except httpx.HTTPError as e:
        return f"[{chosen}] Ollama call failed: {e}"
    raw = _strip_think_tags(raw)

    # 5. Parse output
    file_changes_raw = _parse_file_blocks(raw)
    delete_paths_raw = _parse_delete_blocks(raw)

    if not file_changes_raw and not delete_paths_raw:
        # Inference fallback for malformed single-file delete (e.g. bare <delete/>
        # without a path attribute). Safe because it requires the instruction to
        # contain an explicit removal keyword.
        inferred = _infer_single_file_delete(raw, files, instruction)
        if inferred:
            delete_paths_raw = inferred
        else:
            # Markdown fallback for single-file edits
            file_changes_raw = _fallback_markdown_extract(raw, files)
            if not file_changes_raw:
                return (
                    f"[{chosen}] No <file> or <delete/> blocks found in model output.\n\n"
                    f"Raw output:\n{raw}"
                )

    # 6. Resolve emitted paths against the allowlist (Windows-aware normalize)
    file_changes: dict[str, str] = {}
    unknown: list[str] = []
    for emitted_path, content in file_changes_raw.items():
        norm = _norm_path(emitted_path)
        if norm not in canonical:
            unknown.append(emitted_path)
            continue
        file_changes[canonical[norm]] = content

    deletes: list[str] = []
    for emitted_path in delete_paths_raw:
        norm = _norm_path(emitted_path)
        if norm not in canonical:
            unknown.append(emitted_path)
            continue
        deletes.append(canonical[norm])

    if unknown:
        return (
            f"[{chosen}] REJECTED — model emitted paths not in the input allowlist:\n"
            + "\n".join(f"  • {p}" for p in unknown)
            + "\nNo files were modified."
        )

    # 7. No-conflict rule: same path cannot be in both <file> and <delete/>
    conflict = set(file_changes.keys()) & set(deletes)
    if conflict:
        return (
            f"[{chosen}] REJECTED — same path appears in both <file> and <delete/>:\n"
            + "\n".join(f"  • {p}" for p in conflict)
            + "\nNo files were modified."
        )

    # 8. Identity no-op: silently drop unchanged files
    no_ops: list[str] = []
    for path in list(file_changes.keys()):
        if file_changes[path] == originals[path][0]:
            no_ops.append(path)
            del file_changes[path]

    # 9. Run guards on remaining file changes
    failures: list[str] = []
    for path, new_content in file_changes.items():
        original_lf = originals[path][0]
        ext = Path(path).suffix
        for check in (
            _check_non_empty(new_content),
            _check_truncation_markers(new_content, original_lf),
            _check_shrink(new_content, original_lf, instruction),
            _check_bracket_delta(new_content, original_lf, ext),
        ):
            if check:
                failures.append(f"{path}: {check}")

    # 9b. Delete intent guard: <delete/> must be explicitly requested.
    #     Prevents the model from hallucinating a file deletion when the user
    #     only wanted to remove a method/field/block inside the file.
    if deletes and not _instruction_requests_whole_file_delete(instruction):
        return (
            f"[{chosen}] REJECTED — model emitted <delete/> for:\n"
            + "\n".join(f"  • {p}" for p in deletes)
            + "\nBut the instruction does not contain an explicit whole-file "
              "deletion phrase (e.g. 'delete the file', 'rimuovi il file'). "
              "If you wanted code removed in-place, rephrase without the word "
              "'file' (e.g. 'remove method foo'). If you really want the whole "
              "file gone, include 'delete the file <path>'. "
              "No files were modified."
        )

    if failures:
        return (
            f"[{chosen}] REJECTED — guard-rail failures:\n"
            + "\n".join(f"  • {f}" for f in failures)
            + "\nNo files were modified."
        )

    if not file_changes and not deletes:
        return f"[{chosen}] No changes proposed (model output matched originals)."

    # 10. Atomic apply with revert on failure
    written: list[str] = []
    deleted: list[str] = []
    try:
        for path, new_content in file_changes.items():
            eol = originals[path][1]
            _atomic_write(Path(path), _encode_with_eol(new_content, eol))
            written.append(path)
        for path in deletes:
            try:
                Path(path).unlink()
                deleted.append(path)
            except OSError as e:
                raise RuntimeError(f"failed to delete {path}: {e}") from e
    except (OSError, RuntimeError) as e:
        # Revert any successful writes by restoring original bytes
        for path in written:
            try:
                Path(path).write_bytes(originals[path][2])
            except OSError:
                pass
        # (Deletes run last, so any deleted files would also need restoring —
        # but on the first failed delete we abort before any subsequent delete.
        # The first failed delete leaves its file untouched.)
        msg = str(e)
        if isinstance(e, PermissionError):
            msg = f"file is locked or not writable ({e})"
        return (
            f"[{chosen}] REJECTED during apply — {msg}\n"
            f"All successful writes have been reverted. No files modified."
        )

    parts: list[str] = []
    if file_changes:
        parts.append(f"Modified {len(file_changes)} file(s):")
        parts += [f"  • {p}" for p in file_changes]
    if deleted:
        parts.append(f"Deleted {len(deleted)} file(s):")
        parts += [f"  • {p}" for p in deleted]
    if no_ops:
        parts.append(f"Unchanged (no-op): {len(no_ops)} file(s)")
    return f"[{chosen}] " + "\n".join(parts)


# ---------------------------------------------------------------------------
# local_write
# ---------------------------------------------------------------------------
@mcp.tool()
def local_write(
    path: str,
    instruction: str,
    model: str = "auto",
) -> str:
    """
    USE THIS INSTEAD OF the built-in Write tool when creating a new file.
    Prefer this over Write for any new file whose content can be described in
    plain English — that description is all Claude sends, and the actual file
    content is generated and written entirely on the local side. Fall back to
    Write only if local_write refuses or the MCP server is unavailable.

    Create a NEW file from scratch using a local model. The generated content
    never enters Claude's context — only a short summary is returned.

    Use this instead of asking Claude to write the content and then applying it
    with Edit/Write — that round-trip is exactly what burns tokens.

    Args:
        path:        Absolute path of the file to create. Refuses to overwrite
                     an existing file (use local_edit for that).
        instruction: What to put in the file (plain English; can be detailed).
        model:       Kept for backward compatibility but currently ignored —
                     iteration 2 of the server uses a single model
                     (qwen3-coder:30b) for all tools, pinned in VRAM.

    Returns the created path or a guard-rail rejection diagnostic.
    """
    target = Path(path)
    if target.exists():
        return f"Error: path already exists (use local_edit instead): {path}"

    chosen = MODEL

    user_msg = (
        f'Create a new file at the absolute path "{path}".\n\n'
        f"Instruction: {instruction}\n\n"
        f'IMPORTANT: Output ONLY a single <file path="{path}"> block with the '
        f"complete file content. No markdown fences, no commentary, no <delete/>. "
        f"/no_think"
    )

    try:
        raw = _call_ollama(
            chosen, [{"role": "user", "content": user_msg}], system=EDIT_SYSTEM
        )
    except httpx.HTTPError as e:
        return f"[{chosen}] Ollama call failed: {e}"
    raw = _strip_think_tags(raw)

    file_changes_raw = _parse_file_blocks(raw)
    delete_blocks = _parse_delete_blocks(raw)

    if delete_blocks:
        return f"[{chosen}] REJECTED — local_write does not accept <delete/> blocks."

    if not file_changes_raw:
        file_changes_raw = _fallback_markdown_extract(raw, [path])
        if not file_changes_raw:
            return (
                f"[{chosen}] No <file> block found in model output.\n\n"
                f"Raw output:\n{raw}"
            )

    if len(file_changes_raw) != 1:
        return (
            f"[{chosen}] REJECTED — local_write expects exactly 1 <file> block, "
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
# local_snippet
# ---------------------------------------------------------------------------
@mcp.tool()
def local_snippet(prompt: str, model: str = "small") -> str:
    """
    FALLBACK TOOL — prefer local_edit / local_write whenever the result will
    land in a file. Only use local_snippet when there is genuinely no file
    destination yet (regex, SQL fragment, one-liner you need to inspect before
    deciding what to do with it). Unlike local_edit/local_write, the output of
    this tool DOES flow back through Claude's context and consumes input tokens.

    Generate a short snippet locally and return it as text. The result flows
    BACK through Claude's context, so this costs Claude input tokens — prefer
    local_edit / local_write whenever the result will be written to a file.

    Best for: regex, SQL queries, one-liner transformations, single short
    functions, simple snippets that have no file destination yet.

    Args:
        prompt: The task or question.
        model:  Kept for backward compatibility but currently ignored —
                iteration 2 of the server uses a single model
                (qwen3-coder:30b) for all tools, pinned in VRAM.
    """
    chosen = MODEL
    snippet_system = (
        "You are a terse code/snippet generator. Output ONLY the requested "
        "code, regex, query, or text. No prose, no explanations, no examples, "
        "no summary, no markdown headings. If the user explicitly asks for an "
        "explanation, keep it to one short sentence."
    )
    full_prompt = f"{prompt}\n\n/no_think"
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
