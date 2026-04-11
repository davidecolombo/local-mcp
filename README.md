# local-mcp

MCP server that delegates implementation work to local Ollama models so Claude (Opus/Sonnet) only orchestrates. The goal is to **save cloud tokens** by keeping file contents on the local machine. They read, edit, and write happen entirely server-side, and Claude only sees a short summary.

## Token-saving philosophy

The big savings come from **never round-tripping file contents through Claude's context**:

- `local_edit` reads and writes existing files on the MCP side. Claude sends a short diff-shaped instruction and gets back a one-line summary. **This is an unconditional token win; always prefer it over the built-in `Edit`.**
- `local_write` creates new files the same way, but savings are conditional: the instruction must be much shorter than the file it produces. Good fits: stubs, boilerplate, scaffolds, config templates (short spec, large output). Bad fit: dictating exact content line-by-line — in that case the instruction approaches the file size and the local-model round-trip adds overhead for no gain. Use the built-in `Write` for dictated content.
- `local_delete` and `local_rename` are pure filesystem operations — no model call at all. They exist so the caller never has to ask the model to decide *whether* to delete or move a file; the caller already knows.
- `local_snippet` returns generated text that flows back through Claude. It's useful as a fallback for regex / SQL / one-liners that have no file destination yet, but every byte of its output costs Claude input tokens. Use sparingly.

### Language normalization

All instruction strings (`local_edit`, `local_write`, `local_snippet`) are checked for non-English content on the server side and translated to English in a tiny pre-pass before they reach the main model and the guard-rails. This means callers can write instructions in Italian, French, Spanish, German, etc. without losing any safety check — the guard-rails only need to reason about English keywords (`delete`, `remove`, `strip`, …) and the model also produces better edits when prompted in English. English instructions skip the pre-pass entirely (zero overhead).

## Models: single-model architecture

All three Ollama-backed tools (`local_edit`, `local_write`, `local_snippet`) call **the same model**. There is no routing, no small/large tier, no model switching. There is no `model` parameter on any tool; the model is server configuration, not a caller concern.

The default model is `qwen3-coder:30b`, but you can swap it via `model-config.json` (see [Model configuration](#model-configuration) below).

### Why a single model

Iteration 1 of this server had a two-tier design: `qwen3:14b` for short single-file edits, `qwen3-coder:30b` for multi-file work. In practice that turned out to be the wrong shape:

1. **VRAM math doesn't allow coexistence.** On a 16 GB GPU, `qwen3:14b` (~10 GB) and `qwen3-coder:30b` (~18 GB) cannot be loaded together. Any workflow that mixes the two tools forces Ollama to evict one to load the other. Windows page cache (in 64 GB DDR5) makes the *reload* faster than a cold-from-NVMe load, but it is still seconds of latency on every alternation.
2. **The actual workload is multi-file feature implementation.** Snippet generation is rare in this setup; most calls are "implement this feature reading the plan." That's `local_edit` / `local_write` territory on the larger coder model. The 14b dense model added VRAM thrashing without buying useful work.

The fix is to pick one model and pin it. `qwen3-coder:30b` wins because it's the right tool for multi-file agentic edits and its MoE design (only ~3B params active per token) absorbs partial CPU offload gracefully. Measured ~48 tok/s on RTX 5070 Ti even with ~27% of layers spilled to CPU.

### VRAM pinning (no eviction, ever)

Every Ollama call from this server passes `keep_alive: -1`. The model is loaded into VRAM on the first call of the session and **never unloaded**. No idle eviction, no thrashing, no cold-load latency on subsequent calls. After the first warmup, all calls go straight to compute.

### Qwen3-specific behavior

When the configured model name contains `qwen3`, the server automatically:

- Appends `/no_think` to user prompts so the model skips the reasoning chain (faster, less VRAM, deterministic edits don't benefit from visible reasoning).
- Strips any `<think>...</think>` tags that leak through in the response.

For non-Qwen3 models (e.g. Devstral), both behaviors are disabled automatically.

### Target hardware

Designed and tested on: i9-14900K, 64 GB DDR5, RTX 5070 Ti 16 GB, **Windows 10**. The default 30b coder fits 73% on GPU / 27% on CPU at Q4_K_M with `num_ctx=32768`; multi-file edits can take a few minutes (timeout is 20 min). Smaller GPUs will see a larger CPU split and proportionally lower tok/s.

## Prerequisites

- [Ollama](https://ollama.com) running locally on `http://localhost:11434`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) in PATH
- Your chosen model pulled:

```bash
ollama pull qwen3-coder:30b    # default
# or
ollama pull devstral-small-2:24b  # alternative
```

Dependencies for the server itself are managed automatically by `uv` via the inline script metadata in `server.py`. No manual `pip install`.

## Model configuration

The server reads an optional `model-config.json` file from the project root at startup. If the file is missing or any field is omitted, built-in defaults (matching `qwen3-coder:30b`) are used. The active config is gitignored; it's a local concern, not committed.

### Switching models

Ready-to-use templates live in `configs/`:

```bash
# Switch to Devstral-Small
cp configs/devstral-small-2-24b.json model-config.json

# Switch back to Qwen3-Coder (or just delete model-config.json for defaults)
cp configs/qwen3-coder-30b.json model-config.json
```

**Restart the MCP server after changing the config** (FastMCP loads it once at startup).

### Available templates

| Template | Model | Notes |
|----------|-------|-------|
| `configs/qwen3-coder-30b.json` | `qwen3-coder:30b` | Default. MoE 30B, ~3B active. Best tested option. |
| `configs/devstral-small-2-24b.json` | `devstral-small-2:24b` | Mistral's code-agent model. Trained for structured output. Lower timeout (10 min). |
| `configs/qwen3-30b.json` | `qwen3:30b` | Base Qwen3 (non-Coder). Same MoE architecture, broader training. |

### Config fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"qwen3-coder:30b"` | Ollama model tag |
| `ollama_url` | string | `"http://localhost:11434/api/chat"` | Ollama API endpoint |
| `edit_ctx` | int | `32768` | Context window for edit/write calls |
| `snippet_ctx` | int | `4096` | Context window for snippet calls |
| `snippet_num_predict` | int | `1024` | Max output tokens for snippets |
| `translate_ctx` | int | `2048` | Context window for translation pre-pass |
| `translate_num_predict` | int | `512` | Max output tokens for translation |
| `timeout` | int | `1200` | HTTP timeout in seconds |

### Custom config example

Only include the fields you want to override:

```json
{
  "model": "devstral-small-2:24b",
  "timeout": 600
}
```

All other fields fall back to defaults.

### Remote providers (planned)

The config schema is designed to accommodate a future `"provider"` field for remote inference APIs. Candidates under consideration:

- **OpenRouter**: aggregator with many models including Qwen3-Coder-Next, DeepSeek-V3, Gemini 2.5 Flash. Some models available on free tier (`:free` suffix, rate-limited). Pricing for paid models: ~$0.005 per edit call at 32K context.
- **Google AI Studio**: generous free tier with Gemini 2.5 Flash and Pro. OpenAI-compatible API. Likely the strongest free option for structured output.
- **Groq / Cerebras**: free tiers with very fast inference (~300 tok/s). Good for snippet-style calls. Limited model selection.
- **Together AI / Fireworks AI**: cheap pay-as-you-go with DeepSeek-V3 and Qwen3 variants. OpenAI-compatible endpoints.

These would trade local GPU compute for network latency, with a potential hybrid approach (remote primary, local fallback).

## Installation

The server is registered globally in `~/.claude.json` via:

```bash
claude mcp add --scope user local-mcp uv run "C:/Users/user/.claude/local-mcp/server.py"
```

## Tools exposed to Claude Code

### `local_edit(files, instruction)`

Edit one or more **existing** files in place. Reads the files, sends them to the configured model, parses `«file»` blocks from the response, validates with guard-rails, and atomically applies the result. Claude only sees a one-line summary or a structured rejection.

```
files:       list of absolute file paths
instruction: what to change (any language; include "delete"/"remove"/"strip"
             — or the equivalent in your language — if you expect a large
             reduction in size, otherwise the shrink guard will reject)
```

`local_edit` **never deletes or renames files**. For deletion call `local_delete`; for rename/move call `local_rename`. The model is forbidden from emitting any tag other than `«file»`, and the parser only recognizes `«file»` blocks.

### `local_write(path, instruction)`

Create a **new** file from scratch entirely on the local side. Saves tokens only when the instruction is much shorter than the file it produces. Best for: stubs, boilerplate, scaffolds, config templates, canonical patterns. If you would dictate content line-by-line, use the built-in `Write` instead: same token cost, no local-model round-trip. Refuses to overwrite an existing file (use `local_edit` for that).

```
path:        absolute path of the file to create
instruction: concise spec in any language; if it approaches the length
             of the file itself, use Write instead
```

### `local_delete(paths)`

Delete one or more files. **No model call** — pure `os.unlink`. All paths are validated up front (must be absolute, must exist, must be regular files); if any validation fails, no file is touched. If a deletion fails midway through the loop (e.g. file is locked), already-deleted files are NOT restored — the report tells you which ones survived.

```
paths: non-empty list of absolute file paths
```

### `local_rename(src, dst)`

Rename or move a file. **No model call** — pure `os.replace`. Refuses to overwrite an existing destination. Creates the destination parent directory if missing. Atomic within a Windows volume; cross-volume moves fall back to copy+delete and are not atomic.

```
src: absolute path of the file to rename (must exist, regular file)
dst: absolute destination path (must NOT exist)
```

### `local_snippet(prompt)`

Generate a short snippet and return it as text. **This costs Claude tokens** because the result flows back into Claude's context. Use only when there's no file destination (regex, SQL, one-liners). Uses a 4k context window and a 1024-token output cap to keep snippet calls fast and prevent the model from rambling in markdown; a terse system prompt steers it toward "code only, no prose."

```
prompt: the task or question (any language)
```

## Guard-rails

The old setup occasionally wrote empty / partially-truncated files, or hollowed out class stubs when the user actually wanted the file *deleted*. All checks below run **inside the server**, so Claude only sees a short accept/reject summary; guard-rails cost zero Claude tokens.

For each `«file»` block emitted by the model:

1. **Non-empty**: empty or whitespace-only content is rejected (use `local_delete` to remove a file).
2. **No truncation markers**: lines whose entire trimmed content matches a lazy-output marker (`... rest of file unchanged`, `// ... existing code ...`, `<TRUNCATED>`, etc.) are rejected, but **only if the same line wasn't already in the original**. So legitimate template files don't trip the check.
3. **No suspicious shrink**: if the new file is less than 50% of the original size AND the (English-normalized) instruction contains no removal keyword (`delete`, `remove`, `strip`, `drop`, `clear`, `empty`, `shrink`, `erase`, `purge`, `discard`), the edit is rejected. Non-English instructions are translated first, so equivalents in other languages also satisfy this guard.
4. **Bracket delta unchanged**: for `.py .java .js .ts .tsx .jsx .go .rs .c .cpp .h .hpp .json`, the unmatched-bracket count `({}, (), [])` of the new file must match the original's. Comparing the *delta* lets strings/comments cancel symmetrically and avoids false positives. Catches mid-stream truncation cheaply.
5. **Semantic parse** (`.py`, `.json` only): the new content is fed to `ast.parse` / `json.loads`. Syntax errors are rejected with the offending line number. This is a real parser — it catches unterminated strings, stray indentation, missing commas, and other truncation patterns the bracket heuristic cannot see. For other extensions the check is a no-op (adding JS/TS would require shelling out to `node --check`).
6. **Identity no-op**: files where the model returned the original verbatim are silently dropped from the batch.
7. **Path allowlist**: the model can only emit `«file»` blocks for paths that were passed in `files`. Any unknown path rejects the entire batch.

If any guard fails on any change, **the entire batch is rejected** and a structured diagnostic is returned. No partial writes ever.

Note: there is no `<delete/>` guard-rail because there is no `<delete/>` block. Deletion goes through the dedicated `local_delete` tool, where the caller — not the model — names the paths to remove. This eliminates an entire class of failure modes (hallucinated deletes, intent guards that depended on language-specific phrase lists, the inference fallback for malformed delete tags).

`local_write` runs the same checks except: no shrink guard (no original to compare), and the bracket check is absolute (`{}=0 ()=0 []=0`) instead of delta.

### Parse-failure retry

If the model returns output that contains no `«file»` block (and no fenced-markdown fallback either), the server **automatically retries once** with a stricter user message (`"Your previous output was MALFORMED..."`) before surfacing an error. This protects Claude's context from seeing the first malformed dump at all. If the retry also fails, the raw output echoed in the error is capped at ~600 chars so a rambling model response can't blow up the context.

### Atomic apply

All `local_edit` changes are validated first; only then are they applied. Each file is written via a temp file in the same directory + `os.replace` (atomic on Windows). If any write fails partway through (e.g., a file is locked by an IDE or antivirus), every successful write is reverted from the captured original bytes. `local_delete` is intentionally non-atomic across multiple files: deletions are reported individually and survivors are not restored.

### Windows-specific notes

- **Line endings preserved**: the dominant line ending of each original file (CRLF or LF) is detected and re-applied on write. The model always sees and emits LF; the server is the only place that handles CRLF. No silent CRLF↔LF conversion.
- **Path normalization**: paths in `local_edit`'s allowlist are matched case-insensitively and slash-agnostically (`os.path.normcase(os.path.abspath(...))`), so `C:/Users/...`, `C:\Users\...`, and `c:\users\...` all resolve to the same entry. `local_delete` and `local_rename` use the same normalization for self-comparison.
- **Locked files**: a `PermissionError` from an editor/AV/indexer holding the file is caught, the batch is reverted, and Claude gets a `file is locked or not writable` diagnostic. No traceback.
- **Long paths**: paths exceeding the Windows 260-char limit will surface as a guard-rail rejection. No `\\?\` workaround; enable Windows long-path support if needed.

## Starting the server manually (for debugging)

```powershell
~\.claude\local-mcp\Start-Server.ps1
```

Or directly:

```powershell
uv run "$env:USERPROFILE\.claude\local-mcp\server.py"
```

The server speaks MCP over stdio. When launched normally by Claude Code it starts automatically. Run it manually only to check for import errors or verify Ollama connectivity.

## Verifying Ollama is reachable

```bash
curl http://localhost:11434/api/tags
```

Should return a JSON list of installed models including your configured model.

## Reuse in other projects

The server is registered at user scope and works in every project automatically. No per-project configuration needed *to make the tools available*.

However, **Claude will not automatically prefer the MCP tools over its built-in `Edit` / `Write`**; that's a model decision, and the built-ins usually win unless we forbid them. To actually realise the token savings on a given project, run the per-project setup script below.

### Per-project setup: forcing the delegation

`Setup-Project.ps1` configures a project so Claude *must* use `local_edit` / `local_write` for file changes (it bans the built-in `Edit` / `Write` tools via `permissions.deny`, and adds a guidance section to `CLAUDE.md` explaining why).

```powershell
# In the project root you want to configure:
~\.claude\local-mcp\Setup-Project.ps1

# Or against an explicit path:
~\.claude\local-mcp\Setup-Project.ps1 -Path C:\dev\my-project

# Undo:
~\.claude\local-mcp\Setup-Project.ps1 -Remove
```

What it does (idempotent; safe to re-run):

1. Creates `<project>\.claude\settings.json` if missing, or merges into the existing one. Adds `"Edit"` and `"Write"` to `permissions.deny`. Other keys (`env`, `allow`, other `deny` entries, etc.) are preserved.
2. Creates `<project>\CLAUDE.md` if missing, or appends to it. The guidance block is delimited by `<!-- BEGIN local-mcp -->` / `<!-- END local-mcp -->` markers, so re-running replaces the block in place rather than duplicating it.
3. With `-Remove`: pulls `Edit`/`Write` back out of the deny array, and strips the marker block from `CLAUDE.md`. Empty files (created from scratch by the setup) are deleted; files with other content are preserved.

After running the script, **restart Claude Code in that project** for the new settings to take effect. Then in a new session, ask Claude to make a non-trivial edit; it will be forced to call `local_edit`, and the result will carry the `[<model>]` prefix (e.g. `[qwen3-coder:30b]`) instead of going through the built-in `Edit` tool.

PowerShell 5.1 (default on Windows 10) is supported. No external dependencies.

## Verification / test plan

Run these checks after pulling the model or after any change to `server.py`. Most are end-to-end through Claude Code itself.

### 0. Preliminary placement & speed benchmark

Before relying on the server in a session, confirm the model loads correctly and runs at acceptable speed:

```powershell
# Ensure no other Ollama models are hogging VRAM
ollama stop qwen3:14b 2>$null

# Warm-load the coder model with a representative prompt
ollama run qwen3-coder:30b --verbose "/no_think write a python function to compute fibonacci numbers iteratively"

# Inspect placement
ollama ps
nvidia-smi
```

What to look for:

- `ollama ps` → `PROCESSOR` column should show a GPU-dominant split, e.g. `27% CPU / 73% GPU`. If it says `100% CPU`, something is wrong (wrong quant, VRAM occupied by another process). On reference hardware (RTX 5070 Ti), Ollama places ~73% on GPU.
- `--verbose` output → `eval rate` should be at least ~15 tok/s. Reference hardware gets ~48 tok/s.
- `nvidia-smi` → `Memory-Usage` near full (~15 GB used) is **expected and fine** with this model. The per-process `GPU Memory Usage` column shows `N/A` on Windows WDDM consumer GPUs; this is a driver limitation, not a problem.

If eval rate is below ~10 tok/s, reconsider quant or context size before proceeding.

### 1. Model present

```bash
ollama pull qwen3-coder:30b
curl http://localhost:11434/api/tags
```

The JSON response should list `qwen3-coder:30b`.

### 2. Server imports cleanly

The MCP server is launched by Claude Code via `uv run server.py`. To check for syntax / import errors without blocking on the stdio loop:

```powershell
python -c "import py_compile; py_compile.compile(r'C:/Users/user/.claude/local-mcp/server.py', doraise=True); print('ok')"
```

### 3. Basic tool round-trip (in a Claude Code session)

After restarting the MCP server (FastMCP loads `server.py` once at startup; `/mcp` reconnect or restart the Claude CLI):

- `local_snippet("write a regex for ISO-8601 dates")` → first call eats the load cost (~5-10 s warmup); subsequent calls return in well under 10 s. The result carries the configured model name as prefix (e.g. `[qwen3-coder:30b]`).
- `local_write` on a fresh scratch path → confirm file is created with sane content and a `[<model>] Created ...` summary.
- `local_edit` on a single small file with a short instruction → confirm `[<model>]` prefix and the edit lands.
- `local_edit` on 3+ files with a longer instruction → same model, same behavior.

Between calls, run `ollama ps` from another terminal: the model should stay loaded with no eviction (no `0%` keep_alive countdown).

### 3b. Pinning verification

After 5+ minutes of idle, run `ollama ps` again. With `keep_alive: -1`, the `UNTIL` column should still show the model loaded ("Forever" or equivalent). If it has unloaded, the `keep_alive` payload field is not being honored; check `_call_ollama` in `server.py`.

### 4. Guard-rail regression tests (these are the failures from the deepseek era)

- **Method removal**: ask `local_edit` to "remove unused method foo" on a small file → confirm the method is removed and the rest of the file is still intact (not truncated).
- **File deletion**: call `local_delete([path])` directly → confirm the file is removed and no LLM call is made (check `ollama ps` token counter is unchanged). `local_edit` itself can no longer delete files; if the model emits any non-`«file»` tag it is silently ignored.
- **File rename**: call `local_rename(src, dst)` → confirm `src` is gone, `dst` exists with the same bytes, and again no LLM call. Then call it again with the same args → expect a clean `dst already exists` error.
- **Non-English instruction**: call `local_edit` with `instruction="rimuovi il metodo foo"` (Italian) or `"supprime la méthode foo"` (French) → confirm the edit succeeds and the shrink guard does NOT reject (the translation pre-pass converts the removal verb to English before the guard runs).
- **Suspicious shrink rejection**: pass a non-trivial file with a vague instruction that causes the model to return near-empty content → confirm the shrink guard rejects and **no file on disk is touched**.
- **Atomic apply**: pass two files where one valid edit and one invalid edit are returned → confirm neither file is modified (all-or-nothing).
- **Semantic parse guard**: ask `local_edit` to make a change on a `.py` file with an instruction that's likely to produce a syntax error (e.g. "delete the `def` keyword from function foo") → confirm rejection with a `python syntax error at line N` diagnostic and no file touched. Same on a `.json` file.

### 5. Windows-specific tests

- **CRLF preservation**: edit a file that uses CRLF line endings → confirm the file still uses CRLF after the edit (no silent conversion to LF, no mixed endings). Check with `python -c "print(repr(open('path','rb').read()[:200]))"`.
- **Locked file**: open a target file in another process holding an exclusive lock → run `local_edit` on it → confirm a clean `file is locked or not writable` rejection (no traceback, no partial state).
- **Path normalization**: call `local_delete` with a path mixing forward and backslashes / different casing (`C:/Users/.../Foo.java`, `C:\Users\...\Foo.java`, `c:\users\...\foo.java`) and verify all three resolve to the same file. For `local_edit`, pass a path with one casing in `files` and verify it accepts the model's edit even if the response echoes a different casing (the allowlist is case-insensitive).

### 6. Token-spend sanity check

After running a non-trivial edit task with `local_edit`, look at the Claude Code session token counter; the `local_edit` call itself should add only a handful of input tokens (the short summary), not the full file contents. That's the win.
