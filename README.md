# local-mcp

MCP server that delegates implementation work to local Ollama models so Claude (Opus/Sonnet) only orchestrates. The goal is to **save cloud tokens** by keeping file contents on the local machine — they read, edit, and write happen entirely server-side, and Claude only sees a short summary.

## Token-saving philosophy

The big savings come from **never round-tripping file contents through Claude's context**:

- `local_edit` and `local_write` read and write files on the MCP side. Claude sends a short instruction and gets back a one-line summary. **These are the tools that save the most tokens — prefer them whenever a result will land in a file.**
- `local_snippet` returns generated text that flows back through Claude. It's useful as a fallback for regex / SQL / one-liners that have no file destination yet, but every byte of its output costs Claude input tokens. Use sparingly.

## Models — single-model architecture

| Tag                  | Type                          | Use for |
|----------------------|-------------------------------|---------|
| `qwen3-coder:30b`    | MoE 30B (3B active per token) | Everything: multi-file edits, scaffolding, snippets |

All three tools (`local_edit`, `local_write`, `local_snippet`) call **the same model**. There is no routing, no small/large tier, no model switching. The `model` parameter on each tool is preserved for backward compatibility but is currently ignored.

### Why a single model

Iteration 1 of this server had a two-tier design: `qwen3:14b` for short single-file edits, `qwen3-coder:30b` for multi-file work. In practice that turned out to be the wrong shape:

1. **VRAM math doesn't allow coexistence.** On a 16 GB GPU, `qwen3:14b` (~10 GB) and `qwen3-coder:30b` (~18 GB) cannot be loaded together. Any workflow that mixes the two tools forces Ollama to evict one to load the other. Windows page cache (in 64 GB DDR5) makes the *reload* faster than a cold-from-NVMe load, but it is still seconds of latency on every alternation.
2. **The actual workload is multi-file feature implementation.** Snippet generation is rare in this setup; most calls are "implement this feature reading the plan." That's `local_edit` / `local_write` territory on the larger coder model. The 14b dense model added VRAM thrashing without buying useful work.

The fix is to pick one model and pin it. `qwen3-coder:30b` wins because it's the right tool for multi-file agentic edits and its MoE design (only ~3B params active per token) absorbs partial CPU offload gracefully — measured ~48 tok/s on RTX 5070 Ti even with ~27% of layers spilled to CPU.

### VRAM pinning (no eviction, ever)

Every Ollama call from this server passes `keep_alive: -1`. The model is loaded into VRAM on the first call of the session and **never unloaded** — no idle eviction, no thrashing, no cold-load latency on subsequent calls. After the first warmup, all calls go straight to compute.

`/no_think` is appended to user prompts so qwen3 skips the reasoning chain (faster, less VRAM, deterministic edits don't benefit from visible reasoning). A defensive `<think>...</think>` stripper runs on the response in case any reasoning leaks through.

### Target hardware

Designed and tested on: i9-14900K, 64 GB DDR5, RTX 5070 Ti 16 GB, **Windows 10**. The 30b coder fits 73% on GPU / 27% on CPU at Q4_K_M with `num_ctx=32768`; multi-file edits can take a few minutes (timeout is 20 min). Smaller GPUs will see a larger CPU split and proportionally lower tok/s.

## Prerequisites

- [Ollama](https://ollama.com) running locally on `http://localhost:11434`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) in PATH
- The coder model pulled:

```bash
ollama pull qwen3-coder:30b
```

Dependencies for the server itself are managed automatically by `uv` via the inline script metadata in `server.py` — no manual `pip install`.

## Installation

The server is registered globally in `~/.claude.json` via:

```bash
claude mcp add --scope user local-mcp uv run "C:/Users/user/.claude/local-mcp/server.py"
```

## Tools exposed to Claude Code

### `local_edit(files, instruction, model?)`

Edit one or more **existing** files in place. Reads the files, sends them to `qwen3-coder:30b`, parses `<file>` and `<delete/>` blocks from the response, validates with guard-rails, and atomically applies the result. Claude only sees a one-line summary or a structured rejection.

```
files:       list of absolute file paths
instruction: what to change (plain English; include "delete"/"remove"/"rimuovi"
             if you expect a large reduction in size)
model:       ignored (kept for backward compatibility)
```

The model can also emit `<delete path="..."/>` blocks to **remove a file entirely** instead of leaving it as an empty stub. Deletes are restricted to paths in the `files` argument — the model cannot invent new paths to delete.

### `local_write(path, instruction, model?)`

Create a **new** file from scratch entirely on the local side. Refuses to overwrite an existing file (use `local_edit` for that). The generated content never enters Claude's context.

```
path:        absolute path of the file to create
instruction: what to put in the file (plain English; can be detailed)
model:       ignored (kept for backward compatibility)
```

### `local_snippet(prompt, model?)`

Generate a short snippet and return it as text. **This costs Claude tokens** because the result flows back into Claude's context. Use only when there's no file destination (regex, SQL, one-liners). Uses a 4k context window and a 1024-token output cap to keep snippet calls fast and prevent the model from rambling in markdown; a terse system prompt steers it toward "code only, no prose."

```
prompt: the task or question
model:  ignored (kept for backward compatibility)
```

## Guard-rails

The old setup occasionally wrote empty / partially-truncated files, or hollowed out class stubs when the user actually wanted the file *deleted*. All checks below run **inside the server**, so Claude only sees a short accept/reject summary — guard-rails cost zero Claude tokens.

For each `<file>` block emitted by the model:

1. **Non-empty** — empty or whitespace-only content is rejected (the model should have used `<delete/>`).
2. **No truncation markers** — lines whose entire trimmed content matches a lazy-output marker (`... rest of file unchanged`, `// ... existing code ...`, `<TRUNCATED>`, etc.) are rejected, but **only if the same line wasn't already in the original** — so legitimate template files don't trip the check.
3. **No suspicious shrink** — if the new file is less than 50% of the original size AND the instruction contains no removal keyword (`delete`, `remove`, `strip`, `cancella`, `rimuovi`, `elimina`, …), the edit is rejected.
4. **Bracket delta unchanged** — for `.py .java .js .ts .tsx .jsx .go .rs .c .cpp .h .hpp .json`, the unmatched-bracket count `({}, (), [])` of the new file must match the original's. Comparing the *delta* lets strings/comments cancel symmetrically and avoids false positives. Catches mid-stream truncation cheaply.
5. **Identity no-op** — files where the model returned the original verbatim are silently dropped from the batch.

For each `<delete/>` block:

1. **Strict allowlist** — the deleted path must exactly match (Windows-normalized) one of the absolute paths passed in `files`. The model cannot delete paths it wasn't given.
2. **Must exist as a regular file**.
3. **No conflict** — the same path must not also appear in a `<file>` block.

If any guard fails on any change, **the entire batch is rejected** and a structured diagnostic is returned. No partial writes ever.

`local_write` runs the same checks except: no shrink guard (no original to compare), and the bracket check is absolute (`{}=0 ()=0 []=0`) instead of delta.

### Atomic apply

All changes are validated first; only then are they applied. Each file is written via a temp file in the same directory + `os.replace` (atomic on Windows). If any write or delete fails partway through (e.g., a file is locked by an IDE or antivirus), every successful write is reverted from the captured original bytes.

### Windows-specific notes

- **Line endings preserved**: the dominant line ending of each original file (CRLF or LF) is detected and re-applied on write. The model always sees and emits LF; the server is the only place that handles CRLF. No silent CRLF↔LF conversion.
- **Path normalization**: `<delete>` paths are matched case-insensitively and slash-agnostically (`os.path.normcase(os.path.abspath(...))`), so `C:/Users/...`, `C:\Users\...`, and `c:\users\...` all resolve to the same allowlist entry.
- **Locked files**: a `PermissionError` from an editor/AV/indexer holding the file is caught, the batch is reverted, and Claude gets a `file is locked or not writable` diagnostic — no traceback.
- **Long paths**: paths exceeding the Windows 260-char limit will surface as a guard-rail rejection. No `\\?\` workaround; enable Windows long-path support if needed.

## Starting the server manually (for debugging)

```powershell
~\.claude\local-mcp\Start-Server.ps1
```

Or directly:

```powershell
uv run "$env:USERPROFILE\.claude\local-mcp\server.py"
```

The server speaks MCP over stdio — when launched normally by Claude Code it starts automatically. Run it manually only to check for import errors or verify Ollama connectivity.

## Verifying Ollama is reachable

```bash
curl http://localhost:11434/api/tags
```

Should return a JSON list of installed models including `qwen3-coder:30b`.

## Reuse in other projects

The server is registered at user scope and works in every project automatically. No per-project configuration needed *to make the tools available*.

However, **Claude will not automatically prefer the MCP tools over its built-in `Edit` / `Write`** — that's a model decision, and the built-ins usually win unless we forbid them. To actually realise the token savings on a given project, run the per-project setup script below.

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

What it does (idempotent — safe to re-run):

1. Creates `<project>\.claude\settings.json` if missing, or merges into the existing one. Adds `"Edit"` and `"Write"` to `permissions.deny`. Other keys (`env`, `allow`, other `deny` entries, etc.) are preserved.
2. Creates `<project>\CLAUDE.md` if missing, or appends to it. The guidance block is delimited by `<!-- BEGIN local-mcp -->` / `<!-- END local-mcp -->` markers, so re-running replaces the block in place rather than duplicating it.
3. With `-Remove`: pulls `Edit`/`Write` back out of the deny array, and strips the marker block from `CLAUDE.md`. Empty files (created from scratch by the setup) are deleted; files with other content are preserved.

After running the script, **restart Claude Code in that project** for the new settings to take effect. Then in a new session, ask Claude to make a non-trivial edit — it will be forced to call `local_edit`, and the result will carry the `[qwen3-coder:30b]` prefix instead of going through the built-in `Edit` tool.

PowerShell 5.1 (default on Windows 10) is supported — no external dependencies.

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
- `nvidia-smi` → `Memory-Usage` near full (~15 GB used) is **expected and fine** with this model. The per-process `GPU Memory Usage` column shows `N/A` on Windows WDDM consumer GPUs — this is a driver limitation, not a problem.

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

After restarting the MCP server (FastMCP loads `server.py` once at startup — `/mcp` reconnect or restart the Claude CLI):

- `local_snippet("write a regex for ISO-8601 dates")` → first call eats the load cost (~5–10 s warmup); subsequent calls return in well under 10 s. All calls go to `qwen3-coder:30b` (`[qwen3-coder:30b]` prefix in the result).
- `local_write` on a fresh scratch path → confirm file is created with sane content and a `[qwen3-coder:30b] Created ...` summary.
- `local_edit` on a single small file with a short instruction → confirm `[qwen3-coder:30b]` prefix and the edit lands.
- `local_edit` on 3+ files with a longer instruction → same model, same behavior.

Between calls, run `ollama ps` from another terminal: the model should stay loaded with no eviction (no `0%` keep_alive countdown).

### 3b. Pinning verification

After 5+ minutes of idle, run `ollama ps` again. With `keep_alive: -1`, the `UNTIL` column should still show the model loaded ("Forever" or equivalent). If it has unloaded, the `keep_alive` payload field is not being honored — check `_call_ollama` in `server.py`.

### 4. Guard-rail regression tests (these are the failures from the deepseek era)

- **Method removal**: ask `local_edit` to "remove unused method foo" on a small file → confirm the method is removed and the rest of the file is still intact (not truncated).
- **File deletion via `<delete/>`**: ask `local_edit` to "delete the file Bar.java" → confirm the model emits `<delete/>`, the file is removed, and is **not** left as an empty class stub.
- **Suspicious shrink rejection**: pass a non-trivial file with a vague instruction that causes the model to return near-empty content → confirm the shrink guard rejects and **no file on disk is touched**.
- **Atomic apply**: pass two files where one valid edit and one invalid edit are returned → confirm neither file is modified (all-or-nothing).

### 5. Windows-specific tests

- **CRLF preservation**: edit a file that uses CRLF line endings → confirm the file still uses CRLF after the edit (no silent conversion to LF, no mixed endings). Check with `python -c "print(repr(open('path','rb').read()[:200]))"`.
- **Locked file**: open a target file in another process holding an exclusive lock → run `local_edit` on it → confirm a clean `file is locked or not writable` rejection (no traceback, no partial state).
- **Path normalization**: call `local_edit` with `files=["C:/Users/.../Foo.java"]` and an instruction the model is likely to satisfy with a `<delete>` block — verify the delete succeeds even if the model emits the path with backslashes or different casing (`C:\Users\...\Foo.java` / `c:\users\...\foo.java`).

### 6. Token-spend sanity check

After running a non-trivial edit task with `local_edit`, look at the Claude Code session token counter — the `local_edit` call itself should add only a handful of input tokens (the short summary), not the full file contents. That's the win.
