# local-mcp

MCP server that delegates file operations to a local Ollama model so Claude only orchestrates. File bodies never pass through Claude's context; Claude sends a short instruction and receives a one-line summary.

## How it works

`local_edit`, `local_write`, `local_read`, and `local_snippet` call a single configured Ollama model. `local_delete` and `local_rename` are pure filesystem operations with no model call. Parallel calls from agents are queued FIFO through a single-worker executor; the GPU processes one request at a time with no contention errors.

## Prerequisites

- [Ollama](https://ollama.com) running on `http://localhost:11434`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) in PATH
- Default model pulled: `ollama pull gemma4:e4b`

## Quick start

```powershell
# 1. Set the active model config
cp configs\gemma4-e4b.json model-config.json

# 2. Register the server globally (once)
claude mcp add --scope user local-mcp uv run "C:/Users/user/.claude/local-mcp/server.py"

# 3. Add per-project routing guidance (run in each project root)
~\.claude\local-mcp\Setup-Project.ps1
```

Restart Claude Code after step 2 or 3. After any change to `model-config.json`, reconnect the MCP server (`/mcp` or restart the CLI).

## Tools

| Tool | When to use |
|------|-------------|
| `local_edit(files, instruction)` | Edit existing files. Best when files are not yet in Claude's context or the change spans many lines/files. |
| `local_write(path, instruction)` | Create a new file from scratch. Saves tokens only when the instruction is much shorter than the output (stubs, boilerplate, scaffolds). |
| `local_read(files, instruction)` | Read-only analysis: summarize, review, find patterns. Output flows back to Claude's context. |
| `local_delete(paths)` | Delete files. No model call. |
| `local_rename(src, dst)` | Rename or move a file. No model call. |
| `local_snippet(prompt)` | Generate a short snippet returned as text. Output costs Claude tokens; use sparingly. |

All instruction strings are translated server-side when non-English, so you can write instructions in any language.

## Model configuration

The server reads `model-config.json` at startup. If missing, built-in defaults are used. The file is gitignored.

### Available templates

| Template | Model | Provider | Notes |
|----------|-------|----------|-------|
| `configs/gemma4-e4b.json` | `gemma4:e4b` | ollama | **Default.** 11 GB, 100% GPU, 32k ctx, 90 s timeout. |
| `configs/qwen3-coder-30b.json` | `qwen3-coder:30b` | ollama | MoE 30B (~3B active). 120 s timeout. |
| `configs/devstral-small-2-24b.json` | `devstral-small-2:24b` | ollama | Mistral code-agent model. 120 s timeout. |
| `configs/qwen3-coder-480b-free.json` | `qwen/qwen3-coder:free` | openrouter | Remote free tier; requires `OPENROUTER_API_KEY`. |
| `configs/openrouter-free.json` | `openrouter/free` | openrouter | Free-models router; non-deterministic model per call. |

Copy a template to `model-config.json` to switch models.

### Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `"ollama"` | `"ollama"` or `"openrouter"` |
| `model` | `"gemma4:e4b"` | Ollama tag or OpenRouter slug |
| `ollama_url` | `"http://localhost:11434/api/chat"` | Ollama endpoint |
| `edit_ctx` | `32768` | Context window for edit/write/read (Ollama only) |
| `snippet_ctx` | `4096` | Context window for snippet calls |
| `snippet_num_predict` | `1024` | Max output tokens for snippets |
| `translate_ctx` | `2048` | Context window for translation pre-pass |
| `translate_num_predict` | `512` | Max output tokens for translation |
| `timeout` | `1200` | Seconds; per-chunk for streaming Ollama, total for OpenRouter |

For OpenRouter, also set `openrouter_url`, `openrouter_referer`, `openrouter_title`, `openrouter_extra_body`, and `OPENROUTER_API_KEY` env var.

## Guard-rails

Applied per `«file»` block before any write. All checks run server-side; Claude only sees accept/reject.

1. **Non-empty**: empty content rejected (use `local_delete` to remove a file).
2. **No truncation markers**: lines matching lazy-output patterns (`... rest unchanged`, `// existing code`, `<TRUNCATED>`, etc.) rejected unless already in the original.
3. **No suspicious shrink**: new size < 50% of original without a removal keyword in the instruction is rejected.
4. **Bracket delta**: unmatched `{}`, `()`, `[]` count must match the original's delta (code files only).
5. **Semantic parse**: `.py` files checked with `ast.parse`; `.json` with `json.loads`. Syntax errors include the line number.
6. **Identity no-op**: files unchanged by the model are silently skipped.
7. **Path allowlist**: model can only emit blocks for paths passed in `files`.

If any check fails, the entire batch is rejected and no file is touched. On parse failure (no `«file»` blocks), the server retries once with a stricter prompt before surfacing an error.

## Windows notes

- CRLF line endings are detected and preserved on write.
- Paths are matched case-insensitively and slash-agnostically.
- Locked files produce a clean `file is locked or not writable` diagnostic.

## License

GNU Affero GPL v3. See [LICENSE](LICENSE).
