# local-mcp

MCP server that delegates file operations to a local Ollama model so Claude only orchestrates. File bodies never pass through Claude's context; Claude sends a short instruction and receives a one-line summary.

## How it works

`local_edit`, `local_write`, `local_read`, and `local_snippet` call a single configured Ollama model. `local_outline` answers structural questions with no model call at all (deterministic parsing). Parallel calls from agents are queued FIFO through a single-worker executor; the GPU processes one request at a time with no contention errors. Deletion and rename are deliberately not tools: they involve no file body, so they save no Claude tokens; use the built-in tools or Bash for those.

## Prerequisites

- [Ollama](https://ollama.com) running on `http://localhost:11434`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) in PATH
- Default model pulled: `ollama pull gemma4:12b`

Python dependencies (including `tree-sitter` and `tree-sitter-java`, used by `local_outline`) are declared inline in `server.py` and installed automatically by `uv run`. If tree-sitter is unavailable for some reason the server still runs; only the Java outline path degrades (it returns a clear error or falls back to chunking).

## Quick start

```powershell
# 1. Set the active model config
cp configs\gemma4-12b.json model-config.json

# 2. Register the server globally (once)
claude mcp add --scope user local-mcp uv run "C:/Users/user/.claude/local-mcp/server.py"

# 3. Add per-project routing guidance (run in each project root)
~\.claude\local-mcp\Setup-Project.ps1
```

Restart Claude Code after step 2 or 3. After any change to `model-config.json`, reconnect the MCP server (`/mcp` or restart the CLI).

## Tools

| Tool | When to use |
|------|-------------|
| `local_outline(files)` | Get an API skeleton (package, types, annotations, fields, method signatures; no bodies) for `.java`/`.py`. Deterministic, no model call. Use instead of Read/local_read when you need an interface or shape, not implementations. |
| `local_edit(files, instruction, context_files=None)` | Edit existing files. Best when files are not yet in Claude's context or the change spans many lines/files. |
| `local_write(path, instruction, context_files=None)` | Create a new file from scratch. Saves tokens only when the instruction is much shorter than the output (stubs, boilerplate, scaffolds). |
| `local_read(files, instruction)` | Read-only analysis: summarize, review, find patterns. Output flows back to Claude's context. Not for verbatim retrieval (use the built-in Read). Inputs larger than one context window are handled by map-reduce over the free local model instead of being refused. |
| `local_snippet(prompt)` | Generate a short snippet returned as text. Output costs Claude tokens; use sparingly. |

All instruction strings are translated server-side when non-English, so you can write instructions in any language.

`local_edit` and `local_write` accept an optional `context_files`: absolute paths the local model may **read** for reference (an interface being implemented, a caller's signature, a constants module) but may **not** modify. They are embedded in the prompt as read-only `«context»` blocks and are never added to the editable allowlist, so a cross-file edit can get the details right (exact method names, signatures) without those files entering Claude's context. If the model emits a change for a context file it is rejected like any other out-of-allowlist path. Context files count toward the input-size budget.

### Structural reads (deterministic + map-reduce)

The local model and deterministic parsing are both free, so reads do not have to be one model pass over raw bytes:

- **`local_outline`** parses `.java` (via `tree-sitter`) and `.py` (via the stdlib `ast`) into a compact API skeleton with no model call at all. When Claude needs an interface, a method signature, or "what does this class expose", this returns a few hundred tokens of exact structure instead of the whole file. It is also the right first step before a cross-file edit: outline a collaborator, then pass it as a `context_file`.
- **`local_read` map-reduce**: when the combined input exceeds one context window, each file is summarized in its own free local call and the partials are synthesized into one answer, instead of refusing with "input too large". Oversized code files are first reduced to their skeleton (no extra model cost); only a file with no usable skeleton is chunked by lines. None of the raw bytes enter Claude's context.

When a `.java` or `.kt` file is the target, a short Java/Spring rules block is appended to the system prompt (emit the package for the file's location, preserve every import and annotation, never omit boilerplate behind a comment, follow idiomatic Spring). It loads only for Java/Kotlin targets, so non-Java calls pay nothing for it.

### Spring scaffolding recipes

Spring is the textbook case for this server: a one-line spec expands into a large, heavily annotated class whose bytes never need to touch Claude's context. Route these to `local_write` (or `local_edit` when the file exists) instead of dictating them:

| One-line spec to `local_write` | Expands to |
|--------------------------------|-----------|
| "a JPA entity `User` with id, email, createdAt" | `@Entity` class with `@Id`/`@GeneratedValue`, `@Column` fields, getters/setters |
| "a Spring Data repository for `User`" | `interface UserRepository extends JpaRepository<User, Long>` |
| "a `@RestController` for `User` with CRUD endpoints and a `UserDto`" | controller plus DTO record |
| "a `@Service` `UserService` skeleton using `UserRepository`" | service with constructor injection |
| "a `@Configuration` exposing a `RestTemplate` bean" | config class with `@Bean` method |
| "a MapStruct mapper between `User` and `UserDto`" | `@Mapper` interface |
| "a JUnit 5 test stub for `UserService`" | test class with mocked repository |

For a multi-class change (e.g. editing a `@Service` that depends on an entity and repository), pass the collaborators as `context_files` so the model sees the types it depends on (exact method names, signatures) without Claude reading them into context and without making them editable.

## Token-saving recipes

The largest wins are not new features; they are using the tools in the cheapest order. The single rule behind all of them: **if the result will land in a file, never let its bytes touch Claude's context.** `local_edit` and `local_write` return a one-line summary; `local_read` and `local_snippet` return text Claude pays for. Prefer the former whenever the goal is a file change.

**1. Investigate-and-fix in one call, not read-then-edit.** `local_edit` takes an *instruction*, not a diff, so it can diagnose and fix in the same call.

| Costly | Cheap |
|--------|-------|
| `local_read("find the off-by-one in the loop")` -> read analysis into context -> reason -> `local_edit(...)` | `local_edit("fix the off-by-one that makes the loop skip the last element")` |

The read step only pays off when Claude genuinely needs the *answer* in context (a decision, a summary). If the outcome is a file change, skip it.

**2. Describe intent / end-state; do not dictate line-by-line.** The bytes never pass through Claude either way, so the shortest description of the desired result is cheapest.

| Costly | Cheap |
|--------|-------|
| Read the file, compute the exact new class, paste it into `local_edit` | `local_edit("make Config a frozen dataclass and add a from_env classmethod")` |

**3. Scaffold from a spec; do not write the file yourself.** A one-line spec to `local_write` expands to a large file locally (see the Spring recipes above).

| Costly | Cheap |
|--------|-------|
| Draft a 60-line JPA entity in context, then `Write` it | `local_write("a JPA entity User with id, email, createdAt")` |

**4. Outline or pass context instead of reading dependencies into context.** To get a collaborator's shape, use `local_outline` (no model call, a few hundred tokens) rather than `Read`. To let an edit see a dependency, pass it as a `context_files` entry rather than reading it into context.

| Costly | Cheap |
|--------|-------|
| `Read(UserRepository.java)` to learn its methods, then dictate the call | `local_edit([Service.java], "add lookup(email) delegating to the repo", context_files=[UserRepository.java])` |

## Model configuration

The server reads `model-config.json` at startup. If missing, built-in defaults are used. The file is gitignored.

### Available templates

| Template | Model | Provider | Notes |
|----------|-------|----------|-------|
| `configs/gemma4-12b.json` | `gemma4:12b` | ollama | **Default.** 7.6 GB, `num_ctx` 65536, 180 s timeout. |
| `configs/gemma4-e4b.json` | `gemma4:e4b` | ollama | Lightweight. `num_ctx` 32768, 90 s timeout. |
| `configs/qwen3.5-27b.json` | `qwen3.5:27b` | ollama | Dense 27B, thinking-capable (`think` forced off, `/no_think` applied). `num_ctx` 32768, 240 s timeout for stability under full-parameter compute. |
| `configs/qwen3-coder-30b.json` | `qwen3-coder:30b` | ollama | MoE 30B (~3B active). 120 s timeout. |
| `configs/devstral-small-2-24b.json` | `devstral-small-2:24b` | ollama | Mistral code-agent model. 120 s timeout. |
| `configs/qwen3-coder-480b-free.json` | `qwen/qwen3-coder:free` | openrouter | Remote free tier; requires `OPENROUTER_API_KEY`. |
| `configs/nemotron-3-ultra-550b-free.json` | `nvidia/nemotron-3-ultra-550b-a55b:free` | openrouter | Hybrid Transformer-Mamba MoE, 550B total / 55B active, 1M context (ignored, remote decides). Remote free tier; requires `OPENROUTER_API_KEY`. NVIDIA logs free-tier traffic for security/product improvement, don't send confidential data. |
| `configs/openrouter-free.json` | `openrouter/free` | openrouter | Free-models router; non-deterministic model per call. |

Copy a template to `model-config.json` to switch models.

### Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `"ollama"` | `"ollama"` or `"openrouter"` |
| `model` | `"gemma4:e4b"` | Ollama tag or OpenRouter slug |
| `ollama_url` | `"http://localhost:11434/api/chat"` | Ollama endpoint |
| `num_ctx` | `32768` | One context window for every Ollama call. Kept constant on purpose: changing it between calls reloads the model (~5 s) and defeats `keep_alive` (Ollama only). |
| `read_num_predict` | `1024` | Max output tokens for `local_read` (caps what flows back to Claude). |
| `snippet_num_predict` | `1024` | Max output tokens for `local_snippet`. |
| `translate_num_predict` | `512` | Max output tokens for the translation pre-pass. |
| `temperature` | `0` | Sampling for edit/write/translate; `0` keeps full-file regeneration deterministic. |
| `read_temperature` | `0.2` | Sampling for `local_read`/`local_snippet`; a little fluency for prose. |
| `repeat_penalty` | `1.0` | `1.0` disables the penalty (code legitimately repeats tokens). |
| `top_p` / `top_k` / `seed` / `num_gpu` | `null` | Optional; when null the model's own default applies. |
| `timeout` | `1200` | Seconds; per-chunk for streaming Ollama, total for OpenRouter. |
| `queue_timeout` | `300` | Seconds to wait in the single-worker queue before giving up (bounds the wait, not generation). |
| `log_level` | `"WARNING"` | `"DEBUG"` traces to `local-mcp.log`. Equivalent to the `LOCAL_MCP_DEBUG=1` env var. |

For OpenRouter, also set `openrouter_url`, `openrouter_referer`, `openrouter_title`, `openrouter_extra_body`, and `OPENROUTER_API_KEY` env var. `num_ctx` is ignored for OpenRouter (the remote endpoint decides context).

## Guard-rails

Applied per `«file»` block before any write. All checks run server-side; Claude only sees accept/reject.

1. **Non-empty**: empty content rejected (use the built-in tools to remove a file).
2. **No truncation markers**: lines matching lazy-output patterns (`... rest unchanged`, `// existing code`, `<TRUNCATED>`, etc.) rejected unless already in the original.
3. **No suspicious shrink**: new size < 50% of original without a removal keyword in the instruction is rejected.
4. **Bracket delta**: unmatched `{}`, `()`, `[]` count must match the original's delta (code files only).
5. **Semantic parse**: `.py` files checked with `ast.parse`; `.json` with `json.loads`. Syntax errors include the line number.
6. **Identity no-op**: files unchanged by the model are silently skipped.
7. **Path allowlist**: model can only emit blocks for paths passed in `files`.

### Java/Kotlin guards

Java and Spring are verbose and boilerplate-heavy, so they are where the server saves the most tokens and where a small model is most likely to slip. For `.java`/`.kt` targets four extra deterministic checks run (no model call):

1. **Omission placeholders**: a comment-only line that reduces to a known placeholder phrase (`// getters and setters`, `// ... rest of the class ...`, `// other methods unchanged`, `// constructors omitted`, `// (unchanged)`, etc.) is rejected unless it was already in the original. Normalization strips comment delimiters, dots, and parentheses, so only pure placeholder comments match.
2. **Package matches path**: under a `src/{main,test}/{java,kotlin}` root, the emitted `package` must mirror the directory. A wrong or missing package is a guaranteed compile break.
3. **Public type matches filename**: `Foo.java` must declare `public class|interface|record|enum Foo` (skipped for `package-info`/`module-info`, and for Kotlin, which does not tie types to filenames).
4. **Import-loss heuristic** (edit only): a material drop in `import` lines with no removal keyword in the instruction is flagged, catching imports the model silently dropped.

If any check fails, the entire batch is rejected and no file is touched. On parse failure (no `«file»` blocks), the server retries once with a stricter prompt before surfacing an error.

## Windows notes

- CRLF line endings are detected and preserved on write.
- Paths are matched case-insensitively and slash-agnostically.
- Locked files produce a clean `file is locked or not writable` diagnostic.

## Troubleshooting

The most common first-run failures return a one-line, actionable message instead of a raw stack trace:

- **Ollama not running**: `Ollama is not reachable at <url>. Is it running? Start Ollama, or fix 'ollama_url' in model-config.json.`
- **Model not pulled**: `Model '<model>' is not available in Ollama (HTTP 404). Pull it first: ollama pull <model>`
- **OpenRouter**: connection failures and `401/403/404` responses include the likely fix (check `OPENROUTER_API_KEY`, verify the model slug).

A transient HTTP 500 while a cold model loads is retried once automatically before the error is surfaced.

### Debug logging

The server is silent by default (no log file, nothing on stdout/stderr, so the MCP stdio protocol is never disturbed). To trace why a result occurred (a no-op, a guard rejection, an ignored config, what the model actually returned), set `LOCAL_MCP_DEBUG=1` (or `"log_level": "DEBUG"` in `model-config.json`) and re-run. Output goes to a rotating `local-mcp.log` next to `server.py` (gitignored). File contents are only ever written there, never at the default level.

```powershell
$env:LOCAL_MCP_DEBUG = "1"   # then restart the MCP server
```

## Tests

A single model-free smoke test covers the corruption-risk path (the `«file»` parser, the guard-rails, the size budget, CRLF/binary handling, and language detection). No Ollama or network is needed:

```powershell
uv run tests/test_guards.py
```

It is deliberately minimal: there is no coverage goal and no CI. It exists only because a silent regression in the guards is the one failure that could corrupt a file.

## License

GNU Affero GPL v3. See [LICENSE](LICENSE).
