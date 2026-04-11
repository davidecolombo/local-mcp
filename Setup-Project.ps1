<#
.SYNOPSIS
  Configures a project to force Claude Code to use the local-mcp server
  (qwen3-coder:30b) for file edits and writes, instead of the built-in
  Edit/Write tools. Idempotent: safe to run multiple times.

.DESCRIPTION
  Creates / merges:
    <project>\.claude\settings.json   -- adds "Edit","Write" to permissions.deny
    <project>\CLAUDE.md               -- appends a guidance block delimited by markers

  With -Remove, the inverse: pulls "Edit","Write" out of the deny array and
  removes the marker block from CLAUDE.md, leaving everything else intact.

  Compatible with PowerShell 5.1 (default Windows 10) -- no external deps.

.PARAMETER Path
  Project root. Defaults to the current directory.

.PARAMETER Remove
  Undo the configuration instead of applying it.

.EXAMPLE
  ~\.claude\local-mcp\Setup-Project.ps1
  ~\.claude\local-mcp\Setup-Project.ps1 -Path C:\dev\my-project
  ~\.claude\local-mcp\Setup-Project.ps1 -Remove
#>

[CmdletBinding()]
param(
    [string]$Path = (Get-Location).Path,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

$DenyEntries        = @('Edit')
# Full set of entries this script has ever managed; used to prune stale denies
# from projects set up with an older version of this script.
$ManagedDenyEntries = @('Edit', 'Write')
$BeginMarker  = '<!-- BEGIN local-mcp -->'
$EndMarker    = '<!-- END local-mcp -->'

$ClaudeMdBlock = @"
$BeginMarker
## Use of the local-mcp server

For file operations, use the tools from the ``local-mcp`` server:
- ``local_edit`` instead of ``Edit`` (modifying existing files in place; always a token win)
- ``local_write`` for NEW files ONLY when a concise instruction expands into a much larger file: stubs, boilerplate, scaffolds, config templates. If you would dictate content line-by-line, use the built-in ``Write`` instead -- the local model round-trip adds overhead without saving tokens.
- ``local_delete`` instead of ``Bash rm`` (deleting files; no model call)
- ``local_rename`` instead of ``Bash mv`` (rename / move; no model call, atomic within a volume)
- ``local_snippet`` only as a fallback for snippets with no file destination

Break-even rule for ``local_write``: savings only occur when ``len(instruction) << len(file)``. When you already know the exact content -- copied pattern, specific function bodies, precise config values -- use ``Write``; same token cost without the extra round-trip.

Reason: for ``local_edit`` the file contents never pass through Claude's context (diffs compress well), and ``local_delete`` / ``local_rename`` are pure syscalls. The built-in ``Edit`` tool is denied via ``.claude/settings.json``; ``Write`` remains available for the cases above.

### Instructions can be in any language

The instruction strings to ``local_edit``, ``local_write``, and
``local_snippet`` may be written in any language (Italian, French, Spanish,
German, etc.). Non-English instructions are detected and translated to
English server-side before they reach the model and the guard-rails. Do NOT
translate instructions yourself before calling these tools -- it wastes
output tokens for no benefit.

### Trust the tools, do not re-read files to verify

All ``local-mcp`` tools are transactional and report success or a structured
error in their return value. ``local_edit`` runs guard-rails server-side,
writes atomically, and reverts every successful write from captured original
bytes if any write in the batch fails. ``local_delete`` validates every path
up front before touching anything. ``local_rename`` is a single atomic
``os.replace`` within a volume.

When the summary begins with ``[qwen3-coder:30b]`` (for ``local_edit`` /
``local_write``) or starts with ``Deleted`` / ``Renamed`` (for
``local_delete`` / ``local_rename``) and lists the affected files, trust it
and move on. Do NOT ``Read``, ``Glob``, or ``Bash ls`` the filesystem
afterwards to "verify"; that defeats the token-saving goal. Read a file
ONLY when:
- the tool returned a line starting with ``REJECTED``, ``Error``, or ``Partial failure``, or
- you genuinely need the current contents for a subsequent, unrelated step.

### Deleting and renaming files

``local_edit`` never deletes or renames files -- it can only modify content
in place. For whole-file deletion call ``local_delete([path, ...])``; for
rename or move call ``local_rename(src, dst)``. Both are pure syscalls
(no LLM involvement) and are strictly more reliable than asking a model to
emit a delete tag. Do NOT use ``Bash`` with ``rm`` / ``mv`` / ``del`` /
``move`` for file operations -- those bypass the MCP layer entirely.
$EndMarker
"@

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Utf8NoBom {
    param([string]$FilePath, [string]$Content)
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($FilePath, $Content, $utf8NoBom)
}

# Read JSON as PSCustomObject (PS 5.1's native ConvertFrom-Json output type).
# We deliberately do NOT convert to hashtable: PS 5.1's auto-unwrap of single
# collections through function returns made the deep-conversion approach
# fragile. PSCustomObject + Add-Member is the canonical PS 5.1 idiom and
# round-trips through ConvertTo-Json without losing types.
function Read-Json {
    param([string]$FilePath)
    $raw = [System.IO.File]::ReadAllText($FilePath)
    if ([string]::IsNullOrWhiteSpace($raw)) { return [PSCustomObject]@{} }
    return ($raw | ConvertFrom-Json)
}

function Write-JsonFile {
    param([string]$FilePath, $Object)
    $json = $Object | ConvertTo-Json -Depth 32
    Write-Utf8NoBom -FilePath $FilePath -Content $json
}

function Test-PSObjectHasProperty {
    param([Parameter(Mandatory)] $Object, [Parameter(Mandatory)][string] $Name)
    return ($null -ne $Object.PSObject.Properties[$Name])
}

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    Write-Error "Project path not found: $Path"
    exit 1
}
$ProjectRoot   = (Resolve-Path -LiteralPath $Path).Path
$ClaudeDir     = Join-Path $ProjectRoot '.claude'
$SettingsPath  = Join-Path $ClaudeDir 'settings.json'
$ClaudeMdPath  = Join-Path $ProjectRoot 'CLAUDE.md'

Write-Host "Target project: $ProjectRoot" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# settings.json -- apply
# ---------------------------------------------------------------------------

function Apply-Settings {
    if (-not (Test-Path -LiteralPath $ClaudeDir)) {
        New-Item -ItemType Directory -Path $ClaudeDir -Force | Out-Null
    }

    if (-not (Test-Path -LiteralPath $SettingsPath)) {
        $obj = [PSCustomObject]@{
            permissions = [PSCustomObject]@{
                deny = @($DenyEntries)
            }
        }
        Write-JsonFile -FilePath $SettingsPath -Object $obj
        Write-Host "[+] Created .claude/settings.json" -ForegroundColor Green
        return
    }

    $obj = Read-Json -FilePath $SettingsPath

    if (-not (Test-PSObjectHasProperty $obj 'permissions')) {
        Add-Member -InputObject $obj -NotePropertyName 'permissions' -NotePropertyValue ([PSCustomObject]@{})
    }
    $perms = $obj.permissions

    if (-not (Test-PSObjectHasProperty $perms 'deny')) {
        Add-Member -InputObject $perms -NotePropertyName 'deny' -NotePropertyValue @()
    }

    # @() forces array form even if deny was a scalar string in the source JSON.
    $existing = @($perms.deny)

    # Remove any entry this script formerly managed but no longer wants denied.
    $pruned = @()
    $existing = @($existing | Where-Object {
        if ($ManagedDenyEntries -contains $_) {
            if ($DenyEntries -contains $_) { $true } else { $pruned += $_; $false }
        } else { $true }
    })

    # Add any newly required entry not already present.
    $added = @()
    foreach ($entry in $DenyEntries) {
        if ($existing -notcontains $entry) {
            $existing += $entry
            $added += $entry
        }
    }
    # Assign back as a real array (force with @() to defeat single-elem unwrap).
    $perms.deny = @($existing)

    if ($added.Count -eq 0 -and $pruned.Count -eq 0) {
        Write-Host "[=] .claude/settings.json deny already up to date" -ForegroundColor DarkGray
    } else {
        Write-JsonFile -FilePath $SettingsPath -Object $obj
        $parts = @()
        if ($added.Count -gt 0) { $parts += "added: $($added -join ', ')" }
        if ($pruned.Count -gt 0) { $parts += "removed: $($pruned -join ', ')" }
        Write-Host "[~] Updated .claude/settings.json ($($parts -join '; '))" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# settings.json -- remove
# ---------------------------------------------------------------------------

function Remove-Settings {
    if (-not (Test-Path -LiteralPath $SettingsPath)) {
        Write-Host "[=] .claude/settings.json does not exist, nothing to remove" -ForegroundColor DarkGray
        return
    }

    $obj = Read-Json -FilePath $SettingsPath

    if (-not (Test-PSObjectHasProperty $obj 'permissions')) {
        Write-Host "[=] .claude/settings.json has no permissions key, nothing to remove" -ForegroundColor DarkGray
        return
    }
    $perms = $obj.permissions
    if (-not (Test-PSObjectHasProperty $perms 'deny')) {
        Write-Host "[=] .claude/settings.json has no permissions.deny, nothing to remove" -ForegroundColor DarkGray
        return
    }

    $existing = @($perms.deny)
    $filtered = @($existing | Where-Object { $DenyEntries -notcontains $_ })

    if ($filtered.Count -eq $existing.Count) {
        Write-Host "[=] .claude/settings.json deny contained none of the managed entries" -ForegroundColor DarkGray
        return
    }

    if ($filtered.Count -eq 0) {
        $perms.PSObject.Properties.Remove('deny')
    } else {
        $perms.deny = @($filtered)
    }

    $permsRemainingNotes = @($perms.PSObject.Properties | Where-Object { $_.MemberType -eq 'NoteProperty' })
    if ($permsRemainingNotes.Count -eq 0) {
        $obj.PSObject.Properties.Remove('permissions')
    }

    $objRemainingNotes = @($obj.PSObject.Properties | Where-Object { $_.MemberType -eq 'NoteProperty' })
    if ($objRemainingNotes.Count -eq 0) {
        Remove-Item -LiteralPath $SettingsPath -Force
        Write-Host "[-] Removed empty .claude/settings.json" -ForegroundColor Yellow
    } else {
        Write-JsonFile -FilePath $SettingsPath -Object $obj
        Write-Host "[~] Updated .claude/settings.json (removed managed deny entries)" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# CLAUDE.md -- apply
# ---------------------------------------------------------------------------

function Apply-ClaudeMd {
    if (-not (Test-Path -LiteralPath $ClaudeMdPath)) {
        Write-Utf8NoBom -FilePath $ClaudeMdPath -Content $ClaudeMdBlock
        Write-Host "[+] Created CLAUDE.md" -ForegroundColor Green
        return
    }

    $content = [System.IO.File]::ReadAllText($ClaudeMdPath)

    $beginIdx = $content.IndexOf($BeginMarker)
    $endIdx   = $content.IndexOf($EndMarker)

    if ($beginIdx -ge 0 -and $endIdx -gt $beginIdx) {
        $endLen   = $EndMarker.Length
        $existing = $content.Substring($beginIdx, ($endIdx + $endLen) - $beginIdx)
        if ($existing -eq $ClaudeMdBlock) {
            Write-Host "[=] CLAUDE.md local-mcp block already up to date" -ForegroundColor DarkGray
            return
        }
        $newContent = $content.Substring(0, $beginIdx) + $ClaudeMdBlock + $content.Substring($endIdx + $endLen)
        Write-Utf8NoBom -FilePath $ClaudeMdPath -Content $newContent
        Write-Host "[~] Updated CLAUDE.md (replaced existing local-mcp block)" -ForegroundColor Yellow
        return
    }

    $separator = ''
    if ($content.Length -gt 0 -and -not $content.EndsWith("`n")) { $separator = "`r`n`r`n" }
    elseif ($content.Length -gt 0) { $separator = "`r`n" }
    $newContent = $content + $separator + $ClaudeMdBlock
    Write-Utf8NoBom -FilePath $ClaudeMdPath -Content $newContent
    Write-Host "[~] Updated CLAUDE.md (appended local-mcp block)" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# CLAUDE.md -- remove
# ---------------------------------------------------------------------------

function Remove-ClaudeMd {
    if (-not (Test-Path -LiteralPath $ClaudeMdPath)) {
        Write-Host "[=] CLAUDE.md does not exist, nothing to remove" -ForegroundColor DarkGray
        return
    }

    $content = [System.IO.File]::ReadAllText($ClaudeMdPath)
    $beginIdx = $content.IndexOf($BeginMarker)
    $endIdx   = $content.IndexOf($EndMarker)

    if ($beginIdx -lt 0 -or $endIdx -le $beginIdx) {
        Write-Host "[=] CLAUDE.md has no local-mcp block, nothing to remove" -ForegroundColor DarkGray
        return
    }

    $endLen = $EndMarker.Length
    $before = $content.Substring(0, $beginIdx).TrimEnd("`r", "`n")
    $after  = $content.Substring($endIdx + $endLen).TrimStart("`r", "`n")

    if ($before.Length -gt 0 -and $after.Length -gt 0) {
        $newContent = $before + "`r`n`r`n" + $after
    } else {
        $newContent = $before + $after
    }

    if ([string]::IsNullOrWhiteSpace($newContent)) {
        Remove-Item -LiteralPath $ClaudeMdPath -Force
        Write-Host "[-] Removed empty CLAUDE.md" -ForegroundColor Yellow
    } else {
        Write-Utf8NoBom -FilePath $ClaudeMdPath -Content $newContent
        Write-Host "[~] Updated CLAUDE.md (removed local-mcp block)" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

try {
    if ($Remove) {
        Remove-Settings
        Remove-ClaudeMd
        Write-Host "Done -- local-mcp configuration removed from project." -ForegroundColor Cyan
    } else {
        Apply-Settings
        Apply-ClaudeMd
        Write-Host "Done -- restart Claude Code in this project for settings to take effect." -ForegroundColor Cyan
    }
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}
