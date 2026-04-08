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

$DenyEntries  = @('Edit', 'Write')
$BeginMarker  = '<!-- BEGIN local-mcp -->'
$EndMarker    = '<!-- END local-mcp -->'

$ClaudeMdBlock = @"
$BeginMarker
## Use of the local-mcp server

For code file modifications, ALWAYS use the tools from the ``local-mcp`` server:
- ``local_edit`` instead of ``Edit`` (existing files)
- ``local_write`` instead of ``Write`` (new files)
- ``local_snippet`` only as a fallback for snippets with no file destination

Reason: token savings -- file contents never pass through Claude's context.
The built-in ``Edit`` and ``Write`` tools are denied via ``.claude/settings.json`` to enforce this workflow.

### Trust local_edit, do not re-read files to verify

``local_edit`` is transactional: guard-rails run server-side, writes are
atomic, and on failure all successful writes are reverted from captured
original bytes. When the summary begins with ``[qwen3-coder:30b]`` and lists
modified/deleted files, trust it and move on. Do NOT ``Read`` the file
afterwards to "verify"; that re-ingests the content into Claude's context
and defeats the token-saving goal. Read a file ONLY when:
- ``local_edit`` returned a line starting with ``REJECTED`` or ``Error``, or
- you genuinely need the current contents for a subsequent, unrelated step.

### Phrasing for removals

``local_edit`` distinguishes in-file code removal from whole-file deletion:
- To remove a method / class / field / block inside a file, phrase it
  *without* the word "file". Examples: "remove method foo", "strip unused
  imports", "delete the password field".
- To delete an entire file, phrase it *with* the word "file". Examples:
  "delete the file Foo.java", "rimuovi il file Bar.py". The server will
  refuse a ``<delete/>`` block if this exact pattern is missing.
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
    $added = @()
    foreach ($entry in $DenyEntries) {
        if ($existing -notcontains $entry) {
            $existing += $entry
            $added += $entry
        }
    }
    # Assign back as a real array (force with @() to defeat single-elem unwrap).
    $perms.deny = @($existing)

    if ($added.Count -eq 0) {
        Write-Host "[=] .claude/settings.json already has Edit/Write in deny" -ForegroundColor DarkGray
    } else {
        Write-JsonFile -FilePath $SettingsPath -Object $obj
        Write-Host "[~] Updated .claude/settings.json (added: $($added -join ', '))" -ForegroundColor Yellow
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
        Write-Host "[=] .claude/settings.json deny did not contain Edit/Write" -ForegroundColor DarkGray
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
        Write-Host "[~] Updated .claude/settings.json (removed Edit/Write from deny)" -ForegroundColor Yellow
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
