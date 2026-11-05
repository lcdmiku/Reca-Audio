# pull_from_server.ps1 - pull back files edited directly on the server
#
#   .\runs\scripts\pull_from_server.ps1 -Files src/reca_audio/train/train.py
#   .\runs\scripts\pull_from_server.ps1 -Files a.py,b.py -ShowDiff
#
# Usage rule: after any server-side hotfix, pull it back immediately, review the
# diff and commit; otherwise the next deploy silently overwrites it (or the local
# copy gets patched twice, which already corrupted api.py once).
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Files,
    [switch]$ShowDiff,
    [string]$HostName = "210.28.133.13",
    [int]$Port = 22869,
    [string]$User = "linchudeng",
    [string]$RemoteRoot = "/home/linchudeng/Reca-Audio"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$stage = "${User}@${HostName}"
$failed = @()

foreach ($f in $Files) {
    $rel = $f -replace '\\', '/'
    $remote = "$RemoteRoot/$rel"
    $local = Join-Path $Root ($rel -replace '/', '\')
    $dir = Split-Path $local -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    scp -P $Port "${stage}:$remote" $local
    if ($LASTEXITCODE -ne 0) { $failed += $rel; continue }
    Write-Host "[pull] $rel"
}

if ($failed.Count -gt 0) {
    Write-Host "[pull] failed, use the ssh_download tool instead:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" }
    exit 1
}

$diff = git status --porcelain -- $Files
if ($diff) {
    Write-Host ""
    Write-Host "[pull] local changes detected, commit them now:" -ForegroundColor Yellow
    git diff --stat -- $Files
    if ($ShowDiff) { git diff -- $Files }
} else {
    Write-Host "[pull] identical to local, nothing to commit"
}
