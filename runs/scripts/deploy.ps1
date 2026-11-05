# deploy.ps1 - local packaging and upload (remote verification lives in deploy.sh)
#
# Safeguard 1 lives here: refuse to deploy when the working tree is dirty.
#
#   .\runs\scripts\deploy.ps1                 # check + pack + manifest, print upload steps
#   .\runs\scripts\deploy.ps1 -Push           # also scp + trigger the remote verification
#   .\runs\scripts\deploy.ps1 -AllowDirty     # debugging only
param(
    [switch]$Push,
    [switch]$AllowDirty,
    [string]$HostName = "210.28.133.13",
    [int]$Port = 22869,
    [string]$User = "linchudeng",
    [string]$RemoteRoot = "/home/linchudeng/Reca-Audio",
    [string]$RemoteStage = "/home/linchudeng/deploy"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

# ---------- safeguard 1: clean tree ----------
$dirty = git status --porcelain
if ($dirty -and -not $AllowDirty) {
    Write-Host "[deploy] working tree is dirty, commit before deploying:" -ForegroundColor Red
    $dirty | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }
    exit 1
}
if ($dirty) { Write-Host "[deploy] WARNING: -AllowDirty, remote will run a version absent from git" -ForegroundColor Yellow }

$commit = (git rev-parse --short HEAD).Trim()
$branch = (git rev-parse --abbrev-ref HEAD).Trim()

# ---------- sync scope ----------
$scope = @("src", "Audio-Omni/audio_omni", "runs/scripts", "runs/Benchmark")
$files = git ls-files -- $scope | Where-Object { $_ -and (Test-Path $_) }
if (-not $files) { Write-Host "[deploy] nothing to sync in scope" -ForegroundColor Red; exit 1 }

# ---------- manifest (input of safeguard 2) ----------
$manifest = Join-Path $Root "deploy_manifest.sha256"
$lines = foreach ($f in $files) {
    $h = (Get-FileHash -Algorithm SHA256 -Path (Join-Path $Root $f)).Hash.ToLower()
    "$h  $($f -replace '\\','/')"
}
[IO.File]::WriteAllText($manifest, (($lines -join "`n") + "`n"), (New-Object Text.UTF8Encoding($false)))

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tarName = "deploy_${commit}_${stamp}.tar.gz"
$tarPath = Join-Path $Root $tarName
$listPath = Join-Path $Root "deploy_files.txt"
[IO.File]::WriteAllText($listPath, (($files -join "`n") + "`n"), (New-Object Text.UTF8Encoding($false)))

tar -czf $tarPath -T $listPath
if ($LASTEXITCODE -ne 0) { Write-Host "[deploy] tar failed" -ForegroundColor Red; exit 1 }
Remove-Item $listPath -ErrorAction SilentlyContinue

$sha = (Get-FileHash -Algorithm SHA256 -Path $tarPath).Hash.ToLower().Substring(0, 12)
Write-Host "[deploy] commit=$commit branch=$branch files=$($files.Count) tar=$tarName sha=$sha"

if (-not $Push) {
    Write-Host ""
    Write-Host "Semi-automatic next steps:"
    Write-Host "  1) ssh_upload $tarPath -> $RemoteStage/"
    Write-Host "  2) ssh_upload $manifest -> $RemoteStage/"
    Write-Host "  3) bash $RemoteRoot/runs/scripts/deploy.sh --tar $RemoteStage/$tarName --manifest $RemoteStage/deploy_manifest.sha256 --commit $commit"
    exit 0
}

# ---------- automatic: scp + remote verification ----------
$stage = "${User}@${HostName}"
ssh -p $Port -o BatchMode=yes $stage "mkdir -p $RemoteStage"
if ($LASTEXITCODE -ne 0) { Write-Host "[deploy] ssh failed, use the semi-automatic flow" -ForegroundColor Red; exit 1 }

scp -P $Port $tarPath "${stage}:$RemoteStage/"
scp -P $Port $manifest "${stage}:$RemoteStage/"
if ($LASTEXITCODE -ne 0) { Write-Host "[deploy] scp failed" -ForegroundColor Red; exit 1 }

ssh -p $Port $stage "bash $RemoteRoot/runs/scripts/deploy.sh --tar $RemoteStage/$tarName --manifest $RemoteStage/deploy_manifest.sha256 --commit $commit"
if ($LASTEXITCODE -ne 0) { Write-Host "[deploy] remote verification failed, deployment aborted" -ForegroundColor Red; exit 1 }

Write-Host "[deploy] done: remote is at $commit" -ForegroundColor Green
