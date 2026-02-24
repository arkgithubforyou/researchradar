# Watchdog script for the full-text extraction pipeline.
# Phase 1: Runs PyMuPDF extraction (if not already done)
# Phase 2: Pulls GROBID Docker image, runs GROBID extraction
# Auto-restarts on crash. Safe to leave running unattended.
#
# Usage (run in PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\watchdog.ps1
#
# Or to run in background:
#   Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File scripts\watchdog.ps1" -WindowStyle Hidden

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

$LogFile = "data\raw\watchdog.log"
$MaxRetries = 20
$RetryDelay = 60
$DockerPath = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "Watchdog started (PID=$PID)"
Log "Working directory: $ProjectDir"

# ── Phase 1: PyMuPDF extraction (skip if already done) ───────────────
$progress = Get-Content "data\raw\fulltext_progress.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
$dlCount = if ($progress.downloaded) { $progress.downloaded.Count } else { 0 }
$exCount = if ($progress.extracted) { $progress.extracted.Count } else { 0 }
$feCount = if ($progress.failed_extract) { $progress.failed_extract.Count } else { 0 }

if ($dlCount -eq 0 -or ($exCount + $feCount) -lt $dlCount) {
    Log "PyMuPDF phase incomplete (dl=$dlCount ex=$exCount fe=$feCount). Running..."

    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        Log "=== PyMuPDF attempt $attempt/$MaxRetries ==="

        $proc = Start-Process -FilePath "python" -ArgumentList "scripts/extract_fulltext.py --workers 3 --delay 0.3" `
            -NoNewWindow -PassThru -RedirectStandardOutput "data\raw\extract_stdout.log" `
            -RedirectStandardError "data\raw\extract_stderr.log"

        Log "PyMuPDF pipeline started (PID=$($proc.Id))"
        $proc.WaitForExit()

        if ($proc.ExitCode -eq 0) {
            Log "PyMuPDF pipeline completed successfully!"
            break
        } else {
            Log "PyMuPDF pipeline exited with code $($proc.ExitCode)"
            if ($attempt -lt $MaxRetries) {
                Log "Restarting in $RetryDelay seconds..."
                Start-Sleep -Seconds $RetryDelay
            }
        }
    }
} else {
    Log "PyMuPDF phase already complete (dl=$dlCount ex=$exCount fe=$feCount). Skipping."
}

# Print PyMuPDF summary
Log "=== PyMuPDF results ==="
$progress = Get-Content "data\raw\fulltext_progress.json" | ConvertFrom-Json
Log "Downloaded: $($progress.downloaded.Count)"
Log "Extracted: $($progress.extracted.Count)"
Log "Failed DL: $($progress.failed_download.Count)"
Log "Failed Extract: $($progress.failed_extract.Count)"

# ── Phase 2: GROBID extraction ───────────────────────────────────────
Log "=== Starting GROBID phase ==="

# Check Docker is working
Log "Checking Docker..."
$dockerOk = $false
for ($i = 1; $i -le 12; $i++) {
    try {
        $result = & $DockerPath version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Log "Docker is ready"
            $dockerOk = $true
            break
        }
    } catch {}
    Log "Docker not ready (attempt $i/12), waiting 10s..."
    Start-Sleep -Seconds 10
}

if (-not $dockerOk) {
    Log "ERROR: Docker not available. Cannot run GROBID. Exiting."
    exit 1
}

# Pull GROBID image
Log "Pulling GROBID Docker image (may take a few minutes)..."
& $DockerPath pull lfoppiano/grobid:0.8.1 2>&1 | ForEach-Object { Log $_ }

# Start GROBID container
Log "Starting GROBID container..."
& $DockerPath rm -f grobid 2>$null
& $DockerPath run -d --name grobid -p 8070:8070 --memory 4g lfoppiano/grobid:0.8.1 2>&1 | ForEach-Object { Log $_ }

# Wait for GROBID to be ready
Log "Waiting for GROBID to start..."
$grobidReady = $false
for ($i = 1; $i -le 60; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8070/api/isalive" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Log "GROBID ready after $($i * 5) seconds"
            $grobidReady = $true
            break
        }
    } catch {}
    Start-Sleep -Seconds 5
}

if (-not $grobidReady) {
    Log "ERROR: GROBID failed to start. Exiting."
    exit 1
}

# Run GROBID extraction with auto-restart
for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
    Log "=== GROBID extraction attempt $attempt/$MaxRetries ==="

    # Make sure GROBID container is running
    & $DockerPath start grobid 2>$null
    Start-Sleep -Seconds 5

    $proc = Start-Process -FilePath "python" -ArgumentList "scripts/fulltext_pipeline.py --skip-download --workers 1" `
        -NoNewWindow -PassThru -RedirectStandardOutput "data\raw\grobid_stdout.log" `
        -RedirectStandardError "data\raw\grobid_stderr.log"

    Log "GROBID pipeline started (PID=$($proc.Id))"
    $proc.WaitForExit()

    if ($proc.ExitCode -eq 0) {
        Log "GROBID pipeline completed successfully!"
        break
    } else {
        Log "GROBID pipeline exited with code $($proc.ExitCode)"
        if ($attempt -lt $MaxRetries) {
            Log "Restarting in $RetryDelay seconds..."
            Start-Sleep -Seconds $RetryDelay
        }
    }
}

# Stop GROBID
Log "Stopping GROBID container..."
& $DockerPath stop grobid 2>$null

Log "=== All phases complete ==="
Log "Full-text extraction pipeline finished. Ready for re-chunking and redeployment."
