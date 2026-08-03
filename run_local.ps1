# apex-trader local runner — BACKUP mode only.
#
# Do NOT run this while GitHub Actions is healthy: both runners share one
# state file via git, and overlapping runs can double-trade. Use this only
# when CI is down for an extended period, then hand control back.
#
# Usage (Task Scheduler, every 15 min):
#   powershell -ExecutionPolicy Bypass -File run_local.ps1
#
# Alternative fully-local mode (independent state, no git sync — only when
# CI is disabled):  python apex_trader.py   # built-in loop every CHECK_INTERVAL s

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

git pull --rebase

$env:RUN_ONCE = "true"
python apex_trader.py

git add apex_state.json apex_decisions.jsonl 2>$null
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m ("state: {0} local" -f (Get-Date -Format "yyyy-MM-dd HH:mm"))
    foreach ($i in 1..5) {
        git pull --rebase
        if ($LASTEXITCODE -eq 0) {
            git push
            if ($LASTEXITCODE -eq 0) { break }
        }
        Write-Host "push attempt $i failed; retrying"
        Start-Sleep -Seconds ($i * 5)
    }
}
