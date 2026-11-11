# One full real night (wake -> sleep -> gates -> recall) with stacked experts, about 20-30 minutes.
# Run from the project folder in your own PowerShell window:
#     powershell -ExecutionPolicy Bypass -File scripts\run_real_night.ps1
# Output: models\real-night.log (progress, gates), models\real-night-memory.log (RAM every 15 s),
# models\real-night\night-summary.json and the dream report under models\real-night\sleeps\.
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)
.\.venv-gpu\Scripts\python.exe -u scripts\real_night.py --facts 20 --server-teacher --trainer experts 2>&1 |
  Where-Object { $_ -notmatch "Loading weights|it/s\]" } | Tee-Object -FilePath models\real-night.log
