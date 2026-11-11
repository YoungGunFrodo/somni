# Runs the forgetting experiments one after another (about 1.5 hours on an RX 6800).
# Run from the project folder in your own PowerShell window:
#     powershell -ExecutionPolicy Bypass -File scripts\run_forgetting_lab.ps1
# Results: models\forgetting-lab\results.jsonl and models\forgetting-lab\lab.log
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv-gpu\Scripts\python.exe"
New-Item -ItemType Directory -Force models\forgetting-lab | Out-Null
$log = "models\forgetting-lab\lab.log"
"=== forgetting lab started $(Get-Date)" | Tee-Object -FilePath $log -Append

$runs = @(
  # name                     extra arguments
  @("A_rehearsal",           "--rehearsal 0.5 --baseline"),
  @("B_rehearsal_lowlr",     "--rehearsal 0.5 --lr 5e-5 --epochs 8"),
  @("C_rehearsal_mlp",       "--rehearsal 0.5 --targets mlp --lr 1e-4 --epochs 6"),
  @("D_lowlr_only",          "--lr 5e-5 --epochs 8"),
  # Gated expert (Google-style new module per night) with all three dream practice types:
  # recall (the facts), stay out (rehearsal + gate penalty), combine (fact inside code).
  @("E_experts_combine",     "--method experts --rehearsal 0.5 --combine --lr 2e-4 --epochs 4"),
  @("F_lora_combine",        "--rehearsal 0.5 --combine --lr 2e-4 --epochs 4")
)
foreach ($r in $runs) {
  "--- $($r[0]) $(Get-Date)" | Tee-Object -FilePath $log -Append
  $cmd = "$py -u scripts\forgetting_lab.py --name $($r[0]) $($r[1])"
  # Exit code 3 = training saved its progress and stopped to protect the PC: continue it.
  for ($try = 0; $try -lt 6; $try++) {
    Invoke-Expression "$cmd 2>&1" | Where-Object { $_ -match "^\[(lab|train|resources)\]|Error|Traceback" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 3) { break }
  }
}
"=== done $(Get-Date)" | Tee-Object -FilePath $log -Append
