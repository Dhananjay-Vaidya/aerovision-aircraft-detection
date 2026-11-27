# Final evaluation (Part B) for the models given as key=weights pairs; a failing step is logged and skipped.
#   powershell -ExecutionPolicy Bypass -File run_final_eval.ps1 yolov8s=C:\...\best.pt yolo26s=C:\...\best.pt
# Logs: runs\final_eval_logs\<key>.<step>.log ; status lines: runs\final_eval_logs\status.txt
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\venv\Scripts\python.exe"
$logs = "runs\final_eval_logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$status = "$logs\status.txt"
# Keep Windows from sleeping while this script runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); released when it exits.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

function Step($key, $step, [string[]]$cmd) {
  $t0 = Get-Date
  & $py @cmd *> "$logs\$key.$step.log"
  $code = $LASTEXITCODE
  $line = "{0:u}  {1,-16} {2,-12} exit {3}  ({4:n1} min)" -f (Get-Date), $key, $step, $code, ((Get-Date) - $t0).TotalMinutes
  Add-Content $status $line
  Write-Output $line
}

foreach ($pair in $args) {
  $key, $w = $pair -split "=", 2
  if (-not (Test-Path $w)) { Add-Content $status "$key  MISSING weights $w"; continue }
  Step $key "eval_clean" @("eval_test.py", "--weights", $w, "--name", $key, "--split-set", "clean")
  Step $key "eval_full"  @("eval_test.py", "--weights", $w, "--name", $key, "--split-set", "full")
  Step $key "aircraft_c" @("evaluate.py", "--weights", $w, "--name", $key)
  Step $key "abstain"    @("abstain.py", "--weights", $w, "--name", $key, "--risk-on", "all", "matched",
                           "--target-risk", "0.05", "0.10")
  # Speed: 3 runs of the same protocol (laptop GPU clocks vary by up to ~50 % between runs); each run is appended
  # to results\speed_rounds.csv with its round number, and make_paper_results.py reports the median.
  foreach ($r in 1..3) {
    $tmp = "runs\speed_tmp_${key}_r$r.csv"
    Remove-Item $tmp -ErrorAction SilentlyContinue
    Step $key "speed_r$r" @("speed.py", "--weights", $w, "--out", $tmp, "--warmup", "50", "--runs", "300", "--replace")
    & $py -c "import pandas as pd,sys,os; d=pd.read_csv(sys.argv[1]); d.insert(0,'round',int(sys.argv[2])); f='results/speed_rounds.csv'; old=pd.read_csv(f) if os.path.exists(f) else None; old=old[~((old['model']==d['model'].iloc[0])&(old['round']==int(sys.argv[2])))] if old is not None else None; pd.concat([x for x in (old,d) if x is not None]).to_csv(f,index=False)" $tmp $r
  }
}
Add-Content $status "ALL DONE"
