# Watchdog for the overnight pipeline.
#
# overnight.py is already resumable -- every stage has a done-marker and is
# skipped once it has succeeded -- so the recovery for anything that kills the
# process (a CUDA fault, an OOM, the box rebooting) is simply to start it again.
# This loop does that until the deadline passes, which is the difference between
# losing one stage and losing the night.
#
# The deadline is absolute and passed through on every restart, so restarting
# never extends the run.
#
#   powershell -ExecutionPolicy Bypass -File scripts\overnight.ps1 -Hours 10

param(
  [string]$Script = "scripts\overnight.py",
  [double]$Hours = 10.0,
  [double]$StartEpoch = 0,
  [string]$ScriptArg = "",
  [int]$MaxRestarts = 40
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = "C:\Users\longd\AppData\Local\Programs\Python\Python313\python.exe"
if ($StartEpoch -le 0) {
  $StartEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}
$deadline = $StartEpoch + $Hours * 3600

New-Item -ItemType Directory -Force -Path "runs" | Out-Null
$wlog = Join-Path $root "runs\watchdog.log"

function Say($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  Write-Output $line
  Add-Content -Path $wlog -Value $line -Encoding utf8
}

Say ("watchdog up; deadline {0}" -f ([DateTimeOffset]::FromUnixTimeSeconds([long]$deadline).LocalDateTime.ToString("HH:mm:ss")))

for ($i = 1; $i -le $MaxRestarts; $i++) {
  $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  if ($now -ge $deadline) { Say "deadline reached; stopping"; break }

  Say ("attempt {0}: launching {1} ({2:N1} h left)" -f $i, $Script, (($deadline - $now) / 3600))
  if ($Script -like "*overnight.py") {
    & $py $Script "--hours" $Hours "--start" $StartEpoch
  } else {
    if ($ScriptArg -ne "") { & $py $Script $Hours $ScriptArg }
    else { & $py $Script $Hours }
  }
  $rc = $LASTEXITCODE

  if ($rc -eq 0) { Say "$Script finished cleanly"; break }

  Say ("$Script exited rc={0}; resuming from its done-markers in 60s" -f $rc)
  Start-Sleep -Seconds 60
}

Say "watchdog done"
