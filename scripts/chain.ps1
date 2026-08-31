# Run the marathon once the run in flight finishes, without a human in between.
#
# This is deliberately a detached process rather than something the interactive
# session holds: if the terminal closes, a session-owned waiter dies with it and
# nothing ever launches. Started with Start-Process it outlives the session.
#
# A lock file makes a double launch impossible -- two watchdogs driving the same
# done-markers would race on the GPU and on state.json.
#
#   powershell -ExecutionPolicy Bypass -File scripts\chain.ps1 `
#       -WaitFor runs\recheck.out -Script scripts\marathon.py -Hours 8

param(
  [string]$WaitFor = "runs\recheck.out",
  [string]$Marker  = "watchdog done",
  [string]$Script  = "scripts\marathon.py",
  [double]$Hours   = 8.0,
  [double]$MaxWaitHours = 6.0
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$log  = Join-Path $root "runs\chain.log"
$lock = Join-Path $root "runs\chain.lock"

function Say($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  Write-Output $line
  Add-Content -Path $log -Value $line -Encoding utf8
}

if (Test-Path $lock) {
  Say "another chain holds $lock; exiting rather than racing it"
  exit 0
}
"$PID $(Get-Date -Format o)" | Set-Content $lock -Encoding utf8

try {
  Say "waiting for '$Marker' in $WaitFor (max $MaxWaitHours h)"
  $deadline = (Get-Date).AddHours($MaxWaitHours)
  while ($true) {
    if (Test-Path $WaitFor) {
      $hit = Select-String -Path $WaitFor -Pattern $Marker -SimpleMatch -Quiet
      if ($hit) { Say "predecessor finished"; break }
    }
    if ((Get-Date) -ge $deadline) {
      Say "waited $MaxWaitHours h with no marker; launching anyway"
      break
    }
    Start-Sleep -Seconds 60
  }

  # A predecessor that exited without writing its marker may still be shutting
  # down; a couple of minutes costs nothing against an 8 hour run.
  Start-Sleep -Seconds 120

  $out = Join-Path $root "runs\marathon.out"
  $err = Join-Path $root "runs\marathon.err"
  Say "launching $Script for $Hours h"
  & powershell -ExecutionPolicy Bypass -File (Join-Path $root "scripts\overnight.ps1") `
      -Script $Script -Hours $Hours *> $out 2> $err
  Say "$Script returned rc=$LASTEXITCODE"
}
finally {
  Remove-Item $lock -Force -ErrorAction SilentlyContinue
  Say "chain done"
}
