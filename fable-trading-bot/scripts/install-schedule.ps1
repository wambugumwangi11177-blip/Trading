<#
.SYNOPSIS
    Register (or remove) the Windows Scheduled Task that runs the bot at the US
    market open.

.DESCRIPTION
    The task fires on weekday mornings a few minutes before the earliest
    possible opening bell and hands over to `fable_bot.cli auto-run`, which owns
    every decision from there: whether the NYSE is open today, how long to wait
    for the real bell, whether TradingView needs launching, and whether this
    session has already been traded.

    Why the trigger is early and fixed:
      This machine runs on UTC+3 (Nairobi), which has no daylight saving. New
      York does. So 09:30 ET lands at 16:30 local in the US summer and 17:30
      local in the US winter. Rather than re-registering the task twice a year,
      it fires at 16:20 local year-round and the runner sleeps until the true
      open -- ten minutes in summer, seventy in winter.

    The task runs as the interactive logged-on user by design. TradingView
    Desktop is a GUI application; a task running in session 0 with no desktop
    cannot drive it. That means the machine has to be powered on and this user
    logged in when the trigger fires.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-schedule.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install-schedule.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\install-schedule.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'FableTradingBot-MarketOpen',
    [string]$Time     = '16:20',
    [switch]$DryRun,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        Write-Host "No task named '$TaskName' is registered."
    } else {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    }
    exit 0
}

# pythonw.exe rather than python.exe: the task runs unattended and a console
# window popping up every weekday (and staying up through the wait for the
# bell) is not acceptable. All output goes to logs/ regardless.
$python = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $python)) {
    throw "No virtualenv interpreter at $python. Create it first (python -m venv .venv) and install requirements.txt."
}

$arguments = '-m fable_bot.cli auto-run'
if ($DryRun) { $arguments += ' --dry-run' }

$action = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $projectRoot

$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $Time

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Interactive: the task needs a desktop to drive TradingView. It therefore only
# runs while this user is logged on -- which is the honest constraint, not a
# limitation to work around with S4U.
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited

$description = @"
Runs the Fable trading bot one cycle at the US market open.
Fires at $Time local; the runner waits for the true 09:30 ET bell (which moves
against local time twice a year), skips NYSE holidays, launches TradingView
Desktop if it is closed, and refuses to trade a session twice.
Logs: $projectRoot\logs\
"@

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Description $description -Force | Out-Null

$task = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName

Write-Host ""
Write-Host "Registered '$TaskName'" -ForegroundColor Green
Write-Host "  Runs      : $python $arguments"
Write-Host "  Workdir   : $projectRoot"
Write-Host "  Trigger   : Mon-Fri at $Time local time"
Write-Host "  Next run  : $($info.NextRunTime)"
Write-Host "  State     : $($task.State)"
if ($DryRun) {
    Write-Host ""
    Write-Host "  DRY RUN MODE: this task will not place orders." -ForegroundColor Yellow
    Write-Host "  Re-run this script without -DryRun to arm it." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Run it now:      Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Check result:    Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Host "Remove:          powershell -File scripts\install-schedule.ps1 -Uninstall"
Write-Host ""
