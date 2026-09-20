<#
.SYNOPSIS
    Install the market-open watcher into this user's Startup folder. Needs no
    administrator rights.

.DESCRIPTION
    The Scheduled Task route (scripts\install-schedule.ps1) is tidier, but
    registering a task requires elevation on this machine. This does the same
    job from the user's Startup folder, which does not:

        logon -> pythonw -m fable_bot.cli watch -> sleeps until each 09:30 ET
                 open, runs one cycle, sleeps again

    Nothing is lost by doing it this way. Driving TradingView Desktop needs a
    logged-in interactive session regardless, so a process living in that
    session has exactly the same availability a task would.

    The watcher is idle almost all the time -- it sleeps between opens -- and
    the runner's own journal keeps it from trading a session twice if it is
    restarted mid-day.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1 -StartNow
    powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [double]$LeadMinutes = 10,
    [switch]$DryRun,
    [switch]$StartNow,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'

$scriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot  = Split-Path -Parent $scriptDir
$startupDir   = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupDir 'FableTradingBot-Watcher.lnk'

if ($Uninstall) {
    if (Test-Path $shortcutPath) {
        Remove-Item $shortcutPath -Force
        Write-Host "Removed $shortcutPath"
    } else {
        Write-Host "No startup entry found at $shortcutPath"
    }
    $running = Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
        Where-Object { $_.CommandLine -like '*fable_bot.cli watch*' }
    if ($running) {
        $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        Write-Host "Stopped $($running.Count) running watcher process(es)."
    }
    exit 0
}

# pythonw.exe, not python.exe: this runs from logon to logoff and must not own
# a console window. Everything it does goes to logs\ anyway.
$python = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $python)) {
    throw "No virtualenv interpreter at $python. Create it (python -m venv .venv) and install requirements.txt first."
}

$arguments = "-m fable_bot.cli watch --lead $LeadMinutes"
if ($DryRun) { $arguments += ' --dry-run' }

$shell    = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath       = $python
$shortcut.Arguments        = $arguments
$shortcut.WorkingDirectory = $projectRoot
$shortcut.WindowStyle      = 7  # minimised
$shortcut.Description      = 'Fable trading bot -- runs one cycle at each US market open'
$shortcut.Save()

Write-Host ""
Write-Host "Installed the market-open watcher" -ForegroundColor Green
Write-Host "  Shortcut : $shortcutPath"
Write-Host "  Runs     : $python $arguments"
Write-Host "  Workdir  : $projectRoot"
Write-Host "  Starts   : automatically at every logon"
if ($DryRun) {
    Write-Host ""
    Write-Host "  DRY RUN MODE: it will not place orders." -ForegroundColor Yellow
}

if ($StartNow) {
    $already = Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
        Where-Object { $_.CommandLine -like '*fable_bot.cli watch*' }
    if ($already) {
        Write-Host ""
        Write-Host "  Already running (PID $($already.ProcessId)); leaving it alone." -ForegroundColor Yellow
    } else {
        Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden
        Start-Sleep -Seconds 3
        $proc = Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
            Where-Object { $_.CommandLine -like '*fable_bot.cli watch*' }
        if ($proc) {
            Write-Host ""
            Write-Host "  Started now, PID $($proc.ProcessId)" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "  Failed to start -- check logs\fable-bot.log" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "Check it:   Get-CimInstance Win32_Process -Filter `"Name='pythonw.exe'`" | Where-Object { `$_.CommandLine -like '*fable_bot*' }"
Write-Host "Logs:       $projectRoot\logs\fable-bot.log"
Write-Host "Remove:     powershell -File scripts\install-startup.ps1 -Uninstall"
Write-Host ""
