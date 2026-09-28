<#
.SYNOPSIS
    Roda um notebook via nbconvert desacoplado do terminal/sessao que o disparou,
    usando uma Tarefa Agendada do Windows (execucao unica, imediata).

.DESCRIPTION
    Para runs longos (ex.: sweeps de horas). Uma execucao lancada direto do
    terminal/sessao de chat fica presa ao ciclo de vida dele; se o terminal
    fechar ou a sessao morrer, o processo filho pode morrer junto sem deixar
    rastro no log de eventos do Windows (foi o caso de duas quedas do
    01_bertopic_folha.ipynb em 2026-08-28/29 sem nenhum evento de
    reboot/sleep/crash correspondente). A Tarefa Agendada roda sob o proprio
    servico do Task Scheduler -- sobrevive ao fechamento do terminal, do
    VS Code ou da sessao que a criou. O worker interno tambem inibe suspensao
    do sistema enquanto roda (ver _run_notebook_worker.ps1).

.PARAMETER NotebookRelPath
    Caminho do notebook relativo a 03-topic-modeling/notebooks/, ex.:
    "bertopic/01_bertopic_folha.ipynb".

.PARAMETER KernelName
    Nome do kernel Jupyter registrado (default: topic-modeling).

.PARAMETER Timeout
    --ExecutePreprocessor.timeout em segundos; -1 = sem limite (default).

.EXAMPLE
    .\run_notebook_detached.ps1 -NotebookRelPath "bertopic/01_bertopic_folha.ipynb"
#>
param(
    [Parameter(Mandatory = $true)][string]$NotebookRelPath,
    [string]$KernelName = "topic-modeling",
    [string]$Timeout = "-1"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$notebook = Join-Path $repoRoot "03-topic-modeling\notebooks\$NotebookRelPath"
if (-not (Test-Path $notebook)) {
    throw "Notebook nao encontrado: $notebook"
}

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "venv nao encontrado em: $pythonExe"
}

$workerScript = Join-Path $PSScriptRoot "_run_notebook_worker.ps1"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$safeName = ($NotebookRelPath -replace '[\\/]', '_') -replace '\.ipynb$', ''
$logDir = Join-Path $repoRoot "03-topic-modeling\data\output\_detached_runs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "$safeName`_$stamp.log"

$taskName = "TopicModeling_$safeName`_$stamp"

$workerArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$workerScript`"",
    "-PythonExe", "`"$pythonExe`"",
    "-NotebookPath", "`"$notebook`"",
    "-KernelName", "`"$KernelName`"",
    "-Timeout", "`"$Timeout`"",
    "-LogPath", "`"$logPath`"",
    "-WorkingDirectory", "`"$repoRoot`""
) -join " "

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $workerArgs
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(3)
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Execucao desacoplada (Topic Modeling): $NotebookRelPath" | Out-Null
Start-ScheduledTask -TaskName $taskName

Write-Output "Tarefa criada e iniciada: $taskName"
Write-Output "Log:      $logPath"
Write-Output "Notebook: $notebook"
Write-Output ""
Write-Output "Status:  Get-ScheduledTask -TaskName '$taskName' | Get-ScheduledTaskInfo"
Write-Output "Remover: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
