<#
.SYNOPSIS
    Worker executado DENTRO da Tarefa Agendada criada por run_notebook_detached.ps1.
    Nao chamar diretamente -- sem valor fora desse fluxo.

.DESCRIPTION
    Impede o sistema de suspender (SetThreadExecutionState) enquanto o
    nbconvert roda e libera a flag ao terminar. Existe porque duas execucoes
    do 01_bertopic_folha.ipynb (2026-08-28/29) foram mortas sem nenhum evento
    de reboot/sleep/crash no Visualizador de Eventos -- a hipotese que sobrou
    foi o processo estar preso ao ciclo de vida da sessao/terminal que o
    disparou. Rodar via Tarefa Agendada quebra esse vinculo; o
    SetThreadExecutionState cobre o caso adicional de suspensao por
    inatividade (o timeout de suspensao na bateria deste notebook e de so 10
    minutos -- ociosidade de teclado/mouse, que uma computacao em background
    nao evita sozinha).
#>
param(
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$NotebookPath,
    [Parameter(Mandatory = $true)][string]$KernelName,
    [Parameter(Mandatory = $true)][string]$Timeout,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory
)

$sig = @"
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
$power = Add-Type -MemberDefinition $sig -Name "Power" -Namespace "TopicModeling" -PassThru

$ES_CONTINUOUS = [uint32]0x80000000
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$ES_AWAYMODE_REQUIRED = [uint32]0x00000040

Set-Location -Path $WorkingDirectory
"[$(Get-Date -Format o)] worker start, cwd=$WorkingDirectory" | Out-File -FilePath $LogPath -Encoding utf8

$power::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED) | Out-Null
try {
    & $PythonExe -m jupyter nbconvert --to notebook --execute --inplace $NotebookPath `
        "--ExecutePreprocessor.kernel_name=$KernelName" `
        "--ExecutePreprocessor.timeout=$Timeout" *>> $LogPath
    $exitCode = $LASTEXITCODE
}
finally {
    $power::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

"[$(Get-Date -Format o)] worker end, exit=$exitCode" | Out-File -FilePath $LogPath -Encoding utf8 -Append
exit $exitCode
