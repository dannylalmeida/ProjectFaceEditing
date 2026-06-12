param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$globalTorchRoot = Join-Path $projectRoot "third_party\StyleCLIP\global_torch"
$matplotlibCache = Join-Path $projectRoot ".cache\matplotlib_styleclip"

if (-not (Test-Path $globalTorchRoot)) {
    Write-Error "StyleCLIP global_torch nao encontrado em $globalTorchRoot"
    exit 1
}

if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
    Write-Host "Uso:"
    Write-Host "  .\scripts\run_styleclip_global_torch.ps1 StyleCLIP.py"
    Write-Host "  .\scripts\run_styleclip_global_torch.ps1 manipulate.py"
    exit 1
}

New-Item -ItemType Directory -Force $matplotlibCache | Out-Null

$env:USERPROFILE = $projectRoot
$env:MPLCONFIGDIR = $matplotlibCache

$scriptPath = $CommandArgs[0]
$scriptArgs = @()
if ($CommandArgs.Count -gt 1) {
    $scriptArgs = $CommandArgs[1..($CommandArgs.Count - 1)]
}

Push-Location $globalTorchRoot
try {
    & conda run -n styleclip python $scriptPath @scriptArgs
}
finally {
    Pop-Location
}
