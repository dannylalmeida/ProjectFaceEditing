param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$styleclipRoot = Join-Path $projectRoot "third_party\StyleCLIP"
$matplotlibCache = Join-Path $projectRoot ".cache\matplotlib_styleclip"

if (-not (Test-Path $styleclipRoot)) {
    Write-Error "StyleCLIP nao encontrado em $styleclipRoot"
    exit 1
}

if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
    Write-Host "Uso:"
    Write-Host "  .\scripts\run_styleclip_repo.ps1 optimization\run_optimization.py --help"
    Write-Host "  .\scripts\run_styleclip_repo.ps1 mapper\scripts\inference.py --help"
    exit 1
}

New-Item -ItemType Directory -Force $matplotlibCache | Out-Null

$env:PYTHONPATH = $styleclipRoot
$env:USERPROFILE = $projectRoot
$env:MPLCONFIGDIR = $matplotlibCache

$scriptPath = $CommandArgs[0]
$scriptArgs = @()
if ($CommandArgs.Count -gt 1) {
    $scriptArgs = $CommandArgs[1..($CommandArgs.Count - 1)]
}

Push-Location $styleclipRoot
try {
    & conda run -n styleclip python $scriptPath @scriptArgs
}
finally {
    Pop-Location
}
