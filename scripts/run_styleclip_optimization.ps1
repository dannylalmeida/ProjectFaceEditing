param(
    [string]$Description = "",
    [string[]]$Preset = @(),
    [switch]$ListPresets,
    [ValidateSet("free_generation", "edit")]
    [string]$Mode = "free_generation",
    [int]$Step = 10,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$CondaEnv = "styleclip",
    [string]$ResultsDir = "",
    [string]$LatentPath = "",
    [string]$SourceDescription = "a person",
    [string]$TargetDescription = "",
    [ValidateSet("absolute", "directional")]
    [string]$ClipLossType = "absolute",
    [double]$L2Lambda = 0.008,
    [ValidateSet("sum", "mean")]
    [string]$L2Reduction = "sum",
    [double]$LearningRate = 0.1,
    [double]$ClipLambda = 1.0,
    [double]$EditStrength = 0.0,
    [int]$LatentLayerMin = 0,
    [int]$LatentLayerMax = 17,
    [double]$MaxLatentDelta = 0.0
)

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "styleclip_prompt_presets.ps1")
$styleclipRoot = Join-Path $projectRoot "third_party\StyleCLIP"
$checkpointPath = Join-Path $styleclipRoot "pretrained_models\stylegan2-ffhq-config-f.pt"
$matplotlibCache = Join-Path $projectRoot ".cache\matplotlib_styleclip"

if ($ListPresets) {
    Show-StyleClipPromptPresets
    exit 0
}

if ($TargetDescription) {
    $resolvedDescription = Resolve-StyleClipPrompt -Description $TargetDescription -Preset @()
}
else {
    $resolvedDescription = Resolve-StyleClipPrompt -Description $Description -Preset $Preset
}

if (-not (Test-Path $styleclipRoot)) {
    Write-Error "StyleCLIP nao encontrado em $styleclipRoot"
    exit 1
}

if (-not (Test-Path $checkpointPath)) {
    Write-Error "Checkpoint StyleGAN2 nao encontrado em $checkpointPath"
    exit 1
}

if (-not $ResultsDir) {
    $ResultsDir = Join-Path $projectRoot "outputs\styleclip_optimization"
}

if ($ResultsDir) {
    $resolvedResultsDir = Resolve-Path -LiteralPath $ResultsDir -ErrorAction SilentlyContinue
    if ($resolvedResultsDir) {
        $ResultsDir = $resolvedResultsDir.Path
    }
    else {
        $ResultsDir = [System.IO.Path]::GetFullPath($ResultsDir)
    }
}

if ($LatentPath) {
    $LatentPath = (Resolve-Path -LiteralPath $LatentPath).Path
}

New-Item -ItemType Directory -Force $ResultsDir | Out-Null
New-Item -ItemType Directory -Force $matplotlibCache | Out-Null

$env:PYTHONPATH = $styleclipRoot
$env:USERPROFILE = $projectRoot
$env:MPLCONFIGDIR = $matplotlibCache

$resolvedDevice = $Device
if ($Device -eq "auto") {
    $resolvedDevice = (& conda run -n $CondaEnv python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')").Trim()
    if (-not $resolvedDevice) {
        $resolvedDevice = "cpu"
    }
}

$promptInfo = @{
    input_description = $Description
    input_preset = $Preset
    source_description_en = $SourceDescription
    target_description_input = $TargetDescription
    resolved_description_en = $resolvedDescription
    mode = $Mode
    step = $Step
    device = $resolvedDevice
    latent_path = $LatentPath
    clip_loss_type = $ClipLossType
    l2_lambda = $L2Lambda
    l2_reduction = $L2Reduction
    learning_rate = $LearningRate
    clip_lambda = $ClipLambda
    edit_strength = $EditStrength
    latent_layer_min = $LatentLayerMin
    latent_layer_max = $LatentLayerMax
    max_latent_delta = $MaxLatentDelta
}
$promptInfo | ConvertTo-Json | Set-Content -Path (Join-Path $ResultsDir "prompt_info.json") -Encoding UTF8

$commandArgs = @(
    "optimization\run_optimization.py",
    "--description", $resolvedDescription,
    "--mode", $Mode,
    "--step", $Step,
    "--device", $resolvedDevice,
    "--ckpt", $checkpointPath,
    "--results_dir", $ResultsDir,
    "--source_description", $SourceDescription,
    "--clip_loss_type", $ClipLossType,
    "--l2_lambda", $L2Lambda,
    "--l2_reduction", $L2Reduction,
    "--lr", $LearningRate,
    "--clip_lambda", $ClipLambda,
    "--edit_strength", $EditStrength,
    "--latent_layer_min", $LatentLayerMin,
    "--latent_layer_max", $LatentLayerMax,
    "--max_latent_delta", $MaxLatentDelta,
    "--save_intermediate_image_every", "0"
)

if ($LatentPath) {
    $commandArgs += @("--latent_path", $LatentPath)
}

Push-Location $styleclipRoot
try {
    & conda run -n $CondaEnv python @commandArgs
}
finally {
    Pop-Location
}
