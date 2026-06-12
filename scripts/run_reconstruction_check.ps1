param(
    [string]$InputImage = "",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$FaceEnv = "face",
    [string]$StyleclipEnv = "styleclip",
    [int]$RefinarLatentePassos = 15,
    [double]$RefinarLatenteLearningRate = 0.015,
    [double]$RefinarLatenteL2 = 0.0001,
    [string]$OutputDir = ""
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$runInversion = Join-Path $PSScriptRoot "run_retinaface_to_e4e.ps1"
$comparisonScript = Join-Path $PSScriptRoot "create_sweep_comparison.py"

if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "outputs\reconstruction_check"
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}

$e4eRoot = Join-Path $OutputDir "e4e"
$pspRoot = Join-Path $OutputDir "psp"
$reportPath = Join-Path $OutputDir "reconstruction_report.json"
$comparisonPath = Join-Path $OutputDir "reconstruction_comparison.jpg"

New-Item -ItemType Directory -Force $OutputDir | Out-Null

Write-Host "RECONSTRUCAO ONLY: RetinaFace -> crop -> e4e/pSp -> latent -> reconstrucao"
Write-Host "Sem StyleCLIP, sem edicao."
Write-Host ""

& $runInversion `
    -InputImage $InputImage `
    -Device $Device `
    -FaceEnv $FaceEnv `
    -StyleclipEnv $StyleclipEnv `
    -EncoderBackend e4e `
    -ReconstruirPreview `
    -RefinarLatentePassos $RefinarLatentePassos `
    -RefinarLatenteLearningRate $RefinarLatenteLearningRate `
    -RefinarLatenteL2 $RefinarLatenteL2 `
    -OutputDir $e4eRoot
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na reconstrucao e4e."
    exit $LASTEXITCODE
}

& $runInversion `
    -InputImage $InputImage `
    -Device $Device `
    -FaceEnv $FaceEnv `
    -StyleclipEnv $StyleclipEnv `
    -EncoderBackend psp `
    -ReconstruirPreview `
    -RefinarLatentePassos $RefinarLatentePassos `
    -RefinarLatenteLearningRate $RefinarLatenteLearningRate `
    -RefinarLatenteL2 $RefinarLatenteL2 `
    -OutputDir $pspRoot
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na reconstrucao pSp."
    exit $LASTEXITCODE
}

$e4eInversionDir = Join-Path $e4eRoot "02_e4e_inversion"
$pspInversionDir = Join-Path $pspRoot "02_psp_inversion"
$cropPath = Join-Path $pspRoot "01_retinaface_crop\primary_face_crop.png"
$e4eMetadataPath = Join-Path $e4eInversionDir "inversion_metadata.json"
$pspMetadataPath = Join-Path $pspInversionDir "inversion_metadata.json"
$cropMetadataPath = Join-Path $pspRoot "01_retinaface_crop\primary_face.json"

$comparisonArgs = @("--output", $comparisonPath)
if (Test-Path -LiteralPath $cropPath) {
    $comparisonArgs += @("--image", ("original_crop=" + $cropPath))
}

$candidateImages = @(
    @("e4e_latent", (Join-Path $e4eInversionDir "e4e_reconstruction_preview.png")),
    @("e4e_residual", (Join-Path $e4eInversionDir "e4e_identity_restored_reconstruction.png")),
    @("psp_latent", (Join-Path $pspInversionDir "psp_reconstruction_preview.png")),
    @("psp_residual", (Join-Path $pspInversionDir "psp_identity_restored_reconstruction.png"))
)

foreach ($item in $candidateImages) {
    if (Test-Path -LiteralPath $item[1]) {
        $comparisonArgs += @("--image", ($item[0] + "=" + $item[1]))
    }
}

& conda run -n $StyleclipEnv python $comparisonScript @comparisonArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha ao criar comparacao de reconstrucoes."
    exit $LASTEXITCODE
}

$e4eMetadata = $null
$pspMetadata = $null
$cropMetadata = $null
if (Test-Path -LiteralPath $e4eMetadataPath) {
    $e4eMetadata = Get-Content -LiteralPath $e4eMetadataPath -Raw | ConvertFrom-Json
}
if (Test-Path -LiteralPath $pspMetadataPath) {
    $pspMetadata = Get-Content -LiteralPath $pspMetadataPath -Raw | ConvertFrom-Json
}
if (Test-Path -LiteralPath $cropMetadataPath) {
    $cropMetadata = Get-Content -LiteralPath $cropMetadataPath -Raw | ConvertFrom-Json
}

$report = @{
    operation = "reconstruction_only_e4e_psp_comparison"
    styleclip_used = $false
    input_image = $(if ($cropMetadata -and $cropMetadata.input_image) { $cropMetadata.input_image } else { $InputImage })
    input_crop = $(if ($cropMetadata -and $cropMetadata.crop_path) { $cropMetadata.crop_path } else { $cropPath })
    crop_metadata = $cropMetadataPath
    comparison = $comparisonPath
    refine_latent_steps = $RefinarLatentePassos
    refine_learning_rate = $RefinarLatenteLearningRate
    refine_latent_l2 = $RefinarLatenteL2
    e4e = @{
        output_dir = $e4eRoot
        metadata = $e4eMetadataPath
        latent = Join-Path $e4eInversionDir "inversion_latent.pt"
        reconstruction = Join-Path $e4eInversionDir "e4e_reconstruction_preview.png"
        perfect_reconstruction = Join-Path $e4eInversionDir "e4e_identity_restored_reconstruction.png"
        metrics = $(if ($e4eMetadata) { $e4eMetadata.reconstruction_metrics } else { $null })
        initial_metrics = $(if ($e4eMetadata) { $e4eMetadata.initial_reconstruction_metrics } else { $null })
    }
    psp = @{
        output_dir = $pspRoot
        metadata = $pspMetadataPath
        latent = Join-Path $pspInversionDir "inversion_latent.pt"
        reconstruction = Join-Path $pspInversionDir "psp_reconstruction_preview.png"
        perfect_reconstruction = Join-Path $pspInversionDir "psp_identity_restored_reconstruction.png"
        metrics = $(if ($pspMetadata) { $pspMetadata.reconstruction_metrics } else { $null })
        initial_metrics = $(if ($pspMetadata) { $pspMetadata.initial_reconstruction_metrics } else { $null })
    }
}
$report | ConvertTo-Json -Depth 8 | Set-Content -Path $reportPath -Encoding UTF8

Write-Host ""
Write-Host "RECONSTRUCAO ONLY OK"
Write-Host "Comparacao: $comparisonPath"
Write-Host "Relatorio: $reportPath"
Write-Host "Latent e4e: $(Join-Path $e4eInversionDir "inversion_latent.pt")"
Write-Host "Latent pSp: $(Join-Path $pspInversionDir "inversion_latent.pt")"
