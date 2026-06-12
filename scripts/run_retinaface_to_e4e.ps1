param(
    [string]$InputImage = "",
    [float]$MarginScale = 0.15,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$FaceEnv = "face",
    [string]$StyleclipEnv = "styleclip",
    [ValidateSet("e4e", "psp")]
    [string]$EncoderBackend = "e4e",
    [string]$PspCheckpoint = "",
    [int]$RefinarLatentePassos = 15,
    [double]$RefinarLatenteLearningRate = 0.015,
    [double]$RefinarLatenteL2 = 0.0001,
    [switch]$ReconstruirPreview,
    [string]$OutputDir = ""
)

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "project_input_resolver.ps1")
$cropScript = Join-Path $projectRoot "scripts\export_primary_face_crop.py"
$invertScript = Join-Path $projectRoot "scripts\invert_face_to_latent.py"

$InputImage = Resolve-ProjectInputImage -InputImage $InputImage -ProjectRoot $projectRoot

if (-not $OutputDir) {
    if ($EncoderBackend -eq "psp") {
        $OutputDir = Join-Path $projectRoot "outputs\retinaface_psp"
    }
    else {
        $OutputDir = Join-Path $projectRoot "outputs\retinaface_e4e"
    }
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}

$cropDir = Join-Path $OutputDir "01_retinaface_crop"
$inversionDirName = if ($EncoderBackend -eq "psp") { "02_psp_inversion" } else { "02_e4e_inversion" }
$inversionDir = Join-Path $OutputDir $inversionDirName
$cropPath = Join-Path $cropDir "primary_face_crop.png"
$cropMetadataPath = Join-Path $cropDir "primary_face.json"
$latentPath = Join-Path $inversionDir "inversion_latent.pt"
$inputCropQualityPath = Join-Path $inversionDir "input_crop_full_quality.png"
$inputPreviewPath = Join-Path $inversionDir "input_resized_256_lanczos.png"
$reconstructionPreviewName = if ($EncoderBackend -eq "psp") { "psp_reconstruction_preview.png" } else { "e4e_reconstruction_preview.png" }
$reconstructionPreviewPath = Join-Path $inversionDir $reconstructionPreviewName

foreach ($path in @($OutputDir, $cropDir, $inversionDir)) {
    New-Item -ItemType Directory -Force $path | Out-Null
}

if (-not (Test-Path $InputImage)) {
    Write-Error "Imagem de entrada nao encontrada em $InputImage"
    exit 1
}

Write-Host "Imagem selecionada: $InputImage"
Write-Host "Etapa 1/2: RetinaFace -> deteccao e crop principal"
& conda run -n $FaceEnv python $cropScript --input $InputImage --output-dir $cropDir --margin-scale $MarginScale
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha ao exportar o crop principal da face com RetinaFace."
    exit $LASTEXITCODE
}

if (-not (Test-Path $cropPath)) {
    Write-Error "O crop principal nao foi gerado em $cropPath"
    exit 1
}

Write-Host "Etapa 2/2: $EncoderBackend -> encode do crop para latent"
$invertArgs = @(
    $invertScript,
    "--input-crop", $cropPath,
    "--output-dir", $inversionDir,
    "--device", $Device,
    "--encoder-backend", $EncoderBackend,
    "--refine-latent-steps", $RefinarLatentePassos,
    "--refine-learning-rate", $RefinarLatenteLearningRate,
    "--refine-latent-l2", $RefinarLatenteL2
)
if ($PspCheckpoint) {
    $invertArgs += @("--psp-checkpoint", $PspCheckpoint)
}
if ($ReconstruirPreview) {
    $invertArgs += "--save-reconstruction-preview"
}
& conda run -n $StyleclipEnv python @invertArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha no encode $EncoderBackend do crop."
    exit $LASTEXITCODE
}

foreach ($requiredPath in @($latentPath, $inputCropQualityPath, $inputPreviewPath)) {
    if (-not (Test-Path $requiredPath)) {
        Write-Error "Output esperado nao encontrado: $requiredPath"
        exit 1
    }
}
if ($ReconstruirPreview -and -not (Test-Path $reconstructionPreviewPath)) {
    Write-Error "Preview de reconstrucao esperado nao encontrado: $reconstructionPreviewPath"
    exit 1
}

Write-Host ""
Write-Host "RETINAFACE + $($EncoderBackend.ToUpperInvariant()) OK"
Write-Host "Crop RetinaFace: $cropPath"
Write-Host "Metadados RetinaFace: $cropMetadataPath"
Write-Host "Latent ${EncoderBackend}: $latentPath"
Write-Host "Crop qualidade original ${EncoderBackend}: $inputCropQualityPath"
Write-Host "Input 256x256 ${EncoderBackend}: $inputPreviewPath"
if ($ReconstruirPreview) {
    Write-Host "Preview reconstrucao ${EncoderBackend}: $reconstructionPreviewPath"
}
else {
    Write-Host "Preview reconstrucao ${EncoderBackend}: nao gerada no modo encode-only"
}
Write-Host "Pasta de output: $OutputDir"
