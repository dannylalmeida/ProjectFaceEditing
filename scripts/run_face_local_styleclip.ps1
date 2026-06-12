param(
    [string]$InputImage = "",
    [ValidateSet("auto", "pele", "sobrancelhas", "olhos", "orelhas", "nariz", "boca", "pescoco", "cabelo")]
    [string[]]$Region = @("auto"),
    [string]$Description = "",
    [string[]]$Preset = @(),
    [switch]$ListPresets,
    [int]$Step = 10,
    [int]$Dilation = 8,
    [int]$Feather = 15,
    [double]$BlendStrength = 0.8,
    [float]$MarginScale = 0.15,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$FaceEnv = "face",
    [string]$StyleclipEnv = "styleclip",
    [switch]$Force,
    [string]$OutputDir = ""
)

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "styleclip_prompt_presets.ps1")
. (Join-Path $PSScriptRoot "project_input_resolver.ps1")
$cropScript = Join-Path $projectRoot "scripts\export_primary_face_crop.py"
$invertScript = Join-Path $projectRoot "scripts\invert_face_to_latent.py"
$styleclipWrapper = Join-Path $projectRoot "scripts\run_styleclip_optimization.ps1"
$localizeScript = Join-Path $projectRoot "scripts\apply_local_text_edit.py"

if ($ListPresets) {
    Show-StyleClipPromptPresets
    exit 0
}

$resolvedDescription = Resolve-StyleClipPrompt -Description $Description -Preset $Preset
$resolvedRegions = @($Region | Where-Object { $_ })
if ($resolvedRegions.Count -eq 0 -or ($resolvedRegions.Count -eq 1 -and $resolvedRegions[0] -eq "auto")) {
    $resolvedRegions = Get-StyleClipTargetRegions -Description $Description -Preset $Preset
}
if (-not $resolvedRegions -or $resolvedRegions.Count -eq 0) {
    $resolvedRegions = @("cabelo")
}

$sensitiveRegions = @("sobrancelhas", "olhos", "nariz", "boca")
$hasSensitiveRegions = @($resolvedRegions | Where-Object { $sensitiveRegions -contains $_ }).Count -gt 0
$effectiveStep = $Step
if ($hasSensitiveRegions -and $Step -gt 20) {
    $effectiveStep = 20
    Write-Host "Passos demasiado altos para regioes faciais pequenas; a limitar StyleCLIP interno a 20 para preservar a imagem."
}

$InputImage = Resolve-ProjectInputImage -InputImage $InputImage -ProjectRoot $projectRoot

if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "outputs\ultima_execucao"
    $resetOutputDir = $true
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
    $resetOutputDir = $false
}

$processDir = Join-Path $OutputDir "_processo"
$cropDir = Join-Path $processDir "01_crop"
$inversionDir = Join-Path $processDir "02_inversion"
$styleclipDir = Join-Path $processDir "03_styleclip"
$localizedDir = Join-Path $processDir "04_localized"

if ($resetOutputDir -and (Test-Path $OutputDir)) {
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
}

foreach ($path in @($OutputDir, $processDir, $cropDir, $inversionDir, $styleclipDir, $localizedDir)) {
    New-Item -ItemType Directory -Force $path | Out-Null
}

if (-not (Test-Path $InputImage)) {
    Write-Error "Imagem de entrada nao encontrada em $InputImage"
    exit 1
}

Write-Host "Imagem selecionada: $InputImage"

$cropPath = Join-Path $cropDir "primary_face_crop.png"
$cropMetadataPath = Join-Path $cropDir "primary_face.json"
$latentPath = Join-Path $inversionDir "inversion_latent.pt"
$editedImagePath = Join-Path $styleclipDir "edited_result.jpg"
$localizedOnImagePath = Join-Path $localizedDir "localized_on_image.png"
$promptRegionMapPath = Join-Path $localizedDir "prompt_region_map.png"
$originalImagePath = Join-Path $OutputDir "01_imagem_original.png"
$promptMapOutputPath = Join-Path $OutputDir "02_mapa_regioes_prompt.png"
$finalImagePath = Join-Path $OutputDir "03_resultado_final.png"

Write-Host "Etapa 1/4: RetinaFace -> crop principal"
if (-not $Force -and (Test-Path $cropPath) -and (Test-Path $cropMetadataPath)) {
    Write-Host "Crop reutilizado da cache existente"
}
else {
    & conda run -n $FaceEnv python $cropScript --input $InputImage --output-dir $cropDir --margin-scale $MarginScale
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Falha ao exportar o crop principal da face."
        exit $LASTEXITCODE
    }
}

Write-Host "Etapa 2/4: e4e -> inversao do crop"
if (-not $Force -and (Test-Path $latentPath)) {
    Write-Host "Inversao reutilizada da cache existente"
}
else {
    & conda run -n $StyleclipEnv python $invertScript --input-crop $cropPath --output-dir $inversionDir --device $Device
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Falha na inversao do crop para latent."
        exit $LASTEXITCODE
    }
}

Write-Host "Etapa 3/4: StyleCLIP -> edicao guiada por texto"
Write-Host "Prompt usado: $resolvedDescription"
Write-Host ("Regioes alvo: " + ($resolvedRegions -join ", "))
& $styleclipWrapper -Description $resolvedDescription -Preset $Preset -Mode edit -Step $effectiveStep -Device $Device -CondaEnv $StyleclipEnv -LatentPath $latentPath -ResultsDir $styleclipDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na edicao StyleCLIP."
    exit $LASTEXITCODE
}

if (-not (Test-Path $editedImagePath)) {
    Write-Error "A imagem editada do StyleCLIP nao foi encontrada em $editedImagePath"
    exit 1
}

Write-Host "Etapa 4/4: composicao localizada por regiao facial"
& conda run -n $FaceEnv python $localizeScript --crop-metadata $cropMetadataPath --edited-image $editedImagePath --output-dir $localizedDir --region @($resolvedRegions) --description $resolvedDescription --dilation $Dilation --feather $Feather --blend-strength $BlendStrength
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na composicao localizada da edicao."
    exit $LASTEXITCODE
}

if (-not (Test-Path $localizedOnImagePath)) {
    Write-Error "A imagem final localizada nao foi encontrada em $localizedOnImagePath"
    exit 1
}

Copy-Item -LiteralPath $InputImage -Destination $originalImagePath -Force
if (Test-Path $promptRegionMapPath) {
    Copy-Item -LiteralPath $promptRegionMapPath -Destination $promptMapOutputPath -Force
}
Copy-Item -LiteralPath $localizedOnImagePath -Destination $finalImagePath -Force

Write-Host ""
Write-Host "Pipeline local StyleCLIP terminado."
Write-Host "Imagem original em: $originalImagePath"
Write-Host "Mapa das regioes do prompt em: $promptMapOutputPath"
Write-Host "Imagem final em: $finalImagePath"
Write-Host "Intermedios tecnicos em: $processDir"
