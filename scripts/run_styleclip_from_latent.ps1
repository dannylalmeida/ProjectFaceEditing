param(
    [string]$LatentPath = "",
    [string]$Description = "",
    [string[]]$Preset = @(),
    [int]$Step = 10,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$StyleclipEnv = "styleclip",
    [string]$OutputDir = "",
    [string]$SourceDescription = "auto",
    [string]$TargetDescription = "",
    [ValidateSet("auto", "absolute", "directional")]
    [string]$ClipLossType = "auto",
    [double]$L2Lambda = -1.0,
    [ValidateSet("sum", "mean")]
    [string]$L2Reduction = "sum",
    [double]$LearningRate = -1.0,
    [double]$ClipLambda = 1.0,
    [double]$EditStrength = -1.0,
    [int]$LatentLayerMin = -1,
    [int]$LatentLayerMax = -1,
    [double]$MaxLatentDelta = -1.0,
    [string]$CropMetadataPath = "",
    [string]$FaceEnv = "face",
    [bool]$UseFaceParsing = $true,
    [ValidateSet("auto", "true", "false", "1", "0", "yes", "no", "sim", "nao", "não", "on", "off")]
    [string]$UseRePaint = "auto",
    [ValidateSet("auto", "hair", "cabelo", "mouth", "boca", "smile", "sorriso", "lips", "lip", "labios", "labio", "face", "pele", "skin", "age", "idade", "older", "younger", "beard", "barba", "mustache", "bigode", "goatee", "cavanhaque", "lower_face", "eyes", "eye", "iris", "irises", "olhos", "olho", "glasses", "oculos", "eyebrows", "eyebrow", "sobrancelhas", "sobrancelha", "nose", "nariz", "ears", "ear", "orelhas", "orelha", "neck", "pescoco")]
    [string]$EditRegion = "auto",
    [int]$MaskDilation = -1,
    [int]$MaskErosion = 0,
    [int]$MaskBlur = -1,
    [int]$MaskThreshold = 1,
    [int]$RePaintSteps = 20,
    [double]$RePaintStrength = 0.35,
    [ValidateSet("opencv", "repaint")]
    [string]$RePaintBackend = "opencv",
    [bool]$AuditDebug = $true,
    [switch]$SkipLocalization,
    [switch]$DisableDirectRefinements,
    [switch]$SaveExtraDebug,
    [switch]$SaveComparisons,
    [switch]$ListPresets
)

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "styleclip_prompt_presets.ps1")
$styleclipWrapper = Join-Path $projectRoot "scripts\run_styleclip_optimization.ps1"
$localizeScript = Join-Path $projectRoot "scripts\run_hybrid_edit.py"
$comparisonScript = Join-Path $projectRoot "scripts\create_image_comparison.py"

if ($ListPresets) {
    Show-StyleClipPromptPresets
    exit 0
}

function Resolve-StyleClipSourceDescription {
    param([string]$TargetDescription)

    $target = $TargetDescription.ToLowerInvariant()
    if ($target -match "\b(blond|blonde|platinum|golden)\s+hair\b") {
        return "a person with dark hair"
    }
    if ($target -match "\b(gray|silver|white)\s+hair\b") {
        return "a person with dark hair"
    }
    if ($target -match "\b(black|dark|brown)\s+hair\b") {
        return "a person with light hair"
    }
    if ($target -match "\b(red|auburn|purple|pink|blue|green)\s+hair\b") {
        return "a person with natural dark hair"
    }
    if ($target -match "\bhair\b") {
        return "a person with hair"
    }

    return "a person"
}

function Convert-StyleClipBlondToBlonde {
    param([string]$Text)

    if (-not $Text) {
        return $Text
    }
    return ($Text -replace "\bblond hair\b", "blonde hair" -replace "\bblond\b", "blonde")
}

function Resolve-StyleClipLayerBounds {
    param(
        [string]$TargetDescription,
        [int]$LayerMin,
        [int]$LayerMax
    )

    $target = $TargetDescription.ToLowerInvariant()
    if ($LayerMin -lt 0) {
        if ($target -match "\bhair\b") {
            $LayerMin = 8
        }
        elseif ($target -match "\b(smile|smiling|mouth|lip|lips|lipstick)\b") {
            $LayerMin = 5
        }
        elseif ($target -match "\b(beard|mustache|goatee|facial hair)\b") {
            $LayerMin = 5
        }
        elseif ($target -match "\b(older|younger|old|young|age|wrinkles)\b") {
            $LayerMin = 4
        }
        else {
            $LayerMin = 4
        }
    }
    if ($LayerMax -lt 0) {
        if ($target -match "\bhair\b") {
            $LayerMax = 17
        }
        elseif ($target -match "\b(smile|smiling|mouth|lip|lips|lipstick)\b") {
            $LayerMax = 10
        }
        elseif ($target -match "\b(beard|mustache|goatee|facial hair)\b") {
            $LayerMax = 12
        }
        elseif ($target -match "\b(older|younger|old|young|age|wrinkles)\b") {
            $LayerMax = 14
        }
        else {
            $LayerMax = 12
        }
    }

    return @($LayerMin, $LayerMax)
}

function Resolve-StyleClipEditRegionToTargetRegion {
    param([string]$EditRegion)

    switch ($EditRegion) {
        { $_ -in @("hair", "cabelo") } { return "cabelo" }
        { $_ -in @("mouth", "boca", "smile", "sorriso", "lips", "lip", "labios", "labio") } { return "boca" }
        { $_ -in @("eyes", "eye", "iris", "irises", "olhos", "olho", "glasses", "oculos") } { return "olhos" }
        { $_ -in @("eyebrows", "eyebrow", "sobrancelhas", "sobrancelha") } { return "sobrancelhas" }
        { $_ -in @("nose", "nariz") } { return "nariz" }
        { $_ -in @("ears", "ear", "orelhas", "orelha") } { return "orelhas" }
        { $_ -in @("neck", "pescoco") } { return "pescoco" }
        { $_ -in @("face", "pele", "skin", "age", "idade", "older", "younger", "beard", "barba", "mustache", "bigode", "goatee", "cavanhaque", "lower_face") } { return "pele" }
        default { return "" }
    }
}

function Resolve-StyleClipPrimaryEditRegion {
    param(
        [string]$Description,
        [string]$EditRegion,
        [string[]]$TargetRegions
    )

    if ($EditRegion -ne "auto") {
        switch ($EditRegion) {
            { $_ -in @("hair", "cabelo") } { return "hair" }
            { $_ -in @("mouth", "boca", "smile", "sorriso", "lips", "lip", "labios", "labio") } { return "mouth" }
            { $_ -in @("beard", "barba", "mustache", "bigode", "goatee", "cavanhaque", "lower_face") } { return "beard" }
            { $_ -in @("iris", "irises") } { return "iris" }
            { $_ -in @("eyes", "eye", "olhos", "olho", "glasses", "oculos") } { return "eyes" }
            { $_ -in @("eyebrows", "eyebrow", "sobrancelhas", "sobrancelha") } { return "eyebrows" }
            { $_ -in @("nose", "nariz") } { return "nose" }
            { $_ -in @("ears", "ear", "orelhas", "orelha") } { return "ears" }
            { $_ -in @("neck", "pescoco") } { return "neck" }
            { $_ -in @("face", "pele", "skin", "age", "idade", "older", "younger") } { return "face" }
            default { return "face" }
        }
    }

    $target = $Description.ToLowerInvariant()
    if ($target -match "\b(hair|bangs)\b") { return "hair" }
    if ($target -match "\b(beard|mustache|goatee|facial hair)\b") { return "beard" }
    if ($target -match "\b(smile|smiling|mouth|lip|lips|lipstick)\b") { return "mouth" }
    if ($target -match "\b(iris|irises)\b") { return "iris" }
    if ($target -match "\b(blue|green|hazel|brown|gray|grey)\s+(eye|eyes)\b") { return "iris" }
    if ($target -match "\b(eye|eyes|eyelash|eyelashes|glasses)\b") { return "eyes" }
    if ($target -match "\b(eyebrow|eyebrows|brows)\b") { return "eyebrows" }
    if ($target -match "\bnose\b") { return "nose" }
    if ($target -match "\bears?\b") { return "ears" }
    if ($target -match "\bneck\b") { return "neck" }
    if ($target -match "\b(skin|age|older|younger|wrinkles|freckles|acne|face)\b") { return "face" }

    if ($TargetRegions -contains "cabelo") { return "hair" }
    if ($TargetRegions -contains "boca") { return "mouth" }
    if ($TargetRegions -contains "olhos") { return "eyes" }
    if ($TargetRegions -contains "sobrancelhas") { return "eyebrows" }
    if ($TargetRegions -contains "nariz") { return "nose" }
    if ($TargetRegions -contains "orelhas") { return "ears" }
    if ($TargetRegions -contains "pescoco") { return "neck" }
    return "face"
}

function Resolve-StyleClipRePaintSetting {
    param(
        [string]$Requested,
        [string]$PrimaryEditRegion
    )

    $normalized = if ($Requested) { $Requested.Trim().ToLowerInvariant() } else { "auto" }
    if ($normalized -in @("1", "true", "yes", "sim", "s", "on")) {
        return @{ Enabled = $true; Mode = "manual_true"; StrengthScale = 1.0 }
    }
    if ($normalized -in @("0", "false", "no", "nao", "não", "n", "off")) {
        return @{ Enabled = $false; Mode = "manual_false"; StrengthScale = 1.0 }
    }

    switch ($PrimaryEditRegion) {
        "hair" { return @{ Enabled = $true; Mode = "auto_hair"; StrengthScale = 1.0 } }
        "beard" { return @{ Enabled = $true; Mode = "auto_beard"; StrengthScale = 0.85 } }
        "mouth" { return @{ Enabled = $true; Mode = "auto_mouth_soft"; StrengthScale = 0.45 } }
        default { return @{ Enabled = $false; Mode = "auto_preserve_fine_details"; StrengthScale = 1.0 } }
    }
}

function Resolve-StyleClipCropMetadataPath {
    param(
        [string]$ExplicitPath,
        [string]$LatentPath,
        [string]$ProjectRoot
    )

    if ($ExplicitPath) {
        return [System.IO.Path]::GetFullPath($ExplicitPath)
    }

    $latentDir = Split-Path -Parent $LatentPath
    $runDir = Split-Path -Parent $latentDir
    $candidates = @(
        (Join-Path $runDir "01_retinaface_crop\primary_face.json"),
        (Join-Path $runDir "01_crop\primary_face.json"),
        (Join-Path $ProjectRoot "outputs\retinaface_psp\01_retinaface_crop\primary_face.json"),
        (Join-Path $ProjectRoot "outputs\retinaface_e4e\01_retinaface_crop\primary_face.json")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return ""
}

if (-not $LatentPath) {
    $LatentPath = Join-Path $projectRoot "outputs\retinaface_psp\02_psp_inversion\inversion_latent.pt"
}
else {
    $LatentPath = [System.IO.Path]::GetFullPath($LatentPath)
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "outputs\styleclip_edit"
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}

$finalResultPath = Join-Path $OutputDir "final_result.jpg"
$latentDebugResultPath = Join-Path $OutputDir "styleclip_latent_debug.jpg"
$originalResultPath = Join-Path $OutputDir "original_result.jpg"
$editedResultPath = Join-Path $OutputDir "edited_result.jpg"
$originalLatentOutPath = Join-Path $OutputDir "original_latent.pt"
$editedLatentOutPath = Join-Path $OutputDir "edited_latent.pt"
$latentDeltaOutPath = Join-Path $OutputDir "latent_delta.pt"
$promptInfoPath = Join-Path $OutputDir "prompt_info.json"
$moduleMetadataPath = Join-Path $OutputDir "styleclip_module_metadata.json"
$alignedComparisonPath = Join-Path $OutputDir "final_aligned_comparison.jpg"
$fullComparisonPath = Join-Path $OutputDir "final_on_original_comparison.jpg"

if (-not (Test-Path $LatentPath)) {
    Write-Error "Latent nao encontrado em $LatentPath. Corre primeiro: .\run.cmd -Comando psp"
    exit 1
}

New-Item -ItemType Directory -Force $OutputDir | Out-Null

$latentShape = (& conda run -n $StyleclipEnv python -c "import torch; x=torch.load(r'$LatentPath', map_location='cpu'); print('x'.join(map(str, x.shape)))").Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha ao validar o latent em $LatentPath"
    exit $LASTEXITCODE
}
if ($latentShape -ne "1x18x512") {
    Write-Error "Latent com shape inesperado: $latentShape. Esperado: 1x18x512."
    exit 1
}

$targetInput = if ($TargetDescription) { $TargetDescription } else { $Description }
$resolvedDescription = Resolve-StyleClipPrompt -Description $targetInput -Preset $(if ($targetInput) { @() } else { $Preset })
$resolvedDescription = Convert-StyleClipBlondToBlonde -Text $resolvedDescription
if (-not $SourceDescription -or $SourceDescription -eq "auto") {
    $SourceDescription = Resolve-StyleClipSourceDescription -TargetDescription $resolvedDescription
}
$SourceDescription = Convert-StyleClipBlondToBlonde -Text $SourceDescription
$targetLower = $resolvedDescription.ToLowerInvariant()
$isHairEdit = $targetLower -match "\bhair\b"
$isMouthEdit = $targetLower -match "\b(smile|smiling|mouth|lip|lips|lipstick)\b"
$isBeardEdit = $targetLower -match "\b(beard|mustache|goatee|facial hair)\b"
$isAgeEdit = $targetLower -match "\b(older|younger|old|young|age|wrinkles)\b"
if ($ClipLossType -eq "auto") {
    $ClipLossType = if ($isHairEdit -or $isBeardEdit -or $isAgeEdit) { "directional" } else { "absolute" }
}
if ($L2Lambda -lt 0) {
    $L2Lambda = if ($isHairEdit -or $isMouthEdit -or $isBeardEdit -or $isAgeEdit) { 0.05 } else { 0.012 }
}
if ($LearningRate -lt 0) {
    $LearningRate = 0.02
}
if ($EditStrength -lt 0) {
    if ($isHairEdit) {
        $EditStrength = 0.04
    }
    elseif ($isMouthEdit) {
        $EditStrength = 0.03
    }
    elseif ($isBeardEdit) {
        $EditStrength = 0.035
    }
    elseif ($isAgeEdit) {
        $EditStrength = 0.04
    }
    else {
        $EditStrength = 0.02
    }
}
if ($MaxLatentDelta -lt 0) {
    if ($isHairEdit -or $isMouthEdit) {
        $MaxLatentDelta = 0.08
    }
    elseif ($isBeardEdit) {
        $MaxLatentDelta = 0.10
    }
    elseif ($isAgeEdit) {
        $MaxLatentDelta = 0.12
    }
    else {
        $MaxLatentDelta = 0.12
    }
}
$layerBounds = Resolve-StyleClipLayerBounds -TargetDescription $resolvedDescription -LayerMin $LatentLayerMin -LayerMax $LatentLayerMax
$LatentLayerMin = $layerBounds[0]
$LatentLayerMax = $layerBounds[1]
$targetRegions = @(Get-StyleClipTargetRegions -Description $resolvedDescription -Preset @())
if ($EditRegion -ne "auto") {
    $explicitTargetRegion = Resolve-StyleClipEditRegionToTargetRegion -EditRegion $EditRegion
    if ($explicitTargetRegion -and ($targetRegions -notcontains $explicitTargetRegion)) {
        $targetRegions += $explicitTargetRegion
    }
}
$primaryEditRegion = Resolve-StyleClipPrimaryEditRegion -Description $resolvedDescription -EditRegion $EditRegion -TargetRegions $targetRegions
$repaintDecision = Resolve-StyleClipRePaintSetting -Requested $UseRePaint -PrimaryEditRegion $primaryEditRegion
$resolvedUseRePaint = [bool]$repaintDecision.Enabled
$resolvedRePaintMode = [string]$repaintDecision.Mode
$resolvedRePaintStrength = $RePaintStrength
if ($UseRePaint.Trim().ToLowerInvariant() -eq "auto" -and $resolvedUseRePaint) {
    $resolvedRePaintStrength = [Math]::Round($RePaintStrength * [double]$repaintDecision.StrengthScale, 4)
}
$resolvedCropMetadataPath = Resolve-StyleClipCropMetadataPath -ExplicitPath $CropMetadataPath -LatentPath $LatentPath -ProjectRoot $projectRoot
$initialCropPath = ""
$sourceInputImage = ""
$cropMetadata = $null

if (-not $resolvedCropMetadataPath -or -not (Test-Path -LiteralPath $resolvedCropMetadataPath)) {
    Write-Error "StyleCLIP bloqueado: metadados do crop nao encontrados. Corre primeiro .\run.cmd -Comando reconstrucao ou .\run.cmd -Comando psp/e4e."
    exit 1
}

try {
    $cropMetadata = Get-Content -Path $resolvedCropMetadataPath -Raw | ConvertFrom-Json
    if ($cropMetadata.crop_path) {
        $initialCropPath = [string]$cropMetadata.crop_path
    }
    if ($cropMetadata.input_image) {
        $sourceInputImage = [string]$cropMetadata.input_image
    }
}
catch {
    Write-Error "StyleCLIP bloqueado: nao consegui ler os metadados do crop em $resolvedCropMetadataPath."
    exit 1
}

if (-not $initialCropPath -or -not (Test-Path -LiteralPath $initialCropPath)) {
    Write-Error "StyleCLIP bloqueado: crop inicial nao encontrado nos metadados ($resolvedCropMetadataPath)."
    exit 1
}
if (-not $sourceInputImage -or -not (Test-Path -LiteralPath $sourceInputImage)) {
    Write-Error "StyleCLIP bloqueado: imagem original nao encontrada nos metadados ($resolvedCropMetadataPath)."
    exit 1
}

Write-Host "Modulo isolado StyleCLIP"
Write-Host "Latent usado: $LatentPath"
Write-Host "Imagem original do dataset: $sourceInputImage"
Write-Host "Crop inicial usado para comparacao/composicao: $initialCropPath"
Write-Host "Prompt usado: $resolvedDescription"
Write-Host "Prompt fonte: $SourceDescription"
Write-Host "CLIP loss: $ClipLossType"
Write-Host "L2 latent: $L2Lambda"
Write-Host "L2 reduction: $L2Reduction"
Write-Host "Learning rate: $LearningRate"
Write-Host "CLIP weight: $ClipLambda"
Write-Host "Forca edicao: $EditStrength"
Write-Host "Camadas W+ editaveis: $LatentLayerMin-$LatentLayerMax"
Write-Host "Max latent delta: $MaxLatentDelta"
Write-Host "Face Parsing: $UseFaceParsing"
Write-Host "Edit region: $EditRegion"
Write-Host "Debug auditoria: $AuditDebug"
Write-Host "Primary edit region: $primaryEditRegion"
Write-Host "Use RePaint/Inpainting pedido: $UseRePaint"
Write-Host "Use RePaint/Inpainting resolvido: $resolvedUseRePaint ($resolvedRePaintMode, strength=$resolvedRePaintStrength)"
if ($targetRegions.Count -gt 0) {
    Write-Host ("Regioes do prompt: " + ($targetRegions -join ", "))
}
Write-Host "Passos: $Step"
Write-Host "Output: $OutputDir"

& $styleclipWrapper -Description $resolvedDescription -TargetDescription $resolvedDescription -Preset @() -Mode edit -Step $Step -Device $Device -CondaEnv $StyleclipEnv -LatentPath $LatentPath -ResultsDir $OutputDir -SourceDescription $SourceDescription -ClipLossType $ClipLossType -L2Lambda $L2Lambda -L2Reduction $L2Reduction -LearningRate $LearningRate -ClipLambda $ClipLambda -EditStrength $EditStrength -LatentLayerMin $LatentLayerMin -LatentLayerMax $LatentLayerMax -MaxLatentDelta $MaxLatentDelta
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na edicao isolada com StyleCLIP."
    exit $LASTEXITCODE
}

foreach ($requiredPath in @($finalResultPath, $originalResultPath, $editedResultPath, $promptInfoPath)) {
    if (-not (Test-Path $requiredPath)) {
        Write-Error "Output esperado do StyleCLIP nao encontrado: $requiredPath"
        exit 1
    }
}
if ($SaveExtraDebug) {
    Copy-Item -LiteralPath $finalResultPath -Destination $latentDebugResultPath -Force
}

$cropAlignedPath = Join-Path $OutputDir "crop_aligned.png"
$pspReconstructionPath = Join-Path $OutputDir "psp_reconstruction.png"
$e4eReconstructionPath = Join-Path $OutputDir "e4e_reconstruction.png"
$encoderReconstructionPath = Join-Path $OutputDir "encoder_reconstruction.png"
$perfectReconstructionPath = Join-Path $OutputDir "perfect_reconstruction.png"
$reconstructionResidualMapPath = Join-Path $OutputDir "reconstruction_residual_map.png"
$styleclipResultPath = Join-Path $OutputDir "styleclip_result.png"
$latentShapePath = Join-Path $OutputDir "latent_shape.txt"
$latentShapeLogPath = Join-Path $OutputDir "latent_shape_log.txt"
$styleclipParamsLogPath = Join-Path $OutputDir "styleclip_params_log.txt"
$inputEncoderPath = Join-Path $OutputDir "input_encoder.png"
$localizedDir = Join-Path $OutputDir "localized"
$localizedCropPath = Join-Path $localizedDir "localized_crop.png"
$localizedOnImagePath = Join-Path $localizedDir "localized_on_image.png"
$localizedAlignedOriginalPath = Join-Path $localizedDir "aligned_original.png"
$localizedStyleclipEditPath = Join-Path $localizedDir "styleclip_edit.png"
$localizedEditMaskPath = Join-Path $localizedDir "edit_mask.png"
$localizedEditMaskOnOriginalPath = Join-Path $localizedDir "edit_mask_on_original.png"
$localizedFinalBlendedAlignedPath = Join-Path $localizedDir "final_blended_aligned.png"
$localizedFinalOnOriginalPath = Join-Path $localizedDir "final_on_original.png"
$localizedMetadataPath = Join-Path $localizedDir "localized_edit_metadata.json"
$localizationEnabled = $false
$directRefinementStrength = if ($EditStrength -gt 0) { [Math]::Min(0.82, [Math]::Max(0.36, 0.36 + $EditStrength * 6.0)) } else { 0.56 }
$inversionMetadataPath = Join-Path (Split-Path -Parent $LatentPath) "inversion_metadata.json"
$inversionMetadata = $null
foreach ($reconstructionAliasPath in @($pspReconstructionPath, $e4eReconstructionPath)) {
    if (Test-Path -LiteralPath $reconstructionAliasPath) {
        Remove-Item -LiteralPath $reconstructionAliasPath -Force
    }
}
if (Test-Path -LiteralPath $inversionMetadataPath) {
    try {
        $inversionMetadata = Get-Content -Path $inversionMetadataPath -Raw | ConvertFrom-Json
        if ($inversionMetadata.reconstruction_preview_path -and (Test-Path -LiteralPath ([string]$inversionMetadata.reconstruction_preview_path))) {
            Copy-Item -LiteralPath ([string]$inversionMetadata.reconstruction_preview_path) -Destination $encoderReconstructionPath -Force
            $encoderBackend = if ($inversionMetadata.encoder_backend) { ([string]$inversionMetadata.encoder_backend).ToLowerInvariant() } else { "" }
            if ($encoderBackend -eq "psp") {
                Copy-Item -LiteralPath ([string]$inversionMetadata.reconstruction_preview_path) -Destination $pspReconstructionPath -Force
            }
            elseif ($encoderBackend -eq "e4e") {
                Copy-Item -LiteralPath ([string]$inversionMetadata.reconstruction_preview_path) -Destination $e4eReconstructionPath -Force
            }
        }
        if ($inversionMetadata.identity_restored_reconstruction_path -and (Test-Path -LiteralPath ([string]$inversionMetadata.identity_restored_reconstruction_path))) {
            Copy-Item -LiteralPath ([string]$inversionMetadata.identity_restored_reconstruction_path) -Destination $perfectReconstructionPath -Force
        }
        if ($inversionMetadata.reconstruction_residual_map_path -and (Test-Path -LiteralPath ([string]$inversionMetadata.reconstruction_residual_map_path))) {
            Copy-Item -LiteralPath ([string]$inversionMetadata.reconstruction_residual_map_path) -Destination $reconstructionResidualMapPath -Force
        }
        if ($inversionMetadata.input_resized_path -and (Test-Path -LiteralPath ([string]$inversionMetadata.input_resized_path))) {
            Copy-Item -LiteralPath ([string]$inversionMetadata.input_resized_path) -Destination $inputEncoderPath -Force
        }
    }
    catch {
        Write-Warning "Nao consegui ler os metadados de inversao em $inversionMetadataPath"
    }
}

if (-not $SkipLocalization -and $targetRegions.Count -gt 0 -and $resolvedCropMetadataPath -and (Test-Path -LiteralPath $resolvedCropMetadataPath)) {
    New-Item -ItemType Directory -Force $localizedDir | Out-Null
    $localizeArgs = @(
        $localizeScript,
        "--crop-metadata", $resolvedCropMetadataPath,
        "--output-dir", $localizedDir,
        "--styleclip-edited-image", $editedResultPath,
        "--description", $resolvedDescription,
        "--target-description", $resolvedDescription,
        "--use-face-parsing", ([string]$UseFaceParsing).ToLowerInvariant(),
        "--use-local-recolor", "false",
        "--use-styleclip", "true",
        "--edit-region", $EditRegion,
        "--mask-dilation", ([string]$MaskDilation),
        "--mask-erosion", ([string]$MaskErosion),
        "--mask-blur", ([string]$MaskBlur),
        "--mask-threshold", ([string]$MaskThreshold),
        "--use-repaint", ([string]$resolvedUseRePaint).ToLowerInvariant(),
        "--repaint-steps", ([string]$RePaintSteps),
        "--repaint-strength", ([string]$resolvedRePaintStrength),
        "--repaint-backend", $RePaintBackend,
        "--debug", ([string]$AuditDebug).ToLowerInvariant()
    )
    & conda run -n $FaceEnv python @localizeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Falha na localizacao da edicao StyleCLIP."
        exit $LASTEXITCODE
    }
    $localizationEnabled = $true
}

if ($initialCropPath -and (Test-Path -LiteralPath $initialCropPath)) {
    Copy-Item -LiteralPath $initialCropPath -Destination $cropAlignedPath -Force
    if ($SaveComparisons) {
        $comparisonRightPath = if ($localizationEnabled -and (Test-Path -LiteralPath $localizedFinalBlendedAlignedPath)) { $localizedFinalBlendedAlignedPath } else { $editedResultPath }
        & conda run -n $StyleclipEnv python $comparisonScript --left $initialCropPath --right $comparisonRightPath --output $alignedComparisonPath --left-label "aligned original" --right-label "final blended aligned"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Falha ao criar comparacao final alinhada com a imagem inicial."
            exit $LASTEXITCODE
        }
    }
}

if ($localizationEnabled -and (Test-Path -LiteralPath $localizedFinalOnOriginalPath)) {
    Copy-Item -LiteralPath $localizedFinalOnOriginalPath -Destination $finalResultPath -Force
    if ($SaveComparisons) {
        & conda run -n $StyleclipEnv python $comparisonScript --left $sourceInputImage --right $localizedFinalOnOriginalPath --output $fullComparisonPath --left-label "original image" --right-label "final on original"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Falha ao criar comparacao final na imagem original."
            exit $LASTEXITCODE
        }
    }
}
else {
    Copy-Item -LiteralPath $editedResultPath -Destination $finalResultPath -Force
}

if (-not $SaveExtraDebug) {
    $staleDebugFiles = @(
        $originalResultPath,
        $editedResultPath,
        $perfectReconstructionPath,
        $reconstructionResidualMapPath,
        $latentDebugResultPath,
        $localizedCropPath,
        $localizedOnImagePath,
        (Join-Path $localizedDir "localized_mask.png"),
        (Join-Path $localizedDir "localized_overlay.png"),
        (Join-Path $localizedDir "edit_mask_on_original.png"),
        (Join-Path $localizedDir "prompt_region_map.png"),
        (Join-Path $localizedDir "styleclip_edited_crop.png")
    )
    foreach ($debugFile in $staleDebugFiles) {
        if (Test-Path -LiteralPath $debugFile) {
            Remove-Item -LiteralPath $debugFile -Force
        }
    }
    if (-not $SaveComparisons) {
        foreach ($comparisonDebugFile in @($alignedComparisonPath, $fullComparisonPath)) {
            if (Test-Path -LiteralPath $comparisonDebugFile) {
                Remove-Item -LiteralPath $comparisonDebugFile -Force
            }
        }
    }
}

foreach ($debugPath in @($cropAlignedPath, $encoderReconstructionPath, $styleclipResultPath, $latentShapePath, $editedLatentOutPath, $originalLatentOutPath)) {
    if (-not (Test-Path -LiteralPath $debugPath)) {
        Write-Warning "Output de debug esperado nao encontrado: $debugPath"
    }
}

if (Test-Path -LiteralPath $latentShapePath) {
    Copy-Item -LiteralPath $latentShapePath -Destination $latentShapeLogPath -Force
}
$styleclipParams = @{
    source_description = $SourceDescription
    target_description = $resolvedDescription
    steps = $Step
    edit_strength = $EditStrength
    learning_rate = $LearningRate
    layers_inicio = $LatentLayerMin
    layers_fim = $LatentLayerMax
    lambda_latent = $L2Lambda
    delta_clamp = $MaxLatentDelta
    clip_loss_type = $ClipLossType
    use_face_parsing = $UseFaceParsing
    edit_region = $EditRegion
    primary_edit_region = $primaryEditRegion
    use_repaint_requested = $UseRePaint
    use_repaint = $resolvedUseRePaint
    repaint_backend = $RePaintBackend
} | ConvertTo-Json
$styleclipParams | Set-Content -Path $styleclipParamsLogPath -Encoding UTF8

$metadata = @{
    operation = "styleclip_edit_from_existing_latent"
    is_isolated_module = $true
    calls_retinaface = $false
    calls_crop = $false
    calls_e4e = $false
    calls_face_parsing_for_localization = $localizationEnabled
    input_latent = $LatentPath
    original_latent = $(if (Test-Path -LiteralPath $originalLatentOutPath) { $originalLatentOutPath } else { $null })
    edited_latent = $(if (Test-Path -LiteralPath $editedLatentOutPath) { $editedLatentOutPath } else { $null })
    latent_delta = $(if (Test-Path -LiteralPath $latentDeltaOutPath) { $latentDeltaOutPath } else { $null })
    latent_shape = $latentShape
    inversion_metadata_path = $(if (Test-Path -LiteralPath $inversionMetadataPath) { $inversionMetadataPath } else { $null })
    encoder_backend = $(if ($inversionMetadata -and $inversionMetadata.encoder_backend) { [string]$inversionMetadata.encoder_backend } else { $null })
    latent_refinement_steps = $(if ($inversionMetadata -and $null -ne $inversionMetadata.latent_refinement_steps) { $inversionMetadata.latent_refinement_steps } else { $null })
    input_description = $Description
    input_target_description = $TargetDescription
    input_preset = $Preset
    resolved_description_en = $resolvedDescription
    source_description_en = $SourceDescription
    clip_loss_type = $ClipLossType
    l2_lambda = $L2Lambda
    l2_reduction = $L2Reduction
    learning_rate = $LearningRate
    clip_lambda = $ClipLambda
    edit_strength = $EditStrength
    latent_layer_min = $LatentLayerMin
    latent_layer_max = $LatentLayerMax
    max_latent_delta = $MaxLatentDelta
    direct_refinements_enabled = (-not $DisableDirectRefinements)
    direct_refinement_strength = $directRefinementStrength
    use_face_parsing = $UseFaceParsing
    use_repaint_requested = $UseRePaint
    use_repaint = $resolvedUseRePaint
    repaint_auto_mode = $resolvedRePaintMode
    primary_edit_region = $primaryEditRegion
    edit_region = $EditRegion
    mask_dilation = $MaskDilation
    mask_erosion = $MaskErosion
    mask_blur = $MaskBlur
    mask_threshold = $MaskThreshold
    debug_saved = $AuditDebug
    repaint_steps = $RePaintSteps
    repaint_strength_requested = $RePaintStrength
    repaint_strength = $resolvedRePaintStrength
    repaint_backend = $RePaintBackend
    extra_debug_saved = $SaveExtraDebug.IsPresent
    comparisons_saved = $SaveComparisons.IsPresent
    target_regions = $targetRegions
    crop_metadata_path = $resolvedCropMetadataPath
    source_input_image = $sourceInputImage
    initial_crop = $initialCropPath
    crop_aligned = $(if (Test-Path -LiteralPath $cropAlignedPath) { $cropAlignedPath } else { $null })
    psp_reconstruction = $(if (Test-Path -LiteralPath $pspReconstructionPath) { $pspReconstructionPath } else { $null })
    e4e_reconstruction = $(if (Test-Path -LiteralPath $e4eReconstructionPath) { $e4eReconstructionPath } else { $null })
    encoder_reconstruction = $(if (Test-Path -LiteralPath $encoderReconstructionPath) { $encoderReconstructionPath } else { $null })
    perfect_reconstruction = $(if (Test-Path -LiteralPath $perfectReconstructionPath) { $perfectReconstructionPath } else { $null })
    reconstruction_residual_map = $(if (Test-Path -LiteralPath $reconstructionResidualMapPath) { $reconstructionResidualMapPath } else { $null })
    styleclip_result = $(if (Test-Path -LiteralPath $styleclipResultPath) { $styleclipResultPath } else { $null })
    latent_shape_log = $(if (Test-Path -LiteralPath $latentShapePath) { $latentShapePath } else { $null })
    latent_shape_log_txt = $(if (Test-Path -LiteralPath $latentShapeLogPath) { $latentShapeLogPath } else { $null })
    styleclip_params_log = $(if (Test-Path -LiteralPath $styleclipParamsLogPath) { $styleclipParamsLogPath } else { $null })
    input_encoder = $(if (Test-Path -LiteralPath $inputEncoderPath) { $inputEncoderPath } else { $null })
    aligned_original = $(if (Test-Path -LiteralPath $localizedAlignedOriginalPath) { $localizedAlignedOriginalPath } else { $cropAlignedPath })
    localized_crop = $(if (Test-Path -LiteralPath $localizedCropPath) { $localizedCropPath } elseif (Test-Path -LiteralPath $localizedFinalBlendedAlignedPath) { $localizedFinalBlendedAlignedPath } else { $null })
    localized_on_image = $(if (Test-Path -LiteralPath $localizedOnImagePath) { $localizedOnImagePath } elseif (Test-Path -LiteralPath $localizedFinalOnOriginalPath) { $localizedFinalOnOriginalPath } else { $null })
    localized_styleclip_edit = $(if ($localizationEnabled) { $localizedStyleclipEditPath } else { $null })
    edit_mask = $(if ($localizationEnabled) { $localizedEditMaskPath } else { $null })
    selected_edit_mask = $(if (Test-Path -LiteralPath (Join-Path $localizedDir "selected_edit_mask.png")) { Join-Path $localizedDir "selected_edit_mask.png" } else { $null })
    final_edit_mask = $(if (Test-Path -LiteralPath (Join-Path $localizedDir "final_edit_mask.png")) { Join-Path $localizedDir "final_edit_mask.png" } else { $null })
    edit_mask_on_original = $(if (Test-Path -LiteralPath $localizedEditMaskOnOriginalPath) { $localizedEditMaskOnOriginalPath } else { $null })
    final_blended_aligned = $(if ($localizationEnabled) { $localizedFinalBlendedAlignedPath } else { $null })
    final_repainted_aligned = $(if (Test-Path -LiteralPath (Join-Path $localizedDir "final_repainted_aligned.png")) { Join-Path $localizedDir "final_repainted_aligned.png" } else { $null })
    final_on_original = $(if ($localizationEnabled) { $localizedFinalOnOriginalPath } else { $null })
    localized_metadata = $(if ($localizationEnabled) { $localizedMetadataPath } else { $null })
    step = $Step
    output_dir = $OutputDir
    original_result = $(if (Test-Path -LiteralPath $originalResultPath) { $originalResultPath } else { $null })
    edited_result = $(if (Test-Path -LiteralPath $editedResultPath) { $editedResultPath } else { $null })
    final_result = $finalResultPath
    final_aligned_comparison = $(if (Test-Path -LiteralPath $alignedComparisonPath) { $alignedComparisonPath } else { $null })
    final_on_original_comparison = $(if (Test-Path -LiteralPath $fullComparisonPath) { $fullComparisonPath } else { $null })
    styleclip_latent_debug = $(if (Test-Path -LiteralPath $latentDebugResultPath) { $latentDebugResultPath } else { $null })
} | ConvertTo-Json
$metadata | Set-Content -Path $moduleMetadataPath -Encoding UTF8

Write-Host ""
Write-Host "STYLECLIP ISOLADO OK"
Write-Host "Crop original: $cropAlignedPath"
Write-Host "Reconstrucao encoder: $encoderReconstructionPath"
Write-Host "Edicao StyleCLIP bruta: $styleclipResultPath"
if ($localizationEnabled) {
    Write-Host "Aligned original: $localizedAlignedOriginalPath"
    Write-Host "StyleCLIP edit crop: $localizedStyleclipEditPath"
    Write-Host "Mascara de edicao: $localizedEditMaskPath"
    Write-Host "Final alinhado sobre original: $localizedFinalBlendedAlignedPath"
    if ($resolvedUseRePaint) {
        Write-Host "Final com RePaint/Inpainting local: $(Join-Path $localizedDir "final_repainted_aligned.png")"
    }
    Write-Host "Final na imagem original: $localizedFinalOnOriginalPath"
    Write-Host "Final principal: $finalResultPath"
}
Write-Host "Metadados: $moduleMetadataPath"
