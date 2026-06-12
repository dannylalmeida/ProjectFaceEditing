param(
    [string]$InputImage = "",
    [ValidateSet("pele", "sobrancelhas", "olhos", "orelhas", "nariz", "boca", "pescoco", "cabelo")]
    [string]$Region = "cabelo",
    [string]$Description = "",
    [string[]]$Preset = @("sorriso"),
    [switch]$ListPresets,
    [ValidateSet("fast", "balanced", "quality")]
    [string]$RepaintProfile = "fast",
    [int]$RepaintTimesteps = 0,
    [int]$RepaintJumpLength = 0,
    [int]$RepaintJumpSamples = 0,
    [int]$StyleclipStep = 10,
    [int]$Dilation = 6,
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
$repaintWrapper = Join-Path $projectRoot "scripts\run_face_to_repaint.ps1"
$invertScript = Join-Path $projectRoot "scripts\invert_face_to_latent.py"
$styleclipWrapper = Join-Path $projectRoot "scripts\run_styleclip_optimization.ps1"

if ($ListPresets) {
    Show-StyleClipPromptPresets
    exit 0
}

$resolvedDescription = Resolve-StyleClipPrompt -Description $Description -Preset $Preset

$InputImage = Resolve-ProjectInputImage -InputImage $InputImage -ProjectRoot $projectRoot

if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "outputs\face_repaint_styleclip"
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}

$repaintDir = Join-Path $OutputDir "01_repaint"
$inversionDir = Join-Path $OutputDir "02_inversion"
$styleclipDir = Join-Path $OutputDir "03_styleclip"
$inversionMetadataPath = Join-Path $inversionDir "inversion_metadata.json"
$styleclipFinalPath = Join-Path $styleclipDir "final_result.jpg"
$finalImagePath = Join-Path $OutputDir "resultado_final.jpg"

foreach ($path in @($repaintDir, $inversionDir, $styleclipDir)) {
    New-Item -ItemType Directory -Force $path | Out-Null
}

if (-not (Test-Path $InputImage)) {
    Write-Error "Imagem de entrada nao encontrada em $InputImage"
    exit 1
}

Write-Host "Imagem selecionada: $InputImage"

Write-Host "Etapa 1/3: RePaint localizado na regiao '$Region'"
& $repaintWrapper -InputImage $InputImage -Region $Region -Dilation $Dilation -Profile $RepaintProfile -Timesteps $RepaintTimesteps -JumpLength $RepaintJumpLength -JumpSamples $RepaintJumpSamples -MarginScale $MarginScale -Device $Device -FaceEnv $FaceEnv -RepaintEnv $StyleclipEnv -OutputDir $repaintDir -Force:$Force
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na etapa RePaint."
    exit $LASTEXITCODE
}

$repaintedFacePath = Join-Path $repaintDir "02_repaint\inpainted\face.png"
if (-not (Test-Path $repaintedFacePath)) {
    Write-Error "O resultado do RePaint nao foi encontrado em $repaintedFacePath"
    exit 1
}

if (-not $Force -and (Test-Path $inversionMetadataPath) -and (Test-Path (Join-Path $inversionDir "inversion_latent.pt"))) {
    try {
        $inversionMetadata = Get-Content $inversionMetadataPath -Raw | ConvertFrom-Json
        $reuseInversion = ($inversionMetadata.input_crop -eq $repaintedFacePath)
    }
    catch {
        $reuseInversion = $false
    }
}
else {
    $reuseInversion = $false
}

if ($reuseInversion) {
    Write-Host "Etapa 2/3: inversao e4e reutilizada da cache existente"
}
else {
    Write-Host "Etapa 2/3: e4e -> inversao do resultado do RePaint"
    & conda run -n $StyleclipEnv python $invertScript --input-crop $repaintedFacePath --output-dir $inversionDir --device $Device
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Falha na inversao do resultado do RePaint."
        exit $LASTEXITCODE
    }
}

$latentPath = Join-Path $inversionDir "inversion_latent.pt"
if (-not (Test-Path $latentPath)) {
    Write-Error "O latent do resultado RePaint nao foi gerado em $latentPath"
    exit 1
}

Write-Host "Etapa 3/3: StyleCLIP -> refinamento guiado por texto"
Write-Host "Prompt usado: $resolvedDescription"
& $styleclipWrapper -Description $resolvedDescription -Preset $Preset -Mode edit -Step $StyleclipStep -Device $Device -CondaEnv $StyleclipEnv -LatentPath $latentPath -ResultsDir $styleclipDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na etapa StyleCLIP."
    exit $LASTEXITCODE
}

if (-not (Test-Path $styleclipFinalPath)) {
    Write-Error "A imagem final do pipeline RePaint + StyleCLIP nao foi encontrada em $styleclipFinalPath"
    exit 1
}

Copy-Item -LiteralPath $styleclipFinalPath -Destination $finalImagePath -Force

Write-Host ""
Write-Host "Pipeline RePaint + StyleCLIP terminado."
Write-Host "RePaint em: $repaintDir"
Write-Host "Latent em: $latentPath"
Write-Host "Resultado final em: $styleclipDir"
Write-Host "Imagem final em: $finalImagePath"
