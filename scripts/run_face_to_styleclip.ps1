param(
    [string]$InputImage = "",
    [string]$Description = "",
    [string[]]$Preset = @("sorriso"),
    [switch]$ListPresets,
    [int]$Step = 10,
    [float]$MarginScale = 0.15,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$FaceEnv = "face",
    [string]$StyleclipEnv = "styleclip",
    [string]$OutputDir = ""
)

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "styleclip_prompt_presets.ps1")
. (Join-Path $PSScriptRoot "project_input_resolver.ps1")
$cropScript = Join-Path $projectRoot "scripts\export_primary_face_crop.py"
$invertScript = Join-Path $projectRoot "scripts\invert_face_to_latent.py"
$styleclipWrapper = Join-Path $projectRoot "scripts\run_styleclip_optimization.ps1"

if ($ListPresets) {
    Show-StyleClipPromptPresets
    exit 0
}

$resolvedDescription = Resolve-StyleClipPrompt -Description $Description -Preset $Preset

$InputImage = Resolve-ProjectInputImage -InputImage $InputImage -ProjectRoot $projectRoot

if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "outputs\face_to_styleclip"
}

$cropOutputDir = Join-Path $OutputDir "01_crop"
$inversionOutputDir = Join-Path $OutputDir "02_inversion"
$styleclipOutputDir = Join-Path $OutputDir "03_styleclip"
$styleclipFinalPath = Join-Path $styleclipOutputDir "final_result.jpg"
$finalImagePath = Join-Path $OutputDir "resultado_final.jpg"

foreach ($path in @($cropOutputDir, $inversionOutputDir, $styleclipOutputDir)) {
    New-Item -ItemType Directory -Force $path | Out-Null
}

if (-not (Test-Path $InputImage)) {
    Write-Error "Imagem de entrada nao encontrada em $InputImage"
    exit 1
}

Write-Host "Imagem selecionada: $InputImage"

Write-Host "Etapa 1/3: RetinaFace -> crop principal"
& conda run -n $FaceEnv python $cropScript --input $InputImage --output-dir $cropOutputDir --margin-scale $MarginScale
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha ao exportar o crop principal da face."
    exit $LASTEXITCODE
}

$cropPath = Join-Path $cropOutputDir "primary_face_crop.png"
if (-not (Test-Path $cropPath)) {
    Write-Error "O crop principal nao foi gerado em $cropPath"
    exit 1
}

Write-Host "Etapa 2/3: e4e -> inversao para latent"
& conda run -n $StyleclipEnv python $invertScript --input-crop $cropPath --output-dir $inversionOutputDir --device $Device
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na inversao do crop para latent."
    exit $LASTEXITCODE
}

$latentPath = Join-Path $inversionOutputDir "inversion_latent.pt"
if (-not (Test-Path $latentPath)) {
    Write-Error "O latent nao foi gerado em $latentPath"
    exit 1
}

Write-Host "Etapa 3/3: StyleCLIP -> edicao guiada por texto"
Write-Host "Prompt usado: $resolvedDescription"
& $styleclipWrapper -Description $resolvedDescription -Preset $Preset -Mode edit -Step $Step -Device $Device -CondaEnv $StyleclipEnv -LatentPath $latentPath -ResultsDir $styleclipOutputDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha na edicao StyleCLIP."
    exit $LASTEXITCODE
}

if (-not (Test-Path $styleclipFinalPath)) {
    Write-Error "A imagem final do StyleCLIP nao foi encontrada em $styleclipFinalPath"
    exit 1
}

Copy-Item -LiteralPath $styleclipFinalPath -Destination $finalImagePath -Force

Write-Host ""
Write-Host "Pipeline completo terminado."
Write-Host "Crop guardado em: $cropOutputDir"
Write-Host "Latent guardado em: $latentPath"
Write-Host "Resultados StyleCLIP em: $styleclipOutputDir"
Write-Host "Imagem final em: $finalImagePath"
