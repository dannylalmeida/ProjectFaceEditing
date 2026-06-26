param(
    [string[]]$Images = @(),
    [string]$OutputBase = "outputs/batch_test",
    [string]$DatasetDir = "dataset"
)

# Sem imagens especificadas → escolhe 1 aleatória do dataset.
if ($Images.Count -eq 0) {
    $candidates = Get-ChildItem $DatasetDir -Filter "*.png" -File
    if ($candidates.Count -eq 0) { $candidates = Get-ChildItem $DatasetDir -Filter "*.jpg" -File }
    if ($candidates.Count -eq 0) { Write-Error "Nenhuma imagem em '$DatasetDir'."; exit 1 }
    $picked = ($candidates | Get-Random).BaseName
    $Images = @($picked)
    Write-Host "Imagem aleatoria selecionada: $picked" -ForegroundColor Cyan
}

$ErrorActionPreference = "Continue"
$results = @()

foreach ($img in $Images) {
    $inputPath = "dataset/$img.png"
    if (-not (Test-Path $inputPath)) { continue }

    $outDir = "$OutputBase/$img"
    Write-Host "Processing $img..." -ForegroundColor Cyan

    $start = Get-Date
    $proc = conda run -n face python scripts/run_hybrid_edit.py `
        --input $inputPath `
        --output-dir $outDir `
        --target "labios maiores e sorriso maior" `
        --edit-region mouth `
        --repaint-backend opencv 2>&1
    $elapsed = (Get-Date) - $start

    $resultFile = "$outDir/resultado_final.png"
    $success = Test-Path $resultFile

    # Check validation report for outside-mask changes
    $validFile = "$outDir/validation_report.json"
    $outsideChanged = "N/A"
    if (Test-Path $validFile) {
        $v = Get-Content $validFile | ConvertFrom-Json
        $outsideChanged = $v.full_image_diff.outside_mask.changed_pixels
    }

    $results += [PSCustomObject]@{
        Image = $img
        Success = $success
        Seconds = [int]$elapsed.TotalSeconds
        OutsideMaskChangedPx = $outsideChanged
    }

    if (-not $success) {
        Write-Host "  FAILED: $img" -ForegroundColor Red
        $proc | Select-Object -Last 10 | Write-Host
    } else {
        Write-Host "  OK ($([int]$elapsed.TotalSeconds)s), outside_changed=$outsideChanged" -ForegroundColor Green
    }
}

Write-Host "`n=== BATCH SUMMARY ===" -ForegroundColor Yellow
$results | Format-Table -AutoSize
