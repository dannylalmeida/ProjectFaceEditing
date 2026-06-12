function Get-ProjectRandomDatasetImage {
    param(
        [string]$ProjectRoot
    )

    $datasetRoots = @(
        (Join-Path $ProjectRoot "novo_dataset"),
        (Join-Path $ProjectRoot "38000")
    ) | Where-Object { Test-Path $_ -PathType Container }

    if (-not $datasetRoots -or $datasetRoots.Count -eq 0) {
        throw "As pastas de imagens nao existem: novo_dataset, 38000"
    }

    $imageFiles = foreach ($datasetRoot in $datasetRoots) {
        Get-ChildItem -Path $datasetRoot -Recurse -File -Include *.png,*.jpg,*.jpeg,*.webp -ErrorAction Stop
    }
    if (-not $imageFiles -or $imageFiles.Count -eq 0) {
        throw "Nao encontrei imagens dentro de $($datasetRoots -join ', ')"
    }

    return ($imageFiles | Get-Random).FullName
}

function Resolve-ProjectInputImage {
    param(
        [string]$InputImage,
        [string]$ProjectRoot
    )

    if (-not $InputImage) {
        return Get-ProjectRandomDatasetImage -ProjectRoot $ProjectRoot
    }

    $normalizedInput = $InputImage.Replace("/", "\").TrimEnd("\")
    if ($normalizedInput -in @("dataset", ".\dataset", "novo_dataset", ".\novo_dataset")) {
        return Get-ProjectRandomDatasetImage -ProjectRoot $ProjectRoot
    }

    $resolvedPath = [System.IO.Path]::GetFullPath($InputImage)
    if (Test-Path $resolvedPath -PathType Container) {
        $imageFiles = Get-ChildItem -Path $resolvedPath -Recurse -File -Include *.png,*.jpg,*.jpeg,*.webp -ErrorAction Stop
        if (-not $imageFiles -or $imageFiles.Count -eq 0) {
            throw "Nao encontrei imagens dentro de $resolvedPath"
        }

        return ($imageFiles | Get-Random).FullName
    }

    return $resolvedPath
}
