param(
    [Alias("Acao", "Modo")]
    [ValidateSet("landmarks", "pontos", "reconstrucao", "reconstruir", "retinaface", "e4e", "psp", "styleclip", "editar", "edit", "styleclip_sweep", "ajuda")]
    [string]$Comando = "landmarks",

    [Alias("InputImage")]
    [string]$Imagem = "dataset",

    [Alias("OutputDir")]
    [string]$Saida = "",

    [Alias("Device")]
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Dispositivo = "auto",

    [Alias("Step")]
    [int]$Passos = 5,

    [Alias("SourceDescription")]
    [string]$SourceDescricao = "",

    [Alias("TargetDescription")]
    [string]$TargetDescricao = "",

    [Alias("Texto", "Description")]
    [string]$Descricao = "",

    [Alias("EditStrength")]
    [double]$ForcaEdicao = -1.0,

    [double]$LearningRate = -1.0,

    [Alias("LayersStart", "LatentLayerMin")]
    [int]$LayersInicio = -1,

    [Alias("LayersEnd", "LatentLayerMax")]
    [int]$LayersFim = -1,

    [Alias("L2Lambda")]
    [double]$LambdaLatent = -1.0,

    [Alias("MaxLatentDelta")]
    [double]$DeltaClamp = -1.0,

    [string]$UseFaceParsing = "true",

    [ValidateSet("auto", "true", "false", "1", "0", "yes", "no", "sim", "nao", "nÃ£o", "on", "off")]
    [string]$UseLocalRecolor = "auto",

    [ValidateSet("auto", "true", "false", "1", "0", "yes", "no", "sim", "nao", "nÃ£o", "on", "off")]
    [string]$UseStyleCLIP = "auto",

    [ValidateSet("auto", "true", "false", "1", "0", "yes", "no", "sim", "nao", "nÃ£o", "on", "off")]
    [string]$UseRePaint = "auto",

    [ValidateSet("auto", "mouth", "boca", "smile", "sorriso", "lips", "lip", "labios", "labio", "face", "pele", "skin", "age", "idade", "older", "younger", "eyes", "eye", "iris", "irises", "olhos", "olho", "glasses", "oculos", "eyebrows", "eyebrow", "sobrancelhas", "sobrancelha", "nose", "nariz", "ears", "ear", "orelhas", "orelha", "neck", "pescoco")]
    [string]$EditRegion = "auto",

    [int]$MaskDilation = -1,

    [int]$MaskErosion = 0,

    [int]$MaskBlur = -1,

    [int]$MaskThreshold = 1,

    [ValidateSet("true", "false", "1", "0", "yes", "no", "sim", "nao", "nÃ£o", "on", "off")]
    [string]$AuditDebug = "false",

    [int]$RePaintSteps = 20,

    [double]$RePaintStrength = 0.35,

    [ValidateSet("opencv", "repaint")]
    [string]$RePaintBackend = "opencv",

    [Alias("EncoderBackend", "Encoder")]
    [ValidateSet("auto", "e4e", "psp")]
    [string]$Inversor = "auto",

    [Alias("RefineLatentSteps")]
    [int]$RefinarLatentePassos = 15,

    [Alias("RefineLatentLearningRate")]
    [double]$RefinarLatenteLearningRate = 0.015,

    [Alias("RefineLatentL2")]
    [double]$RefinarLatenteL2 = 0.0001,

    [switch]$ReconstruirPreview,

    [switch]$Batch,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Prompt
)

function Show-ProjectHelp {
    Write-Host ""
    Write-Host "Comando principal do projeto"
    Write-Host ""
    Write-Host "Foco atual: landmarks numerados antes de qualquer edicao."
    Write-Host ""
    Write-Host "Uso simples:"
    Write-Host "  .\run.cmd `"big lips`""
    Write-Host "  .\run.cmd -Descricao `"blue eyes`""
    Write-Host ""
    Write-Host "Uso rapido:"
    Write-Host "  .\run.cmd"
    Write-Host "  .\run.cmd -Comando landmarks"
    Write-Host "  .\run.cmd -Comando landmarks -Batch -EditRegion nariz"
    Write-Host "  .\run.cmd -Comando reconstrucao"
    Write-Host "  .\run.cmd -Comando psp"
    Write-Host "  .\run.cmd -Comando e4e"
    Write-Host "  .\run.cmd -Comando retinaface"
    Write-Host ""
    Write-Host "Edicao explicita:"
    Write-Host "  .\run.cmd -Comando edit -Descricao `"big lips`" -EditRegion boca -AuditDebug true"
    Write-Host "  .\run.cmd -Comando edit -Descricao `"blue eyes`" -EditRegion iris -UseLocalRecolor true -UseStyleCLIP false -AuditDebug true"
    Write-Host "  .\run.cmd -Comando edit -Descricao `"sorriso`" -EditRegion mouth -UseStyleCLIP false -AuditDebug true"
    Write-Host ""
    Write-Host "Defaults:"
    Write-Host "  comando sem prompt: landmarks"
    Write-Host "  comando com prompt: styleclip"
    Write-Host "  imagem: dataset"
    Write-Host "  passos StyleCLIP: 5"
    Write-Host "  inversor: auto"
    Write-Host ""
}

function Convert-RunBool {
    param(
        [string]$Value,
        [bool]$Default
    )

    if ($null -eq $Value -or $Value -eq "") {
        return $Default
    }

    $normalized = $Value.ToString().Trim().ToLowerInvariant()
    if ($normalized -in @("1", "true", "yes", "sim", "s", "on")) {
        return $true
    }
    if ($normalized -in @("0", "false", "no", "nao", "nÃ£o", "n", "off")) {
        return $false
    }
    throw "Valor booleano invalido: $Value. Usa true/false."
}

function Normalize-RunAutoBool {
    param([string]$Value)

    if ($null -eq $Value -or $Value -eq "") {
        return "auto"
    }

    $normalized = $Value.ToString().Trim().ToLowerInvariant()
    if ($normalized -eq "auto") {
        return "auto"
    }
    if ($normalized -in @("1", "true", "yes", "sim", "s", "on")) {
        return "true"
    }
    if ($normalized -in @("0", "false", "no", "nao", "nÃ£o", "n", "off")) {
        return "false"
    }
    throw "Valor invalido: $Value. Usa auto, true ou false."
}

function Convert-RunNormalizedBool {
    param(
        [string]$Value,
        [bool]$Default
    )

    if ($Value -eq "auto") {
        return $Default
    }
    return Convert-RunBool -Value $Value -Default $Default
}

function Normalize-RunText {
    param([string]$Value)

    if ($null -eq $Value) {
        return ""
    }
    $decomposed = $Value.ToString().Normalize([System.Text.NormalizationForm]::FormD)
    return ([regex]::Replace($decomposed, "\p{Mn}", "")).ToLowerInvariant().Replace("-", " ")
}

function Test-RunLocalRecolorCandidate {
    param(
        [string]$Description,
        [string]$EditRegion
    )

    $rawDescription = if ($null -ne $Description) { $Description } else { "" }
    $rawRegion = if ($null -ne $EditRegion) { $EditRegion } else { "auto" }
    $text = " " + (Normalize-RunText -Value $rawDescription) + " "
    $region = Normalize-RunText -Value $rawRegion
    $colorPattern = "\b(blue|green|brown|hazel|gray|grey|black|white|blond|blonde|gold|azul|azuis|verde|verdes|castanho|castanha|castanhos|castanhas|cinza|cinzento|cinzenta|preto|preta|loiro|loira|dourado|dourada)\b"

    if ($region -in @("iris", "irises", "eyes", "eye", "olhos", "olho") -and $text -match $colorPattern) {
        return $true
    }
    if ($text -match "$colorPattern\s+(eye|eyes|olho|olhos)\b") {
        return $true
    }
    return $false
}

function Test-RunLocalGeometryCandidate {
    param(
        [string]$Description,
        [string]$EditRegion
    )

    $rawDescription = if ($null -ne $Description) { $Description } else { "" }
    $rawRegion = if ($null -ne $EditRegion) { $EditRegion } else { "auto" }
    $text = " " + (Normalize-RunText -Value $rawDescription) + " "
    $region = Normalize-RunText -Value $rawRegion
    $isNose = $region -in @("nose", "nariz") -or $text -match "\b(nose|nariz)\b"
    $isDecrease = $text -match "\b(smaller|small|thin|thinner|narrow|narrower|less|shorter|decrease|reduce|reduced|shrink|menor|menores|pequeno|pequena|fino|fina|afinar|diminuir|diminuido|diminuida|reduzir|reduzido|reduzida|encolher|encolhido|encolhida)\b"
    $isIncrease = $text -match "\b(big|large|larger|bigger|wide|wider|widder|broad|broader|full|fuller|thick|thicker|plump|plumper|pouty|maior|maiores|grande|grandes|largo|larga|largos|largas|cheio|cheia|cheios|cheias|grosso|grossa|grossos|grossas|carnudo|carnuda|carnudos|carnudas)\b"
    $isLengthEdit = $text -match "\b(long|longer|short|shorter|comprido|comprida|compridos|compridas|longo|longa|longos|longas|curto|curta|curtos|curtas)\b"
    $isMouth = $region -in @("mouth", "boca", "smile", "sorriso", "lips", "lip", "labios", "labio") -or $text -match "\b(mouth|boca|smile|smiling|sorriso|sorridente|lips|lip|labios|labio)\b"
    $isSmileEdit = $text -match "\b(smile|smiling|sorriso|sorridente|sorrir)\b"
    $isMouthShapeEdit = $isMouth -and ($isSmileEdit -or $isDecrease -or $isIncrease -or $text -match "\b(open|opened|closed|close|aberta|aberto|fechada|fechado|abrir|fechar)\b")
    return (($isNose -and ($isDecrease -or $isLengthEdit)) -or $isMouthShapeEdit)
}

function Resolve-RunHybridRePaint {
    param(
        [string]$Requested,
        [string]$Description,
        [string]$EditRegion
    )

    if ($Requested -eq "true") {
        return $true
    }
    if ($Requested -eq "false") {
        return $false
    }

    $rawDescription = if ($null -ne $Description) { $Description } else { "" }
    $rawRegion = if ($null -ne $EditRegion) { $EditRegion } else { "auto" }
    $text = " " + (Normalize-RunText -Value $rawDescription) + " "
    $region = Normalize-RunText -Value $rawRegion
    if ($region -in @("iris", "irises", "eyes", "eye", "olhos", "olho")) {
        return $false
    }
    if ($region -in @("mouth", "boca", "smile", "sorriso")) {
        return $true
    }
    if ($text -match "\b(mouth|boca|smile|sorriso)\b") {
        return $true
    }
    return $false
}

$projectRoot = $PSScriptRoot
$scriptsDir = Join-Path $projectRoot "scripts"
$runLandmarks = Join-Path $scriptsDir "export_numbered_landmarks.py"
$runRetinaface = Join-Path $scriptsDir "run_retinaface_check.py"
$runInversion = Join-Path $scriptsDir "run_retinaface_to_e4e.ps1"
$runReconstruction = Join-Path $scriptsDir "run_reconstruction_check.ps1"
$runStyleclip = Join-Path $scriptsDir "run_styleclip_from_latent.ps1"
$runHybridEdit = Join-Path $scriptsDir "run_hybrid_edit.py"
$sweepComparisonScript = Join-Path $scriptsDir "create_sweep_comparison.py"

# Quando $Imagem é uma directoria, selecciona uma imagem PNG aleatória dentro dela.
$resolvedImagem = $Imagem
if (Test-Path $resolvedImagem -PathType Container) {
    $candidates = Get-ChildItem $resolvedImagem -Filter "*.png" -File
    if ($candidates.Count -eq 0) {
        $candidates = Get-ChildItem $resolvedImagem -Filter "*.jpg" -File
    }
    if ($candidates.Count -eq 0) {
        Write-Error "Nenhuma imagem encontrada em '$resolvedImagem'."
        exit 1
    }
    $resolvedImagem = ($candidates | Get-Random).FullName
    Write-Host "Imagem aleatoria: $(Split-Path $resolvedImagem -Leaf)" -ForegroundColor Cyan
}
$Imagem = $resolvedImagem

if ($Prompt -and $Prompt.Count -gt 0) {
    $promptText = ($Prompt -join " ").Trim()
    if ($promptText) {
        if (-not $Descricao -and -not $TargetDescricao) {
            $Descricao = $promptText
        }
        elseif ($Descricao) {
            $Descricao = (($Descricao, $promptText) -join " ").Trim()
        }
        elseif ($TargetDescricao) {
            $TargetDescricao = (($TargetDescricao, $promptText) -join " ").Trim()
        }
    }
}

if ($Comando -in @("landmarks", "pontos", "reconstrucao", "reconstruir") -and ($Descricao -or $TargetDescricao -or $SourceDescricao)) {
    $Comando = "styleclip"
}

switch ($Comando) {
    "ajuda" {
        Show-ProjectHelp
    }

    { $_ -in @("landmarks", "pontos") } {
        $landmarkArgs = @($runLandmarks, "--input", $Imagem, "--device", $Dispositivo, "--edit-region", $EditRegion)
        if ($Batch) {
            $landmarkArgs += @("--batch")
        }
        if ($Saida) {
            $landmarkArgs += @("--output-dir", $Saida)
        }
        & conda run -n face python @landmarkArgs
        exit $LASTEXITCODE
    }

    "retinaface" {
        $retinafaceArgs = @($runRetinaface, "--input", $Imagem)
        if ($Saida) {
            $retinafaceArgs += @("--output-dir", $Saida)
        }
        & conda run -n face python @retinafaceArgs
        exit $LASTEXITCODE
    }

    { $_ -in @("reconstrucao", "reconstruir") } {
        & $runReconstruction `
            -InputImage $Imagem `
            -Device $Dispositivo `
            -OutputDir $Saida `
            -RefinarLatentePassos $RefinarLatentePassos `
            -RefinarLatenteLearningRate $RefinarLatenteLearningRate `
            -RefinarLatenteL2 $RefinarLatenteL2
        exit $LASTEXITCODE
    }

    "e4e" {
        & $runInversion `
            -InputImage $Imagem `
            -Device $Dispositivo `
            -OutputDir $Saida `
            -EncoderBackend e4e `
            -ReconstruirPreview `
            -RefinarLatentePassos $RefinarLatentePassos `
            -RefinarLatenteLearningRate $RefinarLatenteLearningRate `
            -RefinarLatenteL2 $RefinarLatenteL2
        exit $LASTEXITCODE
    }

    "psp" {
        & $runInversion `
            -InputImage $Imagem `
            -Device $Dispositivo `
            -OutputDir $Saida `
            -EncoderBackend psp `
            -ReconstruirPreview `
            -RefinarLatentePassos $RefinarLatentePassos `
            -RefinarLatenteLearningRate $RefinarLatenteLearningRate `
            -RefinarLatenteL2 $RefinarLatenteL2
        exit $LASTEXITCODE
    }

    { $_ -in @("styleclip", "editar", "edit") } {
        if (-not $TargetDescricao -and $Descricao) {
            $TargetDescricao = $Descricao
        }
        if (-not $TargetDescricao) {
            Write-Error "Falta a descricao da edicao. Usa, por exemplo: .\run.cmd `"big lips`""
            exit 1
        }
        $resolvedUseFaceParsing = Convert-RunBool -Value $UseFaceParsing -Default $true
        $resolvedUseRePaint = Normalize-RunAutoBool -Value $UseRePaint
        $resolvedUseLocalRecolorSetting = Normalize-RunAutoBool -Value $UseLocalRecolor
        $resolvedUseStyleCLIPSetting = Normalize-RunAutoBool -Value $UseStyleCLIP
        $localRecolorCandidate = Test-RunLocalRecolorCandidate -Description $TargetDescricao -EditRegion $EditRegion
        $localGeometryCandidate = Test-RunLocalGeometryCandidate -Description $TargetDescricao -EditRegion $EditRegion
        $resolvedUseLocalRecolor = Convert-RunNormalizedBool -Value $resolvedUseLocalRecolorSetting -Default $localRecolorCandidate
        if ($resolvedUseStyleCLIPSetting -eq "auto") {
            $resolvedUseStyleCLIP = if ($localGeometryCandidate) { $false } elseif ($Comando -in @("styleclip", "editar")) { $true } else { -not ($resolvedUseLocalRecolor -and $localRecolorCandidate) }
        }
        else {
            $resolvedUseStyleCLIP = Convert-RunNormalizedBool -Value $resolvedUseStyleCLIPSetting -Default $true
        }
        if ($localGeometryCandidate -and $resolvedUseStyleCLIP) {
            if ($resolvedUseStyleCLIPSetting -eq "auto") {
                Write-Host "Nota: edicao geometrica local detectada; a usar geometria local em vez de StyleCLIP para preservar qualidade."
            }
            else {
                Write-Host "Nota: UseStyleCLIP=true foi pedido, mas esta edicao de forma usa landmarks locais para preservar qualidade."
            }
            $resolvedUseStyleCLIP = $false
        }
        if ($resolvedUseStyleCLIP -and $resolvedUseLocalRecolorSetting -eq "auto") {
            $resolvedUseLocalRecolor = $false
        }
        if (-not $Saida) {
            $Saida = if ($resolvedUseStyleCLIP) { Join-Path $projectRoot "outputs\styleclip_edit" } else { Join-Path $projectRoot "outputs\hybrid_edit" }
        }
        $resolvedDebug = Convert-RunBool -Value $AuditDebug -Default $true

        Write-Host "Edicao por prompt simples"
        Write-Host "Descricao: $TargetDescricao"
        Write-Host "A assumir: Inversor=$Inversor, Passos=$Passos, imagem=$Imagem, output=$Saida"
        Write-Host "Mascara: UseFaceParsing=$resolvedUseFaceParsing, EditRegion=$EditRegion, UseRePaint=$resolvedUseRePaint"
        Write-Host "Router: UseLocalRecolor=$resolvedUseLocalRecolor, UseStyleCLIP=$resolvedUseStyleCLIP, LocalCandidate=$localRecolorCandidate, GeometryCandidate=$localGeometryCandidate"
        Write-Host "A composicao final usa a imagem original/crop como base."

        if (-not $resolvedUseStyleCLIP) {
            $resolvedHybridUseRePaint = if ($localGeometryCandidate) { $false } else { Resolve-RunHybridRePaint -Requested $resolvedUseRePaint -Description $TargetDescricao -EditRegion $EditRegion }
            $hybridArgs = @(
                $runHybridEdit,
                "--input", $Imagem,
                "--output-dir", $Saida,
                "--description", $TargetDescricao,
                "--target-description", $TargetDescricao,
                "--edit-region", $EditRegion,
                "--use-face-parsing", ([string]$resolvedUseFaceParsing).ToLowerInvariant(),
                "--use-local-recolor", ([string]$resolvedUseLocalRecolor).ToLowerInvariant(),
                "--use-styleclip", "false",
                "--mask-dilation", ([string]$MaskDilation),
                "--mask-erosion", ([string]$MaskErosion),
                "--mask-blur", ([string]$MaskBlur),
                "--mask-threshold", ([string]$MaskThreshold),
                "--use-repaint", ([string]$resolvedHybridUseRePaint).ToLowerInvariant(),
                "--repaint-steps", ([string]$RePaintSteps),
                "--repaint-strength", ([string]$RePaintStrength),
                "--repaint-backend", $RePaintBackend,
                "--debug", ([string]$resolvedDebug).ToLowerInvariant()
            )
            if ($SourceDescricao) {
                $hybridArgs += @("--source-description", $SourceDescricao)
            }
            & conda run -n face python @hybridArgs
            exit $LASTEXITCODE
        }

        $selectedInversor = $Inversor
        if ($selectedInversor -eq "auto") {
            $selectedInversor = "psp"
            $reconstructionReport = Join-Path $projectRoot "outputs\reconstruction_check\reconstruction_report.json"
            if (Test-Path -LiteralPath $reconstructionReport) {
                try {
                    $report = Get-Content -LiteralPath $reconstructionReport -Raw | ConvertFrom-Json
                    $e4eScore = if ($report.e4e.metrics.global_ssim) { [double]$report.e4e.metrics.global_ssim } else { -1.0 }
                    $pspScore = if ($report.psp.metrics.global_ssim) { [double]$report.psp.metrics.global_ssim } else { -1.0 }
                    if ($e4eScore -gt $pspScore) {
                        $selectedInversor = "e4e"
                    }
                }
                catch {
                    $selectedInversor = "psp"
                }
            }
            Write-Host "Inversor auto selecionado para StyleCLIP: $selectedInversor"
        }
        $inversionRoot = if ($selectedInversor -eq "psp") { Join-Path $projectRoot "outputs\retinaface_psp" } else { Join-Path $projectRoot "outputs\retinaface_e4e" }
        $inversionDirName = if ($selectedInversor -eq "psp") { "02_psp_inversion" } else { "02_e4e_inversion" }
        $latentPath = Join-Path $inversionRoot "$inversionDirName\inversion_latent.pt"
        $latentSourceFile = Join-Path $inversionRoot "$inversionDirName\inversion_source.txt"
        $cropMetadata = Join-Path $inversionRoot "01_retinaface_crop\primary_face.json"

        $needsInversion = -not (Test-Path -LiteralPath $latentPath)
        if (-not $needsInversion) {
            if (Test-Path -LiteralPath $latentSourceFile) {
                $storedSource = (Get-Content -LiteralPath $latentSourceFile -Raw).Trim()
                if ($storedSource -ne $Imagem) {
                    $needsInversion = $true
                    Write-Host "Imagem alterada ($(Split-Path $storedSource -Leaf) → $(Split-Path $Imagem -Leaf)). A regenerar latent..." -ForegroundColor Yellow
                }
            } else {
                # Latent existe mas sem registo de origem — proveniência desconhecida, re-inverter.
                $needsInversion = $true
                Write-Host "Fonte do latent desconhecida. A regenerar para $(Split-Path $Imagem -Leaf)..." -ForegroundColor Yellow
            }
        }

        if ($needsInversion) {
            if (-not (Test-Path -LiteralPath $latentPath)) {
                Write-Host "Latent $selectedInversor ainda nao existe. A gerar reconstrucao primeiro..."
            }
            & $runInversion `
                -InputImage $Imagem `
                -Device $Dispositivo `
                -EncoderBackend $selectedInversor `
                -ReconstruirPreview `
                -RefinarLatentePassos $RefinarLatentePassos `
                -RefinarLatenteLearningRate $RefinarLatenteLearningRate `
                -RefinarLatenteL2 $RefinarLatenteL2
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
            $Imagem | Set-Content -LiteralPath $latentSourceFile -Encoding utf8 -NoNewline
        }

        $disableDirectRefinements = -not $resolvedUseLocalRecolor
        & $runStyleclip `
            -LatentPath $latentPath `
            -TargetDescription $TargetDescricao `
            -SourceDescription $(if ($SourceDescricao) { $SourceDescricao } else { "auto" }) `
            -Step $Passos `
            -Device $Dispositivo `
            -OutputDir $Saida `
            -CropMetadataPath $cropMetadata `
            -EditStrength $ForcaEdicao `
            -LearningRate $LearningRate `
            -LatentLayerMin $LayersInicio `
            -LatentLayerMax $LayersFim `
            -L2Lambda $LambdaLatent `
            -MaxLatentDelta $DeltaClamp `
            -UseFaceParsing $resolvedUseFaceParsing `
            -UseRePaint $resolvedUseRePaint `
            -EditRegion $EditRegion `
            -MaskDilation $MaskDilation `
            -MaskErosion $MaskErosion `
            -MaskBlur $MaskBlur `
            -MaskThreshold $MaskThreshold `
            -RePaintSteps $RePaintSteps `
            -RePaintStrength $RePaintStrength `
            -RePaintBackend $RePaintBackend `
            -AuditDebug $resolvedDebug `
            -DisableDirectRefinements:$disableDirectRefinements `
            -SaveComparisons:$resolvedDebug
        exit $LASTEXITCODE
    }

    "styleclip_sweep" {
        Write-Error "O sweep de StyleCLIP esta pausado enquanto validamos apenas a reconstrucao."
        exit 1
    }
}

