param(
    [string]$InputImage = "",
    [ValidateSet("pele", "sobrancelhas", "olhos", "orelhas", "nariz", "boca", "pescoco", "cabelo")]
    [string]$Region = "cabelo",
    [int]$Dilation = 6,
    [ValidateSet("fast", "balanced", "quality")]
    [string]$Profile = "fast",
    [int]$Timesteps = 0,
    [int]$JumpLength = 0,
    [int]$JumpSamples = 0,
    [float]$MarginScale = 0.15,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$FaceEnv = "face",
    [string]$RepaintEnv = "styleclip",
    [switch]$Force,
    [string]$OutputDir = ""
)

function Convert-ToYamlPath {
    param(
        [string]$PathValue,
        [string]$RootPath
    )

    if ($RootPath) {
        $rootUri = New-Object System.Uri(($RootPath.TrimEnd("\") + "\"))
        $targetUri = New-Object System.Uri($PathValue)
        return [System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($targetUri).ToString())
    }

    return $PathValue.Replace("\", "/")
}

function Resolve-RepaintSchedule {
    param(
        [string]$SelectedProfile,
        [int]$RequestedTimesteps,
        [int]$RequestedJumpLength,
        [int]$RequestedJumpSamples
    )

    $presets = @{
        fast = @{ Timesteps = 20; JumpLength = 10; JumpSamples = 2 }
        balanced = @{ Timesteps = 40; JumpLength = 10; JumpSamples = 4 }
        quality = @{ Timesteps = 100; JumpLength = 10; JumpSamples = 5 }
    }

    $preset = $presets[$SelectedProfile]
    return @{
        Timesteps = $(if ($RequestedTimesteps -gt 0) { $RequestedTimesteps } else { $preset.Timesteps })
        JumpLength = $(if ($RequestedJumpLength -gt 0) { $RequestedJumpLength } else { $preset.JumpLength })
        JumpSamples = $(if ($RequestedJumpSamples -gt 0) { $RequestedJumpSamples } else { $preset.JumpSamples })
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "project_input_resolver.ps1")
$repaintRoot = Join-Path $projectRoot "third_party\RePaint"
$repaintModel = Join-Path $repaintRoot "data\pretrained\celeba256_250000.pt"
$prepareScript = Join-Path $projectRoot "scripts\prepare_repaint_face_inputs.py"

$InputImage = Resolve-ProjectInputImage -InputImage $InputImage -ProjectRoot $projectRoot

if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "outputs\face_to_repaint"
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}

$schedule = Resolve-RepaintSchedule -SelectedProfile $Profile -RequestedTimesteps $Timesteps -RequestedJumpLength $JumpLength -RequestedJumpSamples $JumpSamples
$Timesteps = $schedule.Timesteps
$JumpLength = $schedule.JumpLength
$JumpSamples = $schedule.JumpSamples

$inputStageDir = Join-Path $OutputDir "01_inputs"
$repaintStageDir = Join-Path $OutputDir "02_repaint"
$configPath = Join-Path $OutputDir "repaint_face_run.yml"
$runMetadataPath = Join-Path $OutputDir "run_metadata.json"
$repaintResultPath = Join-Path $repaintStageDir "inpainted\face.png"
$finalImagePath = Join-Path $OutputDir "resultado_final.png"

foreach ($path in @($inputStageDir, $repaintStageDir)) {
    New-Item -ItemType Directory -Force $path | Out-Null
}

if (-not (Test-Path $InputImage)) {
    Write-Error "Imagem de entrada nao encontrada em $InputImage"
    exit 1
}

Write-Host "Imagem selecionada: $InputImage"

if (-not (Test-Path $repaintRoot)) {
    Write-Error "Repositorio RePaint nao encontrado em $repaintRoot"
    exit 1
}

if (-not (Test-Path $repaintModel)) {
    Write-Error "Checkpoint do RePaint nao encontrado em $repaintModel"
    exit 1
}

if (-not $Force -and (Test-Path $repaintResultPath)) {
    Write-Host "RePaint reutilizado a partir da cache existente em $OutputDir"
}
else {
    Write-Host "Etapa 1/2: crop facial, parsing e keep mask para RePaint"
    & conda run -n $FaceEnv python $prepareScript --input $InputImage --output-dir $inputStageDir --region $Region --margin-scale $MarginScale --dilation $Dilation
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Falha a preparar os inputs do RePaint."
        exit $LASTEXITCODE
    }

    $resolvedDevice = $Device
    if ($Device -eq "auto") {
        $resolvedDevice = (& conda run -n $RepaintEnv python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')").Trim()
        if (-not $resolvedDevice) {
            $resolvedDevice = "cpu"
        }
    }

    $yaml = @"
attention_resolutions: 32,16,8
class_cond: false
diffusion_steps: 1000
learn_sigma: true
noise_schedule: linear
num_channels: 256
num_head_channels: 64
num_heads: 4
num_res_blocks: 2
resblock_updown: true
use_fp16: false
use_scale_shift_norm: true
classifier_scale: 4.0
lr_kernel_n_std: 2
num_samples: 1
show_progress: false
timestep_respacing: '$Timesteps'
use_kl: false
predict_xstart: false
rescale_timesteps: false
rescale_learned_sigmas: false
classifier_use_fp16: false
classifier_width: 128
classifier_depth: 2
classifier_attention_resolutions: 32,16,8
classifier_use_scale_shift_norm: true
classifier_resblock_updown: true
classifier_pool: attention
num_heads_upsample: -1
channel_mult: ''
dropout: 0.0
use_checkpoint: false
use_new_attention_order: false
clip_denoised: true
use_ddim: false
latex_name: RePaint
method_name: Repaint
image_size: 256
model_path: '$(Convert-ToYamlPath $repaintModel $repaintRoot)'
name: face_repaint
device: $resolvedDevice
inpa_inj_sched_prev: true
n_jobs: 0
print_estimated_vars: true
inpa_inj_sched_prev_cumnoise: false
schedule_jump_params:
  t_T: $Timesteps
  n_sample: 1
  jump_length: $JumpLength
  jump_n_sample: $JumpSamples
data:
  eval:
    project_face_mask:
      mask_loader: true
      gt_path: '$(Convert-ToYamlPath (Join-Path $inputStageDir "gts") $repaintRoot)'
      mask_path: '$(Convert-ToYamlPath (Join-Path $inputStageDir "gt_keep_masks") $repaintRoot)'
      image_size: 256
      class_cond: false
      deterministic: true
      random_crop: false
      random_flip: false
      return_dict: true
      drop_last: false
      batch_size: 1
      return_dataloader: true
      offset: 0
      max_len: 1
      paths:
        srs: '$(Convert-ToYamlPath (Join-Path $repaintStageDir "inpainted") $repaintRoot)'
        lrs: '$(Convert-ToYamlPath (Join-Path $repaintStageDir "gt_masked") $repaintRoot)'
        gts: '$(Convert-ToYamlPath (Join-Path $repaintStageDir "gt") $repaintRoot)'
        gt_keep_masks: '$(Convert-ToYamlPath (Join-Path $repaintStageDir "gt_keep_mask") $repaintRoot)'
"@

    Set-Content -Path $configPath -Value $yaml -Encoding UTF8

    Write-Host "Etapa 2/2: RePaint inference"
    Push-Location $repaintRoot
    try {
        & conda run -n $RepaintEnv python test.py --conf_path $configPath
    }
    finally {
        Pop-Location
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Falha ao correr o RePaint."
        exit $LASTEXITCODE
    }

    $runMetadata = @{
        input_image = $InputImage
        region = $Region
        dilation = $Dilation
        margin_scale = $MarginScale
        profile = $Profile
        timesteps = $Timesteps
        jump_length = $JumpLength
        jump_samples = $JumpSamples
        result_image = $repaintResultPath
    } | ConvertTo-Json
    Set-Content -Path $runMetadataPath -Value $runMetadata -Encoding UTF8
}

if (-not (Test-Path $repaintResultPath)) {
    Write-Error "A imagem final do RePaint nao foi encontrada em $repaintResultPath"
    exit 1
}

Copy-Item -LiteralPath $repaintResultPath -Destination $finalImagePath -Force

Write-Host ""
Write-Host "Pipeline RePaint terminado."
Write-Host "Config usada: $configPath"
Write-Host "Inputs preparados em: $inputStageDir"
Write-Host "Resultados gerados em: $repaintStageDir"
Write-Host "Imagem final em: $finalImagePath"
