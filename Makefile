FACE_ENV ?= face
STYLECLIP_ENV ?= styleclip
INPUT ?= novo_dataset
DEVICE ?= auto
MARGIN ?= 0.15
REGION ?= cabelo
PRESET ?= sorriso
DESCRIPTION ?=
STEP ?= 10
STYLECLIP_STEP ?= 10
REPAINT_PROFILE ?= fast
REPAINT_TIMESTEPS ?= 0
REPAINT_JUMP_LENGTH ?= 0
REPAINT_JUMP_SAMPLES ?= 0
DILATION ?= 6
LATENT ?= outputs/retinaface_psp/02_psp_inversion/inversion_latent.pt
OUTPUT_STYLECLIP ?= outputs/face_to_styleclip
OUTPUT_REPAINT ?= outputs/face_to_repaint
OUTPUT_REPAINT_STYLECLIP ?= outputs/face_repaint_styleclip
OUTPUT_LOCAL_STYLECLIP ?= outputs/face_local_styleclip

.PHONY: help landmarks retinaface retinaface-psp retinaface-e4e styleclip-latent parsing crop invert styleclip-free styleclip-edit presets \
	repaint repaint-force repaint-styleclip repaint-styleclip-force local-styleclip local-styleclip-force clean-outputs

help:
	@echo Available targets:
	@echo   make landmarks
	@echo   make retinaface
	@echo   make retinaface-psp
	@echo   make retinaface-e4e
	@echo   make styleclip-latent PRESET=sorriso STEP=10
	@echo   make parsing
	@echo   make crop INPUT=novo_dataset
	@echo   make invert LATENT=outputs/inversion_test/inversion_latent.pt
	@echo   make styleclip-free PRESET=sorriso STEP=10
	@echo   make styleclip-edit PRESET=sorriso LATENT=outputs/inversion_test/inversion_latent.pt STEP=10
	@echo   make presets
	@echo   make repaint REGION=cabelo REPAINT_PROFILE=fast
	@echo   make repaint-force REGION=cabelo
	@echo   make repaint-styleclip REGION=cabelo PRESET=sorriso STYLECLIP_STEP=10
	@echo   make repaint-styleclip-force REGION=cabelo PRESET=sorriso
	@echo   make local-styleclip REGION=cabelo PRESET=sorriso STEP=10
	@echo   make local-styleclip-force REGION=cabelo PRESET=sorriso STEP=10
	@echo   make clean-outputs

landmarks:
	powershell -NoProfile -ExecutionPolicy Bypass -Command "conda run -n $(FACE_ENV) python scripts/export_numbered_landmarks.py --input \"$(INPUT)\" --output-dir outputs/landmarks --device \"$(DEVICE)\""

retinaface:
	powershell -NoProfile -ExecutionPolicy Bypass -Command "conda run -n $(FACE_ENV) python scripts/run_retinaface_check.py --input \"$(INPUT)\" --output-dir outputs/retinaface_check"

retinaface-psp:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_retinaface_to_e4e.ps1 -InputImage "$(INPUT)" -MarginScale $(MARGIN) -Device $(DEVICE) -FaceEnv $(FACE_ENV) -StyleclipEnv $(STYLECLIP_ENV) -OutputDir outputs/retinaface_psp -EncoderBackend psp -ReconstruirPreview

retinaface-e4e:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_retinaface_to_e4e.ps1 -InputImage "$(INPUT)" -MarginScale $(MARGIN) -Device $(DEVICE) -FaceEnv $(FACE_ENV) -StyleclipEnv $(STYLECLIP_ENV) -OutputDir outputs/retinaface_e4e

styleclip-latent:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_styleclip_from_latent.ps1 -LatentPath "$(LATENT)" -Preset "$(PRESET)" -Description "$(DESCRIPTION)" -Step $(STEP) -Device $(DEVICE) -StyleclipEnv $(STYLECLIP_ENV) -OutputDir outputs/styleclip_edit

parsing:
	powershell -NoProfile -ExecutionPolicy Bypass -Command "conda run -n $(FACE_ENV) python scripts/run_face_pipeline.py"

crop:
	powershell -NoProfile -ExecutionPolicy Bypass -Command "conda run -n $(FACE_ENV) python scripts/export_primary_face_crop.py --input \"$(INPUT)\" --output-dir outputs/crop_test --margin-scale $(MARGIN)"

invert:
	powershell -NoProfile -ExecutionPolicy Bypass -Command "conda run -n $(STYLECLIP_ENV) python scripts/invert_face_to_latent.py --input-crop outputs/crop_test/primary_face_crop.png --output-dir outputs/inversion_test --device $(DEVICE) --encoder-backend psp --save-reconstruction-preview --refine-latent-steps 15 --refine-learning-rate 0.015 --refine-latent-l2 0.0001"

styleclip-free:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_styleclip_optimization.ps1 -Preset "$(PRESET)" -Description "$(DESCRIPTION)" -Mode free_generation -Step $(STEP) -Device $(DEVICE) -CondaEnv $(STYLECLIP_ENV)

styleclip-edit:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_styleclip_optimization.ps1 -Preset "$(PRESET)" -Description "$(DESCRIPTION)" -Mode edit -Step $(STEP) -Device $(DEVICE) -CondaEnv $(STYLECLIP_ENV) -LatentPath "$(LATENT)"

presets:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -ListPresets

repaint:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_face_to_repaint.ps1 -InputImage "$(INPUT)" -Region $(REGION) -Profile $(REPAINT_PROFILE) -Timesteps $(REPAINT_TIMESTEPS) -JumpLength $(REPAINT_JUMP_LENGTH) -JumpSamples $(REPAINT_JUMP_SAMPLES) -Dilation $(DILATION) -MarginScale $(MARGIN) -Device $(DEVICE) -FaceEnv $(FACE_ENV) -RepaintEnv $(STYLECLIP_ENV) -OutputDir "$(OUTPUT_REPAINT)"

repaint-force:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_face_to_repaint.ps1 -InputImage "$(INPUT)" -Region $(REGION) -Profile $(REPAINT_PROFILE) -Timesteps $(REPAINT_TIMESTEPS) -JumpLength $(REPAINT_JUMP_LENGTH) -JumpSamples $(REPAINT_JUMP_SAMPLES) -Dilation $(DILATION) -MarginScale $(MARGIN) -Device $(DEVICE) -FaceEnv $(FACE_ENV) -RepaintEnv $(STYLECLIP_ENV) -OutputDir "$(OUTPUT_REPAINT)" -Force

repaint-styleclip:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_face_repaint_styleclip.ps1 -InputImage "$(INPUT)" -Region $(REGION) -Preset "$(PRESET)" -Description "$(DESCRIPTION)" -RepaintProfile $(REPAINT_PROFILE) -RepaintTimesteps $(REPAINT_TIMESTEPS) -RepaintJumpLength $(REPAINT_JUMP_LENGTH) -RepaintJumpSamples $(REPAINT_JUMP_SAMPLES) -StyleclipStep $(STYLECLIP_STEP) -Dilation $(DILATION) -MarginScale $(MARGIN) -Device $(DEVICE) -FaceEnv $(FACE_ENV) -StyleclipEnv $(STYLECLIP_ENV) -OutputDir "$(OUTPUT_REPAINT_STYLECLIP)"

repaint-styleclip-force:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_face_repaint_styleclip.ps1 -InputImage "$(INPUT)" -Region $(REGION) -Preset "$(PRESET)" -Description "$(DESCRIPTION)" -RepaintProfile $(REPAINT_PROFILE) -RepaintTimesteps $(REPAINT_TIMESTEPS) -RepaintJumpLength $(REPAINT_JUMP_LENGTH) -RepaintJumpSamples $(REPAINT_JUMP_SAMPLES) -StyleclipStep $(STYLECLIP_STEP) -Dilation $(DILATION) -MarginScale $(MARGIN) -Device $(DEVICE) -FaceEnv $(FACE_ENV) -StyleclipEnv $(STYLECLIP_ENV) -OutputDir "$(OUTPUT_REPAINT_STYLECLIP)" -Force

local-styleclip:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -InputImage "$(INPUT)" -Region $(REGION) -Preset "$(PRESET)" -Description "$(DESCRIPTION)" -Step $(STEP) -Dilation $(DILATION) -MarginScale $(MARGIN) -Device $(DEVICE) -FaceEnv $(FACE_ENV) -StyleclipEnv $(STYLECLIP_ENV) -OutputDir "$(OUTPUT_LOCAL_STYLECLIP)"

local-styleclip-force:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -InputImage "$(INPUT)" -Region $(REGION) -Preset "$(PRESET)" -Description "$(DESCRIPTION)" -Step $(STEP) -Dilation $(DILATION) -MarginScale $(MARGIN) -Device $(DEVICE) -FaceEnv $(FACE_ENV) -StyleclipEnv $(STYLECLIP_ENV) -OutputDir "$(OUTPUT_LOCAL_STYLECLIP)" -Force

clean-outputs:
	powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path outputs) { Remove-Item -LiteralPath outputs -Recurse -Force }; New-Item -ItemType Directory -Force outputs | Out-Null"
