# Face Editing Pipeline

## Visao Geral

Este projeto junta dois fluxos principais:

1. `imagem -> RetinaFace -> crop automatico -> Face Parsing`
2. `imagem -> RetinaFace -> crop automatico -> pSp inversao/latent -> StyleCLIP`
3. `imagem -> RetinaFace -> crop automatico -> Face Parsing -> keep mask -> RePaint`
4. `imagem -> RetinaFace -> crop automatico -> Face Parsing -> keep mask -> RePaint -> pSp/e4e inversao/latent -> StyleCLIP`
5. `imagem -> RetinaFace -> crop automatico -> pSp/e4e inversao/latent -> StyleCLIP -> mascara da regiao -> composicao localizada`

O objetivo do repositorio e ter um pipeline facial modular, em que conseguimos:

- detetar a face principal numa imagem
- recortar automaticamente essa face
- validar a segmentacao facial com `Face Parsing`
- inverter o crop para o espaco latente com `encoder4editing (e4e)` ou `pixel2style2pixel (pSp)`
- editar atributos faciais via texto com `StyleCLIP`
- editar regioes faciais localizadas com `RePaint`
- encadear `RePaint` e `StyleCLIP` numa unica pipeline
- controlar semanticamente a alteracao por texto e limitar o efeito a uma regiao facial

O projeto esta organizado para que a parte de deteccao/parsing viva no ambiente `face` e a parte de inversao/edicao viva no ambiente `styleclip`.

## Comando Principal

Agora tens um ponto de entrada unico na raiz do projeto:

- [run.ps1](/c:/Users/danny/Desktop/Projeto/run.ps1)
- [run.cmd](/c:/Users/danny/Desktop/Projeto/run.cmd)

O objetivo e deixares de decorar varios scripts grandes. Nesta fase, o comando principal esta focado apenas em reconstrucao, antes de qualquer edicao:

```powershell
.\run.cmd
```

Se estiveres explicitamente num terminal PowerShell, tambem podes usar:

```powershell
.\run.ps1
```

## Pipeline Hibrida de Alta Preservacao

A edicao hibrida passa a ter um caminho local que nao usa a reconstrucao StyleGAN como imagem final. A regra e:

```text
final = original_alinhado * (1 - mask) + fonte_editada * mask
```

Ou seja, `aligned_original.png` e sempre a base do resultado alinhado, e a imagem original completa e sempre a base de `final_on_original.png`. e4e, pSp e StyleCLIP ficam como fontes de edicao, nunca como substitutos da imagem inteira.

Novo caminho local:

```text
imagem -> RetinaFace -> aligned_original -> Face Parsing -> router -> mascara -> recolor/inpainting local -> blend -> inverse warp
```

Exemplos:

```powershell
.\run.cmd -Comando edit -Descricao "blue eyes" -EditRegion iris -UseLocalRecolor true -UseStyleCLIP false -UseFaceParsing true -Debug true
.\run.cmd -Comando edit -Descricao "blonde hair" -EditRegion hair -UseLocalRecolor true -UseStyleCLIP false -UseFaceParsing true -UseRePaint true -MaskDilation 8 -MaskBlur 15 -Debug true
.\run.cmd -Comando edit -TargetDescricao "a smiling person" -EditRegion mouth -UseStyleCLIP true -UseFaceParsing true -Passos 5 -ForcaEdicao 0.03 -MaskBlur 12 -Debug true
.\run.cmd -Comando edit -TargetDescricao "a person with a beard" -EditRegion beard -UseStyleCLIP true -UseFaceParsing true -Passos 5 -ForcaEdicao 0.035 -MaskBlur 7 -Debug true
.\run.cmd -Comando edit -TargetDescricao "an older person" -EditRegion face -UseStyleCLIP true -UseFaceParsing true -Passos 5 -ForcaEdicao 0.04 -UseRePaint false -Debug true
.\run.cmd -Comando edit -Descricao "blonde hair" -EditRegion hair -UseLocalRecolor true -UseStyleCLIP false -UseFaceParsing true -UseRePaint true -RePaintBackend repaint -RePaintSteps 20 -MaskDilation 8 -MaskBlur 15 -Debug true
```

`-RePaintBackend opencv` e o default rapido. `-RePaintBackend repaint` chama o RePaint real como refinamento local da borda da mascara e cai para OpenCV se o backend externo falhar, registando a razao em `params_log.txt`.

Outputs de debug principais:

- `original_image.png`
- `aligned_original.png`
- `landmarks_overlay.png`
- `parsing_map.png`
- `hair_mask.png`, `face_skin_mask.png`, `eyes_mask.png`, `mouth_mask.png`, `teeth_mask.png`, `neck_mask.png`, `background_mask.png`
- `iris_mask.png` e `iris_overlay.png` para edicoes de olhos
- `selected_mask.png`, `selected_mask_overlay.png`, `final_edit_mask.png`
- `final_blended_aligned.png`, `final_repainted_aligned.png`, `final_on_original.png`
- `difference_outside_mask.png`, `params_log.txt`, `latent_shape_log.txt`, `validation_report.json`

O `validation_report.json` deve reportar `changed_pixels = 0` em `outside_mask` para o resultado final.

Nota importante:

- em `cmd.exe` ou nalguns terminais do IDE, correr `.ps1` diretamente pode abrir o ficheiro no editor associado
- para evitar isso no Windows, usa `.\run.cmd`

Este comando faz por defeito:

- selecao da imagem unica em `novo_dataset/`
- deteccao com `RetinaFace`
- crop principal da face
- inversao do crop com `encoder4editing (e4e)`
- inversao do crop com `pixel2style2pixel (pSp)`
- escrita dos latents W+ e previews de reconstrucao
- comparacao lado a lado entre crop original, e4e e pSp

Para testar apenas um encoder:

```powershell
.\run.cmd -Comando e4e
.\run.cmd -Comando psp
```

Para editar com um prompt simples, podes usar diretamente:

```powershell
.\run.cmd "blonde hair"
.\run.cmd -Descricao "cabelo loiro"
```

Quando passas `-Descricao` ou um texto solto, o comando muda automaticamente para StyleCLIP e assume os valores seguros por defeito:

- `-Inversor auto`
- `-Passos 5`
- `-LayersInicio 8`
- `-LayersFim 17`
- `-LambdaLatent 0.05`
- `-DeltaClamp 0.08`

Se quiseres controlar tudo explicitamente:

```powershell
.\run.cmd -Comando reconstrucao
.\run.cmd -Comando styleclip -Inversor auto -TargetDescricao "a person with blonde hair"
```

Na edicao, `-Inversor auto` escolhe o melhor encoder a partir de `outputs\reconstruction_check\reconstruction_report.json` quando esse relatorio existe. Se ainda nao houver relatorio, usa pSp como fallback.

Se nao indicares `-Saida`, o comando principal de reconstrucao usa automaticamente:

- `outputs\reconstruction_check`

E nessa pasta deixa:

- `reconstruction_comparison.jpg`
- `reconstruction_report.json`
- `e4e\01_retinaface_crop\primary_face_crop.png`
- `e4e\02_e4e_inversion\inversion_latent.pt`
- `e4e\02_e4e_inversion\e4e_reconstruction_preview.png`
- `psp\01_retinaface_crop\primary_face_crop.png`
- `psp\02_psp_inversion\inversion_latent.pt`
- `psp\02_psp_inversion\psp_reconstruction_preview.png`

O comando `retinaface` continua disponivel para testar apenas a deteccao e os landmarks.
Os metodos `StyleCLIP`, `RePaint`, `Face Parsing` e composicao localizada ficam pausados no comando principal enquanto validamos a reconstrucao dos modelos um a um.

Defaults:

- comando sem prompt: `reconstrucao`
- comando com prompt: `styleclip`
- inversor StyleCLIP: `auto`
- passos StyleCLIP: `5`
- refinamento latent: `15` passos, quando for preciso gerar o latent
- imagem: `novo_dataset`

Para ver ajuda:

```powershell
.\run.cmd -Comando ajuda
```

## Regra Atual de Input

O comando principal usa agora, por defeito, a imagem dentro de `novo_dataset/`.
Como essa pasta ficou com uma unica imagem, a validacao do RetinaFace fica deterministica.

Isto aplica-se quando nao passas `-InputImage` ou `INPUT`, por exemplo em:

- `.\run.cmd`
- `.\run.cmd -Comando psp`
- `.\run.cmd -Comando e4e`
- `.\run.cmd -Comando retinaface`
- `make retinaface-psp`

Se quiseres usar uma imagem especifica, podes continuar a passar um caminho manual:

- `INPUT=novo_dataset\39864.png`
- `-InputImage "novo_dataset\39864.png"`

Por compatibilidade temporaria, `dataset` tambem e tratado como alias para `novo_dataset`.

## Arquitetura Atual

### Pipeline de parsing

`imagem -> RetinaFace -> crop -> Face Parsing`

Serve para validar deteccao e segmentacao.

### Pipeline de edicao facial

`imagem -> RetinaFace -> crop -> pSp inversao/latent -> StyleCLIP`

Serve para editar a face principal com texto.

### Pipeline de inpainting facial

`imagem -> RetinaFace -> crop -> Face Parsing -> keep mask -> RePaint`

Serve para apagar uma regiao facial especifica e deixa-la ser regenerada pelo modelo de inpainting.

### Pipeline combinada RePaint + StyleCLIP

`imagem -> RetinaFace -> crop -> Face Parsing -> keep mask -> RePaint -> pSp/e4e inversao -> StyleCLIP`

Serve para:

- primeiro alterar localmente uma regiao com `RePaint`
- depois refinar semanticamente o resultado com `StyleCLIP`

### Pipeline local por texto

`imagem -> RetinaFace -> crop -> pSp inversao -> StyleCLIP -> mascara da regiao -> composicao localizada`

Este e agora o metodo recomendado quando queres:

- dizer por texto o que queres alterar
- escolher em que regiao facial a alteracao deve aparecer
- evitar que a edicao afete a face toda

### Ambientes usados

- `face`
  contem `RetinaFace`, `tensorflow`, `facexlib` e o codigo de parsing

- `styleclip`
  contem `StyleCLIP`, `pixel2style2pixel`, `encoder4editing`, `RePaint`, `torch` e os scripts de inversao/edicao

Os wrappers PowerShell novos chamam os ambientes certos automaticamente, por isso nao precisas de estar sempre a ativar manualmente o ambiente correto para correr o pipeline integrado.

## Estrutura do Projeto

```text
Projeto/
  Makefile
  run.cmd
  run.ps1
  .cache/
    clip/
    matplotlib/
    matplotlib_styleclip/
    torch_extensions/
  novo_dataset/
  models/
    facexlib/
      detection_Resnet50_Final.pth
      parsing_bisenet.pth
      parsing_parsenet.pth
  outputs/
  scripts/
    export_primary_face_crop.py
    face_pipeline_utils.py
    invert_face_to_latent.py
    apply_local_text_edit.py
    prepare_repaint_face_inputs.py
    project_input_resolver.ps1
    run_face_pipeline.py
    run_face_local_styleclip.ps1
    run_face_repaint_styleclip.ps1
    run_face_to_repaint.ps1
    run_face_to_styleclip.ps1
    styleclip_prompt_presets.ps1
    run_styleclip_global_torch.ps1
    run_styleclip_optimization.ps1
    run_styleclip_repo.ps1
  third_party/
    RePaint/
    StyleCLIP/
    encoder4editing/
  README.md
```

## Makefile

Existe um [Makefile](/c:/Users/danny/Desktop/Projeto/Makefile) para facilitar os comandos mais usados do projeto.

Nota:

- para uso diario no Windows, o recomendado passou a ser `.\run.cmd`
- o comando `styleclip` no wrapper principal passa automaticamente para composicao localizada quando o prompt menciona regioes faciais
- o `Makefile` continua util para tarefas tecnicas e automatizacao

Se tiveres `make` instalado:

```powershell
make help
```

Se no Windows o comando `make` nao existir, tenta:

```powershell
mingw32-make help
```

Nesta maquina, o comando `make` ja ficou exposto globalmente atraves de `C:\MinGW\bin`, por isso deve funcionar:

```powershell
make help
```

Tambem foi validado dentro dos ambientes:

```powershell
conda run -n face make --version
conda run -n styleclip make --version
```

Targets principais:

- `make parsing`
- `make crop`
- `make invert`
- `make presets`
- `make styleclip-free PRESET=sorriso`
- `make styleclip-edit PRESET=sorriso LATENT=outputs/inversion_test/inversion_latent.pt`
- `make repaint REGION=cabelo REPAINT_PROFILE=fast`
- `make repaint-force REGION=cabelo`
- `make repaint-styleclip REGION=cabelo PRESET=sorriso STYLECLIP_STEP=10`
- `make repaint-styleclip-force REGION=cabelo PRESET=sorriso`
- `make clean-outputs`

Exemplos:

```powershell
make parsing
make repaint REGION=cabelo REPAINT_PROFILE=fast
make repaint-styleclip REGION=cabelo PRESET=sorriso STYLECLIP_STEP=1
make local-styleclip REGION=cabelo DESCRIPTION="uma pessoa com cabelo vermelho comprido e liso" STEP=1
```

## Ficheiros Principais

### `scripts/run_face_pipeline.py`

Pipeline de validacao do parsing.

Faz:

- selecao da imagem em `novo_dataset/`
- leitura da imagem
- deteccao de faces com `RetinaFace`
- crop automatico com margem
- `Face Parsing` com `facexlib`
- escrita de `crop`, `mask_gray`, `mask_color` e `overlay`

Comando:

```powershell
conda run -n face python scripts/run_face_pipeline.py
```

### `scripts/export_primary_face_crop.py`

Script auxiliar para exportar apenas o crop principal da face.

Faz:

- usa a imagem de `novo_dataset/` se `--input` vier vazio, `dataset` ou `novo_dataset`
- deteccao com `RetinaFace`
- ordenacao por `score`
- exportacao do melhor crop
- escrita de metadados em JSON

Comando:

```powershell
conda run -n face python scripts/export_primary_face_crop.py --input novo_dataset --output-dir outputs\crop_test
```

### `scripts/invert_face_to_latent.py`

Script de inversao da face para latent com `encoder4editing (e4e)`.

Faz:

- leitura do crop exportado
- resize e normalizacao para o encoder
- carregamento do checkpoint `e4e_ffhq_encode.pt`
- geracao do latent W+
- geracao de um preview da inversao
- escrita de metadados em JSON

Comando:

```powershell
conda run -n styleclip python scripts/invert_face_to_latent.py --input-crop outputs\crop_test\primary_face_crop.png --output-dir outputs\inversion_test --device auto
```

### `scripts/run_styleclip_optimization.ps1`

Wrapper simplificado para o metodo `optimization` do `StyleCLIP`.

Faz:

- escolhe `cpu` ou `cuda` automaticamente quando `-Device auto`
- aceita presets simples em portugues para nao teres de escrever prompts completos
- aceita `Description` em portugues ou ingles
- normaliza descricoes em portugues para um prompt final em ingles mais controlado
- passou a ter um dicionario portugues muito mais amplo, focado em edicao facial
- se a descricao em portugues tiver termos nao reconhecidos com seguranca, falha em vez de enviar um prompt ambiguo
- configura `PYTHONPATH`, `MPLCONFIGDIR` e cache local
- aponta para o checkpoint StyleGAN2 ja instalado
- corre `optimization/run_optimization.py`

Categorias de vocabulário em portugues agora cobertas:

- pessoas e sujeito: `pessoa`, `homem`, `mulher`, `rapaz`, `rapariga`
- regioes faciais: `cabelo`, `franja`, `pestanas`, `sobrancelhas`, `olhos`, `nariz`, `boca`, `labios`, `pele`, `barba`, `bigode`, `cavanhaque`, `bochechas`, `queixo`, `testa`, `mandibula`
- estilo e forma: `liso`, `ondulado`, `encaracolado`, `volumoso`, `fino`, `grosso`, `definido`, `arqueado`, `redondo`, `oval`, `quadrado`, `estreito`, `largo`, `simetrico`
- intensidade e comparativos: `mais`, `menos`, `muito`, `ligeiramente`, `mais comprido`, `mais fino`, `mais grosso`, `mais arqueado`, `mais cheio`
- cores e tons: `ruivo`, `loiro`, `castanho`, `preto`, `branco`, `cinzento`, `grisalho`, `prateado`, `rosa`, `roxo`, `lilas`, `dourado`
- maquilhagem e detalhe: `batom`, `delineador`, `rimel`, `blush`, `rugas`, `sardas`, `acne`

Mesmo assim, isto continua a ser um dicionario orientado ao dominio facial, nao um parser completo de portugues geral. Quando aparecer um termo fora deste dominio ou demasiado ambiguo, o projeto continua a falhar de proposito.

O prompt final usado fica guardado em:

- `prompt_info.json`

Listar presets:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_styleclip_optimization.ps1 -ListPresets
```

Comando para gerar uma face do zero:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_styleclip_optimization.ps1 -Preset sorriso -Mode free_generation -Step 10 -Device auto
```

Comando para editar a partir de um latent:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_styleclip_optimization.ps1 -Preset sorriso,cabelo_loiro -Mode edit -Step 10 -Device auto -LatentPath outputs\inversion_test\inversion_latent.pt
```

### `scripts/prepare_repaint_face_inputs.py`

Script que prepara os inputs do `RePaint` a partir do teu pipeline facial.

Faz:

- deteccao da face principal com `RetinaFace`
- crop automatico
- parsing facial com `facexlib`
- construcao de uma `keep mask`
  `255 = manter`
  `0 = regenerar`
- escrita de imagens e metadados de debug

Comando:

```powershell
conda run -n face python scripts/prepare_repaint_face_inputs.py --input novo_dataset\39864.png --output-dir outputs\repaint_prepare_test --region cabelo
```

### `scripts/run_face_to_repaint.ps1`

Wrapper do pipeline `RetinaFace + Face Parsing + RePaint`.

Faz:

1. usa a imagem de `novo_dataset/` quando nao passas `-InputImage`
1. usa o ambiente `face` para gerar o crop e a `keep mask`
2. cria automaticamente a config `yaml` do `RePaint`
3. usa o ambiente `styleclip` para correr o `RePaint`
4. guarda o resultado final em `outputs/`

Otimizacoes:

- perfil `fast` por defeito
- reutiliza automaticamente o ultimo resultado do `RePaint` no mesmo `OutputDir`
- recalculo forcado so quando usas `-Force`

Regioes suportadas:

- `pele`
- `sobrancelhas`
- `olhos`
- `orelhas`
- `nariz`
- `boca`
- `pescoco`
- `cabelo`

Comando:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_repaint.ps1 -Region cabelo -Profile fast -Device auto
```

### `scripts/run_face_repaint_styleclip.ps1`

Wrapper da cadeia completa `RePaint + StyleCLIP`.

Faz:

1. `RetinaFace + Face Parsing + keep mask`
2. `RePaint` para alterar localmente a regiao escolhida
3. `e4e` para inverter o resultado do `RePaint`
4. `StyleCLIP` para refinar com texto ou presets

Otimizacoes:

- reutiliza automaticamente o output do `RePaint` se ele ja existir no mesmo `OutputDir`
- reutiliza automaticamente a inversao `e4e` se o resultado do `RePaint` nao mudou
- para recalcular tudo, usa `-Force`

Comando:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_repaint_styleclip.ps1 -Region cabelo -Preset sorriso -RepaintProfile fast -StyleclipStep 10 -Device auto
```

### `scripts/apply_local_text_edit.py`

Script que limita a edicao do `StyleCLIP` a uma regiao facial.

Faz:

- parsing do crop original
- parsing do crop editado
- uniao das mascaras da regiao escolhida
- composicao suave da regiao editada sobre o crop original
- escrita do resultado localizado e da imagem final recolocada na imagem original

### `scripts/run_face_local_styleclip.ps1`

Este e o fluxo recomendado para controlo por texto.

Faz:

1. usa a imagem de `novo_dataset/` quando nao passas `-InputImage`
1. deteta a face principal e cria o crop
2. inverte esse crop com `e4e`
3. usa `StyleCLIP` para gerar a alteracao semantica a partir do teu texto
4. usa `Face Parsing` para aplicar essa alteracao so na regiao facial escolhida

Notas sobre a descricao:

- podes escrever `Description` em portugues ou ingles
- se estiver em portugues, o projeto converte para ingles controlado antes de enviar ao `StyleCLIP`
- o prompt final em ingles fica guardado em `03_styleclip/prompt_info.json`
- o projeto tenta detetar automaticamente as regioes mencionadas no prompt e aplica a composicao nessas zonas
- se o prompt mencionar varias zonas, por exemplo `olhos` e `cabelo`, a mascara final passa a ser a uniao dessas regioes
- se o texto em portugues for ambiguo ou tiver termos nao suportados, o script falha com erro claro

Comando:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -Region cabelo -Preset sorriso,cabelo_loiro -Step 10 -Device auto
```

Exemplo com descricao em portugues:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -Region cabelo -Description "uma pessoa com cabelo vermelho comprido e liso" -Step 10 -Device auto
```

Exemplo com descricao em ingles:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -Region cabelo -Description "a person with long straight bright red hair" -Step 10 -Device auto
```

### `scripts/run_face_to_styleclip.ps1`

Este e o comando que fecha a cadeia do projeto para edicao facial.

Faz:

1. usa a imagem de `novo_dataset/` quando nao passas `-InputImage`
2. usa o ambiente `face` para encontrar a face e criar o crop
3. usa o ambiente `styleclip` para inverter esse crop para latent com `e4e`
4. usa o `StyleCLIP` para editar o latent com base no texto

Tambem aceita presets simples em portugues, por exemplo:

- `sorriso`
- `jovem`
- `velho`
- `cabelo_loiro`
- `cabelo_preto`
- `cabelo_grisalho`
- `barba`
- `oculos`
- `batom_vermelho`
- `maquilhagem`

Podes ver a lista no terminal com:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -ListPresets
```

Comando:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset sorriso,cabelo_loiro -Step 10 -Device auto
```

## Fluxos de Trabalho Recomendados

### 0. Comando mais simples

Para o uso normal do projeto, usa:

```powershell
.\run.cmd
```

Exemplos rapidos:

```powershell
.\run.cmd -Comando psp
.\run.cmd -Comando retinaface
conda run -n face python scripts/run_retinaface_check.py
make retinaface-psp
make retinaface
```

### 1. Validar RetinaFace + pSp

Usa este comando quando quiseres confirmar que a face e detetada, recortada e invertida pelo `pixel2style2pixel`:

```powershell
.\run.cmd -Comando psp
```

Por defeito, este comando refina o latent W+ durante 15 passos para reduzir a perda de reconstrucao. Para uma projeção mais agressiva, podes aumentar:

```powershell
.\run.cmd -Comando psp -RefinarLatentePassos 40
```

O projeto guarda dois tipos de reconstrucao:

- `psp_reconstruction_preview.png`: reconstrucao pura a partir do latent W+.
- `psp_identity_restored_reconstruction.png`: reconstrucao com residual de detalhe, usada como diagnostico da informacao que ficou fora do latent.

### Pipeline completa de edicao controlada

Para reconstruir com pSp refinado e editar com StyleCLIP preservando identidade:

```powershell
.\run.cmd -Comando editar -SourceDescricao "a person with dark hair" -TargetDescricao "a person with blonde hair" -Passos 5 -ForcaEdicao 0.04 -LearningRate 0.02 -LayersInicio 8 -LayersFim 17 -LambdaLatent 0.05 -DeltaClamp 0.08
```

Outputs principais:

- `encoder_reconstruction.png`: reconstrucao pura do latent refinado.
- `perfect_reconstruction.png`: reconstrucao com residual de detalhe para preservar a imagem original.
- `styleclip_result.png`: resultado bruto do StyleCLIP.
- `localized/localized_crop.png`: edicao localizada na regiao do prompt.
- `original_latent.pt`, `edited_latent.pt`, `latent_delta.pt`: latents para analise e futuras edicoes.
- `styleclip_module_metadata.json`: parametros usados, prompts, layers, forca, LR e regularizacao.

Para comparar intensidades:

```powershell
.\run.cmd -Comando styleclip_sweep -TargetDescricao "a person with blonde hair" -Passos 5
```

### 2. Validar so o RetinaFace

Usa este comando quando quiseres confirmar que a deteccao da face e os landmarks estao corretos:

```powershell
conda run -n face python scripts/run_retinaface_check.py
```

### 3. Validar so o crop principal

Usa este comando quando quiseres testar apenas o `RetinaFace`:

```powershell
conda run -n face python scripts/export_primary_face_crop.py --input novo_dataset --output-dir outputs\crop_test
```

### 4. Validar so a inversao para latent

Depois de gerar o crop, usa:

```powershell
conda run -n styleclip python scripts/invert_face_to_latent.py --input-crop outputs\crop_test\primary_face_crop.png --output-dir outputs\inversion_test --device auto
```

### 4. Validar so o StyleCLIP a partir de um latent

Depois de gerar o latent, usa:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_styleclip_optimization.ps1 -Preset sorriso -Mode edit -Step 10 -Device auto -LatentPath outputs\inversion_test\inversion_latent.pt
```

### 5. Correr o RePaint sobre uma regiao facial

Este e o comando principal do `RePaint` no projeto:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_repaint.ps1 -Region cabelo -Profile fast -Device auto
```

### 6. Correr a cadeia local por texto

Este e o comando recomendado para o tipo de controlo que queres:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -Region cabelo -Preset sorriso,cabelo_loiro -Step 10 -Device auto
```

### 7. Correr a cadeia completa do RePaint + StyleCLIP

Este e o comando mais completo do projeto neste momento:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_repaint_styleclip.ps1 -Region cabelo -Preset sorriso -RepaintProfile fast -StyleclipStep 10 -Device auto
```

### 8. Correr a cadeia completa do StyleCLIP

Este e o comando principal do projeto para edicao facial:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset sorriso,cabelo_loiro -Step 10 -Device auto
```

### 9. Testes mais simples e rapidos

Se quiseres testar sem pensar no texto, usa um destes exemplos:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset sorriso -Step 1 -Device auto
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset barba -Step 1 -Device auto
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset jovem -Step 1 -Device auto
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset sorriso,cabelo_loiro -Step 1 -Device auto
```

## O Que Cada Output Significa

### Outputs do parsing

Quando corres `scripts/run_face_pipeline.py`, os resultados vao para `outputs/`:

- `face_1_crop.png`
  crop da face detetada

- `face_1_mask_gray.png`
  mascara de parsing em cinzento

- `face_1_mask_color.png`
  mascara colorida

- `face_1_overlay.png`
  overlay da mascara sobre o crop

### Outputs do pipeline integrado face -> StyleCLIP

Quando corres `scripts/run_face_to_styleclip.ps1`, os resultados vao por defeito para:

- `outputs/face_to_styleclip/01_crop/`
- `outputs/face_to_styleclip/02_inversion/`
- `outputs/face_to_styleclip/03_styleclip/`

Conteudo esperado:

- `01_crop/primary_face_crop.png`
  crop principal detetado pelo `RetinaFace`

- `01_crop/primary_face.json`
  score e bbox da deteccao

- `02_inversion/input_resized.png`
  versao redimensionada do crop para o encoder

- `02_inversion/inversion_preview.png`
  preview da reconstrucao feita pelo `e4e`

- `02_inversion/inversion_latent.pt`
  latent W+ que alimenta o `StyleCLIP`

- `02_inversion/inversion_metadata.json`
  metadados da inversao

- `03_styleclip/final_result.jpg`
  resultado final da edicao

- `resultado_final.jpg`
  atalho direto para a imagem final principal desta execucao

### Outputs do pipeline integrado face -> RePaint

Quando corres `scripts/run_face_to_repaint.ps1`, os resultados vao por defeito para:

- `outputs/face_to_repaint/01_inputs/`
- `outputs/face_to_repaint/02_repaint/`

Conteudo esperado:

- `01_inputs/gts/face.png`
  crop preparado para o `RePaint`

- `01_inputs/gt_keep_masks/face.png`
  mascara onde `255` significa manter e `0` significa regenerar

- `01_inputs/debug/parsing_labels.png`
  labels do parsing redimensionados para debug

- `01_inputs/debug/target_region.png`
  preview da regiao escolhida

- `02_repaint/inpainted/face.png`
  resultado final do `RePaint`

- `02_repaint/gt_masked/face.png`
  visualizacao da entrada mascarada

- `resultado_final.png`
  atalho direto para a imagem final principal desta execucao

### Outputs do pipeline integrado RePaint -> StyleCLIP

Quando corres `scripts/run_face_repaint_styleclip.ps1`, os resultados vao por defeito para:

- `outputs/face_repaint_styleclip/01_repaint/`
- `outputs/face_repaint_styleclip/02_inversion/`
- `outputs/face_repaint_styleclip/03_styleclip/`

Conteudo esperado:

- `01_repaint/01_inputs/...`
  inputs preparados para o `RePaint`

- `01_repaint/02_repaint/inpainted/face.png`
  saida do `RePaint`

- `02_inversion/inversion_latent.pt`
  latent gerado a partir do resultado do `RePaint`

- `02_inversion/inversion_preview.png`
  preview da inversao desse resultado

- `03_styleclip/final_result.jpg`
  resultado final apos o refinamento com `StyleCLIP`

- `resultado_final.jpg`
  atalho direto para a imagem final principal desta execucao

### Outputs do pipeline local por texto

Quando corres `scripts/run_face_local_styleclip.ps1`, os resultados vao por defeito para:

- `outputs/face_local_styleclip/01_crop/`
- `outputs/face_local_styleclip/02_inversion/`
- `outputs/face_local_styleclip/03_styleclip/`
- `outputs/face_local_styleclip/04_localized/`

Conteudo esperado:

- `03_styleclip/edited_result.jpg`
  crop editado pelo `StyleCLIP`

- `03_styleclip/prompt_info.json`
  descricao original e prompt final em ingles usado pelo modelo

- `04_localized/localized_crop.png`
  crop final limitado a regiao escolhida

- `04_localized/localized_mask.png`
  mascara usada na composicao

- `04_localized/localized_on_image.png`
  resultado final recolocado na imagem original

- `resultado_final.png`
  atalho direto para a imagem final principal desta execucao

## Modelos e Pesos Ja Preparados

### Face parsing

Em `models/facexlib/`:

- `detection_Resnet50_Final.pth`
- `parsing_bisenet.pth`
- `parsing_parsenet.pth`

### StyleCLIP

Em `third_party/StyleCLIP/pretrained_models/`:

- `stylegan2-ffhq-config-f.pt`

Em `.cache/clip/`:

- `ViT-B-32.pt`

### e4e

Em `third_party/encoder4editing/pretrained_models/`:

- `e4e_ffhq_encode.pt`

### RePaint

Em `third_party/RePaint/data/pretrained/`:

- `celeba256_250000.pt`

## Limpeza do Projeto

Foi feita uma limpeza dos ficheiros nao essenciais para execucao.

Removido:

- `outputs/` gerados por testes e validacoes anteriores
- `__pycache__/`
- `.cache/matplotlib/`
- `.cache/matplotlib_styleclip/`
- `.cache/torch_extensions/`

Mantido de proposito:

- `novo_dataset/`
- `models/`
- `third_party/`
- `.cache/clip/ViT-B-32.pt`

Nota:

- o ficheiro `.cache/clip/ViT-B-32.pt` foi mantido porque e usado pelo `StyleCLIP` e evita novo download
- a pasta `tests/` foi removida porque nao faz parte do funcionamento normal dos metodos

## Estado Atual do Projeto

Neste momento o projeto ja consegue:

- `imagem -> RetinaFace -> crop -> Face Parsing`
- `imagem -> RetinaFace -> crop -> pSp inversao/latent -> StyleCLIP`
- `imagem -> RetinaFace -> crop -> Face Parsing -> keep mask -> RePaint`
- `imagem -> RetinaFace -> crop -> Face Parsing -> keep mask -> RePaint -> pSp/e4e inversao/latent -> StyleCLIP`
- `imagem -> RetinaFace -> crop -> pSp/e4e inversao -> StyleCLIP -> mascara da regiao -> composicao localizada`

O segundo fluxo ja foi validado localmente com um teste curto em:

- `outputs/face_to_styleclip_smoke/01_crop/primary_face_crop.png`
- `outputs/face_to_styleclip_smoke/02_inversion/inversion_latent.pt`
- `outputs/face_to_styleclip_smoke/02_inversion/inversion_preview.png`
- `outputs/face_to_styleclip_smoke/03_styleclip/final_result.jpg`

O terceiro fluxo tambem ja foi validado localmente com um teste curto em:

- `outputs/face_to_repaint_smoke/01_inputs/gts/face.png`
- `outputs/face_to_repaint_smoke/01_inputs/gt_keep_masks/face.png`
- `outputs/face_to_repaint_smoke/02_repaint/inpainted/face.png`

O quarto fluxo tambem ja foi validado localmente com um teste curto em:

- `outputs/face_repaint_styleclip_smoke/01_repaint/02_repaint/inpainted/face.png`
- `outputs/face_repaint_styleclip_smoke/02_inversion/inversion_latent.pt`
- `outputs/face_repaint_styleclip_smoke/03_styleclip/final_result.jpg`

O quinto fluxo tambem ja foi validado localmente com um teste curto em:

- `outputs/face_local_styleclip_smoke/03_styleclip/edited_result.jpg`
- `outputs/face_local_styleclip_smoke/04_localized/localized_crop.png`
- `outputs/face_local_styleclip_smoke/04_localized/localized_on_image.png`

## Limitacoes Atuais

- o ambiente `styleclip` ja esta a ver a GPU nesta maquina
- com `-Device auto`, `StyleCLIP` e `RePaint` passam a usar `cuda` quando a sessao tem acesso a GPU
- o wrapper `run_face_to_styleclip.ps1` trabalha sobre a face principal da imagem
- o `StyleCLIP` esta funcional no fluxo `optimization`
- o `RePaint` oficial e guiado por mascara, nao por texto

Isto significa que:

- o `StyleCLIP` continua a ser o bloco semantico guiado por prompt
- o `RePaint` e o bloco de inpainting localizado guiado por mascara
- a cadeia combinada funciona, mas a regiao local ainda tem de ser escolhida manualmente com `-Region`
- se reutilizares o mesmo `OutputDir`, o `RePaint` passa a usar cache por defeito

Recomendacao pratica:

- se queres controlo por texto sobre cor, tamanho, estilo ou atributo, usa o fluxo local `StyleCLIP + composicao por mascara`
- se queres apenas regenerar uma regiao com inpainting, usa `RePaint`
- se escreveres em portugues, usa descricoes curtas e objetivas para a normalizacao ficar mais segura

Nota pratica:

- `torch.cuda.is_available()` no ambiente `styleclip` esta a devolver `True`
- a GPU vista pelo `torch` e `NVIDIA GeForce RTX 4060 Laptop GPU`
- um teste curto do `StyleCLIP` em `cuda` ja foi validado em [final_result.jpg](/c:/Users/danny/Desktop/Projeto/outputs/styleclip_cuda_check/final_result.jpg)
- um teste curto do `RePaint` em `cuda` ja foi validado em [face.png](/c:/Users/danny/Desktop/Projeto/outputs/face_to_repaint_cuda_check/02_repaint/inpainted/face.png)

## Como Verificar Se Esta Tudo Funcional

### Parsing

```powershell
conda run -n face python scripts/run_face_pipeline.py
```

### Edicao facial com a cadeia completa

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset sorriso -Step 1 -Device auto -OutputDir outputs\face_to_styleclip_smoke
```

Se estiver tudo bem, deves ver:

- crop gerado em `01_crop`
- latent e preview em `02_inversion`
- imagem final em `03_styleclip/final_result.jpg`

### Inpainting facial com RePaint

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_repaint.ps1 -Region cabelo -Profile fast -Device auto -OutputDir outputs\face_to_repaint_smoke
```

Se estiver tudo bem, deves ver:

- crop em `01_inputs/gts`
- keep mask em `01_inputs/gt_keep_masks`
- resultado final em `02_repaint/inpainted/face.png`

### Cadeia completa RePaint + StyleCLIP

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_repaint_styleclip.ps1 -Region cabelo -Preset sorriso -RepaintProfile fast -StyleclipStep 1 -Device auto -OutputDir outputs\face_repaint_styleclip_smoke
```

Se estiver tudo bem, deves ver:

- resultado do `RePaint` em `01_repaint/02_repaint/inpainted/face.png`
- latent em `02_inversion/inversion_latent.pt`
- resultado final do `StyleCLIP` em `03_styleclip/final_result.jpg`

### Validacao atual RetinaFace

```powershell
.\run.cmd -Saida outputs\retinaface_check
```

Se estiver tudo bem, deves ver:

- confirmacao `RETINAFACE OK` no terminal
- overlay em `retinaface_landmarks.png`
- metadados em `retinaface_detections.json`

## Troubleshooting

### `conda` nao reconhecido

Fecha e volta a abrir o terminal. Se precisares:

```powershell
. $PROFILE
```

### `ModuleNotFoundError` no parsing

Confirma que estas a correr a etapa de parsing no ambiente `face`, ou usa diretamente:

```powershell
conda run -n face python ...
```

### `CUDA nao esta disponivel`

Neste momento, nesta maquina, o `styleclip` ja consegue usar `cuda`.

Se numa sessao futura isso falhar, valida com:

```powershell
conda run -n styleclip python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem gpu')"
```

### `Nenhuma face foi detetada`

Confirma que:

- o `novo_dataset/` tem uma imagem valida
- a face esta visivel
- a face nao esta demasiado pequena ou tapada
- se quiseres testar outro ficheiro, passa `-InputImage` com um caminho concreto

### O pipeline integrado demora muito

Se o `styleclip` estiver a usar `cuda`, o tempo melhora bastante.

Para testes rapidos, usa poucos passos:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset sorriso -Step 1 -Device auto
```

Se quiseres controlo por texto e por regiao, usa:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -Region cabelo -Preset sorriso,cabelo_loiro -Step 1 -Device auto
```

Ou com descricao direta em portugues:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -Region cabelo -Description "uma pessoa com cabelo vermelho comprido e liso" -Step 1 -Device auto
```

Para o `RePaint`, o mais rapido e usar o perfil `fast`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_repaint.ps1 -Region cabelo -Profile fast -Device auto
```

Se quiseres obrigar o recalculo, usa:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_repaint.ps1 -Region cabelo -Profile fast -Device auto -Force
```

Na cadeia completa, combina `RePaint` curto com `StyleCLIP` curto:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_repaint_styleclip.ps1 -Region cabelo -Preset sorriso -RepaintProfile fast -StyleclipStep 1 -Device auto
```

## Comandos Rapidos

### Comando principal recomendado

```powershell
.\run.cmd
```

### Comando explicito RetinaFace + pSp

```powershell
.\run.cmd -Comando psp
```

### Alternativa RetinaFace + e4e

```powershell
.\run.cmd -Comando e4e
```

### Comando explicito RetinaFace

```powershell
.\run.cmd -Comando retinaface
```

### Atalho `.cmd`

```powershell
.\run.cmd
```

### RetinaFace direto

```powershell
conda run -n face python scripts/run_retinaface_check.py
```

### Crop principal

```powershell
conda run -n face python scripts/export_primary_face_crop.py --input novo_dataset --output-dir outputs\crop_test
```

### Inversao para latent

```powershell
conda run -n styleclip python scripts/invert_face_to_latent.py --input-crop outputs\crop_test\primary_face_crop.png --output-dir outputs\inversion_test --device auto
```

### StyleCLIP livre

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_styleclip_optimization.ps1 -Preset sorriso -Mode free_generation -Step 10 -Device auto
```

### StyleCLIP a partir de latent

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_styleclip_optimization.ps1 -Preset sorriso -Mode edit -Step 10 -Device auto -LatentPath outputs\inversion_test\inversion_latent.pt
```

### StyleCLIP localizado por regiao

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -Region cabelo -Preset sorriso,cabelo_loiro -Step 10 -Device auto
```

### StyleCLIP localizado por regiao com descricao em portugues

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_local_styleclip.ps1 -Region cabelo -Description "uma pessoa com cabelo vermelho comprido e liso" -Step 10 -Device auto
```

### RePaint localizado

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_repaint.ps1 -Region cabelo -Profile fast -Device auto
```

### Cadeia completa RePaint + StyleCLIP

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_repaint_styleclip.ps1 -Region cabelo -Preset sorriso -RepaintProfile fast -StyleclipStep 10 -Device auto
```

### Cadeia completa imagem -> StyleCLIP

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -Preset sorriso,cabelo_loiro -Step 10 -Device auto
```

### Ver presets disponiveis

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_face_to_styleclip.ps1 -ListPresets
```

### Makefile para edicao local por texto

```powershell
make local-styleclip REGION=cabelo PRESET=sorriso,cabelo_loiro STEP=1
```

Ou com descricao completa:

```powershell
make local-styleclip REGION=cabelo DESCRIPTION="uma pessoa com cabelo vermelho comprido e liso" STEP=1
```

## Regra de Documentacao

Sempre que houver alteracoes relevantes no projeto, este `README.md` deve ser atualizado com:

- o que mudou
- novos scripts e pastas
- novos comandos
- a forma correta de testar
- o impacto no pipeline global
