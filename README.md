# Project Face Editing

Pipeline local para edicao facial controlada por texto, com foco em preservar a imagem original fora da regiao editada.

O projeto combina deteccao facial, alinhamento, segmentacao, mascaras por FaceMesh/Face Parsing, edicoes locais e, quando necessario, fluxos com pSp/e4e, StyleCLIP e RePaint.

## Estado Atual

- O ponto de entrada principal e `run.cmd` / `run.ps1`.
- O comando sem argumentos gera landmarks numerados.
- A base de imagens curada fica em `dataset/`.
- `dataset/`, `outputs/`, `models/` e `third_party/` sao locais e nao vao para o GitHub.
- As edicoes locais usam a imagem original como base e alteram apenas a mascara selecionada.
- A geometria do nariz foi reforcada para que prompts como `nariz mais curto` e `nariz mais comprido` sejam mais visiveis.

## Estrutura

```text
Projeto/
  README.md
  Makefile
  run.cmd
  run.ps1
  scripts/
    audit_dataset_nose_edit.py
    move_acceptable_images_to_dataset.py
    run_hybrid_edit.py
    run_retinaface_check.py
    run_reconstruction_check.ps1
    ...
  src/
    blending/
    cli/
    core/
    editors/
    encoders/
    evaluation/
    pipeline/
    segmentation/
  dataset/       # imagens aceites, local, ignorado pelo Git
  outputs/       # resultados/debug, local, ignorado pelo Git
  models/        # pesos locais, ignorado pelo Git
  third_party/   # repos externos locais, ignorado pelo Git
```

## O Que Vai Para O GitHub

O reposititorio guarda codigo, scripts, README, Makefile e wrappers.

Nao guarda:

- imagens do `dataset/`
- outputs de testes
- modelos `.pt`, `.pth`, `.onnx`, etc.
- caches locais
- repos externos em `third_party/`

Isto evita enviar muitos GB para o GitHub e mantem o repositorio leve.

## Ambientes

O projeto usa dois ambientes Conda principais:

- `face`: deteccao, alinhamento, FaceMesh, Face Parsing e edicoes locais.
- `styleclip`: pSp/e4e, StyleCLIP, RePaint e fluxo de latent.

Os wrappers chamam o ambiente certo automaticamente na maioria dos casos.

## Comandos Principais

Ver ajuda:

```powershell
.\run.cmd -Comando ajuda
```

Gerar landmarks numerados numa imagem aleatoria do `dataset/`:

```powershell
.\run.cmd
```

Gerar landmarks em lote:

```powershell
.\run.cmd -Comando landmarks -Batch -EditRegion nariz
```

Validar RetinaFace:

```powershell
.\run.cmd -Comando retinaface
```

Gerar reconstrucao e comparar e4e/pSp:

```powershell
.\run.cmd -Comando reconstrucao
```

Testar apenas um encoder:

```powershell
.\run.cmd -Comando e4e
.\run.cmd -Comando psp
```

## Edicao Local Recomendada

Para alteracoes localizadas, usa `-Comando edit` com a regiao explicita.

Olhos azuis sem StyleCLIP:

```powershell
.\run.cmd -Comando edit -Descricao "blue eyes" -EditRegion iris -UseLocalRecolor true -UseStyleCLIP false -AuditDebug true
```

Cabelo loiro com recolor local:

```powershell
.\run.cmd -Comando edit -Descricao "blonde hair" -EditRegion hair -UseLocalRecolor true -UseStyleCLIP false -AuditDebug true
```

Nariz mais curto:

```powershell
.\run.cmd -Comando edit -Descricao "nariz mais curto" -EditRegion nariz -UseStyleCLIP false -AuditDebug true
```

Nariz mais comprido:

```powershell
.\run.cmd -Comando edit -Descricao "nariz mais comprido" -EditRegion nariz -UseStyleCLIP false -AuditDebug true
```

Nariz maior e mais curto:

```powershell
.\run.cmd -Comando edit -Descricao "nariz maior e mais curto" -EditRegion nariz -UseStyleCLIP false -AuditDebug true
```

O resultado final usa sempre a imagem original como base. A regra e:

```text
final = original * (1 - mask) + edicao * mask
```

O `validation_report.json` deve manter `outside_mask.changed_pixels = 0`.

## Edicao Com StyleCLIP

Para edicoes sem caminho local direto, podes usar StyleCLIP:

```powershell
.\run.cmd -Comando styleclip -TargetDescricao "a smiling person" -EditRegion mouth
```

Quando `-Inversor auto` esta ativo, o script tenta escolher o melhor encoder com base em `outputs/reconstruction_check/reconstruction_report.json`. Se esse relatorio nao existir, usa pSp como fallback.

## Dataset Curado

A pasta `dataset/` e a base de imagens do projeto.

Regras atuais:

- imagens com resultado `bom` ou `mediano` sao consideradas aceitaveis
- imagens com resultado `mau` ficam fora do dataset
- os nomes originais das imagens sao preservados
- nao sao criados CSVs ou manifests dentro de `dataset/`

Contar imagens no dataset:

```powershell
Get-ChildItem -LiteralPath dataset -File |
  Where-Object { $_.Extension.ToLowerInvariant() -in '.png','.jpg','.jpeg','.webp' } |
  Measure-Object
```

## Testar Imagens Novas Para O Dataset

O script abaixo testa imagens novas com o pipeline de nariz e move apenas as aceitaveis para `dataset/`, mantendo os nomes originais.

Processar todas as pastas com prefixo `images1024x1024-`:

```powershell
conda run -n face python scripts\move_acceptable_images_to_dataset.py
```

Processar uma pasta especifica:

```powershell
conda run -n face python scripts\move_acceptable_images_to_dataset.py --source-dir caminho\para\pasta
```

Smoke test com poucas imagens:

```powershell
conda run -n face python scripts\move_acceptable_images_to_dataset.py --limit 10 --progress-every 1
```

O script usa uma pasta temporaria em `outputs/` e apaga-a no fim.

## Auditoria Com Relatorio

Para uma auditoria completa com relatorio, paineis e CSV fora do dataset:

```powershell
conda run -n face python scripts\audit_dataset_nose_edit.py --output-dir outputs\dataset_audit_nose
```

Usa isto quando quiseres analisar qualidade com mais detalhe. O `dataset/` continua limpo.

## Outputs

Os outputs sao separados por pasta para evitar sobrescrever resultados.

Exemplos:

- `outputs/hybrid_edit/`
- `outputs/reconstruction_check/`
- `outputs/param_tuning/`
- `outputs/dataset_audit_nose/`

Ficheiros comuns:

- `original_image.png`
- `resultado_final.png`
- `selected_mask_overlay.png`
- `edit_mask.png`
- `difference_inside_mask.png`
- `difference_outside_mask.png`
- `validation_report.json`
- `params_log.txt`

As pastas em `outputs/` sao debug/historico local; nao fazem parte do repositorio.

## Repositorio GitHub

Remote atual:

```text
https://github.com/dannylalmeida/ProjectFaceEditing.git
```

Fluxo normal:

```powershell
git status
git add README.md src scripts run.ps1 run.cmd Makefile .gitignore
git commit -m "Update project documentation"
git push
```

Antes de qualquer push, confirma que `dataset/`, `outputs/`, `models/` e `third_party/` continuam ignorados.

## Notas Importantes

- Em `cmd.exe`, usa `.\run.cmd`.
- Em PowerShell, podes usar `.\run.ps1`.
- Nao edites outputs como fonte de verdade; trata-os como diagnostico.
- Se uma alteracao afetar pixels fora da mascara, considera o resultado invalido.
- Para prompts de nariz, o caminho local geometrico e preferido a StyleCLIP porque preserva melhor a imagem original.
