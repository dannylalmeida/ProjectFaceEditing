# Project Face Editing

Pipeline local para edicao facial controlada por texto. A ideia central e simples:
editar apenas a regiao indicada (nariz, boca, olhos, cabelo, ...) e manter o resto
da imagem original intacto.

## Como Funciona

1. Deteta e alinha o rosto na imagem.
2. Cria uma mascara da regiao a editar (via FaceMesh / Face Parsing).
3. Aplica a edicao so dentro dessa mascara.
4. Junta o resultado a imagem original usando a mascara:

```text
final = original * (1 - mask) + edicao * mask
```

Assim, tudo o que esta fora da mascara nao muda. O `validation_report.json`
de cada resultado deve manter `outside_mask.changed_pixels = 0`.

Ha dois caminhos de edicao:

- **Local (recomendado):** geometria (nariz, boca, labios, sorriso) e recolor
  (olhos, cabelo). Rapido e preserva melhor o original.
- **StyleCLIP:** para edicoes mais abertas sem caminho local (ex.: "make the
  person older").

## Estrutura

```text
Projeto/
  run.cmd / run.ps1   # ponto de entrada
  src/                # codigo do pipeline
    core/             # deteccao e alinhamento
    segmentation/     # mascaras (FaceMesh, Face Parsing)
    editors/          # edicoes locais, StyleCLIP, RePaint
    encoders/         # pSp / e4e
    blending/         # juncao com a imagem original
    pipeline/         # orquestracao
  scripts/            # utilitarios e testes
  dataset/            # imagens de entrada (local, ignorado pelo Git)
  outputs/            # resultados e debug (local, ignorado pelo Git)
  models/             # pesos (local, ignorado pelo Git)
  third_party/        # repos externos (local, ignorado pelo Git)
```

Apenas codigo, scripts e wrappers vao para o GitHub. `dataset/`, `outputs/`,
`models/` e `third_party/` ficam sempre locais.

## Ambientes (Conda)

- `face`: deteccao, alinhamento, mascaras e edicoes locais.
- `styleclip`: pSp/e4e, StyleCLIP, RePaint.

Os wrappers escolhem o ambiente certo automaticamente.

## Comandos Principais

Sem argumentos, o comando escolhe uma imagem aleatoria do `dataset/`.

```powershell
.\run.cmd -Comando ajuda            # ver ajuda
.\run.cmd                           # landmarks numa imagem aleatoria
.\run.cmd -Comando reconstrucao     # comparar e4e / pSp
```

Edicao local (regiao explicita + descricao):

```powershell
# Nariz
.\run.cmd -Comando edit -Descricao "nariz mais curto" -EditRegion nariz -UseStyleCLIP false

# Boca / labios / sorriso
.\run.cmd -Comando edit -Descricao "sorriso" -EditRegion mouth -UseStyleCLIP false
.\run.cmd -Comando edit -Descricao "labios mais cheios" -EditRegion mouth -UseStyleCLIP false

# Olhos / cabelo (recolor local)
.\run.cmd -Comando edit -Descricao "blue eyes" -EditRegion iris -UseLocalRecolor true -UseStyleCLIP false
.\run.cmd -Comando edit -Descricao "blonde hair" -EditRegion hair -UseLocalRecolor true -UseStyleCLIP false
```

Edicao com StyleCLIP (quando nao ha caminho local):

```powershell
.\run.cmd -Comando styleclip -TargetDescricao "make the person older" -EditRegion face
```

Adiciona `-AuditDebug true` para guardar mascaras e imagens de diferenca.

## Outputs

Cada execucao escreve numa subpasta de `outputs/` (ex.: `outputs/hybrid_edit/`)
com ficheiros como:

- `original_image.png`, `resultado_final.png`
- `edit_mask.png`, `selected_mask_overlay.png`
- `difference_inside_mask.png`, `difference_outside_mask.png`
- `validation_report.json`

Trata os outputs como diagnostico, nao como fonte de verdade.

## Notas

- Em `cmd.exe` usa `.\run.cmd`; em PowerShell podes usar `.\run.ps1`.
- Se uma edicao alterar pixels fora da mascara, o resultado e invalido.
- Para nariz, boca, labios e olhos/cabelo, o caminho local e preferido ao StyleCLIP.
