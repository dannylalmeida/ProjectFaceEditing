function Get-StyleClipPromptPresetMap {
    return [ordered]@{
        sorriso         = @{ Type = "quality"; Value = "smiling"; Note = "Adiciona um sorriso" }
        jovem           = @{ Type = "quality"; Value = "younger"; Note = "Rejuvenesce a face" }
        velho           = @{ Type = "quality"; Value = "older"; Note = "Envelhece a face" }
        oculos          = @{ Type = "feature"; Value = "glasses"; Note = "Adiciona oculos" }
        batom_vermelho  = @{ Type = "feature"; Value = "red lipstick"; Note = "Realca batom vermelho" }
        maquilhagem     = @{ Type = "feature"; Value = "makeup"; Note = "Realca maquilhagem" }
    }
}

function Convert-StyleClipToAscii {
    param([string]$Text)

    if (-not $Text) {
        return ""
    }

    $normalized = $Text.Normalize([System.Text.NormalizationForm]::FormD)
    $builder = New-Object System.Text.StringBuilder
    foreach ($char in $normalized.ToCharArray()) {
        if ([System.Globalization.CharUnicodeInfo]::GetUnicodeCategory($char) -ne [System.Globalization.UnicodeCategory]::NonSpacingMark) {
            [void]$builder.Append($char)
        }
    }

    return $builder.ToString().Normalize([System.Text.NormalizationForm]::FormC)
}

function Get-StyleClipPtToEnPhraseMap {
    return @(
        @{ Pattern = "\bgostava de\b"; Replacement = "" }
        @{ Pattern = "\bgostaria de\b"; Replacement = "" }
        @{ Pattern = "\bquero\b"; Replacement = "" }
        @{ Pattern = "\bdeixar\b"; Replacement = "" }
        @{ Pattern = "\btornar\b"; Replacement = "" }
        @{ Pattern = "\balterar\b"; Replacement = "" }
        @{ Pattern = "\bmudar\b"; Replacement = "" }
        @{ Pattern = "\btransformar\b"; Replacement = "" }
        @{ Pattern = "\bfazer\b"; Replacement = "" }
        @{ Pattern = "\buma pessoa com\b"; Replacement = "a person with" }
        @{ Pattern = "\buma pessoa\b"; Replacement = "a person" }
        @{ Pattern = "\bpessoa com\b"; Replacement = "a person with" }
        @{ Pattern = "\bpessoa\b"; Replacement = "a person" }
        @{ Pattern = "\bum homem com\b"; Replacement = "a man with" }
        @{ Pattern = "\bum homem\b"; Replacement = "a man" }
        @{ Pattern = "\buma mulher com\b"; Replacement = "a woman with" }
        @{ Pattern = "\buma mulher\b"; Replacement = "a woman" }
        @{ Pattern = "\bum rapaz com\b"; Replacement = "a man with" }
        @{ Pattern = "\bum rapaz\b"; Replacement = "a man" }
        @{ Pattern = "\buma rapariga com\b"; Replacement = "a woman with" }
        @{ Pattern = "\buma rapariga\b"; Replacement = "a woman" }
        @{ Pattern = "\bque tem\b"; Replacement = "with" }
        @{ Pattern = "\bcom oculos\b"; Replacement = "with glasses" }
        @{ Pattern = "\bsem oculos\b"; Replacement = "without glasses" }
        @{ Pattern = "\bcom maquilhagem\b"; Replacement = "with makeup" }
        @{ Pattern = "\bsem maquilhagem\b"; Replacement = "without makeup" }
        @{ Pattern = "\bmais jovem\b"; Replacement = "younger" }
        @{ Pattern = "\bmais velho\b"; Replacement = "older" }
        @{ Pattern = "\bmais velha\b"; Replacement = "older" }
        @{ Pattern = "\bmuito comprido\b"; Replacement = "very long" }
        @{ Pattern = "\bmuito comprida\b"; Replacement = "very long" }
        @{ Pattern = "\bmuito curtas\b"; Replacement = "very short" }
        @{ Pattern = "\bmuito curto\b"; Replacement = "very short" }
        @{ Pattern = "\bmuito curta\b"; Replacement = "very short" }
        @{ Pattern = "\bmuito escuro\b"; Replacement = "very dark" }
        @{ Pattern = "\bmuito escura\b"; Replacement = "very dark" }
        @{ Pattern = "\bmuito claro\b"; Replacement = "very light" }
        @{ Pattern = "\bmuito clara\b"; Replacement = "very light" }
        @{ Pattern = "\bmuito volumoso\b"; Replacement = "very voluminous" }
        @{ Pattern = "\bmuito volumosa\b"; Replacement = "very voluminous" }
        @{ Pattern = "\bmuito definido\b"; Replacement = "very defined" }
        @{ Pattern = "\bmuito definida\b"; Replacement = "very defined" }
        @{ Pattern = "\bmais comprido\b"; Replacement = "longer" }
        @{ Pattern = "\bmais comprida\b"; Replacement = "longer" }
        @{ Pattern = "\bmais compridos\b"; Replacement = "longer" }
        @{ Pattern = "\bmais compridas\b"; Replacement = "longer" }
        @{ Pattern = "\bmais curto\b"; Replacement = "shorter" }
        @{ Pattern = "\bmais curta\b"; Replacement = "shorter" }
        @{ Pattern = "\bmais curtos\b"; Replacement = "shorter" }
        @{ Pattern = "\bmais curtas\b"; Replacement = "shorter" }
        @{ Pattern = "\bmais fino\b"; Replacement = "thinner" }
        @{ Pattern = "\bmais fina\b"; Replacement = "thinner" }
        @{ Pattern = "\bmais finos\b"; Replacement = "thinner" }
        @{ Pattern = "\bmais finas\b"; Replacement = "thinner" }
        @{ Pattern = "\bmais grosso\b"; Replacement = "thicker" }
        @{ Pattern = "\bmais grossa\b"; Replacement = "thicker" }
        @{ Pattern = "\bmais grossos\b"; Replacement = "thicker" }
        @{ Pattern = "\bmais grossas\b"; Replacement = "thicker" }
        @{ Pattern = "\bmais escuro\b"; Replacement = "darker" }
        @{ Pattern = "\bmais escura\b"; Replacement = "darker" }
        @{ Pattern = "\bmais escuros\b"; Replacement = "darker" }
        @{ Pattern = "\bmais escuras\b"; Replacement = "darker" }
        @{ Pattern = "\bmais claro\b"; Replacement = "lighter" }
        @{ Pattern = "\bmais clara\b"; Replacement = "lighter" }
        @{ Pattern = "\bmais claros\b"; Replacement = "lighter" }
        @{ Pattern = "\bmais claras\b"; Replacement = "lighter" }
        @{ Pattern = "\bmais cheio\b"; Replacement = "fuller" }
        @{ Pattern = "\bmais cheia\b"; Replacement = "fuller" }
        @{ Pattern = "\bmais cheios\b"; Replacement = "fuller" }
        @{ Pattern = "\bmais cheias\b"; Replacement = "fuller" }
        @{ Pattern = "\bmais definido\b"; Replacement = "more defined" }
        @{ Pattern = "\bmais definida\b"; Replacement = "more defined" }
        @{ Pattern = "\bmais definidos\b"; Replacement = "more defined" }
        @{ Pattern = "\bmais definidas\b"; Replacement = "more defined" }
        @{ Pattern = "\bmais arqueado\b"; Replacement = "more arched" }
        @{ Pattern = "\bmais arqueada\b"; Replacement = "more arched" }
        @{ Pattern = "\bmais arqueados\b"; Replacement = "more arched" }
        @{ Pattern = "\bmais arqueadas\b"; Replacement = "more arched" }
        @{ Pattern = "\bmais redondo\b"; Replacement = "rounder" }
        @{ Pattern = "\bmais redonda\b"; Replacement = "rounder" }
        @{ Pattern = "\bmais estreito\b"; Replacement = "narrower" }
        @{ Pattern = "\bmais estreita\b"; Replacement = "narrower" }
        @{ Pattern = "\bmais estreitos\b"; Replacement = "narrower" }
        @{ Pattern = "\bmais estreitas\b"; Replacement = "narrower" }
        @{ Pattern = "\bmais largo\b"; Replacement = "wider" }
        @{ Pattern = "\bmais larga\b"; Replacement = "wider" }
        @{ Pattern = "\bmais largos\b"; Replacement = "wider" }
        @{ Pattern = "\bmais largas\b"; Replacement = "wider" }
        @{ Pattern = "\bmais pequeno\b"; Replacement = "smaller" }
        @{ Pattern = "\bmais pequena\b"; Replacement = "smaller" }
        @{ Pattern = "\bmais pequenos\b"; Replacement = "smaller" }
        @{ Pattern = "\bmais pequenas\b"; Replacement = "smaller" }
        @{ Pattern = "\bmais grande\b"; Replacement = "larger" }
        @{ Pattern = "\bmais grandes\b"; Replacement = "larger" }
        @{ Pattern = "\bmais volumoso\b"; Replacement = "more voluminous" }
        @{ Pattern = "\bmais volumosa\b"; Replacement = "more voluminous" }
        @{ Pattern = "\bmais volumosos\b"; Replacement = "more voluminous" }
        @{ Pattern = "\bmais volumosas\b"; Replacement = "more voluminous" }
        @{ Pattern = "\bmais suave\b"; Replacement = "softer" }
        @{ Pattern = "\bmais suaves\b"; Replacement = "softer" }
        @{ Pattern = "\bpouco mais\b"; Replacement = "slightly more" }
        @{ Pattern = "\bum pouco mais\b"; Replacement = "slightly more" }
        @{ Pattern = "\bligeiramente mais\b"; Replacement = "slightly more" }
        @{ Pattern = "\bpouco menos\b"; Replacement = "slightly less" }
        @{ Pattern = "\bum pouco menos\b"; Replacement = "slightly less" }
        @{ Pattern = "\bligeiramente menos\b"; Replacement = "slightly less" }
        @{ Pattern = "\bmacas do rosto\b"; Replacement = "cheekbones" }
        @{ Pattern = "\blinha mandibular\b"; Replacement = "jawline" }
    )
}

function Get-StyleClipIgnoredPortugueseTokens {
    return @(
        "eu", "me", "mim", "por", "para", "de", "do", "da", "dos", "das",
        "o", "os", "as", "um", "uma", "uns", "umas",
        "no", "na", "nos", "nas", "ao", "aos", "que"
    )
}

function Get-StyleClipPtTokenMap {
    return @{
        "persona" = "person"
        "pessoa" = "person"
        "homem" = "man"
        "mulher" = "woman"
        "rapaz" = "man"
        "rapariga" = "woman"
        "rosto" = "face"
        "face" = "face"
        "pestana" = "eyelashes"
        "pestanas" = "eyelashes"
        "cilio" = "eyelashes"
        "cilios" = "eyelashes"
        "sobrancelha" = "eyebrows"
        "sobrancelhas" = "eyebrows"
        "olho" = "eyes"
        "olhos" = "eyes"
        "nariz" = "nose"
        "boca" = "mouth"
        "labio" = "lips"
        "labios" = "lips"
        "pele" = "skin"
        "oculos" = "glasses"
        "maquilhagem" = "makeup"
        "maquiagem" = "makeup"
        "batom" = "lipstick"
        "delineador" = "eyeliner"
        "rimel" = "mascara"
        "blush" = "blush"
        "rugas" = "wrinkles"
        "sardas" = "freckles"
        "acne" = "acne"
        "espinhas" = "acne"
        "testa" = "forehead"
        "bochecha" = "cheeks"
        "bochechas" = "cheeks"
        "maca" = "cheekbones"
        "macas" = "cheekbones"
        "queixo" = "chin"
        "mandibula" = "jawline"
        "maxilar" = "jawline"
        "sorridente" = "smiling"
        "sorriso" = "smile"
        "jovem" = "young"
        "velho" = "old"
        "velha" = "old"
        "novo" = "young"
        "nova" = "young"
        "comprido" = "long"
        "comprida" = "long"
        "compridos" = "long"
        "compridas" = "long"
        "longo" = "long"
        "longa" = "long"
        "longos" = "long"
        "longas" = "long"
        "curto" = "short"
        "curta" = "short"
        "curtos" = "short"
        "curtas" = "short"
        "liso" = "straight"
        "lisa" = "straight"
        "lisos" = "straight"
        "lisas" = "straight"
        "ondulado" = "wavy"
        "ondulada" = "wavy"
        "ondulados" = "wavy"
        "onduladas" = "wavy"
        "encaracolado" = "curly"
        "encaracolada" = "curly"
        "encaracolados" = "curly"
        "encaracoladas" = "curly"
        "volumoso" = "voluminous"
        "volumosa" = "voluminous"
        "volumosos" = "voluminous"
        "volumosas" = "voluminous"
        "fino" = "thin"
        "fina" = "thin"
        "finos" = "thin"
        "finas" = "thin"
        "grosso" = "thick"
        "grossa" = "thick"
        "grossos" = "thick"
        "grossas" = "thick"
        "claro" = "light"
        "clara" = "light"
        "claros" = "light"
        "claras" = "light"
        "escuro" = "dark"
        "escura" = "dark"
        "escuros" = "dark"
        "escuras" = "dark"
        "brilhante" = "bright"
        "brilhantes" = "bright"
        "natural" = "natural"
        "naturais" = "natural"
        "grande" = "large"
        "grandes" = "large"
        "pequeno" = "small"
        "pequena" = "small"
        "pequenos" = "small"
        "pequenas" = "small"
        "maior" = "larger"
        "maiores" = "larger"
        "menor" = "smaller"
        "menores" = "smaller"
        "cheio" = "full"
        "cheia" = "full"
        "cheios" = "full"
        "cheias" = "full"
        "definido" = "defined"
        "definida" = "defined"
        "definidos" = "defined"
        "definidas" = "defined"
        "afiado" = "sharp"
        "afiada" = "sharp"
        "afiados" = "sharp"
        "afiadas" = "sharp"
        "suave" = "soft"
        "suaves" = "soft"
        "intenso" = "intense"
        "intensa" = "intense"
        "intensos" = "intense"
        "intensas" = "intense"
        "dramatico" = "dramatic"
        "dramatica" = "dramatic"
        "dramaticos" = "dramatic"
        "dramaticas" = "dramatic"
        "arqueado" = "arched"
        "arqueada" = "arched"
        "arqueados" = "arched"
        "arqueadas" = "arched"
        "redondo" = "round"
        "redonda" = "round"
        "redondos" = "round"
        "redondas" = "round"
        "oval" = "oval"
        "ovais" = "oval"
        "quadrado" = "square"
        "quadrada" = "square"
        "quadrados" = "square"
        "quadradas" = "square"
        "estreito" = "narrow"
        "estreita" = "narrow"
        "estreitos" = "narrow"
        "estreitas" = "narrow"
        "largo" = "wide"
        "larga" = "wide"
        "largos" = "wide"
        "largas" = "wide"
        "simetrico" = "symmetrical"
        "simetrica" = "symmetrical"
        "simetricos" = "symmetrical"
        "simetricas" = "symmetrical"
        "vermelho" = "red"
        "vermelha" = "red"
        "vermelhos" = "red"
        "vermelhas" = "red"
        "azul" = "blue"
        "azuis" = "blue"
        "verde" = "green"
        "verdes" = "green"
        "amarelo" = "yellow"
        "amarela" = "yellow"
        "amarelos" = "yellow"
        "amarelas" = "yellow"
        "laranja" = "orange"
        "alaranjado" = "orange"
        "alaranjada" = "orange"
        "loiro" = "blond"
        "loira" = "blond"
        "loiros" = "blond"
        "loiras" = "blond"
        "castanho" = "brown"
        "castanha" = "brown"
        "castanhos" = "brown"
        "castanhas" = "brown"
        "preto" = "black"
        "preta" = "black"
        "pretos" = "black"
        "pretas" = "black"
        "branco" = "white"
        "branca" = "white"
        "brancos" = "white"
        "brancas" = "white"
        "cinzento" = "gray"
        "cinzenta" = "gray"
        "cinzentos" = "gray"
        "cinzentas" = "gray"
        "grisalho" = "gray"
        "grisalha" = "gray"
        "grisalhos" = "gray"
        "grisalhas" = "gray"
        "prateado" = "silver"
        "prateada" = "silver"
        "prateados" = "silver"
        "prateadas" = "silver"
        "rosa" = "pink"
        "roxo" = "purple"
        "roxa" = "purple"
        "roxos" = "purple"
        "roxas" = "purple"
        "lilas" = "lilac"
        "dourado" = "golden"
        "dourada" = "golden"
        "dourados" = "golden"
        "douradas" = "golden"
        "ruivo" = "auburn"
        "ruiva" = "auburn"
        "ruivos" = "auburn"
        "ruivas" = "auburn"
        "platinado" = "platinum"
        "platinada" = "platinum"
        "platinados" = "platinum"
        "platinadas" = "platinum"
        "mais" = "more"
        "menos" = "less"
        "muito" = "very"
        "muita" = "very"
        "muitos" = "very"
        "muitas" = "very"
        "pouco" = "slightly"
        "pouca" = "slightly"
        "ligeiramente" = "slightly"
        "subtil" = "subtle"
        "subtis" = "subtle"
        "discreto" = "subtle"
        "discreta" = "subtle"
        "discretos" = "subtle"
        "discretas" = "subtle"
        "com" = "with"
        "sem" = "without"
        "e" = "and"
        "ou" = "or"
    }
}

function Test-LikelyPortuguesePrompt {
    param([string]$Description)

    if ($Description -match "[áàãâéêíóôõúç]") {
        return $true
    }

    $lower = (Convert-StyleClipToAscii $Description).ToLowerInvariant()
    $markers = @(
        " olhos ", " sobrancelhas ", " oculos ", " pele ",
        " nariz ", " boca ", " labios ", " maquilhagem ", " pessoa ", " homem ", " mulher ",
        " com ", " sem ", " mais ", " muito ", " jovem ", " velho ", " velha ",
        " pestanas ", " bochechas ", " queixo ", " castanho "
    )

    foreach ($marker in $markers) {
        if ((" " + $lower + " ").Contains($marker)) {
            return $true
        }
    }

    return $false
}

function Get-StyleClipAllowedEnglishTokens {
    return @(
        "a", "an", "person", "man", "woman", "with", "without", "and", "or",
        "very", "more", "less", "slightly", "subtle", "intense", "dramatic",
        "young", "younger", "old", "older", "smiling", "smile",
        "face", "eyelashes", "eyebrows", "eyes", "nose", "mouth", "lips", "skin",
        "glasses", "makeup", "lipstick", "eyeliner", "mascara", "blush",
        "wrinkles", "freckles", "acne", "forehead", "cheeks", "cheekbones", "chin", "jawline",
        "long", "longer", "short", "shorter", "straight", "wavy", "curly", "voluminous", "full", "fuller",
        "thin", "thinner", "thick", "thicker", "light", "lighter", "dark", "darker", "bright", "natural",
        "large", "larger", "small", "smaller", "defined", "sharp", "soft", "softer", "arched", "round", "rounder",
        "oval", "square", "narrow", "narrower", "wide", "wider", "symmetrical",
        "red", "blue", "green", "yellow", "orange", "blond", "brown", "black", "white", "gray", "silver",
        "pink", "purple", "lilac", "golden", "auburn", "platinum"
    )
}

function Format-StyleClipEnglishPrompt {
    param([string]$Prompt)

    if (-not $Prompt) {
        return $Prompt
    }

    $regionTerms = @(
        "eyelashes", "eyebrows", "eyes", "nose", "mouth", "lips", "skin",
        "glasses", "makeup", "lipstick", "eyeliner", "mascara",
        "blush", "wrinkles", "freckles", "acne", "forehead", "cheeks", "cheekbones", "chin", "jawline", "face"
    )
    $subjectTerms = @("person", "man", "woman")
    $articles = @("a", "an")
    $connectors = @("with", "without", "and", "or")
    $tokens = $Prompt -split "\s+"
    $formatted = New-Object System.Collections.Generic.List[string]

    $i = 0
    while ($i -lt $tokens.Count) {
        $token = $tokens[$i]

        if (-not $token) {
            $i++
            continue
        }

        if ($regionTerms -contains $token) {
            $noun = $token
            $modifiers = New-Object System.Collections.Generic.List[string]
            $i++

            while ($i -lt $tokens.Count) {
                $next = $tokens[$i]
                if (($regionTerms -contains $next) -or ($subjectTerms -contains $next) -or ($articles -contains $next)) {
                    break
                }

                if (($next -in @("with", "without")) -or (($next -in @("and", "or")) -and ($i + 1 -lt $tokens.Count) -and ($regionTerms -contains $tokens[$i + 1]))) {
                    break
                }

                $modifiers.Add($next)
                $i++
            }

            while ($modifiers.Count -gt 0 -and $modifiers[$modifiers.Count - 1] -in @("and", "or")) {
                $modifiers.RemoveAt($modifiers.Count - 1)
            }

            if ($modifiers.Count -gt 0) {
                $formatted.Add(($modifiers -join " "))
                $formatted.Add($noun)
            }
            else {
                $formatted.Add($noun)
            }

            continue
        }

        $formatted.Add($token)
        $i++
    }

    $result = (($formatted -join " ") -replace "\s+", " ").Trim()
    if ($result -and $result -notmatch "^(a|an)\s+" -and $result -notmatch "^(person|man|woman)\b") {
        if ($result -match "^(with|without)\b") {
            $result = "a person $result"
        }
        elseif (($result -split "\s+") | Where-Object { $regionTerms -contains $_ }) {
            $result = "a person with $result"
        }
    }

    return ($result -replace "\s+", " ").Trim()
}

function Convert-StyleClipDescriptionToEnglish {
    param([string]$Description)

    $original = $Description.Trim()
    if (-not $original) {
        return $original
    }

    if (-not (Test-LikelyPortuguesePrompt -Description $original)) {
        return (Format-StyleClipEnglishPrompt -Prompt $original)
    }

    $normalized = (Convert-StyleClipToAscii $original).ToLowerInvariant()
    $normalized = $normalized -replace "[-_/]", " "
    $normalized = $normalized -replace "[,;:(){}\[\]!?.'""`´]", " "
    $normalized = $normalized -replace "\s+", " "
    $normalized = " $normalized "

    foreach ($entry in (Get-StyleClipPtToEnPhraseMap)) {
        $normalized = [regex]::Replace($normalized, $entry.Pattern, $entry.Replacement)
    }

    $translatedTokens = New-Object System.Collections.Generic.List[string]
    $tokenMap = Get-StyleClipPtTokenMap
    $ignoredTokens = Get-StyleClipIgnoredPortugueseTokens
    $allowedTokens = Get-StyleClipAllowedEnglishTokens

    foreach ($token in ($normalized -split "\s+")) {
        if (-not $token) {
            continue
        }

        if ($ignoredTokens -contains $token) {
            continue
        }

        if ($tokenMap.ContainsKey($token)) {
            $translated = $tokenMap[$token]
            if ($translated) {
                $translatedTokens.Add($translated)
            }
            continue
        }

        if ($allowedTokens -contains $token) {
            $translatedTokens.Add($token)
            continue
        }

        if ($token -match "^[a-z]+$") {
            throw "Descricao em portugues contem um termo que nao foi convertido com seguranca: '$token'. Usa ingles direto ou simplifica a descricao."
        }
    }

    $normalized = ($translatedTokens -join " ").Trim()
    $normalized = $normalized -replace "\s+", " "
    $normalized = $normalized.Trim()

    $normalized = Format-StyleClipEnglishPrompt -Prompt $normalized

    $normalized = $normalized -replace "\bwith with\b", "with"
    $normalized = $normalized -replace "\bwithout without\b", "without"
    $normalized = $normalized -replace "\band and\b", "and"
    $normalized = $normalized -replace "\bor or\b", "or"
    $normalized = $normalized -replace "\bwith a glasses\b", "with glasses"
    $normalized = $normalized -replace "\bwithout a glasses\b", "without glasses"
    $normalized = $normalized -replace "\ba person with young\b", "a young person with"
    $normalized = $normalized -replace "\ba person with old\b", "an old person with"
    $normalized = $normalized -replace "\ba person smiling\b", "a smiling person"
    $normalized = $normalized -replace "\ba person with smiling\b", "a smiling person with"
    $normalized = $normalized -replace "\ba person with with\b", "a person with"
    $normalized = $normalized -replace "\ba person with without\b", "a person without"
    $normalized = $normalized -replace "\bface of\b", "face"
    $normalized = $normalized -replace "\s+", " "
    $normalized = $normalized.Trim()

    if ($normalized -notmatch "^(a|an)\s+") {
        if ($normalized -match "^(young|old|smiling)\b") {
            $normalized = "a $normalized person"
        }
        elseif ($normalized -match "^(person|man|woman)\b") {
            $normalized = "a $normalized"
        }
        elseif ($normalized -match "^(with|without)\b") {
            $normalized = "a person $normalized"
        }
        else {
            $normalized = "a person with $normalized"
        }
    }

    if ($normalized -match "[áàãâéêíóôõúç]") {
        throw "Descricao em portugues ainda contem caracteres nao resolvidos. Usa um ingles mais direto ou simplifica a frase."
    }

    foreach ($token in ($normalized -split "\s+")) {
        if (-not $token) {
            continue
        }

        if ($token -match "^[a-z-]+$" -and $allowedTokens -notcontains $token) {
            throw "Descricao em portugues contem um termo que nao foi convertido com seguranca: '$token'. Usa ingles direto ou simplifica a descricao."
        }
    }

    return ($normalized -replace "\s+", " ").Trim()
}

function Show-StyleClipPromptPresets {
    $map = Get-StyleClipPromptPresetMap
    Write-Host "Presets disponiveis para testes rapidos:"
    foreach ($entry in $map.GetEnumerator()) {
        Write-Host ("- {0}: {1}" -f $entry.Key, $entry.Value.Note)
    }
}

function Get-StyleClipRegionKeywordMap {
    return [ordered]@{
        sobrancelhas = @("sobrancelha", "sobrancelhas", "eyebrow", "eyebrows")
        olhos = @("olho", "olhos", "eyes", "eye", "eyelash", "eyelashes", "pestana", "pestanas", "cilio", "cilios", "glasses", "oculos", "eyeliner", "delineador", "mascara", "rimel")
        orelhas = @("orelha", "orelhas", "ear", "ears")
        nariz = @("nariz", "nose")
        boca = @("boca", "labio", "labios", "lips", "lip", "mouth", "lipstick", "batom", "smile", "smiling", "sorriso", "sorridente")
        pescoco = @("pescoco", "pescoço", "neck")
        pele = @("pele", "skin", "freckles", "sardas", "wrinkles", "rugas", "acne", "espinhas", "face", "rosto", "cheeks", "bochechas", "cheekbones", "queixo", "chin", "jawline", "mandibula", "maxilar", "older", "younger", "old", "young", "age", "idade")
    }
}

function Get-StyleClipTargetRegions {
    param(
        [string]$Description,
        [string[]]$Preset
    )

    $regions = New-Object System.Collections.Generic.List[string]
    $text = ""
    if ($Description) {
        $text = (Convert-StyleClipToAscii $Description).ToLowerInvariant()
    }
    elseif ($Preset -and $Preset.Count -gt 0) {
        $text = (($Preset -join " ") -replace "_", " ").ToLowerInvariant()
    }

    if (-not $text) {
        return @()
    }

    $regionMap = Get-StyleClipRegionKeywordMap
    foreach ($entry in $regionMap.GetEnumerator()) {
        foreach ($keyword in $entry.Value) {
            if ((" " + $text + " ") -match ("\b" + [regex]::Escape($keyword) + "\b")) {
                $regions.Add($entry.Key)
                break
            }
        }
    }

    return @($regions | Select-Object -Unique)
}

function Resolve-StyleClipPrompt {
    param(
        [string]$Description,
        [string[]]$Preset
    )

    if ($Description) {
        return (Convert-StyleClipDescriptionToEnglish -Description $Description)
    }

    if (-not $Preset -or $Preset.Count -eq 0) {
        return "a smiling person"
    }

    $map = Get-StyleClipPromptPresetMap
    $normalizedPresets = New-Object System.Collections.Generic.List[string]

    foreach ($presetEntry in $Preset) {
        foreach ($presetName in ($presetEntry -split ",")) {
            $trimmedPreset = $presetName.Trim()
            if ($trimmedPreset) {
                $normalizedPresets.Add($trimmedPreset)
            }
        }
    }

    $qualities = New-Object System.Collections.Generic.List[string]
    $features = New-Object System.Collections.Generic.List[string]

    foreach ($presetName in $normalizedPresets) {
        if (-not $map.Contains($presetName)) {
            throw "Preset desconhecido: $presetName"
        }

        $entry = $map[$presetName]
        if ($entry.Type -eq "quality") {
            $qualities.Add($entry.Value)
        }
        elseif ($entry.Type -eq "feature") {
            $features.Add($entry.Value)
        }
    }

    if ($qualities.Count -gt 0) {
        $personDescription = "a " + ($qualities -join ", ") + " person"
    }
    else {
        $personDescription = "a person"
    }

    if ($features.Count -gt 0) {
        return $personDescription + " with " + ($features -join " and ")
    }

    return $personDescription
}
