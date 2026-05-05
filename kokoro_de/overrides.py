"""
Overrides for the German Kokoro frontend.

Priority: BRAND_OVERRIDES > EN_OVERRIDES > DE_FOREIGN
"""

from __future__ import annotations

import unicodedata

EN_OVERRIDES: dict[str, str] = {
    'accelerate': 'ɐksˈɛlɚɹˌAt',
    'amd': 'ˌAˌɛmdˈiː',
    'apache': 'ɐpˈæʧi',
    'api': 'eɪpiːˈaɪ',
    'bert': 'bˈɜːt',
    'checkpoint': 'tʃˈɛkpɔɪnt',
    'cli': 'tseːɛlˈaɪ',
    'debian': 'dˈɛbiən',
    'disneyplus': 'dˈɪzni plˈʌs',
    'dropout': 'dɹˈɑːpWt',
    'fallback': 'fˈɔːlbæk',
    'finetuning': 'fˈIn tˈuːnɪŋ',
    'gan': 'ɡˈæn',
    'github': 'ɡˈɪthab',
    'githubactions': 'ɡˈɪt hˈʌb ˈɛkʃəns',
    'gpu': 'dʒiːpiːjˈuː',
    'https': 'ˌAʧtˌiːtˈiːpˌiːˈɛs',
    'huggingface': 'hˈaɡɪŋfeɪs',
    'ipad': 'ˈI pˈæd',
    'jameswebb': 'ʤˈAmz wˈɛb',
    'json': 'dʒˈeɪsən',
    'kde': 'kˌAdˌiːˈiː',
    'louisvuitton': 'lwˈi vyitˈɔ̃',
    'macos': 'mˈɛk oː ˈɛs',
    'moetchandon': 'mɔˈɛ ʃɑ̃dˈɔ̃',
    'nvidia': 'ɛnˈviːdiːa',
    'ollama': 'olˈaːma',
    'pipeline': 'pˈaɪplaɪn',
    'primevideo': 'pɹˈIm vˈɪdɪO',
    'protocol': 'pʁotokˈɔl',
    'pytorch': 'pˈaɪtɔːɹtʃ',
    'rag': 'ɹˈæɡ',
    'repository': 'ɹᵻpˈɑːzɪtˌɔːɹi',
    'review': 'ɹᵻvjˈuː',
    'rnn': 'ˌɑːɹɹˌɛnˈɛn',
    'runtime': 'ˈɹantaɪm',
    'styletts': 'stˈaɪl tiːtiːˈɛs',
    'styletts2': 'stˈaɪl tiːtiːˈɛs tsvai',
    'surface': 'sˈɜːfɪs',
    'tcp': 'tˌiːsˌiːpˈiː',
    'thread': 'θɹˈɛd',
    'tpu': 'tˌiːpˌiːjˈuː',
    'transformers': 'tɹænsfˈɔːɹmɚz',
    'ubuntu': 'uːbˈuːntuː',
    'ui': 'juːˈaɪ',
    'wavlm': 'wˈɛɪv ɛlˈɛm',
    'wsl': 'dˌʌbəljˌuːˌɛsˈɛl',
    'zero-shot': 'zˈiːɹo ʃˈɔt',
}

DE_FOREIGN: dict[str, str] = {
    'diathese': 'diaˈteːzə',
    'ekstase': 'ɛkstˈaːzə',
    'epiklese': 'epiˈkleːzə',
    'epithese': 'epiˈteːzə',
    'glucose': 'ɡlukˈoːzə',
    'hypnose': 'hˈyːpnoːzə',
    'metamorphose': 'metamɔʁfˈoːzə',
    'oase': 'oˈaːzə',
    'prosthese': 'pʁɔstˈeːzə',
    'prothese': 'pʁotˈeːzə',
    'symbiose': 'zʏmbɪˈoːzə',
    'synthese': 'zʏntˈeːzə',
}

BRAND_OVERRIDES: dict[str, str] = {
    'bark': 'bˈaːɐk',
    'claude': 'klˈoːt',
    'coqui': 'kˈoːki',
    'cuda': 'kˈuːda',
    'deepseek': 'dˈiːpsiːk',
    'espeak-ng': 'ˈiːspiːk ɛndʒiː',
    'fastpitch': 'fˈaːstpɪtʃ',
    'geforce': 'dʒiːfˈɔɐs',
    'gemini': 'dʒˈɛmɪnaɪ',
    'hifigan': 'haɪfˈaɪɡæn',
    'intellij': 'ˈɪntɛlaɪdʒ',
    'kikiri': 'kɪkˈiːʁi',
    'kokoro': 'kˈoːkoːʁoː',
    'llama': 'lˈaːma',
    'mistral': 'mˈɪstʁal',
    'mixtral': 'mˈɪkstʁal',
    'neovim': 'nˈiːovɪm',
    'phonemizer': 'foːnəmˈaɪzɐ',
    'piper': 'pˈaɪpɐ',
    'pycharm': 'pˈaɪtʃaːɐm',
    'qwen': 'kwˈɛn',
    'radeon': 'ɹˈeɪdɪɔn',
    'ryzen': 'ɹˈaɪzən',
    'tacotron': 'tˈækotʁɔn',
    'tacotron2': 'tˈækotʁɔn tuː',
    'triton': 'tʁˈaɪtɔn',
    'typescript': 'tˈaɪpskʁɪpt',
    'unsloth': 'ˈʌnslɔːθ',
    'vits': 'vˈɪts',
    'vscode': 'viːˈɛs koːt',
}


def override_for(word: str) -> str | None:
    key = word.lower().strip()
    if key in BRAND_OVERRIDES:
        return BRAND_OVERRIDES[key]
    if key in EN_OVERRIDES:
        return EN_OVERRIDES[key]
    if key in DE_FOREIGN:
        return DE_FOREIGN[key]
    normalized = normalize_for_lookup(key)
    canonical = _NORMALIZED_ALIASES.get(normalized, normalized)
    return _NORMALIZED_OVERRIDES.get(canonical)


def normalize_for_lookup(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    parts: list[str] = []
    replacements = {
        "+": "plus",
        "&": "and",
        "@": "at",
    }
    for char in text:
        if unicodedata.category(char) == "Mn":
            continue
        replacement = replacements.get(char)
        if replacement is not None:
            parts.append(replacement)
            continue
        if char.isalnum():
            parts.append(char)
    return "".join(parts)


_NORMALIZED_ALIASES: dict[str, str] = {
    "moetandchandon": "moetchandon",
}


def _build_normalized_overrides() -> dict[str, str]:
    normalized: dict[str, str] = {}
    for mapping in (BRAND_OVERRIDES, EN_OVERRIDES, DE_FOREIGN):
        for key, value in mapping.items():
            normalized.setdefault(normalize_for_lookup(key), value)
    return normalized


_NORMALIZED_OVERRIDES = _build_normalized_overrides()
