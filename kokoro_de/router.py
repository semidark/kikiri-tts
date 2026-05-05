"""
German TTS frontend with a small override layer on top of espeak-de.

Design goals:
- Import-safe: no G2P backend is loaded at module import time.
- Real-world matching: overrides can match `Louis Vuitton`, `Disney+`,
  `Prime Video`, `GitHub Actions`, etc., not only collapsed forms.
- Conservative fallback: unknown tokens go straight to the configured G2P.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .normalizer import normalize_text_de
from .overrides import override_for

_TOKEN = re.compile(
    r"\d+(?:[.,]\d+)*"
    r"|[A-Za-zÀ-ÖØ-öø-ÿß]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿß0-9]+)*"
    r"|[^\w\s]",
    flags=re.UNICODE,
)
_WORDISH = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿß]", flags=re.UNICODE)
_TRAILING_PUNCT = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "»", "”"}
_MATCHABLE_TAILS = {"+", "&", "@", "/"}


class G2PCallable(Protocol):
    def __call__(self, text: str): ...


@dataclass(frozen=True)
class _RawToken:
    text: str
    start: int
    end: int


@dataclass
class Token:
    text: str
    phonemes: str
    is_override: bool = False
    source: str = "g2p"


def tokenize(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN.finditer(text)]


def _scan_tokens(text: str) -> list[_RawToken]:
    return [_RawToken(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _is_wordish(text: str) -> bool:
    return bool(_WORDISH.search(text))


def _span_text(text: str, tokens: list[_RawToken], start: int, end: int) -> str:
    return text[tokens[start].start:tokens[end - 1].end]


def _is_matchable_tail(text: str) -> bool:
    return _is_wordish(text) or text in _MATCHABLE_TAILS


def _render_phonemes(parts: list[str]) -> str:
    rendered = ""
    for part in parts:
        if not part:
            continue
        if not rendered:
            rendered = part
        elif part in _TRAILING_PUNCT:
            rendered += part
        else:
            rendered += " " + part
    return rendered


class Router:
    def __init__(
        self,
        g2p: G2PCallable | None = None,
        max_phrase_tokens: int = 6,
        normalizer=normalize_text_de,
    ):
        self._g2p = g2p
        self.max_phrase_tokens = max_phrase_tokens
        self.normalizer = normalizer

    def _ensure_g2p(self) -> G2PCallable:
        if self._g2p is None:
            try:
                from misaki import espeak
            except ImportError as exc:
                raise ImportError(
                    "Router requires `misaki` for runtime phonemization. "
                    "Install the project dependencies first."
                ) from exc
            self._g2p = espeak.EspeakG2P(language="de")
        return self._g2p

    def _g2p_phonemes(self, text: str) -> str:
        result = self._ensure_g2p()(text)
        if isinstance(result, tuple):
            return result[0] or ""
        return result or ""

    def _fallback_token(self, text: str) -> Token:
        if not _is_wordish(text):
            return Token(text=text, phonemes=text, is_override=False, source="punct")
        return Token(
            text=text,
            phonemes=self._g2p_phonemes(text),
            is_override=False,
            source="g2p",
        )

    def token(self, word: str) -> Token:
        ov = override_for(word)
        if ov is not None:
            return Token(text=word, phonemes=ov, is_override=True, source="override")
        return self._fallback_token(word)

    def route(self, text: str) -> list[Token]:
        text = self.normalizer(text) if self.normalizer else text
        raw_tokens = _scan_tokens(text)
        routed: list[Token] = []
        i = 0
        while i < len(raw_tokens):
            matched = False
            if not _is_wordish(raw_tokens[i].text):
                routed.append(self._fallback_token(raw_tokens[i].text))
                i += 1
                continue
            max_end = min(len(raw_tokens), i + self.max_phrase_tokens)
            for end in range(max_end, i, -1):
                if not _is_matchable_tail(raw_tokens[end - 1].text):
                    continue
                span = _span_text(text, raw_tokens, i, end)
                ov = override_for(span)
                if ov is None:
                    continue
                routed.append(Token(text=span, phonemes=ov, is_override=True, source="override"))
                i = end
                matched = True
                break
            if matched:
                continue
            routed.append(self._fallback_token(raw_tokens[i].text))
            i += 1
        return routed

    def phonemize(self, text: str) -> str:
        return _render_phonemes([token.phonemes for token in self.route(text)])


_default_router: Router | None = None


def _get_default_router() -> Router:
    global _default_router
    if _default_router is None:
        _default_router = Router()
    return _default_router


def route(text: str) -> list[Token]:
    return _get_default_router().route(text)


def phonemize(text: str) -> str:
    return _get_default_router().phonemize(text)
