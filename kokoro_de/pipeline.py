"""
KokoroDEPipeline wraps Kokoro's upstream KPipeline with a German override router.

The wrapper stays close to upstream behavior:
- non-German languages are delegated unchanged,
- German text is routed through `kokoro_de.Router`,
- phoneme sequences are split on token boundaries instead of being sliced.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Generator, Optional, Union

from .router import Router, Token


def _load_kpipeline_class():
    try:
        from kokoro import KPipeline
        return KPipeline
    except ImportError:
        local_submodule = Path(__file__).resolve().parents[1] / "kokoro"
        if local_submodule.exists():
            local_path = str(local_submodule)
            if local_path not in sys.path:
                sys.path.insert(0, local_path)
            from kokoro import KPipeline
            return KPipeline
        raise ImportError(
            "KokoroDEPipeline requires the upstream `kokoro` package. "
            "Install `kokoro>=0.9.4` or work from a checkout with the `kokoro/` submodule."
        )


class KokoroDEPipeline:
    """German-aware wrapper for Kokoro's upstream pipeline."""

    def __init__(
        self,
        lang_code: str = "d",
        repo_id: Optional[str] = None,
        model=None,
        trf: bool = False,
        device: Optional[str] = None,
        router: Optional[Router] = None,
    ):
        kpipeline_cls = _load_kpipeline_class()
        self._base = kpipeline_cls(
            lang_code=lang_code,
            repo_id=repo_id,
            model=model,
            trf=trf,
            device=device,
        )
        self.lang_code = self._base.lang_code
        self.model = self._base.model
        self.voices = self._base.voices
        self.repo_id = self._base.repo_id
        self.router = router if self.lang_code == "d" else None
        if self.lang_code == "d" and self.router is None:
            self.router = Router()

    def load_voice(self, voice, delimiter=","):
        return self._base.load_voice(voice, delimiter)

    @property
    def Result(self):
        return self._base.Result

    @staticmethod
    def _render_phoneme_tokens(tokens: list[Token]) -> str:
        rendered = ""
        for token in tokens:
            part = token.phonemes
            if not part:
                continue
            if not rendered:
                rendered = part
            elif len(part) == 1 and not part[0].isalnum():
                rendered += part
            else:
                rendered += " " + part
        return rendered

    @staticmethod
    def _token_text(tokens: list[Token]) -> str:
        text = ""
        for token in tokens:
            part = token.text
            if not text:
                text = part
            elif len(part) == 1 and part in ".,!?;:%)]}»”":
                text += part
            else:
                text += " " + part
        return text

    @classmethod
    def _split_routed_tokens(cls, tokens: list[Token], max_phonemes: int = 510) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        current: list[Token] = []
        last_break_index: int | None = None
        break_tokens = {".", "!", "?", ":", ";", ","}

        def flush(count: int | None = None):
            nonlocal current, last_break_index
            if not current:
                return
            use_count = len(current) if count is None else count
            head = current[:use_count]
            chunks.append((cls._token_text(head), cls._render_phoneme_tokens(head)))
            current = current[use_count:]
            last_break_index = None
            for index, token in enumerate(current):
                if token.text in break_tokens:
                    last_break_index = index + 1

        for token in tokens:
            current.append(token)
            if token.text in break_tokens:
                last_break_index = len(current)
            rendered = cls._render_phoneme_tokens(current)
            if len(rendered) <= max_phonemes:
                continue
            if len(current) == 1:
                raise ValueError(
                    f"Single routed token exceeds Kokoro limit ({len(rendered)} > {max_phonemes}): {token.text!r}"
                )
            if last_break_index and last_break_index < len(current):
                overflow = current[last_break_index:]
                flush(last_break_index)
                current = overflow
            else:
                overflow = current[-1:]
                flush(len(current) - 1)
                current = overflow
            rendered = cls._render_phoneme_tokens(current)
            if len(rendered) > max_phonemes:
                raise ValueError(
                    f"Single routed token exceeds Kokoro limit ({len(rendered)} > {max_phonemes}): {current[0].text!r}"
                )

        flush()
        return [(text, phonemes) for text, phonemes in chunks if phonemes]

    def _german_chunks(self, graphemes: str) -> list[tuple[str, str]]:
        routed = self.router.route(graphemes)
        return self._split_routed_tokens(routed)

    def __call__(
        self,
        text: Union[str, list[str]],
        voice: Optional[str] = None,
        speed: Union[float, Callable] = 1,
        split_pattern: Optional[str] = r"\n+",
        model=None,
    ) -> Generator:
        if self.router is None:
            yield from self._base(
                text,
                voice=voice,
                speed=speed,
                split_pattern=split_pattern,
                model=model,
            )
            return

        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: pipeline(text="Hallo", voice="df_voice")')
        pack = self.load_voice(voice).to(model.device) if model else None

        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]

        kpipeline_cls = _load_kpipeline_class()
        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():
                continue
            for chunk_text, chunk_phonemes in self._german_chunks(graphemes):
                output = kpipeline_cls.infer(model, chunk_phonemes, pack, speed) if model else None
                yield self._base.Result(
                    graphemes=chunk_text,
                    phonemes=chunk_phonemes,
                    output=output,
                    text_index=graphemes_index,
                )
