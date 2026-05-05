"""German frontend helpers for Kokoro TTS."""

from .router import Router, Token, route, phonemize
from .overrides import (
    EN_OVERRIDES,
    DE_FOREIGN,
    BRAND_OVERRIDES,
    normalize_for_lookup,
    override_for,
)
from .normalizer import normalize_text_de

try:
    from .pipeline import KokoroDEPipeline
except ImportError:
    KokoroDEPipeline = None

__version__ = "0.3.0"

__all__ = [
    "Router", "Token", "route", "phonemize",
    "KokoroDEPipeline",
    "EN_OVERRIDES", "DE_FOREIGN", "BRAND_OVERRIDES",
    "normalize_text_de",
    "normalize_for_lookup",
    "override_for",
]
