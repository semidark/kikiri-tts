from __future__ import annotations

import unittest

from kokoro_de import Router, normalize_for_lookup, override_for, phonemize


class FakeG2P:
    def __call__(self, text: str):
        return f"<{text.casefold()}>"


class RouterImportTests(unittest.TestCase):
    def test_package_level_phonemizer_is_lazy(self):
        import kokoro_de

        self.assertTrue(hasattr(kokoro_de, "Router"))
        self.assertTrue(callable(kokoro_de.override_for))


class OverrideLookupTests(unittest.TestCase):
    def test_normalization_collapses_spaces(self):
        self.assertEqual(normalize_for_lookup("Louis Vuitton"), "louisvuitton")
        self.assertEqual(override_for("Louis Vuitton"), override_for("louisvuitton"))

    def test_normalization_handles_plus_sign(self):
        self.assertEqual(normalize_for_lookup("Disney+"), "disneyplus")
        self.assertEqual(override_for("Disney+"), override_for("disneyplus"))

    def test_normalization_handles_accents_and_ampersands(self):
        self.assertEqual(normalize_for_lookup("Moët & Chandon"), "moetandchandon")
        self.assertEqual(override_for("Moët & Chandon"), override_for("moetchandon"))

    def test_phrase_lookup_handles_split_brands(self):
        self.assertEqual(override_for("Prime Video"), override_for("primevideo"))
        self.assertEqual(override_for("GitHub Actions"), override_for("githubactions"))
        self.assertEqual(override_for("James Webb"), override_for("jameswebb"))


class RouterBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.router = Router(g2p=FakeG2P())

    def test_route_matches_multiword_brands(self):
        tokens = self.router.route("Louis Vuitton und Disney+.")
        self.assertEqual(tokens[0].text, "Louis Vuitton")
        self.assertTrue(tokens[0].is_override)
        self.assertEqual(tokens[1].phonemes, "<und>")
        self.assertEqual(tokens[2].text, "Disney+")
        self.assertTrue(tokens[2].is_override)
        self.assertEqual(tokens[3].phonemes, ".")

    def test_route_matches_diacritic_phrase(self):
        tokens = self.router.route("Moët & Chandon bleibt teuer.")
        self.assertEqual(tokens[0].text, "Moët & Chandon")
        self.assertTrue(tokens[0].is_override)

    def test_phonemize_keeps_sentence_punctuation_tight(self):
        rendered = self.router.phonemize("Hallo, Disney+!")
        self.assertTrue(rendered.endswith("!"))
        self.assertIn(",", rendered)


if __name__ == "__main__":
    unittest.main()
