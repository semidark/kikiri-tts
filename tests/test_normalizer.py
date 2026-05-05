from __future__ import annotations

import unittest

from kokoro_de import normalize_text_de


class NormalizerTests(unittest.TestCase):
    def test_abbreviation_expansion(self):
        self.assertEqual(
            normalize_text_de("Dr. Müller kommt z. B. ca. 3 Min. später."),
            "Doktor Müller kommt zum beispiel circa drei Min. später.",
        )

    def test_time_expansion(self):
        self.assertEqual(
            normalize_text_de("Treffen um 14:30"),
            "Treffen um vierzehn uhr dreißig",
        )

    def test_date_expansion(self):
        self.assertEqual(
            normalize_text_de("Termin am 03.05.2026"),
            "Termin am dritten mai zweitausendsechsundzwanzig",
        )

    def test_euro_expansion(self):
        self.assertEqual(
            normalize_text_de("Kostet 29,99 €"),
            "Kostet neunundzwanzig euro und neunundneunzig cent",
        )

    def test_integer_expansion(self):
        self.assertEqual(
            normalize_text_de("Es sind 42 Beispiele."),
            "Es sind zweiundvierzig Beispiele.",
        )

    def test_decimal_expansion(self):
        self.assertEqual(
            normalize_text_de("Wert 3,5 ist okay."),
            "Wert drei komma fünf ist okay.",
        )

    def test_percent_expansion(self):
        self.assertEqual(
            normalize_text_de("Akku bei 87%."),
            "Akku bei siebenundachtzig prozent.",
        )

    def test_unit_expansion(self):
        self.assertEqual(
            normalize_text_de("Es sind 12 km und 21°C."),
            "Es sind zwölf kilometer und einundzwanzig grad celsius.",
        )

    def test_simple_ordinal_expansion(self):
        self.assertEqual(
            normalize_text_de("Der 3. Versuch klappt."),
            "Der dritte Versuch klappt.",
        )

    def test_acronym_expansion(self):
        self.assertEqual(
            normalize_text_de("AI, TTS und GPU für LLM-Tests."),
            "a i, t t s und g p u für l l m-Tests.",
        )

    def test_tech_unit_expansion(self):
        self.assertEqual(
            normalize_text_de("24 kHz, 3 ms und 16 GB für v1.2."),
            "vierundzwanzig kilohertz, drei millisekunden und sechzehn gigabyte für version eins punkt zwei.",
        )


if __name__ == "__main__":
    unittest.main()
