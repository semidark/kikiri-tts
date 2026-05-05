# PROGRESS

> Stand: 2026-05-05
> Status: Sanierungsstand nach technischem Review

## Was geändert wurde

Der frühere `kokoro_de`-Stand war als Rohmaterial brauchbar, aber nicht GitHub-tauglich:

- das Paket wurde nicht gebaut
- `import kokoro_de` konnte schon beim Import crashen
- die Tests waren lokale Demo-Skripte mit privaten Pfaden
- die Coverage-Claims waren größer als die tatsächlich abgedeckte Schreibweise
- der Pipeline-Wrapper schnitt zu lange Phonemstrings roh ab

Dieser Stand wurde jetzt auf einen kleineren, belastbaren Kern zurückgebaut.

## Aktueller technischer Stand

### Packaging

- `kokoro_de` wird jetzt als eigenes Paket gebaut
- der kaputte Wheel-Target auf `kokoro/kokoro` wurde entfernt
- das Root-Paket behauptet nicht mehr, selbst das Upstream-CLI auszuliefern

### Router

- kein G2P-Backend mehr beim Modulimport
- `Router()` lädt `misaki/espeak` erst zur Laufzeit
- phrase-basiertes Matching über normalisierte Spans
- reale Schreibweisen wie `Louis Vuitton`, `Disney+`, `Prime Video`, `GitHub Actions` und `James Webb` werden als Schreibvarianten erkannt, wenn passende Lexikoneinträge vorhanden sind

### Normalisierung

- repo-eigener deutscher Vorverarbeitungslayer vor dem Router
- aktuell abgedeckt: Datum, Uhrzeit, Euro-Beträge, Prozent, einfache Dezimalzahlen, Ordnungszahlen im Satz, ausgewählte Technik-Einheiten und häufige TTS/ML-Abkürzungen
- Ziel: die fachlich richtige Idee aus `feat/misaki-integration` übernehmen, ohne das Repo direkt an einen losen Fork-Branch zu koppeln

### Pipeline

- `KokoroDEPipeline` hängt sich als dünner Wrapper an Kokoro an
- lokaler Source-Checkout wird sauberer behandelt als zuvor
- zu lange Phonemketten werden nicht mehr stumpf mit `[:510]` abgeschnitten
- Chunking passiert auf Token-Grenzen

### Tests

- die lokalen Testskripte wurden entfernt
- stattdessen gibt es echte Unit-Tests in `tests/`
- die Tests brauchen keine privaten Checkpoints

## Was bewusst noch nicht behauptet wird

Folgende Aussagen werden absichtlich nicht mehr gemacht:

- keine Behauptung, dass alle `2531` Einträge vollständig validiert seien
- keine Behauptung `0 false positives`
- keine Behauptung, dass `100+` Domänen in natürlicher Schreibweise vollständig abgedeckt seien
- keine Behauptung, dass der Override-Layer `espeak-de` allgemein ersetzt

Die richtige Formulierung ist aktuell:

- `espeak-de` macht die Hauptarbeit
- `kokoro_de` ergänzt gezielte, kuratierte Korrekturen
- der große Override-Bestand wurde auf einen aktiven Kern reduziert

Aktueller Override-Stand:

- `90` aktive Einträge im Lexikon
- davon `34` bereits als `keep` bestätigt
- `56` verbleiben als gezielter `review`-Rest
- `2441` frühere Kandidaten sind aus dem aktiven Lexikon entfernt oder bewusst verworfen

## Nächste sinnvolle Schritte

1. Den verbleibenden `review`-Rest weiter gegen Hörproben absichern oder verwerfen.
2. Für jede neue Override-Familie einen kleinen Testfall ergänzen.
3. Akustische A/B-Tests mit öffentlichen Beispielsätzen dokumentieren.
4. Mehrwort- und Markenformen nur dann in die README aufnehmen, wenn sie testseitig abgesichert sind.

## Kurzfazit

Vorher: ambitionierter, aber technischer Prototyp mit überzogenen Claims.  
Jetzt: kleinerer, ehrlicherer und testbarer Frontend-Layer, der als Grundlage für ein öffentliches `kokoro-deutsch`-Repo tragfähig ist.
