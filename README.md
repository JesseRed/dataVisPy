# dataVisPy

Interaktive Python/Dash-Anwendung zur statistischen Auswertung von MEG-Konnektivitaetsdaten auf Basis von JSON- und CSV-Exporten.

## Lokal starten

```bash
source .venv/bin/activate
python app.py
```

Optional mit `src`-Layout direkt als Modul:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m data_vis_py.app
```

## Tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## Erwartete Datendateien

Fuer einen Datensatz wie `data/raw/REST_24_Stroke/` erwartet die App aktuell mindestens:

- `data_coh.json` oder `export_coh.json` mit den Konnektivitaetswerten
- `info.csv` mit den Probanden- und Verhaltensdaten

## Pflichtspalten in `info.csv`

Die ersten vier Spalten muessen immer vorhanden sein:

1. `ID`
   Diese Spalte enthaelt die eindeutige Messungs-ID, also genau die Kennung, die auch im JSON unter `subject_id` vorkommt.
2. `Group`
   Diese Spalte beschreibt die Gruppenzugehoerigkeit der Messung, zum Beispiel `Stroke`, `Control` oder numerische Gruppen wie `1` und `2`.
3. `IDX`
   Diese Spalte ist die stabile Personen-ID ueber mehrere Messungen hinweg. Gleicher `IDX` bedeutet gleiche Person. Diese Spalte wird fuer longitudinale Vergleiche verwendet und ist deshalb Pflicht.
4. `MTime`
   Diese Spalte beschreibt den Messzeitpunkt, zum Beispiel `1` und `2` oder `M1` und `M2`. Intern wird sie auf Werte wie `M1`, `M2` normalisiert.

## Weitere Spalten

Alle weiteren Spalten in `info.csv` werden als zusaetzliche Behavior- oder Kovariatenfelder behandelt. Sie koennen in der App verwendet werden fuer:

- Korrelationsanalysen
- lineare Regressionsmodelle
- longitudinale Selektion ueber eine frei waehlbare Spalte

Beispiele sind `age`, `MoCA`, `NIHSS`, `DSS`, `gender` oder projektspezifische Labordaten.

## Longitudinale Analyse

Wenn in der App die longitudinale Analyse aktiviert wird:

- waehlt man zuerst eine Spalte aus der Behavior-Tabelle
- danach zwei konkrete Werte aus dieser Spalte
- die App berechnet dann pro Person zunaechst den Trial-Kontrast `Trial B - Trial A`
- anschliessend wird dieses Delta zwischen den beiden ausgewaehlten Messungen derselben Person verglichen

Die Zuordnung derselben Person erfolgt dabei ausschliesslich ueber `IDX`.
