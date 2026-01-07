# K Means Clustering Projekt

Meine Umsetzung der K Means Clustering Übung aus dem Udemy-Kurs "Python für Data Science, Maschinelles Lernen & Visualization" im Rahmen der Angleichungsleistung.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/3-K_Means_Clustering/main?filepath=K_Means_Clustering_Solution.ipynb)

## Überblick
Universitäten werden anhand ihrer Merkmale in zwei Gruppen (Privat / Öffentlich) geclustert. Die Cluster-Zuweisungen werden dann mit den tatsächlichen Labels verglichen. Modell: K-Means (scikit-learn).

## Inhalt
* `K_Means_Clustering_Solution.ipynb` - Haupt-Notebook mit der Clustering-Analyse
* `College_Data` - Datensatz mit Universitäts-Merkmalen (777 Einträge)

## Ausführung

1. Auf den **Binder-Badge** oben klicken, um das Notebook in myBinder zu starten.
2. Warten, bis die Umgebung geladen ist (kann 1-2 Minuten dauern).
3. `K_Means_Clustering_Solution.ipynb` öffnen.
4. Alle Zellen nacheinander ausführen (*Run > Run All Cells*).
5. **Erwartete Ergebnisse:**
   - Scatterplots der Universitäts-Merkmale
   - K-Means Clustering mit k=2
   - Vergleich der Cluster mit den echten Privat/Öffentlich-Labels
   - Classification Report und Confusion Matrix
   - Übereinstimmung von ca. **0.77 - 0.80**

---

## Prüfungsaufgabe 2: Automatisierung und Testen

Ich habe das Projekt für Aufgabe 2 um Unit-Tests und Logging erweitert, nach dem Ansatz aus dem Artikel "Unit Testing and Logging for Data Science".

### Dateien
| Datei | Beschreibung |
|---|---|
| `model_logic.py` | K-Means Logik mit `my_logger` und `my_timer` Dekoratoren |
| `test_model.py` | Unit-Tests für `predict()` (Cluster-Accuracy) und `fit()` (Laufzeit) |
| `generate_test_data.py` | Skript zur Erzeugung der Testdaten |
| `train_data.csv` | Trainingsdaten (543 Zeilen) |
| `test_data.csv` | Testdaten (234 Zeilen) |
| `training.log` | Log-File mit Trainingsereignissen |

### Testfälle

**Testfall 1 - predict():** K-Means wird auf den Trainingsdaten trainiert und die Übereinstimmung der Cluster mit den echten Labels auf `test_data.csv` geprüft. Ziel: Accuracy > 0.70.

**Testfall 2 - fit():** Die Laufzeit der Trainingsfunktion wird gemessen und geprüft, ob sie unter 120% der Normzeit (0.5s) bleibt.

### Testergebnisse
```text
[Test predict()] Gemessene Accuracy: 0.7863
.
[Test fit()] Gemessene Dauer: 0.0193s (Limit: 0.6000s)
.
----------------------------------------------------------------------
Ran 2 tests in 1.410s

OK
```

### Tests ausführen

1. Binder-Umgebung über den Badge oben starten.
2. **Terminal** öffnen (*File > New > Terminal*).
3. Folgenden Befehl ausführen:
   ```bash
   python -m unittest test_model -v
   ```
4. Die Tests laden die Daten aus `test_data.csv` und `train_data.csv`.
5. Beide Tests sollten mit `OK` durchlaufen.

Um die Testdaten neu zu generieren: `python generate_test_data.py`
