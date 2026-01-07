# K Means Clustering Projekt

Dieses Repository enthält ein K Means Clustering Projekt als Teil der Angleichungsleistungen im Modul "Data Science und Engineering mit Python".

## Projektüberblick
Das Ziel dieses Projekts ist es, Universitäten basierend auf ihren Merkmalen in zwei Gruppen (Privat / Öffentlich) zu clustern.

## Inhalt
* `K_Means_Clustering_Solution.ipynb`: Das Haupt-Notebook mit der Clustering-Analyse.
* `College_Data`: Der Datensatz, der für das Projekt verwendet wurde.

## Prüfungsaufgabe 2: Automatisierung und Testen

Dieses Projekt wurde gemäß den Anforderungen für Aufgabe 2 refaktoriert und mit automatisierten Tests sowie Logging ausgestattet.

### Struktur
- `model_logic.py`: Enthält die Kernlogik (K-Means Clustering) sowie Logging-Funktionalität.
- `test_model.py`: Führt Unit-Tests zur Validierung der Modellgüte und der Trainingslaufzeit durch.
- `training.log`: Protokolliert Trainingsereignisse.

### Testergebnisse
Die Tests wurden erfolgreich ausgeführt:
```text
[Test Cluster] Gemessene Accuracy: 0.7786357786357786
[Test Fit] Gemessene Dauer: 0.0197s (Limit: 0.6000s)
Ran 2 tests in 1.300s
OK
```

## Nutzung
Das Notebook kann direkt über [myBinder](https://mybinder.org/v2/gh/Johannes-Steinle/3-K_Means_Clustering/main?filepath=K_Means_Clustering_Solution.ipynb) ausgeführt werden.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/3-K_Means_Clustering/main?filepath=K_Means_Clustering_Solution.ipynb)
