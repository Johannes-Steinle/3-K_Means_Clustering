import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import logging
import time
import os

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

def load_data(filepath):
    """Lädt die Daten und führt Vorverarbeitung durch."""
    logger = logging.getLogger()
    try:
        df = pd.read_csv(filepath, index_col=0)
        logger.info(f"Daten erfolgreich von {filepath} geladen.")
        
        # Wir behalten die Labels für die Evaluation, trainieren aber ohne sie
        y = df['Private'].apply(converter)
        X = df.drop('Private', axis=1)
        
        return X, y
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten: {e}")
        raise

def fit_model(X, n_clusters=2):
    """Trainiert das KMeans Modell und misst die Zeit."""
    logger = logging.getLogger()
    start_time = time.time()
    
    logger.info(f"Starte Modelltraining (KMeans, k={n_clusters})...")
    model = KMeans(n_clusters=n_clusters, n_init=10)
    model.fit(X)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Training beendet in {duration:.4f} Sekunden.")
    
    return model, duration

def predict_model(model, X):
    """Erstellt Cluster-Zuweisungen."""
    logger = logging.getLogger()
    logger.info("Erstelle Cluster-Zuweisungen...")
    predictions = model.predict(X)
    return predictions

if __name__ == "__main__":
    X, y = load_data('College_Data')
    model, duration = fit_model(X)
    preds = predict_model(model, X)
    
    # Da die Cluster-Labels 0 und 1 beliebig sind, prüfen wir beide Varianten
    acc = accuracy_score(y, preds)
    acc_inv = accuracy_score(y, 1-preds)
    final_acc = max(acc, acc_inv)
    
    print(f"Beste Accuracy (nach Label-Matching): {final_acc}")
    logging.info(f"KMeans Accuracy im Testlauf: {final_acc}")
