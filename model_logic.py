import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import logging
import time
import os
from functools import wraps

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def my_logger(orig_func):
    """Loggt den Funktionsnamen und die übergebenen Argumente."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    """Loggt die Ausführungszeit der Funktion."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        logging.info(f'{orig_func.__name__} ran in: {t2:.4f} sec')
        return result
    return wrapper

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

@my_logger
@my_timer
def load_data(filepath):
    """Lädt die Daten und führt Vorverarbeitung durch."""
    try:
        df = pd.read_csv(filepath, index_col=0)
        # Labels für Evaluation behalten, Training ohne sie
        y = df['Private'].apply(converter)
        X = df.drop('Private', axis=1)
        return X, y
    except Exception as e:
        logging.error(f"Fehler beim Laden der Daten: {e}")
        raise

@my_logger
@my_timer
def fit_model(X, n_clusters=2):
    """Trainiert das KMeans Modell."""
    model = KMeans(n_clusters=n_clusters, n_init=10)
    model.fit(X)
    return model

@my_logger
@my_timer
def predict_model(model, X):
    """Erstellt Cluster-Zuweisungen."""
    predictions = model.predict(X)
    return predictions

if __name__ == "__main__":
    X, y = load_data('College_Data')
    model = fit_model(X)
    preds = predict_model(model, X)
    
    acc = accuracy_score(y, preds)
    acc_inv = accuracy_score(y, 1-preds)
    final_acc = max(acc, acc_inv)
    print(f"Beste Accuracy: {final_acc}")
