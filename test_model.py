import unittest
import time
from sklearn.metrics import accuracy_score
from model_logic import load_data, fit_model, predict_model

class TestKMeansModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten."""
        cls.X, cls.y = load_data('College_Data')
        # Norm-Zeit für KMeans ca. 0.1s
        cls.norm_fit_time = 0.5 

    def test_1_cluster_accuracy(self):
        """
        Ziel: Eine Form der Übereinstimmung (Accuracy) muss > 0.70 sein.
        Da Cluster-Zahlen (0,1) zufällig sind, prüfen wir Original und Invertiert.
        """
        model, _ = fit_model(self.X)
        predictions = predict_model(model, self.X)
        
        acc = accuracy_score(self.y, predictions)
        acc_inv = accuracy_score(self.y, 1-predictions)
        final_acc = max(acc, acc_inv)
        
        print(f"\n[Test Cluster] Gemessene Accuracy: {final_acc}")
        self.assertGreater(final_acc, 0.70, f"Übereinstimmung zu niedrig: {final_acc} < 0.70")

    def test_2_fit_runtime(self):
        """
        Ziel: Laufzeit < 120% der Normzeit.
        """
        _, duration = fit_model(self.X)
        
        limit = self.norm_fit_time * 1.2
        print(f"\n[Test Fit] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")
        
        self.assertLess(duration, limit, f"Training dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
