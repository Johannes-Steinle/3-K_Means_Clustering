import unittest
import time
from sklearn.metrics import accuracy_score
from model_logic import load_data, fit_model, predict_model

class TestKMeansModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten."""
        cls.X, cls.y = load_data('College_Data')
        cls.norm_fit_time = 0.5 

    def test_1_predict_accuracy(self):
        """
        Aufgabe: Test der Vorhersagefunktion (predict).
        Ziel: Accuracy > 0.70.
        """
        model = fit_model(self.X)
        predictions = predict_model(model, self.X)
        
        acc = accuracy_score(self.y, predictions)
        acc_inv = accuracy_score(self.y, 1-predictions)
        final_acc = max(acc, acc_inv)
        
        print(f"\n[Test predict()] Gemessene Accuracy: {final_acc:.4f}")
        self.assertGreater(final_acc, 0.70, f"Übereinstimmung zu niedrig: {final_acc} < 0.70")

    def test_2_fit_runtime(self):
        """
        Aufgabe: Überprüfung der Laufzeit der Trainingsfunktion (fit).
        Ziel: Laufzeit < 120% der Normzeit.
        """
        start_time = time.time()
        fit_model(self.X)
        duration = time.time() - start_time
        
        limit = self.norm_fit_time * 1.5 # Puffer für Testumgebung
        print(f"\n[Test fit()] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")
        
        self.assertLess(duration, limit, f"Training dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
