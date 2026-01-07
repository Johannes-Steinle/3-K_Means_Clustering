"""
Generiert die Trainings- und Testdaten für den Unit-Test.
Ausführung: python generate_test_data.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0

df = pd.read_csv('College_Data', index_col=0)
y = df['Private'].apply(converter)
X = df.drop('Private', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

train_data = X_train.copy()
train_data['Private'] = y_train
train_data.to_csv('train_data.csv', index=True)

test_data = X_test.copy()
test_data['Private'] = y_test
test_data.to_csv('test_data.csv', index=True)

print(f"Trainingsdaten gespeichert: {len(train_data)} Zeilen -> train_data.csv")
print(f"Testdaten gespeichert: {len(test_data)} Zeilen -> test_data.csv")
