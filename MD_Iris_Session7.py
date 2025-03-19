import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class IrisModel:
    def __init__(self):
        """Inisialisasi model dan dataset"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.load_data()
    
    def load_data(self):
        """Load dataset Iris dan buat DataFrame"""
        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.df['species'] = iris.target  # Tambahkan kolom target
        
        # Pisahkan fitur (X) dan label (y)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df['species']
    
    def train(self, test_size=0.2):
        """Latih model dengan data yang telah dibagi"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluasi model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Classification Report: {classification_report(y_test, y_pred)}")
        print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}")
    
    def save_model(self, filename='iris_model.pkl'):
        """Simpan model ke file pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model telah disimpan sebagai {filename}")

class IrisPredictor:
    def __init__(self, model_path='iris_model.pkl'):
        """Load model dari file pickle"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model {model_path} berhasil dimuat!")

    def predict(self, sample):
        """Melakukan prediksi berdasarkan input fitur"""
        pred = self.model.predict([sample])[0]
        return pred

iris_model = IrisModel()
iris_model.train()
iris_model.save_model()
