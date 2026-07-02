import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_driver_status_knn():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    ai_engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'ai_engine'))
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    model_output_path = os.path.join(ai_engine_dir, 'knn_body_model.joblib')
    scaler_output_path = os.path.join(ai_engine_dir, 'data_scaler_knn.joblib')
    
    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}. Veuillez vérifier l'exécution de merge_heart_data.py.")
        return

    print("Chargement de la base de données d'entraînement...")
    df = pd.read_csv(train_data_path)
    
    X = df[['HeartRate', 'HRV']]
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Entraînement du classifieur KNN en cours...")
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("Entraînement du modèle terminé.")
    
    y_pred = model.predict(X_test_scaled)
    print("\n================ RAPPORT D'ÉVALUATION ================")
    print(classification_report(y_test, y_pred, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))
    print("======================================================")
    
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"Le modèle KNN et le standardiseur ont été enregistrés avec succès dans le dossier ai_engine.")

if __name__ == "__main__":
    train_driver_status_knn()