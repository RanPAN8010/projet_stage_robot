import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_driver_status_model():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    ai_engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'ai_engine'))
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    model_output_path = os.path.join(ai_engine_dir, 'random_forest_body_model.joblib')
    scaler_output_path = os.path.join(ai_engine_dir, 'data_scaler.joblib')
    
    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}. ")
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
    
    print("Entraînement du classifieur random forest en cours...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("Entraînement du modèle terminé.")
    
    y_pred = model.predict(X_test_scaled)
    print("\n================ RAPPORT D'ÉVALUATION ================")
    print(classification_report(y_test, y_pred, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))
    print("==============================================")
    
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"Le modèle KNN et le standardiseur ont été enregistrés avec succès dans le dossier ai_engine.")

if __name__ == "__main__":
    train_driver_status_model()