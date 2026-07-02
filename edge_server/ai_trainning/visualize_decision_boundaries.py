import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def plot_model_boundaries():
    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        return
        
    data_path = os.path.join(base_project_dir, 'data', 'driver_body_status_train.csv')
    ai_engine_dir = os.path.join(base_project_dir, 'ai_engine')
    
    if not os.path.exists(data_path):
        print(f"Erreur : Fichier de données introuvable - {data_path}")
        return
        
    model_files = glob.glob(os.path.join(ai_engine_dir, '*_model.joblib'))
    if not model_files:
        print(f"Erreur : Aucun modèle trouvé dans {ai_engine_dir}")
        return
        
    print("=== Modèles disponibles pour la visualisation ===")
    for idx, file_path in enumerate(model_files):
        print(f"[{idx}] {os.path.basename(file_path)}")
    print("=================================================")
    
    try:
        choix = int(input("Entrez le numéro du modèle à visualiser : "))
        if choix < 0 or choix >= len(model_files):
            print("Choix invalide.")
            return
    except ValueError:
        print("Veuillez entrer un nombre valide.")
        return
        
    selected_model_path = model_files[choix]
    model_name = os.path.basename(selected_model_path).replace('.joblib', '')
    
    scaler_name = 'data_scaler.joblib' if 'random_forest' in model_name else f"data_scaler_{model_name.split('_')[0]}.joblib"
    selected_scaler_path = os.path.join(ai_engine_dir, scaler_name)
    
    if not os.path.exists(selected_scaler_path):
        selected_scaler_path = os.path.join(ai_engine_dir, 'data_scaler.joblib')
        
    model = joblib.load(selected_model_path)
    scaler = joblib.load(selected_scaler_path)
    df = pd.read_csv(data_path)
    
    X = df[['HeartRate', 'HRV']].values
    y = df['Label'].values
    
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = -10, df['HRV'].quantile(0.98) + 50
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.5),
        np.arange(y_min, y_max, 2.0)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    Z = model.predict(grid_points_scaled)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(11, 8))
    sns.set_theme(style="white")
    
    cmap_background = plt.cm.colors.ListedColormap(['#a3e4d7', '#f9e79f', '#f9e79f' if 'knn' in model_name else '#f5b7b1'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
    
    mapping_labels = {
        0: "Normal (Sain)",
        1: "Fatigue Mentale",
        2: "Crise Cardiaque"
    }
    df['État du Conducteur'] = df['Label'].map(mapping_labels)
    
    palette_points = {
        "Normal (Sain)": "#2ecc71",
        "Fatigue Mentale": "#f1c40f",
        "Crise Cardiaque": "#e74c3c"
    }
    
    sns.scatterplot(
        data=df,
        x='HeartRate',
        y='HRV',
        hue='État du Conducteur',
        palette=palette_points,
        alpha=0.3,
        s=20,
        edgecolor='none'
    )
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.title(f"Frontières de Décision : {model_name.upper()}", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Fréquence Cardiaque (HeartRate - BPM)", fontsize=11, labelpad=8)
    plt.ylabel("Variabilité de la Fréquence Cardiaque (HRV)", fontsize=11, labelpad=8)
    plt.legend(title="État réel du sujet", loc='upper right')
    plt.tight_layout()
    
    output_image_name = f"frontieres_decision_{model_name}.png"
    output_image_path = os.path.join(base_project_dir, 'data', output_image_name)
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    
    print("\n==================================================")
    print("Visualisation des frontières générée avec succès !")
    print(f"Modèle analysé : {model_name}")
    print(f"Image enregistrée sous : {output_image_path}")
    print("==================================================")

if __name__ == "__main__":
    plot_model_boundaries()