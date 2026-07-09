import os
import pandas as pd
import numpy as np

# dataset: https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final
# Extrait les caractéristiques clés du jeu de données cardiaques, les aligne avec 
# celles de la fatigue, et fusionne le tout dans le jeu d'entraînement final.
def merge_heart_and_fatigue():
    current_path = os.path.abspath(__file__)
    
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
        
    data_dir = os.path.join(base_project_dir, 'data')
    
    fatigue_cleaned_path = os.path.join(data_dir, 'fatigueset_cleaned.csv')
    heart_raw_path = os.path.join(data_dir, 'heart_statlog_cleveland_hungary_final.csv')
    output_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    
    # 检查前置文件是否存在
    if not os.path.exists(fatigue_cleaned_path):
        print(f"Erreur : Fichier nettoyé introuvable.")
        print(f"Chemin attendu : {fatigue_cleaned_path}")
        print("Veuillez vérifier si clean_fatigueset.py a fonctionné.")
        return
    if not os.path.exists(heart_raw_path):
        print(f"Erreur : Le jeu de données initial de la maladie cardiaque est introuvable dans le répertoire data. Le chemin attendu est :\n  -> {heart_raw_path}")
        return
        
    print("Lecture commencée...")
    df_fatigue = pd.read_csv(fatigue_cleaned_path)
    df_heart_raw = pd.read_csv(heart_raw_path)
    
    # 通过列的物理位置索引 (iloc) 强制提取核心特征
    df_heart_prepared = pd.DataFrame()
    df_heart_prepared['HeartRate'] = df_heart_raw.iloc[:, 7]
    df_heart_prepared['HRV'] = df_heart_raw.iloc[:, 9].abs() * 100.0
    df_heart_prepared['Label'] = df_heart_raw.iloc[:, -1].apply(lambda x: 2 if x == 1 else 0)
    
    # 精简列结构
    df_fatigue_reduced = df_fatigue[['HeartRate', 'HRV', 'Label']]
    df_heart_prepared = df_heart_prepared[['HeartRate', 'HRV', 'Label']]
    
    print(f"[Progression] Fin de l'extraction des caractéristiques cardiaques.")
    print(f"[Progression] Total : {len(df_heart_prepared)} échantillons.")
    print("[Progression] Fusion des deux jeux de données.")
    print("[Progression] Création du jeu d'entraînement final.")
    final_train_df = pd.concat([df_fatigue_reduced, df_heart_prepared], ignore_index=True)
    final_train_df.dropna(inplace=True)
    
    # 7. 写入最终大表
    final_train_df.to_csv(output_path, index=False)
    
    print("\n==========================================")
    print("Fusion des jeux de données réussie !")
    print(f"Chemin du jeu final : {output_path}")
    print(f"Total : {len(final_train_df)} lignes")
    print("==========================================")

if __name__ == "__main__":
    merge_heart_and_fatigue()