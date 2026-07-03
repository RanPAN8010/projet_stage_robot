import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

def train_final_model():
    # ==========================================
    # 动态路径定位（基于当前脚本提取绝对路径）
    # ==========================================
    current_path = os.path.abspath(__file__)
    
    # 准确定位项目根目录 'edge_server'
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("Erreur : Le script n'est pas placé dans le dossier 'edge_server' !")
        return
        
    # 定位数据文件夹路径：edge_server/data/
    data_dir = os.path.join(base_project_dir, 'data')
    
    train_path = os.path.join(data_dir, 'final_train_data_5_features.csv')
    test_path = os.path.join(data_dir, 'final_test_data_5_features.csv')
    
    # 修改为自然的法语输出
    print("Chargement des données de train et test...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 定义训练所需的 5 个核心特征
    feature_cols = ['Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index']
    
    X_train = df_train[feature_cols]
    y_train = df_train['Label']
    
    X_test = df_test[feature_cols]
    y_test = df_test['Label']
    
    # 修改为自然的法语输出
    print(f"Échantillons d'entraînement : {X_train.shape[0]} lignes")
    print(f"Échantillons de test : {X_test.shape[0]} lignes")
    
    # 4. 初始化 XGBoost 多分类器（针对 0=Sécurité, 1=Canicule, 2=Feu）
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',  # 配置多分类概率输出接口
        num_class=3,                 # 明确总类别数为 3
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # 5. 训练模型
    # 修改为自然的法语输出
    print("Entraînement du modèle XGBoost à 3 classes (Sécurité, Canicule, Feu)...")
    model.fit(X_train, y_train)
    
    # 6. 模型预测与评估
    y_pred = model.predict(X_test)
    
    # 修改为自然的法语输出
    print("\n=== Rapport d'évaluation du modèle ===")
    print(f"Précision globale (Accuracy) : {accuracy_score(y_test, y_pred):.4f}")
    print("\nRapport détaillé par classe :")
    print(classification_report(y_test, y_pred, target_names=['Sécurité (0)', 'Canicule (1)', 'Feu/Fumée (2)']))
    
    # ==========================================
    # 模型导出（精准投递至 edge_server/ai_engine/env/ 目录下）
    # ==========================================
    model_output_dir = os.path.join(base_project_dir, 'ai_engine', 'env')
    
    # 自动检查并创建最终的推理环境目标夹，防止路径不存在报错
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        
    model_output_path = os.path.join(model_output_dir, 'car_safety_xgboost_model.json')
    model.save_model(model_output_path)
    
    # 修改为自然的法语输出
    print(f"\nModèle exporté avec succès vers : {model_output_path}")

if __name__ == "__main__":
    train_final_model()