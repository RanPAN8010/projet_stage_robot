import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

def run_hyperparameter_tuning():
    # 获取当前脚本的绝对路径 (edge_server/ai_trainning/med/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    data_dir = os.path.join(base_project_dir, 'data')
    ai_engine_med_dir = os.path.join(base_project_dir, 'ai_engine', 'med')
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    
    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}.")
        return

    print("Chargement de la base de données...")
    df = pd.read_csv(train_data_path)
    X = df[['HeartRate', 'HRV']]
    y = df['Label']
    
    # 保持统一的 80/20 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 计算用于样本权重的比例 (对抗不平衡)
    classes_counts = y_train.value_counts()
    total_samples = len(y_train)
    n_classes = len(classes_counts)
    
    # ==========================================
    # 1. OPTIMISATION DE XGBOOST
    # ==========================================
    print("\n" + "="*15 + " RECHERCHE DES HYPERPARAMÈTRES : XGBOOST " + "="*15)
    
    # 生成 XGBoost 的样本权重向量
    xgb_sample_weights = y_train.map(lambda label: total_samples / (n_classes * classes_counts[label]))
    
    xgb_model = XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
    
    # 定义 XGBoost 搜索网格
    xgb_param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 150]
    }
    
    # scoring='f1_macro' 确保重点优化少数类
    xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, scoring='f1_macro', cv=3, verbose=1, n_jobs=-1)
    # XGBoost 传入样本权重需要放在 fit_params 中
    xgb_grid.fit(X_train_scaled, y_train, sample_weight=xgb_sample_weights)
    
    print(f"\nMeilleurs paramètres XGBoost : {xgb_grid.best_params_}")
    y_pred_xgb = xgb_grid.predict(X_test_scaled)
    print("\nRapport d'évaluation XGBoost Optimisé :")
    print(classification_report(y_test, y_pred_xgb, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))

    # ==========================================
    # 2. OPTIMISATION DE LIGHTGBM
    # ==========================================
    print("\n" + "="*15 + " RECHERCHE DES HYPERPARAMÈTRES : LIGHTGBM " + "="*15)
    
    # LightGBM 可以通过内置的 class_weight='balanced' 自动处理权重，省去手动计算
    lgb_model = LGBMClassifier(objective='multiclass', num_class=3, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    
    # 定义 LightGBM 搜索网格
    lgb_param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 150],
        'num_leaves': [15, 31, 63]  # LightGBM 核心参数，通常小于 2^max_depth
    }
    
    lgb_grid = GridSearchCV(lgb_model, lgb_param_grid, scoring='f1_macro', cv=3, verbose=1, n_jobs=-1)
    lgb_grid.fit(X_train_scaled, y_train)
    
    print(f"\nMeilleurs paramètres LightGBM : {lgb_grid.best_params_}")
    y_pred_lgb = lgb_grid.predict(X_test_scaled)
    print("\nRapport d'évaluation LightGBM Optimisé :")
    print(classification_report(y_test, y_pred_lgb, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))

    # ==========================================
    # 3. SAUVEGARDE DU MEILLEUR MODÈLE GLOBAL
    # ==========================================
    os.makedirs(ai_engine_med_dir, exist_ok=True)
    
    # 比较两者的最佳宏观 F1 得分，自动保存最优者
    if xgb_grid.best_score_ > lgb_grid.best_score_:
        best_model = xgb_grid.best_estimator_
        model_name = "best_tuned_xgboost.joblib"
        print(f"\n[SUCCÈS] XGBoost est globalement meilleur (F1 Macro Mean = {xgb_grid.best_score_:.4f})")
    else:
        best_model = lgb_grid.best_estimator_
        model_name = "best_tuned_lightgbm.joblib"
        print(f"\n[SUCCÈS] LightGBM est globalement meilleur (F1 Macro Mean = {lgb_grid.best_score_:.4f})")
        
    joblib.dump(best_model, os.path.join(ai_engine_med_dir, model_name))
    joblib.dump(scaler, os.path.join(ai_engine_med_dir, "tuned_med_scaler.joblib"))
    print(f"Modèle sauvegardé dans ai_engine/med/ sous le nom : {model_name}")

if __name__ == "__main__":
    run_hyperparameter_tuning()