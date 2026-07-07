import pandas as pd
import numpy as np
import os
import xgboost as xgb

def calculate_heat_index(T_celsius, RH):
    """标准热指数计算公式（与训练完全一致）"""
    T_f = T_celsius * 1.8 + 32
    HI_f = 0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094))
    
    mask = T_f >= 80
    if mask.any():
        T = T_f
        R = RH
        HI_full = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 
                   6.83783e-3*T**2 - 5.481717e-2*R**2 + 1.22874e-3*T**2*R + 
                   8.5282e-4*T*R**2 - 1.99e-6*T**2*R**2)
        
        adj_dry = ((13 - R) / 4) * np.sqrt((17 - np.abs(T - 91.)) / 14)
        adj_humid = ((R - 85) / 10) * ((87 - T) / 5)
        
        HI_full = np.where((R < 13) & (T >= 80) & (T <= 112), HI_full - adj_dry, HI_full)
        HI_full = np.where((R > 85) & (T >= 80) & (T <= 87), HI_full + adj_humid, HI_full)
        
        HI_f = np.where(mask, HI_full, HI_f)
        
    return (HI_f - 32) / 1.8

def test_inference_performance():
    # ==========================================
    # 动态路径截取定位（基于你严谨的 edge_server 切分法）
    # ==========================================
    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("Erreur : Le script n'est pas placé dans le dossier 'edge_server' !")
        return

    # 定位输入与模型路径
    data_path = os.path.join(base_project_dir, 'data', 'sensor_data_for_ai 1.csv')
    model_path = os.path.join(base_project_dir, 'ai_engine', 'env', 'car_safety_xgboost_model.json')
    output_path = os.path.join(base_project_dir, 'data', 'sensor_inference_output.csv')

    if not os.path.exists(data_path):
        print(f"Erreur : Dataset d'inférence introuvable à {data_path}")
        return
    if not os.path.exists(model_path):
        print(f"Erreur : Modèle entraîné introuvable à {model_path}")
        return

    print("Chargement du dataset de test...")
    df = pd.read_csv(data_path)
    
    # 兼容处理原始小写字段并转换为时间标准序列
    df = df.rename(columns={'timestamp': 'Timestamp', 'temperature': 'Temperature', 'humidity': 'Humidity'})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # ==========================================
    # 动态流式特征提取
    # ==========================================
    print("Calcul des caractéristiques dynamiques (Rates & Heat Index)...")
    df['dt'] = df['Timestamp'].diff().dt.total_seconds()
    df['dt'] = df['dt'].replace(0, 1.0) # 避免分母为0
    
    # 计算一阶差分变化率
    df['Temp_Rate'] = df['Temperature'].diff() / df['dt']
    df['Humidity_Rate'] = df['Humidity'].diff() / df['dt']
    df['Heat_Index'] = calculate_heat_index(df['Temperature'], df['Humidity'])
    
    # 用第二行填充首行的一阶差分空值
    df['Temp_Rate'] = df['Temp_Rate'].bfill()
    df['Humidity_Rate'] = df['Humidity_Rate'].bfill()

    # ==========================================
    # 导入完全体模型进行判定
    # ==========================================
    print("Chargement du modèle XGBoost Optimisé...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    feature_cols = ['Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index']
    
    print("Exécution des prédictions sur le flux de données...")
    df['Predicted_Label'] = model.predict(df[feature_cols])
    
    # 预测概率矩阵
    prob_matrix = model.predict_proba(df[feature_cols])
    df['Prob_Sécurité(0)'] = prob_matrix[:, 0]
    df['Prob_Canicule(1)'] = prob_matrix[:, 1]
    df['Prob_Feu(2)'] = prob_matrix[:, 2]

    # 法语标签化文本映射
    status_map = {0: 'Sécurité', 1: 'Canicule', 2: 'Feu/Fumée'}
    df['Status'] = df['Predicted_Label'].map(status_map)

    # 保存预测详情结果至 data/ 文件夹
    df.to_csv(output_path, index=False)
    print(f"\nEvaluation terminée ! Résultats enregistrés sous : {output_path}")
    
    # ==========================================
    # 控制台统计分析展示
    # ==========================================
    print("\n" + "="*40)
    print("📊 STATISTIQUES DE DIAGNOSTIC AUTOMATIQUE")
    print("="*40)
    counts = df['Status'].value_counts()
    for status_type in ['Sécurité', 'Canicule', 'Feu/Fumée']:
        print(f"  -> {status_type} : {counts.get(status_type, 0)} lignes détectées")
    print("="*40)
    
    # 抽样观察突变区域（主要观察湿度波动的那个时间段）
    print("\n👀 Extrait des prédictions (Lignes 5 à 15) :")
    print(df[['Timestamp', 'Temperature', 'Humidity', 'Status', 'Prob_Feu(2)']].iloc[5:16])

if __name__ == "__main__":
    test_inference_performance()