import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

def train_driver_status_xgboost():
    # 获取当前脚本的绝对路径 (当前在 edge_server/ai_trainning/med/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 向上回溯两级定位到 edge_server 根目录
    base_project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # 精准定位数据目录与模型输出目录
    data_dir = os.path.join(base_project_dir, 'data')
    ai_engine_med_dir = os.path.join(base_project_dir, 'ai_engine', 'med')
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    
    # 保持原文件名，确保评估脚本自动检索和覆盖
    model_output_path = os.path.join(ai_engine_med_dir, 'xgboost_body_model.joblib')
    scaler_output_path = os.path.join(ai_engine_med_dir, 'data_scaler_xgboost.joblib')
    
    # 检查训练集是否存在
    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}. Veuillez vérifier l'emplacement de votre fichier CSV.")
        return

    print("Chargement de la base de données d'entraînement...")
    df = pd.read_csv(train_data_path)
    
    X = df[['HeartRate', 'HRV']]
    y = df['Label']
    
    # 划分训练集与测试集（保持 80/20 比例与分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 尽管树模型对特征缩放不敏感，但为了后续多模态流水线的一致性，依旧保留标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 计算类别权重以应对严重的数据不平衡 (Fatigue 与 Crise_Cardiaque 的样本差异)
    # 计算公式：总样本数 / (类别数 * 该类样本数)
    classes_counts = y_train.value_counts()
    total_samples = len(y_train)
    n_classes = len(classes_counts)
    
    # 为训练集中的每个样本分配对应的权重权重
    sample_weights = y_train.map(lambda label: total_samples / (n_classes * classes_counts[label]))
    
    print("Entraînement du classifieur XGBoost en cours...")
    # multi:softprob 用于多分类任务，输出每个类别的概率
    # 此处已更新为网格搜索得出的最优参数组合：learning_rate=0.2, max_depth=6, n_estimators=150
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.2,
        random_state=42,
        n_jobs=-1
    )
    
    # 在 fit 时传入样本权重，强制模型强力关注稀少的心脏病样本
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    print("Entraînement du modèle terminé.")
    
    # 在测试集上进行预测并输出评估报告
    y_pred = model.predict(X_test_scaled)
    print("\n================ RAPPORT D'ÉVALUATION ================")
    print(classification_report(y_test, y_pred, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))
    print("======================================================")
    
    # 确保输出目录存在
    os.makedirs(ai_engine_med_dir, exist_ok=True)
    
    # 将模型与标准化工具保存至指定的 ai_engine/med/ 文件夹下
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"Le modèle XGBoost et le standardiseur ont été enregistrés avec succès dans le dossier ai_engine/med.")

if __name__ == "__main__":
    train_driver_status_xgboost()