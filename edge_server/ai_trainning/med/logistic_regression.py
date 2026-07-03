import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_driver_status_logistic_regression():
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 动态定位数据和模型的存储路径
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    ai_engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'ai_engine'))
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    model_output_path = os.path.join(ai_engine_dir, 'logistic_regression_model.joblib')
    scaler_output_path = os.path.join(ai_engine_dir, 'data_scaler_logistic.joblib')
    
    # 检查训练集是否存在
    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}. Veuillez vérifier l'exécution de merge_heart_data.py.")
        return

    print("Chargement de la base de données d'entraînement...")
    df = pd.read_csv(train_data_path)
    
    X = df[['HeartRate', 'HRV']]
    y = df['Label']
    
    # 划分训练集与测试集（保持 80/20 比例与分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 逻辑回归对特征缩放极度敏感，必须进行标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Entraînement du classifieur Régression Logistique en cours...")
    # 使用 class_weight='balanced' 来对抗严重的数据不平衡问题
    # multi_class='multinomial' 用于处理三分类任务
    model = LogisticRegression(
        class_weight='balanced',
        solver='lbfgs',
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    print("Entraînement du modèle terminé.")
    
    # 在测试集上进行预测并输出评估报告
    y_pred = model.predict(X_test_scaled)
    print("\n================ RAPPORT D'ÉVALUATION ================")
    print(classification_report(y_test, y_pred, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))
    print("======================================================")
    
    # 将模型与标准化工具保存至指定的 ai_engine 文件夹下
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"Le modèle Régression Logistique et le standardiseur ont été enregistrés avec succès dans le dossier ai_engine.")

if __name__ == "__main__":
    train_driver_status_logistic_regression()