import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Permet à l'utilisateur de choisir un modèle enregistré, charge son standardiseur 
# associé, puis évalue ses performances sur le jeu de test via un rapport et une matrice.
def evaluate_selected_model():
    # 获取当前脚本的绝对路径
    current_path = os.path.abspath(__file__)
    # 动态定位项目根目录 edge_server
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        return
        
    # 定义数据和模型所在的绝对路径
    data_path = os.path.join(base_project_dir, 'data', 'driver_body_status_train.csv')
    ai_engine_dir = os.path.join(base_project_dir, 'ai_engine', 'med')
    
    # 检查训练集文件是否存在
    if not os.path.exists(data_path):
        print(f"Erreur : Fichier de données introuvable - {data_path}")
        return
        
    # 扫描 ai_engine/med 文件夹下所有以 _model.joblib 结尾的模型文件
    model_files = glob.glob(os.path.join(ai_engine_dir, '*_model.joblib'))
    if not model_files:
        print(f"Erreur : Aucun modèle trouvé dans {ai_engine_dir}")
        return
        
    # 在终端打印当前所有可用的模型列表
    print("=== Modèles disponibles pour l'évaluation ===")
    for idx, file_path in enumerate(model_files):
        print(f"[{idx}] {os.path.basename(file_path)}")
    print("=============================================")
    
    # 获取用户输入的模型编号
    try:
        choix = int(input("Entrez le numéro du modèle à évaluer : "))
        if choix < 0 or choix >= len(model_files):
            print("Choix invalide.")
            return
    except ValueError:
        print("Veuillez entrer un nombre valide.")
        return
        
    # 获取所选模型的路径和名称
    selected_model_path = model_files[choix]
    model_name = os.path.basename(selected_model_path).replace('.joblib', '')
    
    # 动态匹配对应的标准化工具 (Scaler)
    scaler_name = 'data_scaler.joblib' if 'random_forest' in model_name else f"data_scaler_{model_name.split('_')[0]}.joblib"
    selected_scaler_path = os.path.join(ai_engine_dir, scaler_name)
    
    # 如果找不到特有的 Scaler，则降级使用默认的 data_scaler.joblib
    if not os.path.exists(selected_scaler_path):
        selected_scaler_path = os.path.join(ai_engine_dir, 'data_scaler.joblib')
        
    print(f"\nChargement du modèle : {os.path.basename(selected_model_path)}")
    print(f"Chargement du standardiseur : {os.path.basename(selected_scaler_path)}")
    
    # 从磁盘反序列化加载模型和标准化工具
    model = joblib.load(selected_model_path)
    scaler = joblib.load(selected_scaler_path)
    
    # 读取训练大表并提取特征与标签
    df = pd.read_csv(data_path)
    X = df[['HeartRate', 'HRV']]
    y = df['Label']
    
    # 使用完全相同的切分参数划分测试集（保持 random_state=42 和分层抽样）
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 使用加载的 Scaler 对测试集特征进行标准化转换
    X_test_scaled = scaler.transform(X_test)
    
    # 模型预测
    y_pred = model.predict(X_test_scaled)
    
    # 打印法语分类评估报告
    print("\n" + "="*20 + " RAPPORT D'ÉVALUATION COMPLET " + "="*20)
    print(f"Modèle évalué : {model_name.upper()}")
    print("-"*70)
    print(classification_report(y_test, y_pred, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))
    
    # 计算并打印混淆矩阵
    print("-"*70)
    print("MATRICE DE CONFUSION :")
    cm = confusion_matrix(y_test, y_pred)
    print("                      Prédit: 0   Prédit: 1   Prédit: 2")
    print(f"Réel 0 (Normal)        {cm[0][0]:<12}{cm[0][1]:<12}{cm[0][2]:<12}")
    print(f"Réel 1 (Fatigue)       {cm[1][0]:<12}{cm[1][1]:<12}{cm[1][2]:<12}")
    print(f"Réel 2 (Cardiaque)     {cm[2][0]:<12}{cm[2][1]:<12}{cm[2][2]:<12}")
    print("="*70 + "\n")

if __name__ == "__main__":
    evaluate_selected_model()