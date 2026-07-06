import os
import pandas as pd
from pycaret.classification import setup, compare_models, tune_model

def run_automl_safety_selection():
    # ==========================================
    # 动态路径定位（采用你指定的严谨切割写法）
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
    
    # 提取我们核心关注的 5 个特征和标签列
    feature_cols = ['Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index', 'Label']
    
    # PyCaret 内部会自动进行交叉验证分割，我们直接将融合好的训练集作为输入
    data_for_automl = df_train[feature_cols]
    
    print("\n================ CONFIGURATION DE L'EXPÉRIENCE ================")
    # 初始化 PyCaret 3分类环境
    # fix_imbalance=True: 自动启用平衡算法处理稀少的 Canicule (1) 样本不平衡问题
    clf_setup = setup(
        data=data_for_automl,
        target='Label',
        train_size=0.8,
        fix_imbalance=True, 
        preprocess=True,
        numeric_features=['Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index'],
        session_id=42,
        html=False,       # 确保在 PowerShell 中以纯文本表格输出
        verbose=True,
    )
    
    print("\n================ COMPARAISON DES MODÈLES D'IA ================")
    print("Évaluation automatique de tous les algorithmes disponibles (LightGBM, RF, XGBoost, etc.)...")
    
    # 自动对比所有分类器模型
    # sort='F1': 鉴于多分类且不平衡场景，使用 F1-macro 作为核心排序标准
    # fold=3: 3折交叉验证，在百万级数据量下取得速度与准确性的平衡
    best_model = compare_models(
        exclude=['svm', 'gpc', 'rbfsvm', 'ridge'], 
        sort='F1',
        fold=3,
        budget_time=15   # 设定时限预算（分钟），防止脚本挂起
    )
    
    print("\n================ OPTIMISATION DU MEILLEUR MODÈLE ================")
    print("Recherche des meilleurs hyperparamètres pour le modèle sélectionné...")
    
    # 对筛选出的第一名模型进行进一步的超参数全自动微调
    tuned_model = tune_model(
        best_model, 
        optimize='F1', 
        fold=3,
        choose_better=True
    )
    
    print("\n================ RÉSULTATS DE L'OPTIMISATION ================")
    print("Voici la configuration finale et optimisée du meilleur modèle :")
    print("--------------------------------------------------")
    print(tuned_model)
    print("--------------------------------------------------")
    print("Veuillez reporter ces paramètres dans votre script d'entraînement principal.")
    print("================================================================")

if __name__ == "__main__":
    run_automl_safety_selection()