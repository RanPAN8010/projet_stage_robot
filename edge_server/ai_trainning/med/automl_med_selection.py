import os
import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model

def run_automl_medical_selection():
    # 获取当前脚本的绝对路径 (edge_server/ai_trainning/med/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 向上回溯两级定位到 edge_server 根目录
    base_project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # 精准定位数据目录与模型输出目录
    data_dir = os.path.join(base_project_dir, 'data')
    ai_engine_med_dir = os.path.join(base_project_dir, 'ai_engine', 'med')
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    model_output_prefix = os.path.join(ai_engine_med_dir, 'best_automl_med_model')
    
    # 检查训练集是否存在
    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}.")
        return

    print("Chargement de la base de données pour l'AutoML...")
    df = pd.read_csv(train_data_path)
    
    # 只提取我们核心关注的二维生理特征和标签
    data_for_automl = df[['HeartRate', 'HRV', 'Label']]
    
    print("\n================ CONFIGURATION DE L'EXPÉRIENCE ================")
    # 初始化 PyCaret 环境
    # target: 预测的目标列
    # train_size: 训练集比例 (80/20 分割)
    # fix_imbalance: 自动启用 SMOTE 等算法处理稀少的心脏病样本不平衡问题
    # html=False: 确保在终端/PowerShell 中以纯文本格式漂亮地打印表格，而不是嵌入网页 HTML
    clf_setup = setup(
        data=data_for_automl,
        target='Label',
        train_size=0.8,
        fix_imbalance=True,
        preprocess=True,
        numeric_features=['HeartRate', 'HRV'],
        session_id=42,
        verbose=True,
    )
    
    print("\n================ COMPARAISON DES MODÈLES D'IA ================")
    print("Évaluation automatique de tous les algorithmes disponibles (RF, KNN, XGBoost, etc.)...")
    
    # 自动对比所有分类器模型
    # sort='F1': 鉴于医疗高危场景，我们让它优先以 F1-score 作为核心排序标准，平衡精确率与召回率
    # include: 显式指定重点考察对比的几个核心模型，你也可以删掉这一行让它跑全库十几种模型
    best_model = compare_models(
        exclude=['svm', 'gpc', 'rbfsvm'],
        sort='F1',
    )
    
    print("\n================ FINALISATION DU MEILLEUR MODÈLE ================")
    print("Entraînement final sur l'intégralité des données...")
    # 锁定并在全部数据上完整重新训练表现最好的那个模型
    final_model = finalize_model(best_model)
    
    # 确保输出目录存在
    os.makedirs(ai_engine_med_dir, exist_ok=True)
    
    # 保存最优模型（PyCaret 会自动把标准化预处理 Pipeline 和模型打包在一起保存为 .pkl 文件）
    save_model(final_model, model_output_prefix)
    print(f"\nLe meilleur modèle a été enregistré avec succès sous : {model_output_prefix}.pkl")
    print("================================================================")

if __name__ == "__main__":
    run_automl_medical_selection()