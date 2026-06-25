import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_driver_status_model():
    # 精准定位路径：当前在 ai_engine，数据在 ../data/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    model_output_path = os.path.join(current_dir, 'random_forest_body_model.joblib')
    scaler_output_path = os.path.join(current_dir, 'data_scaler.joblib')
    
    if not os.path.exists(train_data_path):
        print(f"错误: 未找到训练数据集 {train_data_path}，请确认 merge_heart_data.py 是否成功运行。")
        return

    print("正在加载合并后的身体状态训练数据集...")
    df = pd.read_csv(train_data_path)
    
    X = df[['HeartRate', 'HRV']]
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("正在训练平衡权重的随机森林分类器...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("模型训练完成。")
    
    y_pred = model.predict(X_test_scaled)
    print("\n================ 模型评估报告 ================")
    print(classification_report(y_test, y_pred, target_names=['0:正常', '1:疲劳', '2:心脏病']))
    print("==============================================")
    
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"权重及标准化工具已成功保存在 ai_engine 文件夹下。")

if __name__ == "__main__":
    train_driver_status_model()