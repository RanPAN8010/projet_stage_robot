import pandas as pd
import numpy as np
import os

# 标准热指数计算函数（Rothfusz回归方程）
def calculate_heat_index(T_celsius, RH):
    T_f = T_celsius * 1.8 + 32
    HI_f = 0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094))
    
    mask = T_f >= 80
    if mask.any():
        T = T_f
        R = RH
        HI_full = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 
                   6.83783e-3*T**2 - 5.481717e-2*R**2 + 1.22874e-3*T**2*R + 
                   8.5282e-4*T*R**2 - 1.99e-6*T**2*R**2)
        
        # 使用 np.clip(..., 0, None) 将小于 0 的负数全部强制转换为 0
        inner_value = (17 - np.abs(T - 91.)) / 14
        adj_dry = ((13 - R) / 4) * np.sqrt(np.clip(inner_value, 0, None))
        adj_humid = ((R - 85) / 10) * ((87 - T) / 5)
        
        HI_full = np.where((R < 13) & (T >= 80) & (T <= 112), HI_full - adj_dry, HI_full)
        HI_full = np.where((R > 85) & (T >= 80) & (T <= 87), HI_full + adj_humid, HI_full)
        
        HI_f = np.where(mask, HI_full, HI_f)
        
    return (HI_f - 32) / 1.8

# 单个文件提取特征的通用清洗函数
def process_single_file(df, is_smoke_dataset=False):
    # 统一列名
    if is_smoke_dataset:
        df['Timestamp'] = pd.to_datetime(df['UTC'], unit='s')
        df = df.rename(columns={'Temperature[C]': 'Temperature', 'Humidity[%]': 'Humidity'})
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.rename(columns={'Ambient_Temperature': 'Temperature', 'Ambient_Humidity': 'Humidity'})
        
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # 计算时间步长和温湿度一阶导数
    df['dt'] = df['Timestamp'].diff().dt.total_seconds()
    # 避免分母为 0 导致无穷大
    df['dt'] = df['dt'].replace(0, 1.0) 
    
    df['Temp_Rate'] = df['Temperature'].diff() / df['dt']
    df['Humidity_Rate'] = df['Humidity'].diff() / df['dt']
    
    # 计算热指数
    df['Heat_Index'] = calculate_heat_index(df['Temperature'], df['Humidity'])
    
    # 定义分类状态标签 (Target Label):
    # 0 = 正常车内环境
    # 1 = 车内极端高温 (此处根据热指数或原始温度 > 40°C 来定义，你可以根据需求修改阈值)
    # 2 = 火灾
    if is_smoke_dataset:
        # 如果火灾数据集里 Fire Alarm 为 1，则为状态 2（火灾）；为 0 则属于正常或高温
        df['Label'] = np.where(df['Fire Alarm'] == 1, 2, np.where(df['Temperature'] >= 40, 1, 0))
    else:
        # 车内环境数据集全都没有火灾，只区分为 1（极端高温）和 0（正常）
        df['Label'] = np.where(df['Temperature'] >= 40, 1, 0)
        
    df.dropna(subset=['Temp_Rate', 'Humidity_Rate'], inplace=True)
    
    cols = ['Timestamp', 'Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index', 'Label']
    return df[cols]

def main():
    # 自动路径定位
    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("Erreur : Le script n'est pas placé dans le dossier 'edge_server' !")
        return

    data_dir = os.path.join(base_project_dir, 'data')
    
    # 原始文件完整路径
    smoke_path = os.path.join(data_dir, 'smoke_detection_iot.csv')
    car_train_path = os.path.join(data_dir, 'train-00000-of-00001.parquet')
    car_test_path = os.path.join(data_dir, 'test-00000-of-00001.parquet')
    
    print("Préparation du nettoyage et de la fusion des données...")
    
    # 处理火灾 CSV 数据
    df_smoke = pd.read_csv(smoke_path)
    processed_smoke = process_single_file(df_smoke, is_smoke_dataset=True)
    
    # 将火灾数据也切分为 80% 训练和 20% 测试，以便和车内环境的 train/test 完美融合
    smoke_train = processed_smoke.sample(frac=0.8, random_state=42)
    smoke_test = processed_smoke.drop(smoke_train.index)
    
    # 处理车内环境 Parquet 数据
    df_car_train = pd.read_parquet(car_train_path)
    processed_car_train = process_single_file(df_car_train, is_smoke_dataset=False)
    
    df_car_test = pd.read_parquet(car_test_path)
    processed_car_test = process_single_file(df_car_test, is_smoke_dataset=False)
    
    # 合并训练集与测试集
    final_train = pd.concat([smoke_train, processed_car_train], ignore_index=True).sort_values('Timestamp')
    final_test = pd.concat([smoke_test, processed_car_test], ignore_index=True).sort_values('Timestamp')
    
    # 保存结果到 data 文件夹
    train_output_path = os.path.join(data_dir, 'final_train_data_5_features.csv')
    test_output_path = os.path.join(data_dir, 'final_test_data_5_features.csv')
    
    final_train.to_csv(train_output_path, index=False)
    final_test.to_csv(test_output_path, index=False)
    
    print(f"Fusion terminée avec succès !")
    print(f"Fichier d'entraînement enregistré : {train_output_path} ({final_train.shape[0]} lignes)")
    print(f"Fichier de test enregistré : {test_output_path} ({final_test.shape[0]} lignes)")

if __name__ == "__main__":
    main()