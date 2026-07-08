import pandas as pd
import numpy as np
import os

def calculate_heat_index(T_celsius, RH):
    T_f = T_celsius * 1.8 + 32
    HI_f = 0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094))
    mask = T_f >= 80
    if mask.any():
        T = T_f; R = RH
        HI_full = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 
                   6.83783e-3*T**2 - 5.481717e-2*R**2 + 1.22874e-3*T**2*R + 
                   8.5282e-4*T*R**2 - 1.99e-6*T**2*R**2)
        inner_value = (17 - np.abs(T - 91.)) / 14
        adj_dry = ((13 - R) / 4) * np.sqrt(np.clip(inner_value, 0, None))
        adj_humid = ((R - 85) / 10) * ((87 - T) / 5)
        HI_full = np.where((R < 13) & (T >= 80) & (T <= 112), HI_full - adj_dry, HI_full)
        HI_full = np.where((R > 85) & (T >= 80) & (T <= 87), HI_full + adj_humid, HI_full)
        HI_f = np.where(mask, HI_full, HI_f)
    return (HI_f - 32) / 1.8

def process_single_file(df, is_smoke_dataset=False):
    if is_smoke_dataset:
        # ==========================================================
        # 🎯 核心净化：如果是火灾数据集，直接剔除所有未着火的干扰行！
        # 只保留真正触发警报（Fire Alarm == 1）的硬核火灾数据
        # ==========================================================
        df = df[df['Fire Alarm'] == 1].copy()
        if df.empty:
            return pd.DataFrame()
            
        df['Timestamp'] = pd.to_datetime(df['UTC'], unit='s')
        df = df.rename(columns={'Temperature[C]': 'Temperature', 'Humidity[%]': 'Humidity'})
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.rename(columns={'Ambient_Temperature': 'Temperature', 'Ambient_Humidity': 'Humidity'})
        
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    df['dt'] = df['Timestamp'].diff().dt.total_seconds().replace(0, 1.0)
    
    temp_diff = df['Temperature'].diff()
    hum_diff = df['Humidity'].diff()
    
    # 保持传感器台阶降噪截断
    df['Temp_Rate'] = np.where(temp_diff.abs() <= 1, 0.0, temp_diff / df['dt'])
    df['Humidity_Rate'] = np.where(hum_diff.abs() <= 1, 0.0, hum_diff / df['dt'])
    
    df['Heat_Index'] = calculate_heat_index(df['Temperature'], df['Humidity'])
    
    if is_smoke_dataset:
        df['Label'] = 2  # 净化后，火灾数据集里的行百分之百是火灾标签
    else:
        df['Label'] = np.where(df['Temperature'] >= 40, 1, 0)
        
    df.dropna(subset=['Temp_Rate', 'Humidity_Rate'], inplace=True)
    
    cols = ['Timestamp', 'Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index', 'Label']
    return df[cols]

def main():
    current_path = os.path.abspath(__file__)
    base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    data_dir = os.path.join(base_project_dir, 'data')
    
    smoke_path = os.path.join(data_dir, 'smoke_detection_iot.csv')
    car_train_path = os.path.join(data_dir, 'train-00000-of-00001.parquet')
    car_test_path = os.path.join(data_dir, 'test-00000-of-00001.parquet')
    
    print("Purge des données normales du dataset de feu en cours...")
    
    df_smoke = pd.read_csv(smoke_path)
    processed_smoke = process_single_file(df_smoke, is_smoke_dataset=True)
    
    smoke_train = processed_smoke.sample(frac=0.8, random_state=42)
    smoke_test = processed_smoke.drop(smoke_train.index)
    
    df_car_train = pd.read_parquet(car_train_path)
    processed_car_train = process_single_file(df_car_train, is_smoke_dataset=False)
    
    df_car_test = pd.read_parquet(car_test_path)
    processed_car_test = process_single_file(df_car_test, is_smoke_dataset=False)
    
    final_train = pd.concat([smoke_train, processed_car_train], ignore_index=True).sort_values('Timestamp')
    final_test = pd.concat([smoke_test, processed_car_test], ignore_index=True).sort_values('Timestamp')
    
    train_output_path = os.path.join(data_dir, 'final_train_data_5_features.csv')
    test_output_path = os.path.join(data_dir, 'final_test_data_5_features.csv')
    
    final_train.to_csv(train_output_path, index=False)
    final_test.to_csv(test_output_path, index=False)
    
    print(f"Pipeline de purification terminé avec succès !")

if __name__ == "__main__":
    main()