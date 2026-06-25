import os
import pandas as pd
import numpy as np

def merge_heart_and_fatigue():
    # 1. 路径定义
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    
    fatigue_cleaned_path = os.path.join(data_dir, 'fatigueset_cleaned.csv')
    heart_raw_path = os.path.join(data_dir, 'heart_statlog_cleveland_hungary_final.csv')
    output_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    
    # 2. 检查前置文件
    if not os.path.exists(fatigue_cleaned_path):
        print(f"错误: 未找到上一步生成的清洗文件 {fatigue_cleaned_path}，请先运行 clean_fatigueset.py")
        return
    if not os.path.exists(heart_raw_path):
        print(f"错误: 未在 data 目录下找到心脏病原始数据集 {heart_raw_path}")
        return
        
    print("开始精确读取 FatigueSet 数据与心脏病数据...")
    df_fatigue = pd.read_csv(fatigue_cleaned_path)
    df_heart_raw = pd.read_csv(heart_raw_path)
    
    # 3. 绕过列名别名，通过列的位置索引 (iloc) 强制提取核心特征
    df_heart_prepared = pd.DataFrame()
    
    # .iloc[:, 7] 提取第8列 (max hr)
    df_heart_prepared['HeartRate'] = df_heart_raw.iloc[:, 7]
    
    # .iloc[:, 9] 提取第10列 (oldpeak)
    df_heart_prepared['HRV'] = df_heart_raw.iloc[:, 9] * 100.0
    
    # .iloc[:, -1] 提取最后一列 (target)
    # 原始 target 中，0代表正常，1代表患心脏病。
    # 合并时，正常(0)保持不变；患病(1)修改为 2 (突发心脏病危险)，防止与疲劳(1)冲突。
    df_heart_prepared['Label'] = df_heart_raw.iloc[:, -1].apply(lambda x: 2 if x == 1 else 0)
    
    # 4. 提取 FatigueSet 并精简列（确保两边都只有 ['HeartRate', 'HRV', 'Label']）
    df_fatigue_reduced = df_fatigue[['HeartRate', 'HRV', 'Label']]
    df_heart_prepared = df_heart_prepared[['HeartRate', 'HRV', 'Label']]
    
    print(f"心脏病数据特征提取完成，共计 {len(df_heart_prepared)} 条样本。")
    
    # 5. 纵向拼合大表 (Concat)
    print("正在合并两个数据集并构建最终多分类训练集...")
    final_train_df = pd.concat([df_fatigue_reduced, df_heart_prepared], ignore_index=True)
    
    # 过滤可能产生的极少数空值
    final_train_df.dropna(inplace=True)
    
    # 6. 保存最终结果
    final_train_df.to_csv(output_path, index=False)
    
    print("\n==========================================")
    print("身体状态数据集最终合并成功！")
    print(f"生成最终训练集路径: {output_path}")
    print(f"总计总样本量: {len(final_train_df)} 行")
    print(f"  -> [Label 0] 正常状态样本数: {len(final_train_df[final_train_df['Label'] == 0])} 行")
    print(f"  -> [Label 1] 疲劳危险样本数: {len(final_train_df[final_train_df['Label'] == 1])} 行")
    print(f"  -> [Label 2] 心脏病发样本数: {len(final_train_df[final_train_df['Label'] == 2])} 行")
    print("==========================================")

if __name__ == "__main__":
    merge_heart_and_fatigue()