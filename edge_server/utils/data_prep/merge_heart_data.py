import os
import pandas as pd
import numpy as np

def merge_heart_and_fatigue():
    # 1. 获取当前脚本的绝对路径
    current_path = os.path.abspath(__file__)
    
    # 2. 精准截取 edge_server 项目根目录，彻底免疫任何层级深度
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("错误: 脚本未放置在 edge_server 文件夹内！")
        return
        
    # 3. 强行锁定正确的 data 路径
    data_dir = os.path.join(base_project_dir, 'data')
    
    fatigue_cleaned_path = os.path.join(data_dir, 'fatigueset_cleaned.csv')
    heart_raw_path = os.path.join(data_dir, 'heart_statlog_cleveland_hungary_final.csv')
    output_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    
    # 4. 检查前置文件是否存在
    if not os.path.exists(fatigue_cleaned_path):
        print(f"错误: 未找到上一步生成的清洗文件，预期路径应为:\n  -> {fatigue_cleaned_path}\n请先确认 clean_fatigueset.py 是否成功跑完。")
        return
    if not os.path.exists(heart_raw_path):
        print(f"错误: 未在 data 目录下找到心脏病原始数据集，预期路径应为:\n  -> {heart_raw_path}")
        return
        
    print("【进度】开始精确读取 FatigueSet 数据与心脏病数据...")
    df_fatigue = pd.read_csv(fatigue_cleaned_path)
    df_heart_raw = pd.read_csv(heart_raw_path)
    
    # 5. 通过列的物理位置索引 (iloc) 强制提取核心特征
    df_heart_prepared = pd.DataFrame()
    df_heart_prepared['HeartRate'] = df_heart_raw.iloc[:, 7]
    df_heart_prepared['HRV'] = df_heart_raw.iloc[:, 9] * 100.0
    df_heart_prepared['Label'] = df_heart_raw.iloc[:, -1].apply(lambda x: 2 if x == 1 else 0)
    
    # 6. 精简列结构
    df_fatigue_reduced = df_fatigue[['HeartRate', 'HRV', 'Label']]
    df_heart_prepared = df_heart_prepared[['HeartRate', 'HRV', 'Label']]
    
    print(f"【进度】心脏病数据特征提取完成，共计 {len(df_heart_prepared)} 条样本。")
    print("【进度】正在合并两个数据集并构建最终多分类训练集...")
    final_train_df = pd.concat([df_fatigue_reduced, df_heart_prepared], ignore_index=True)
    final_train_df.dropna(inplace=True)
    
    # 7. 写入最终大表
    final_train_df.to_csv(output_path, index=False)
    
    print("\n==========================================")
    print("身体状态数据集最终合并成功！")
    print(f"生成最终训练集路径: {output_path}")
    print(f"总计总样本量: {len(final_train_df)} 行")
    print("==========================================")

if __name__ == "__main__":
    merge_heart_and_fatigue()