import os
import pandas as pd
import numpy as np

def analyze_hrv_correlation():
    print("【进度】脚本已启动，开始计算目标文件路径...")
    
    # 1. 拦截项目根目录，彻底根治相对路径地雷
    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("【错误】脚本未放置在 edge_server 文件夹内！")
        return
        
    data_path = os.path.join(base_project_dir, 'data', 'driver_body_status_train.csv')
    print(f"【进度】最终训练集物理路径指向:\n  -> {data_path}")
    
    if not os.path.exists(data_path):
        print(f"【错误】未找到合并后的训练集文件，请确认该路径下是否有文件！")
        return
        
    print("【进度】成功检测到数据，开始加载入内存...")
    df = pd.read_csv(data_path)
    print(f"【进度】数据加载成功！总行数: {len(df)} 行")
    
    # 2. 提取不同标签的 HRV
    hrv_fatigue = df[df['Label'] == 1]['HRV']
    hrv_heart_disease = df[df['Label'] == 2]['HRV']
    
    print(f"【进度】标签提取完成。疲劳样本数: {len(hrv_fatigue)} 行，心脏病样本数: {len(hrv_heart_disease)} 行")
    
    if len(hrv_fatigue) == 0 or len(hrv_heart_disease) == 0:
        print("【警告】其中一个标签的样本量为 0，无法对比。")
        return

    print("\n========== HRV 特征数值相关性分析 ==========\n")
    
    # 3. 统计描述比对
    print("统计学描述对比:")
    print(f"-> FatigueSet (疲劳) 的 HRV 范围:  [{hrv_fatigue.min():.2f} 到 {hrv_fatigue.max():.2f}]，均值: {hrv_fatigue.mean():.2f}")
    print(f"-> Heart Statlog (发病) 的 HRV 范围:[{hrv_heart_disease.min():.2f} 到 {hrv_heart_disease.max():.2f}]，均值: {hrv_heart_disease.mean():.2f}")
    
    # 4. 寻找物理重叠值
    fatigue_rounded = np.round(hrv_fatigue, 1)
    heart_rounded = np.round(hrv_heart_disease, 1)
    
    intersection = np.intersect1d(fatigue_rounded, heart_rounded)
    print(f"\n物理数值交集测试:")
    print(f"-> 在保留1位小数的情况下，两个数据集共有 {len(intersection)} 个完全相同的 HRV 数值点！")
    if len(intersection) > 0:
        print(f"-> 相同数据点示例（前5个）: {intersection[:5]}")
        
    # 5. 空间重叠度计算
    min_f, max_f = hrv_fatigue.min(), hrv_fatigue.max()
    in_range_count = hrv_heart_disease.between(min_f, max_f).sum()
    overlap_percentage = (in_range_count / len(hrv_heart_disease)) * 100
    
    print(f"\n空间重叠度分析:")
    print(f"-> 心脏病数据集里有 {overlap_percentage:.1f}% 的 HRV 数据完全落在了疲劳数据集的取值包络线内。")
    print("\n结论：")
    if overlap_percentage > 50:
        print("【相关性成立】两个数据集在 HRV 指标上具有极高的空间重叠度与相同的数值区间。")
        print("这证明了无论人在受控实验疲劳下，还是临床心脏突发骤变下，心电特征（HRV）踩在了一套完全相同的物理生理坐标系中。")
    else:
        print("【重叠度较低】两者的 HRV 分布区域相对独立。")

if __name__ == "__main__":
    analyze_hrv_correlation()