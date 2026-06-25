import os
import glob
import pandas as pd
import numpy as np

def clean_and_merge_fatigueset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'fatigueset'))
    output_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    
    print(f"数据扫描根目录: {base_data_dir}")
    print("开始扫描 Fatigueset 目录...")
    
    all_sessions_compiled = []
    
    search_pattern = os.path.join(base_data_dir, '*', '*').replace('\\', '/')
    all_paths = glob.glob(search_pattern)
    
    session_paths = []
    for p in all_paths:
        parts = p.split(os.sep)
        if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
            session_paths.append(p)
            
    print(f"共发现 {len(session_paths)} 个有效的数字场次(Session)目录。")
    
    if not session_paths:
        print("错误: 未找到任何有效的 '数字/数字' 结构的子文件夹。")
        return

    for session_path in session_paths:
        parts = session_path.split(os.sep)
        participant = parts[-2]
        session = parts[-1]
        
        hr_file = os.path.join(session_path, 'wrist_hr.csv')
        temp_file = os.path.join(session_path, 'wrist_skin_temperature.csv')
        rr_file = os.path.join(session_path, 'chest_rr_interval.csv')
        fatigue_survey_file = os.path.join(session_path, 'exp_fatigue.csv')
        
        if not (os.path.exists(hr_file) and os.path.exists(temp_file) and os.path.exists(rr_file) and os.path.exists(fatigue_survey_file)):
            continue
            
        print(f"正在处理: 参与者 {participant} -> 场次 {session}")
        
        # 读取疲劳分数
        df_survey = pd.read_csv(fatigue_survey_file)
        avg_fatigue = df_survey.iloc[:, 1].mean() if 'fatigue_score' not in df_survey.columns else df_survey['fatigue_score'].mean()
        
        if avg_fatigue <= 4.0:
            current_label = 0  
        elif avg_fatigue >= 7.0:
            current_label = 1  
        else:
            print(f"  [过滤] {participant}/{session} 处于中度疲劳 ({avg_fatigue:.1f}分)")
            continue
            
        # 读取时序生理信号
        df_hr = pd.read_csv(hr_file)
        df_temp = pd.read_csv(temp_file)
        df_rr = pd.read_csv(rr_file)
        
        df_hr['time'] = pd.to_datetime(df_hr['timestamp'], unit='s')
        df_temp['time'] = pd.to_datetime(df_temp['timestamp'], unit='s')
        df_rr['time'] = pd.to_datetime(df_rr['timestamp'], unit='s')
        
        # 【核心修正】将 '1S' 修改为小写的 '1s' 以兼容新版 Pandas
        resampled_hr = df_hr.resample('1s', on='time')['hr'].mean().reset_index()
        resampled_temp = df_temp.resample('1s', on='time')['temp'].mean().reset_index()
        resampled_hrv = df_rr.resample('1s', on='time')['duration'].std().reset_index()
        
        resampled_hrv.rename(columns={'duration': 'hrv'}, inplace=True)
        resampled_hrv['hrv'] = resampled_hrv['hrv'].fillna(0)
        
        # 合并特征
        session_merged = pd.merge(resampled_hr, resampled_temp, on='time', how='inner')
        session_merged = pd.merge(session_merged, resampled_hrv, on='time', how='inner')
        
        session_merged['Label'] = current_label
        session_merged = session_merged[['hr', 'temp', 'hrv', 'Label']]
        session_merged.columns = ['HeartRate', 'Temperature', 'HRV', 'Label']
        
        all_sessions_compiled.append(session_merged)

    # 汇总并保存
    if all_sessions_compiled:
        final_fatigue_df = pd.concat(all_sessions_compiled, ignore_index=True)
        final_fatigue_df.dropna(inplace=True)
        
        output_file_path = os.path.join(output_dir, 'fatigueset_cleaned.csv')
        final_fatigue_df.to_csv(output_file_path, index=False)
        print("\n==========================================")
        print("Fatigueset 特征清洗与对齐完成！")
        print(f"生成文件路径: {output_file_path}")
        print(f"总计提取样本数: {len(final_fatigue_df)} 行 (秒)")
        print("==========================================")
    else:
        print("\n[提示] 未提取到任何满足过滤标准的有效时序数据。")

if __name__ == "__main__":
    clean_and_merge_fatigueset()