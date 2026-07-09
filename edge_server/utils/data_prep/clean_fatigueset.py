import os
import glob
import pandas as pd
import numpy as np

# dataset: https://www.kaggle.com/datasets/tanjemahamed/mental-fatigue-level-detection-fatigueset-data
# Scanne les dossiers de sessions, filtre la fatigue selon l'enquête, rééchantillonne 
# les données physiologiques à 1s et les fusionne dans un fichier nettoyé.
def clean_and_merge_fatigueset():
    current_path = os.path.abspath(__file__)
    
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
        
    base_data_dir = os.path.join(base_project_dir, 'data', 'fatigueset')
    output_dir = os.path.join(base_project_dir, 'data')
    
    print(f"Répertoire racine du scan de données : {base_data_dir}")
    print("Début du scan du répertoire Fatigueset...")
    
    all_sessions_compiled = []
    
    search_pattern = os.path.join(base_data_dir, '*', '*').replace('\\', '/')
    all_paths = glob.glob(search_pattern)
    
    session_paths = []
    for p in all_paths:
        parts = p.split(os.sep)
        if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
            session_paths.append(p)
            
    print(f"Au total, {len(session_paths)} répertoires de sessions numériques valides ont été trouvés.")
    
    if not session_paths:
        print(f"Erreur : Aucun sous-dossier valide n'a été trouvé.")
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
            
        df_survey = pd.read_csv(fatigue_survey_file)
        avg_fatigue = df_survey.iloc[:, 1].mean() if 'fatigue_score' not in df_survey.columns else df_survey['fatigue_score'].mean()
        
        if avg_fatigue <= 4.0:
            current_label = 0  
        elif avg_fatigue >= 7.0:
            current_label = 1  
        else:
            continue
            
        df_hr = pd.read_csv(hr_file)
        df_temp = pd.read_csv(temp_file)
        df_rr = pd.read_csv(rr_file)
        
        # 强制将时间戳列转为数值
        df_hr.iloc[:, 0] = pd.to_numeric(df_hr.iloc[:, 0], errors='coerce')
        df_temp.iloc[:, 0] = pd.to_numeric(df_temp.iloc[:, 0], errors='coerce')
        df_rr.iloc[:, 0] = pd.to_numeric(df_rr.iloc[:, 0], errors='coerce')
        
        df_hr.dropna(subset=[df_hr.columns[0]], inplace=True)
        df_temp.dropna(subset=[df_temp.columns[0]], inplace=True)
        df_rr.dropna(subset=[df_rr.columns[0]], inplace=True)

        # 智能识别单位
        hr_unit = 'ms' if df_hr.iloc[0, 0] > 1e11 else 's'
        temp_unit = 'ms' if df_temp.iloc[0, 0] > 1e11 else 's'
        rr_unit = 'ms' if df_rr.iloc[0, 0] > 1e11 else 's'
        
        df_hr['time'] = pd.to_datetime(df_hr.iloc[:, 0], unit=hr_unit, errors='coerce')
        df_temp['time'] = pd.to_datetime(df_temp.iloc[:, 0], unit=temp_unit, errors='coerce')
        df_rr['time'] = pd.to_datetime(df_rr.iloc[:, 0], unit=rr_unit, errors='coerce')
        
        df_hr.dropna(subset=['time'], inplace=True)
        df_temp.dropna(subset=['time'], inplace=True)
        df_rr.dropna(subset=['time'], inplace=True)
        
        if df_hr.empty or df_temp.empty or df_rr.empty:
            continue

        print(f"正在处理: 参与者 {participant} -> 场次 {session}")

        # 【核心修正】显式调用具体列名，剔除不兼容的 .iloc
        # wrist_hr.csv 的列名为 timestamp, hr
        resampled_hr = df_hr.resample('1s', on='time')['hr'].mean().reset_index()
        
        # wrist_skin_temperature.csv 的列名为 timestamp, temp
        resampled_temp = df_temp.resample('1s', on='time')['temp'].mean().reset_index()
        
        # chest_rr_interval.csv 的列名为 timestamp, duration
        resampled_hrv = df_rr.resample('1s', on='time')['duration'].std().reset_index()
        resampled_hrv.rename(columns={'duration': 'hrv'}, inplace=True)
        resampled_hrv['hrv'] = resampled_hrv['hrv'].fillna(0)
        
        # 横向拼接特征
        session_merged = pd.merge(resampled_hr, resampled_temp, on='time', how='inner')
        session_merged = pd.merge(session_merged, resampled_hrv, on='time', how='inner')
        
        session_merged['Label'] = current_label
        session_merged = session_merged[['hr', 'temp', 'hrv', 'Label']]
        session_merged.columns = ['HeartRate', 'Temperature', 'HRV', 'Label']
        
        all_sessions_compiled.append(session_merged)

    if all_sessions_compiled:
        final_fatigue_df = pd.concat(all_sessions_compiled, ignore_index=True)
        final_fatigue_df.dropna(inplace=True)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_file_path = os.path.join(output_dir, 'fatigueset_cleaned.csv')
        final_fatigue_df.to_csv(output_file_path, index=False)
        print("\n==========================================")
        print("Nettoyage et alignement des caractéristiques de Fatigueset terminés !")
        print(f"Chemin du fichier généré : {output_file_path}")
        print(f"Nombre total d'échantillons extraits : {len(final_fatigue_df)} lignes (secondes)")
        print("==========================================")
    else:
        print("\nErreur : Impossible d'extraire des lignes de données physiologiques temporelles valides.")

if __name__ == "__main__":
    clean_and_merge_fatigueset()