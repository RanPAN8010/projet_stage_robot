import os
import pandas as pd

def debug_false_alarms():
    # 动态定位路径
    current_path = os.path.abspath(__file__)
    base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    
    # 读取上一步推理生成的完整输出结果
    output_csv_path = os.path.join(base_project_dir, 'data', 'sensor_inference_output.csv')
    
    if not os.path.exists(output_csv_path):
        print(f"Erreur : Le fichier de résultats {output_csv_path} n'existe pas. Veuillez d'abord exécuter predict_service.py")
        return

    df = pd.read_csv(output_csv_path)
    
    # 强行筛选出模型判断为“火灾 (Feu/Fumée)”的所有行
    false_alarms = df[df['Status'] == 'Feu/Fumée']
    
    print("="*60)
    print(f"🔍 COUPE-FILE : {len(false_alarms)} LIGNES D'ANOMALIES DÉTECTÉES")
    print("="*60)
    
    if len(false_alarms) == 0:
        print("Aucune fausse alarme trouvée.")
        return
        
    # 遍历并打印每一行引发误报的详细特征参数
    for idx, row in false_alarms.iterrows():
        print(f"📍 Ligne originale dans le CSV : {idx + 2}") # 加2对齐Excel等软件的行号
        print(f"  -> Horodatage (Timestamp) : {row['Timestamp']}")
        print(f"  -> Température brute : {row['Temperature']} °C")
        print(f"  -> Humidité brute : {row['Humidity']} %")
        print(f"  -> Temp_Rate (Taux de Temp) : {row['Temp_Rate']:.4f} °C/s")
        print(f"  -> Humidity_Rate (Taux d'Hum) : {row['Humidity_Rate']:.4f} %s")
        print(f"  -> Heat_Index (Indice Chaleur) : {row['Heat_Index']:.2f} °C")
        print(f"  -> Probabilité de Feu calculée : {row['Prob_Feu(2)']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    debug_false_alarms()