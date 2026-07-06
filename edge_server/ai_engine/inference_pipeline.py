import os
import numpy as np
import joblib
from xgboost import XGBClassifier

class MultiModalSafetyEngine:
    def __init__(self):
        # 获取当前脚本的绝对路径并定位各模型
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        env_model_path = os.path.join(current_dir, 'env', 'car_safety_xgboost_model.json')
        med_model_path = os.path.join(current_dir, 'med', 'xgboost_body_model.joblib')
        med_scaler_path = os.path.join(current_dir, 'med', 'data_scaler_xgboost.joblib')
        
        print("Chargement des modèles multi-modaux...")
        
        # 1. 加载医学模型及配套标准化器 (Joblib 格式)
        self.med_scaler = joblib.load(med_scaler_path)
        self.med_model = joblib.load(med_model_path)
        
        # 2. 加载环境模型 (原生 JSON 格式)
        self.env_model = XGBClassifier()
        self.env_model.load_model(env_model_path)
        
        print("Modèles chargés avec succès sur le Raspberry Pi.")

    def predict_safety_status(self, raw_med_data, raw_env_data, alpha=0.6):
        """
        融合判定核心逻辑
        :param raw_med_data: 列表 [HeartRate, HRV]
        :param raw_env_data: 列表 [Temperature, Humidity, Light, etc.] 匹配环境训练特征
        :param alpha: 生理数据的决策权重系数 (0 到 1 之间)，环境权重则为 (1 - alpha)
        """
        # ----------------------------------------------------
        # 步骤 1：生理模型推理 (医学低维数据)
        # ----------------------------------------------------
        # 特征预处理
        med_features = np.array([raw_med_data])
        med_features_scaled = self.med_scaler.transform(med_features)
        
        # 获取多分类概率：[P(Normal), P(Fatigue), P(Crise)]
        med_probs = self.med_model.predict_proba(med_features_scaled)[0]
        
        # 提取高危状态（疲劳或心脏病）的综合生理风险分值
        # 这里以 心脏病概率 和 疲劳概率 的某种组合作为生理基础分
        med_risk_score = med_probs[2] * 1.5 + med_probs[1] * 0.5  # 赋权放大突发心脏病风险
        
        # ----------------------------------------------------
        # 步骤 2：环境模型推理 (环境高维数据)
        # ----------------------------------------------------
        env_features = np.array([raw_env_data])
        # 假设环境模型输出 [P(Secured), P(Dangerous)]
        env_probs = self.env_model.predict_proba(env_features)[0]
        env_risk_score = env_probs[1]  # 提取环境危险概率作为加分项
        
        # ----------------------------------------------------
        # 步骤 3：多模态加权融合 (按照博士朋友的说法)
        # ----------------------------------------------------
        combined_risk = (alpha * med_risk_score) + ((1 - alpha) * env_risk_score)
        
        # ----------------------------------------------------
        # 步骤 4：决策树形门槛输出级别
        # ----------------------------------------------------
        if combined_risk > 0.75:
            status = "DANGER_CRITIQUE (Alerte d'urgence)"
            level = 2
        elif combined_risk > 0.40:
            status = "ATTENTION (Fatigue ou Risque Environnemental)"
            level = 1
        else:
            status = "SÉCURITÉ (Normal)"
            level = 0
            
        return {
            "status": status,
            "level": level,
            "combined_risk": round(float(combined_risk), 4),
            "med_prob_crise": round(float(med_probs[2]), 4),
            "env_prob_danger": round(float(env_risk_score), 4)
        }

# 模拟测试
if __name__ == "__main__":
    engine = MultiModalSafetyEngine()
    
    # 模拟树莓派从传感器接收到的实时数据
    mock_med = [88.5, 35.0]  # 心率偏高，HRV偏低
    mock_env = [38.5, 75.0, 1.0, 0.0, 120.0]  # 假设的高温、高湿环境特征数据
    
    result = engine.predict_safety_status(mock_med, mock_env)
    print("\n--- Résultat de l'analyse en temps réel ---")
    print(result)