# Système de surveillance d'urgence basé sur une détection bi-puce et l'IA embarquée

## Présentation du projet

Ce projet est un système IoT complet développé entièrement en Python. Il utilise une architecture de détection à deux nœuds, composée d'un **ESP32-S3** et d'un **Pycom FiPy 1.0**, permettant de collecter des données environnementales sur plusieurs canaux de manière redondante. Les données physiques mesurées sont transmises en temps réel via le réseau local à un serveur de calcul en périphérie (**Raspberry Pi**). Ce dernier exécute des **algorithmes d'IA** pour détecter et analyser les situations d'urgence en quelques secondes.

---

## Technologies et environnement de développement

### 1. Couche de détection matérielle (Dual-Node Sensing)

* **Nœud de détection A :** Carte de développement ESP32-S3 (sous MicroPython).
* **Nœud de détection B :** Module Pycom FiPy 1.0 branché sur une carte d'extension Pytrack (sous MicroPython).
* **Environnement de développement (IDE) :** Thonny.

### 2. Couche Serveur et IA embarquée (Raspberry Pi Server)

* **Matériel central :** Raspberry Pi.
* **Système d'exploitation :** Raspberry Pi OS (Debian).
* **Langage de programmation :** Python 3.
* **Bibliothèques et protocoles :** TensorFlow Lite / OpenCV (pour l'inférence de l'IA), Sockets Python / MQTT.

---

## 📐 Architecture du système

```text
 [ Nœud ESP32-S3 ] (MicroPython) ──────┐
                                       ├── (Wi-Fi / Réseau Local) ──▶ [ Serveur Raspberry Pi (Python 3) ] ──▶ [ Inférence IA ] ──▶ Alerte d'urgence
 [ Nœud Pycom FiPy 1.0 ] (MicroPython) ┘

```

## 📂 Structure du projet

```text
├── .gitignore               # Configuration des exclusions Git
├── README.md                # Fichier principal de description du projet
├── edge_server/             # Code source du serveur de calcul en périphérie (Python 3)
│   ├── ai_engine/           # Dossier de stockage des modèles d'IA entraînés (ignoré sur Git)
│   │   ├── env/             # Structure pour les modèles liés aux données environnementales (.gitkeep)
│   │   └── med/             # Structure pour les modèles liés aux données médicales (.gitkeep)
│   ├── ai_trainning/        # Scripts d'entraînement des modèles d'IA
│   │   ├── env/             # Entraînement des modèles environnementaux (ex: XGBoost)
│   │   └── med/             # Entraînement des modèles médicaux (ex: Random Forest, KNN)
│   ├── network/             # Scripts de communication réseau (Serveur de réception)
│   ├── utils/               # Outils de traitement de données et pilotes matériels
│   │   ├── data_prep/       # Nettoyage, corrélation et fusion des données (HRV, Fatigueset, etc.)
│   │   └── hardware/        # Pilotes et tests des capteurs (SpO2, ECG, Tension artérielle)
│   ├── data/                # Dossier de stockage des données brutes (.gitkeep)
│   └── requirements.txt     # Liste des dépendances Python pour le serveur
└── firmware/                # Code source des nœuds de détection embarqués (MicroPython)
    ├── esp32_s3/            # Programme et configuration pour le nœud ESP32-S3
    └── fipy_10/             # Programme, configuration et bibliothèques capteurs (L76GNSS, LIS2HH12, etc.) pour le nœud FiPy 1.0
```

---

## 👤 Organisation et répartition des tâches

* **Pan RAN** : Développement complet et autonome du projet de bout en bout. Réalisation des scripts MicroPython sur PyCharm pour la détection bi-puce (ESP32-S3 et FiPy 1.0), mise en place du protocole de communication sur le réseau local, configuration du serveur Raspberry Pi et déploiement du modèle d'IA pour la détection des urgences.

---

## 📝 Licence

Ce projet est sous licence [MIT]. Développé uniquement dans un cadre académique et expérimental.</Nom_du_Projet>
