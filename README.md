# Système de surveillance d'urgence basé sur une détection bi-puce et l'IA embarquée

## Présentation du projet

Ce projet est un système IoT complet développé entièrement en Python. Il utilise une architecture de détection à deux nœuds, composée d'un **ESP32-S3** et d'un **Pycom FiPy 1.0**, permettant de collecter des données environnementales sur plusieurs canaux de manière redondante. Les données physiques mesurées sont transmises en temps réel via le réseau local à un serveur de calcul en périphérie (**Raspberry Pi**). Ce dernier exécute des **algorithmes d'IA** pour détecter et analyser les situations d'urgence en quelques secondes.

---

## Technologies et environnement de développement

### 1. Couche de détection matérielle (Dual-Node Sensing)

* **Nœud de détection A :** Carte de développement ESP32-S3 (sous MicroPython).
* **Nœud de détection B :** Module Pycom FiPy 1.0 branché sur une carte d'extension Pytrack (sous MicroPython).
* **Environnement de développement (IDE) :** PyCharm (avec l'extension MicroPython).

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
├── README.md               # Fichier principal de description du projet
├── firmware/               # Code source des nœuds de détection (MicroPython)
│   ├── esp32_s3_node/      # Programme pour le nœud ESP32-S3
│   │   ├── boot.py
│   │   └── main.py
│   └── fipy_node/          # Programme pour le nœud FiPy 1.0
│       ├── boot.py
│       └── main.py
└── edge-ai/                # Code source du serveur Raspberry Pi (Python 3)
    ├── models/             # Modèles d'IA entraînés pour l'inférence
    ├── main.py             # Script principal (réception bi-puce et détection IA)
    └── requirements.txt    # Liste des dépendances Python pour le Raspberry Pi

```

---

## 👤 Organisation et répartition des tâches

* **Pan RAN** : Développement complet et autonome du projet de bout en bout. Réalisation des scripts MicroPython sur PyCharm pour la détection bi-puce (ESP32-S3 et FiPy 1.0), mise en place du protocole de communication sur le réseau local, configuration du serveur Raspberry Pi et déploiement du modèle d'IA pour la détection des urgences.

---

## 📝 Licence

Ce projet est sous licence [MIT]. Développé uniquement dans un cadre académique et expérimental.</Nom_du_Projet>
