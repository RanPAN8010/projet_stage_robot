import os
import csv
import json
from datetime import datetime
import paho.mqtt.client as mqtt

# ==================== Configuration ====================
MQTT_BROKER = "localhost"  # S'exécute localement sur le Raspberry Pi
MQTT_TOPIC = "esp32/data"
CSV_FILE_PATH = "sensor_data_for_ai.csv"

# ==================== Initialisation du fichier CSV ====================
# Si le fichier n'existe pas, on le crée et on ajoute l'en-tête
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["horodatage", "temperature", "humidite"])
    print(f"[Initialisation] Nouveau fichier créé : {CSV_FILE_PATH}")

# ==================== Fonctions de rappel MQTT ====================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[Connexion Réussie] Connecté au Broker MQTT local")
        # Abonnement au sujet
        client.subscribe(MQTT_TOPIC)
        print(f"[Abonnement Réussi] Écoute du sujet : {MQTT_TOPIC} ...")
    else:
        print(f"[Échec de Connexion] Code d'erreur : {rc}")

def on_message(client, userdata, msg):
    try:
        # 1. Décodage du message reçu
        payload_str = msg.payload.decode('utf-8')
        print(f"[Données brutes reçues] {payload_str}")
        
        # 2. Analyse des données JSON
        data = json.loads(payload_str)
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        
        # 3. Vérification de la validité des données
        if temperature is None or humidity is None:
            print("[Attention] Données incomplètes reçues, ignorées.")
            return
            
        # 4. Obtention de l'horodatage actuel (Format ISO 8601)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 5. Écriture dans le fichier CSV
        with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, temperature, humidity])
            
        print(f"[Succès] {current_time} -> Temp : {temperature}°C, Hum : {humidity}%")
        
    except json.JSONDecodeError:
        print(f"[Erreur] Impossible d'analyser le JSON, format incorrect : {msg.payload}")
    except Exception as e:
        print(f"[Exception] Une erreur est survenue lors du traitement : {e}")

# ==================== Logique Principale ====================
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Démarrage du programme de collecte de données...")
client.connect(MQTT_BROKER, 1883, 60)

# Maintien de la boucle pour attendre et traiter les messages
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n[Arrêt] Le programme de collecte a été arrêté manuellement.")