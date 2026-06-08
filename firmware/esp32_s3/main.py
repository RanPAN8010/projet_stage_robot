import machine
import dht
import time
import network
from umqtt.simple import MQTTClient
import config

# 初始化 DHT11 传感器，连接到 GPIO 5
sensor = dht.DHT11(machine.Pin(5))

def read_esp32_sensors():
    try:
        # 触发传感器测量
        sensor.measure()
        
        # 读取数据
        temp = sensor.temperature()
        humidity = sensor.humidity()
        
        print("====== Données du capteur ESP32-S3 ======")
        print("Température: {} C".format(temp))
        print("Humidité: {} %".format(humidity))
        print("=========================================\n")
        
        return temp, humidity
    except Exception as e:
        print("[Erreur] Impossible de lire les données du DHT11: {}".format(e))
        return None, None

# ==================== 2. MQTT 主程序逻辑 ====================
print("Connexion au Broker MQTT du Raspberry Pi: {}...".format(config.RASPBERRY_PI_IP))
try:
    # 实例化 MQTT 客户端
    # 参数分别为：客户端唯一ID，服务器IP地址
    client = MQTTClient("ESP32S3_Client", config.RASPBERRY_PI_IP)
    client.connect()
    print("[Succès] Connecté avec succès au serveur MQTT du Raspberry Pi !\n")
    
    # 循环读取并发送数据
    while True:
        temp, humidity = read_esp32_sensors()
        
        if temp is not None and humidity is not None:
            # 拼接成类似 JSON 的字符串格式
            payload = "{{\"temperature\": {}, \"humidity\": {}}}".format(temp, humidity)
            
            print("Envoi des données sur le sujet {} ...".format(config.MQTT_TOPIC))
            # 传输数据（注意：MicroPython 中发送数据需要转换为 bytes 类型）
            client.publish(config.MQTT_TOPIC, payload.encode('utf-8'))
            print("[Terminé] Données envoyées.\n")
            
        time.sleep(5) # 每隔 5 秒发送一次
        
except Exception as e:
    print("[Erreur Critique] Le processus MQTT s'est effondré ou la connexion a été coupée: {}".format(e))