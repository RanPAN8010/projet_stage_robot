import time
import gc

# 1. 先尝试继承 boot.py 里已经占住网络通道的客户端
try:
    import boot
    client = boot.mqtt_client
except Exception as e:
    client = None

if client is None:
    print("[Attention] Pas de client MQTT. Mode autonome.")
else:
    print("[Info] Client MQTT initialisé avec succès.")

# 2. 【核心修复】在导入大型硬件库之前，再次清理内存，并设置延迟
print("Libération de la mémoire avant chargement des pilotes...")
gc.collect()
time.sleep(1)

# 3. 此时网络已经连上，我们在最后一步才把沉重的 GPS 驱动请进内存
print("Chargement des pilotes Pytrack et GPS...")
from pycoproc_1 import Pycoproc
from L76GNSS import L76GNSS

# 4. 初始化硬件
py = Pycoproc(Pycoproc.PYTRACK)
gnss = L76GNSS(py, timeout=30)
print("Matériel prêt. Démarrage de la collecte GPS...")

while True:
    coord = gnss.coordinates()
    
    if coord[0] is not None and coord[1] is not None:
        payload = "{{\"latitude\": {}, \"longitude\": {}}}".format(coord[0], coord[1])
    else:
        payload = "{\"latitude\": null, \"longitude\": null}"
        
    print("Envoi: {}".format(payload))
    
    if client is not None and getattr(client, 'sock', None) is not None:
        try:
            client.publish("fipy/gps", payload.encode('utf-8'))
        except Exception as e:
            print("[Erreur] Perte de connexion : {}".format(e))
            client.sock = None
            
    # 定期在循环底部回收垃圾，防止内存泄漏导致网卡再次挂掉
    gc.collect()
    time.sleep(5)