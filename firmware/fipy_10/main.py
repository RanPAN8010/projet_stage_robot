import time
from pycoproc_1 import Pycoproc
from L76GNSS import L76GNSS
import usocket as socket
import config

# 提取树莓派的 IP 
target_ip = config.RASPBERRY_PI_IP
target_port = 5000

# 手动实现一个极简的 HTTP POST 函数
def http_post_json(ip, port, path, json_data):
    # 1. 序列化 JSON 字符串
    import json
    body = json.dumps(json_data)
    
    # 2. 拼接标准的 HTTP 请求报文
    req = (
        "POST {} HTTP/1.1\r\n"
        "Host: {}:{}\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: {}\r\n"
        "Connection: close\r\n\r\n"
        "{}"
    ).format(path, ip, port, len(body), body)
    
    # 3. 创建短连接 Socket 并发送
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 解析地址
        addr = socket.getaddrinfo(ip, port)[0][-1]
        s.connect(addr)
        s.send(req.encode('utf-8'))
        
        # 接收服务器的响应（即用即断）
        resp = s.recv(1024)
        if b"200 OK" in resp:
            print("[Succès HTTP] Données envoyées avec succès !")
        else:
            print("[HTTP] Réponse du serveur :", resp.split(b"\r\n")[0])
    except Exception as e:
        print("[Erreur HTTP] :", e)
    finally:
        s.close() # 必须关闭连接，释放网络栈资源

# 初始化 Pytrack GPS 硬件
py = Pycoproc(Pycoproc.PYTRACK)
gnss = L76GNSS(py, timeout=30)
print("Démarrage de la collecte GPS via HTTP...")

while True:
    coord = gnss.coordinates()
    
    if coord[0] is not None and coord[1] is not None:
        payload = {"latitude": coord[0], "longitude": coord[1]}
    else:
        payload = {"latitude": None, "longitude": None}
        
    print("Tentative d'envoi HTTP: {}".format(payload))
    
    # 调用手动实现的 HTTP 函数发包
    http_post_json(target_ip, target_port, "/gps", payload)
            
    time.sleep(5)