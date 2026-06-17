import socket
import time

# 填入刚刚抓到的经典蓝牙地址
DEVICE_ADDRESS = "00:07:80:8C:06:BA"
# 经典蓝牙通用串口服务端口
PORT = 1 

def main():
    print(f"Connexion via Bluetooth Classique a [{DEVICE_ADDRESS}]...")
    
    # 创建 RFCOMM 经典蓝牙套接字
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    
    try:
        sock.connect((DEVICE_ADDRESS, PORT))
        print("Connexion reussie ! Le flux de donnees est ouvert.")
        print("Lecture des premiers octets bruts (Raw Bytes)...")
        print("-" * 50)
        
        # 连续读取 100 次数据包，观察有没有原始字节吐出来
        for _ in range(100):
            data = sock.recv(1024)
            if data:
                # 打印出接收到的原始十进制字节数组
                print(f"Octets recus : {list(data)}")
            time.sleep(0.01)
            
    except Exception as e:
        print(f"\nErreur de connexion : {e}")
    finally:
        sock.close()
        print("-" * 50)
        print("Connexion fermee.")

if __name__ == "__main__":
    main()
