import machine
import dht
import time
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
        
        print("====== ESP32-S3 传感器数据 ======")
        print("温度: {} C".format(temp))
        print("湿度: {} %".format(humidity))
        print("================================\n")
        
        return temp, humidity
    except Exception as e:
        print("[错误] 无法从 DHT11 读取数据: {}".format(e))
        return None, None

# 测试循环
while True:
    read_esp32_sensors()
    time.sleep(2)