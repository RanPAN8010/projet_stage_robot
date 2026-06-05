from network import WLAN
import network
import time
import machine
import config  # 确保你的 config.py 里面有 WIFI_SSID 和 WIFI_PASSWORD

def connect_wifi():
    # 1. 实例化 WLAN 对象 (标准 MicroPython 使用 network.STA_IF)
    wlan = WLAN(network.STA_IF)
    
    # 2. 显式激活 Wi-Fi 射频芯片（这一步至关重要，能解决 Internal State Error）
    wlan.active(True)
    
    # 给底层驱动 500ms 的稳定时间
    time.sleep_ms(500)
    
    if not wlan.isconnected():
        print("ESP32-S3 正在尝试连接至 Wi-Fi: {} ...".format(config.WIFI_SSID))
        # 3. 标准 MicroPython 连接只需传入 ssid 和 password 参数
        wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)
        
        # 设置 15 秒超时机制
        timeout = 15
        while not wlan.isconnected() and timeout > 0:
            print(".", end="")
            time.sleep(1)
            timeout -= 1
        print()

    # 4. 验证最终连接状态
    if wlan.isconnected():
        ip_info = wlan.ifconfig()
        print("\n====== [BOOT] ESP32-S3 连网成功 ======")
        print("ESP32-S3 局域网 IP : {}".format(ip_info[0]))
        print("======================================\n")
    else:
        print("\n[错误] ESP32-S3 Wi-Fi 连接超时。")
        print("当前状态码: {}".format(wlan.status()))

# 执行连网函数
connect_wifi()