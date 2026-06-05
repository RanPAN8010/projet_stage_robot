from network import WLAN
import time
import machine
import config  # 导入配置

def connect_wifi():
    # 实例化 WLAN 对象，设置为终端模式
    wlan = WLAN(mode=WLAN.STA)
    
    if not wlan.isconnected():
        # 修正：将未定义的 ssid 改为 config.WIFI_SSID
        print("FiPy 正在尝试连接至 Wi-Fi: {} ...".format(config.WIFI_SSID))
        wlan.connect(ssid=config.WIFI_SSID, auth=(WLAN.WPA2, config.WIFI_PASSWORD))
        
        # 设置超时机制
        timeout = 15
        while not wlan.isconnected() and timeout > 0:
            print(".", end="")
            time.sleep(1)
            timeout -= 1
        print()

    # 验证最终连接状态
    if wlan.isconnected():
        ip_info = wlan.ifconfig()
        print("\n====== [BOOT] FiPy 连网成功 ======")
        # 修正：将 f-string 替换为 .format() 写法
        print("FiPy 局域网 IP : {}".format(ip_info[0]))
        print("==================================\n")
    else:
        print("\n[错误] Wi-Fi 连接超时，请检查密码或路由器状态。")

# 执行连网函数
connect_wifi()