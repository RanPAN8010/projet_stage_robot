# boot.py
import network
import time
import config  # 导入刚才写的账号密码配置


def connect_wifi():
    # 实例化一个无线网卡对象，设置为 STA (Station) 终端模式
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)  # 激活网卡

    # 检查是否已经连上网络
    if not wlan.isconnected():
        print(f"正在尝试连接至 Wi-Fi: {config.WIFI_SSID} ...")
        wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)

        # 设置超时机制（防止死循环卡死单片机）
        timeout = 15
        while not wlan.isconnected() and timeout > 0:
            print(".", end="")
            time.sleep(1)
            timeout -= 1
        print()  # 换行

    # 判断最终连接结果
    if wlan.isconnected():
        # wlan.ifconfig() 返回一个元组: (IP地址, 子网掩码, 网关, DNS服务器)
        ip_info = wlan.ifconfig()
        print("\n====== [BOOT] 连网成功 ======")
        print(f"ESP32-S3 局域网 IP : {ip_info[0]}")
        print(f"子网掩码            : {ip_info[1]}")
        print(f"默认网关            : {ip_info[2]}")
        print("=============================\n")
    else:
        print("\n[错误] Wi-Fi 连接超时，请检查密码或实验室网络状态！")
        # 实际工程中，这里可以选择使用 machine.reset() 重启硬件再次尝试


# 运行连网函数
connect_wifi()