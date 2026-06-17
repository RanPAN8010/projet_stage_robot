from network import WLAN
import time
import config

def connect_wifi():
    wlan = WLAN(mode=WLAN.STA)
    time.sleep_ms(500)
    
    if not wlan.isconnected():
        print("FiPy tente de se connecter au Wi-Fi : {} ...".format(config.WIFI_SSID))
        wlan.connect(config.WIFI_SSID, auth=(None, config.WIFI_PASSWORD))
        
        timeout = 15
        while not wlan.isconnected() and timeout > 0:
            print(".", end="")
            time.sleep(1)
            timeout -= 1
        print()

    if wlan.isconnected():
        print("\n====== [BOOT] FiPy Connecté avec succès ======")
        print("IP : {}".format(wlan.ifconfig()[0]))
        print("==================================================\n")
    else:
        print("\n[Erreur] Wi-Fi Timeout.")

connect_wifi()