from network import WLAN
import network
import time
import machine
import config

def connect_wifi():
    wlan = WLAN(network.STA_IF)
    wlan.active(True)
    time.sleep_ms(500)
    
    if not wlan.isconnected():
        print("ESP32-S3 tente de se connecter au Wi-Fi : {} ...".format(config.WIFI_SSID))
        wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)
        
        # Configuration du mécanisme de temps d'attente (15 secondes)
        timeout = 15
        while not wlan.isconnected() and timeout > 0:
            print(".", end="")
            time.sleep(1)
            timeout -= 1
        print()

    if wlan.isconnected():
        ip_info = wlan.ifconfig()
        print("\n====== [BOOT] ESP32-S3 Connecté avec succès ======")
        print("Adresse IP locale de l'ESP32-S3 : {}".format(ip_info[0]))
        print("==================================================\n")
    else:
        print("\n[Erreur] Délai d'attente dépassé pour la connexion Wi-Fi.")
        print("Code d'état actuel : {}".format(wlan.status()))

connect_wifi()