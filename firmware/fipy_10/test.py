import machine
import time

# Initialisation de l'I2C
i2c = machine.I2C(0, pins=('P22', 'P21'))

print("Analyse du bus I2C en cours...")
try:
    devices = i2c.scan()
    if not devices:
        print("[Info] Le bus fonctionne correctement, mais aucun périphérique I2C n'a été détecté.")
    else:
        print("Périphérique(s) trouvé(s), adresse(s) hexadécimale(s) :")
        for addr in devices:
            print(hex(addr))
except Exception as e:
    print("[Erreur] Anomalie sur le bus I2C, veuillez vérifier l'inversion des câbles ou un court-circuit : {}".format(e))