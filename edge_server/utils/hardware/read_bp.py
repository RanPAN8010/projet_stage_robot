import asyncio
from bleak import BleakClient

DEVICE_ADDRESS = "50:8C:B1:6A:F1:30"
BP_CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# 备选组合 A：ASCII 字符 'S' (Start) 
CMD_A = b'S'

# 备选组合 B：标准类 Libelium/微芯透传握手帧
CMD_B = bytes([0xAA, 0x01, 0x01, 0xAC])

# 备选组合 C：部分经典 BLE 血压计的物理启动序列
CMD_C = bytes([0x02, 0x40, 0xdd, 0x04])

def notification_handler(sender, data):
    data_list = list(data)
    print(f"\n[Donnees brutes recues] -> {data_list}")
    
    if len(data_list) >= 4:
        print("Analyse du paquet...")
        for idx, val in enumerate(data_list):
            print(f"Index {idx} : {val}")
        print("-" * 40)

async def main():
    print(f"Connexion au tensiometre [{DEVICE_ADDRESS}]...")
    try:
        async with BleakClient(DEVICE_ADDRESS) as client:
            if client.is_connected:
                print("Connexion reussie ! Indicateur bleu fixe.")
                
                await client.start_notify(BP_CHARACTERISTIC_UUID, notification_handler)
                await asyncio.sleep(1.0)
                
                # 提示用户在终端手动输入选择
                print("\nChoisissez la commande a tester :")
                print("1 : Envoyer b'S'")
                print("2 : Envoyer [0xAA, 0x01, 0x01, 0xAC]")
                print("3 : Envoyer [0x02, 0x40, 0xdd, 0x04]")
                
                loop = asyncio.get_event_loop()
                choice = await loop.run_in_executor(None, input, "Entrez votre choix (1, 2 ou 3) : ")
                
                if choice == '1':
                    cmd = CMD_A
                elif choice == '2':
                    cmd = CMD_B
                else:
                    cmd = CMD_C
                    
                print(f"Envoi de la commande selectionnee...")
                await client.write_gatt_char(BP_CHARACTERISTIC_UUID, cmd, response=False)
                
                print("Attente du demarrage du moteur (90s)...")
                await asyncio.sleep(90.0)
                
                try:
                    await client.stop_notify(BP_CHARACTERISTIC_UUID)
                except:
                    pass
    except Exception as e:
        print(f"\nErreur : {e}")

if __name__ == "__main__":
    asyncio.run(main())
