import asyncio
from bleak import BleakClient

# MAC adresse de votre tensiometre
DEVICE_ADDRESS = "50:8C:B1:6A:E6:77" 

async def main():
    print(f"Connexion au tensiometre [{DEVICE_ADDRESS}]...")
    
    async with BleakClient(DEVICE_ADDRESS) as client:
        if client.is_connected:
            print("Connexion reussie ! Analyse des services et caracteristiques...")
            print("-" * 60)
            
            for service in client.services:
                print(f"\n[Service] UUID: {service.uuid} ({service.description})")
                
                for char in service.characteristics:
                    print(f"  └── [Characteristic] UUID: {char.uuid}")
                    print(f"      Properties: {char.properties}")
            
            print("-" * 60)
            print("Analyse terminee.")

if __name__ == "__main__":
    asyncio.run(main())
