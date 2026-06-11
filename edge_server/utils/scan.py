import asyncio
from bleak import BleakScanner

async def main():
    print("正在扫描周围的蓝牙设备，请按下血氧仪的白色按钮...")
    devices = await BleakScanner.discover()
    for d in devices:
        # 寻找名字包含 Libelium 或 MySignals 的设备
        if d.name and ("Libelium" in d.name or "MySignals" in d.name):
            print(f"找到血氧仪！设备名称: {d.name}, MAC地址: {d.address}")

asyncio.run(main())
