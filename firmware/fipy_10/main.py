import machine
import time
from bmp180 import BMP180

# 1. 初始化 Pytrack 专属的 I2C 接口
i2c = machine.I2C(0, pins=('P22', 'P21'))

try:
    # 2. 传入 I2C 实例
    bmp = BMP180(i2c)
    # 设置精度（0：低功耗，3：高分辨率）
    bmp.oversample_sett = 2
    print("[系统] 成功加载 micropython-IMU 开源库并完成初始化")
except Exception as e:
    print("[错误] 硬件初始化失败: {}".format(e))
    bmp = None

# 3. 数据轮询采集
while True:
    if bmp:
        try:
            # 必须调用库提供的阻塞读取函数刷新内部 Generator 状态
            bmp.blocking_read()
            
            # 读取温度属性
            current_temp = bmp.temperature
            
            print("====== FiPy 传感器数据 (BMP180) ======")
            print("当前环境温度: {} C".format(current_temp))
            print("======================================\n")
        except Exception as e:
            print("[错误] 无法获取传感器温度: {}".format(e))
            
    time.sleep(2)