import sys
import threading
import ctypes
import numpy as np
import cv2

# 尝试导入 SDK
try:
    from MvImport.MvCameraControl_class import *
except ImportError:
    pass # 可以在主程序中处理这个错误

class HikCamera:
    def __init__(self):
        self.handle = None
        self.is_opened = False
        self.data_buf = None
        self.n_payload_size = 0
        
    def open_by_ip(self, ip):
        """通过 IP 连接相机"""
        if 'MvCameraControl_class' not in sys.modules:
            raise Exception("未找到 MvImport 库，请先运行 setup_hik_sdk.py 或手动复制库文件")

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE
        
        # 枚举设备
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            raise Exception(f"枚举设备失败: {ret}")
            
        if deviceList.nDeviceNum == 0:
            raise Exception("未发现 GigE 相机")
            
        # 寻找匹配 IP 的设备
        target_device = None
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                # 获取当前 IP
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                str_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
                
                if str_ip == ip:
                    target_device = mvcc_dev_info
                    break
        
        if target_device is None:
            raise Exception(f"未找到 IP 为 {ip} 的相机")
            
        # 创建句柄
        self.handle = MvCamera()
        ret = self.handle.MV_CC_CreateHandle(target_device)
        if ret != 0:
            raise Exception(f"创建句柄失败: {ret}")
            
        # 打开设备
        ret = self.handle.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception(f"打开设备失败: {ret}")
            
        # 设置探测网络包大小(GigE必须)
        nPacketSize = self.handle.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = self.handle.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            
        # 获取 Payload Size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.handle.MV_CC_GetIntValue("PayloadSize", stParam)
        self.n_payload_size = stParam.nCurValue
        
        # 分配缓存
        self.data_buf = (c_ubyte * self.n_payload_size)()
        
        # 开始取流
        ret = self.handle.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception(f"开始取流失败: {ret}")
            
        self.is_opened = True
        return True

    def read(self):
        """读取一帧图像，返回 (ret, frame) 格式兼容 OpenCV"""
        if not self.is_opened:
            return False, None
            
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(MV_FRAME_OUT_INFO_EX))
        
        # 超时时间 1000ms
        ret = self.handle.MV_CC_GetOneFrameTimeout(byref(self.data_buf), self.n_payload_size, stFrameInfo, 1000)
        
        if ret == 0:
            # 转换图像格式
            h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
            pixel_type = stFrameInfo.enPixelType
            
            # 将 ctypes 数组转为 numpy
            data = np.frombuffer(self.data_buf, count=int(self.n_payload_size), dtype=np.uint8)
            
            # 格式转换
            if PixelType_Gvsp_Mono8 == pixel_type:
                img = data.reshape((h, w))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return True, img
                
            elif PixelType_Gvsp_BayerGR8 == pixel_type:
                img = data.reshape((h, w))
                img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
                return True, img
                
            elif PixelType_Gvsp_BayerRG8 == pixel_type:
                img = data.reshape((h, w))
                img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
                return True, img
            
            elif PixelType_Gvsp_BayerGB8 == pixel_type:
                img = data.reshape((h, w))
                img = cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
                return True, img
                
            elif PixelType_Gvsp_BayerBG8 == pixel_type:
                img = data.reshape((h, w))
                img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
                return True, img
                
            else:
                # 尝试软解码
                return False, None # 暂不支持其他格式，需要增加转换逻辑
        else:
            return False, None

    def release(self):
        if self.handle:
            self.handle.MV_CC_StopGrabbing()
            self.handle.MV_CC_CloseDevice()
            self.handle.MV_CC_DestroyHandle()
        self.is_opened = False
        
    def isOpened(self):
        return self.is_opened
