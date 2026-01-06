import sys
import os
import glob
import shutil
import socket
import struct
import threading
import time
import json
import ctypes
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# =============================================================================
# 模块一：海康 SDK 核心加载器 (DLL 强力搜索版)
# =============================================================================
HAS_HIK_SDK = False
SDK_ERROR_MSG = ""

try:
    possible_dll_paths = [
        r"C:\Program Files (x86)\MVS\Runtime\Win64_x64",
        r"C:\Program Files (x86)\MVS\Runtime\Win32_x86",
        r"D:\MVS\Runtime\Win64_x64",
        r"D:\MVS\Runtime\Win32_x86",
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_x86"
    ]
    
    dll_loaded = False
    for dll_path in possible_dll_paths:
        if os.path.exists(dll_path):
            os.environ['PATH'] = dll_path + ";" + os.environ['PATH']
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(dll_path)
            print(f"[系统] 成功加载海康驱动 DLL 路径: {dll_path}")
            dll_loaded = True
            break
    
    if not dll_loaded:
        print("[警告] 未找到海康 MVS Runtime 路径，如果连接失败请检查驱动安装。")

    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    if os.path.exists(os.path.join(current_dir, "MvImport")):
        from MvImport.MvCameraControl_class import *
        HAS_HIK_SDK = True
        print("[系统] 海康 SDK Python 接口导入成功！")
    else:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.exists(os.path.join(desktop_path, "MvImport")):
            sys.path.append(desktop_path)
            from MvImport.MvCameraControl_class import *
            HAS_HIK_SDK = True
            print("[系统] 在桌面找到并加载了 SDK！")
        else:
            raise ImportError("未找到 MvImport 文件夹")

except Exception as e:
    SDK_ERROR_MSG = str(e)
    print(f"[系统错误] SDK 加载严重失败: {e}")
    HAS_HIK_SDK = False


# =============================================================================
# 模块二：海康相机驱动封装 (增强版 - 拍照参数强制生效)
# =============================================================================
class HikCameraWrapper:
    def __init__(self):
        self.handle = None
        self.is_opened = False
        self.data_buf = None
        self.n_payload_size = 0
        # 记录当前设置的曝光和增益（用于拍照时校验）
        self.current_exposure = 1000.0  # 默认1000μs
        self.current_gain = 0.0         # 默认0dB
        
    def open_by_ip(self, ip):
        if not HAS_HIK_SDK: 
            raise Exception(f"SDK 环境未就绪: {SDK_ERROR_MSG}")

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0: 
            raise Exception(f"枚举设备失败，错误码: {hex(ret)}")
        
        if deviceList.nDeviceNum == 0: 
            raise Exception("未发现 GigE 网口相机，请检查网线连接")
            
        target_device = None
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = ctypes.cast(deviceList.pDeviceInfo[i], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                str_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
                if str_ip == ip:
                    target_device = mvcc_dev_info
                    break
        
        if target_device is None: 
            raise Exception(f"未找到 IP 为 {ip} 的相机，请确认 IP 设置正确")
            
        self.handle = MvCamera()
        ret = self.handle.MV_CC_CreateHandle(target_device)
        if ret != 0: 
            raise Exception(f"创建句柄失败: {hex(ret)}")
            
        ret = self.handle.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0: 
            raise Exception(f"打开设备失败: {hex(ret)}")
            
        # 防丢包配置
        ret = self.handle.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        if ret != 0: print(f"[警告] 设置包长失败: {hex(ret)}")
        
        ret = self.handle.MV_CC_SetEnumValue("TriggerMode", 0) # 关闭触发，连续采集
        if ret != 0: print(f"[警告] 设置触发模式失败: {hex(ret)}")
        
        self.handle.MV_CC_SetIntValue("GevSCPD", 2000)
            
        # 分配缓存
        stParam = MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_INTVALUE))
        ret = self.handle.MV_CC_GetIntValue("PayloadSize", stParam)
        self.n_payload_size = stParam.nCurValue
        self.data_buf = (ctypes.c_ubyte * self.n_payload_size)()
        
        # 开始取流
        ret = self.handle.MV_CC_StartGrabbing()
        if ret != 0: 
            raise Exception(f"开始取流失败: {hex(ret)}")
            
        self.is_opened = True
        # 初始化曝光和增益
        self.set_exposure_time(self.current_exposure)
        self.set_gain(self.current_gain)
        print(f"[SDK] 相机 {ip} 打开成功 | 初始曝光: {self.current_exposure}μs | 初始增益: {self.current_gain}dB")

    def read(self):
        if not self.is_opened: return False, None
        
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ctypes.memset(ctypes.byref(stFrameInfo), 0, ctypes.sizeof(MV_FRAME_OUT_INFO_EX))
        
        ret = self.handle.MV_CC_GetOneFrameTimeout(ctypes.byref(self.data_buf), self.n_payload_size, stFrameInfo, 2000)
        
        if ret == 0:
            h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
            data = np.frombuffer(self.data_buf, count=int(self.n_payload_size), dtype=np.uint8)
            pt = stFrameInfo.enPixelType
            
            if PixelType_Gvsp_Mono8 == pt:
                img = data.reshape((h, w))
                return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif PixelType_Gvsp_BayerGR8 == pt:
                img = data.reshape((h, w))
                return True, cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
            elif PixelType_Gvsp_BayerRG8 == pt:
                img = data.reshape((h, w))
                return True, cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
            elif PixelType_Gvsp_RGB8_Packed == pt:
                img = data.reshape((h, w, 3))
                return True, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            try:
                img = data.reshape((h, w))
                return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except:
                return False, None
        else:
            if ret != 0x80000007: 
                print(f"[SDK Warning] GetFrame 异常: {hex(ret)}")
            return False, None

    def set_exposure_time(self, exposure_us):
        """设置曝光时间（拍照时强制生效）"""
        if not self.is_opened or not self.handle:
            return False
        try:
            # 限制曝光范围：100μs ~ 50000μs（工业相机常用范围）
            exposure_us = max(100.0, min(50000.0, exposure_us))
            # 关闭自动曝光
            ret = self.handle.MV_CC_SetEnumValue("ExposureAuto", 0)
            if ret != 0:
                print(f"[警告] 关闭自动曝光失败，错误码: {hex(ret)}")
                return False
            # 设置曝光时间
            ret = self.handle.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
            if ret != 0:
                print(f"[警告] 设置曝光时间失败，错误码: {hex(ret)}")
                return False
            # 更新当前记录的曝光值
            self.current_exposure = exposure_us
            return True
        except Exception as e:
            print(f"[错误] 设置曝光时间异常: {e}")
            return False

    def get_exposure_time(self):
        """获取当前实际曝光时间"""
        if not self.is_opened or not self.handle:
            return 0.0
        try:
            stParam = MVCC_FLOATVALUE()
            ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_FLOATVALUE))
            ret = self.handle.MV_CC_GetFloatValue("ExposureTime", stParam)
            if ret != 0:
                print(f"[警告] 获取曝光时间失败，错误码: {hex(ret)}")
                return self.current_exposure  # 返回记录值
            return stParam.fCurValue
        except Exception as e:
            print(f"[错误] 获取曝光时间异常: {e}")
            return self.current_exposure

    def set_gain(self, gain_db):
        """设置增益（拍照时强制生效）"""
        if not self.is_opened or not self.handle:
            return False
        try:
            # 限制增益范围：0dB ~ 30dB（避免图像过曝）
            gain_db = max(0.0, min(30.0, gain_db))
            # 关闭自动增益
            ret = self.handle.MV_CC_SetEnumValue("GainAuto", 0)
            if ret != 0:
                print(f"[警告] 关闭自动增益失败，错误码: {hex(ret)}")
                return False
            # 设置增益值
            ret = self.handle.MV_CC_SetFloatValue("Gain", float(gain_db))
            if ret != 0:
                print(f"[警告] 设置增益失败，错误码: {hex(ret)}")
                return False
            # 更新当前记录的增益值
            self.current_gain = gain_db
            return True
        except Exception as e:
            print(f"[错误] 设置增益异常: {e}")
            return False

    def get_gain(self):
        """获取当前实际增益值"""
        if not self.is_opened or not self.handle:
            return 0.0
        try:
            stParam = MVCC_FLOATVALUE()
            ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_FLOATVALUE))
            ret = self.handle.MV_CC_GetFloatValue("Gain", stParam)
            if ret != 0:
                print(f"[警告] 获取增益失败，错误码: {hex(ret)}")
                return self.current_gain  # 返回记录值
            return stParam.fCurValue
        except Exception as e:
            print(f"[错误] 获取增益异常: {e}")
            return self.current_gain

    def release(self):
        if self.handle:
            self.handle.MV_CC_StopGrabbing()
            self.handle.MV_CC_CloseDevice()
            self.handle.MV_CC_DestroyHandle()
        self.is_opened = False


# =============================================================================
# 模块三：GigE 扫描器 (UDP 广播)
# =============================================================================
class GigEScanner:
    def scan(self, timeout=1.5):
        devices = []
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)
            
            msg = struct.pack('>BBHHH', 0x42, 0x11, 0x0002, 0x0000, 0x0001)
            sock.sendto(msg, ('255.255.255.255', 3956))
            
            start = time.time()
            while time.time() - start < timeout:
                try:
                    data, addr = sock.recvfrom(1024)
                    if len(data) > 0:
                        if addr[0] not in [d['ip'] for d in devices]:
                            devices.append({'ip': addr[0]})
                except socket.timeout:
                    break
            sock.close()
        except Exception as e:
            print(f"扫描异常: {e}")
        return devices


# =============================================================================
# 模块四：主程序界面与逻辑 (拍照参数强制生效版，已修复 KeyError)
# =============================================================================
DEFAULT_CONFIG = {
    "robot_net": {
        "bind_ip": "0.0.0.0",
        "port": 8000,
        "robot_ip": "192.168.1.100",  # 机械臂IP地址（用于客户端连接）
        "robot_port": 8080             # 机械臂端口（用于客户端连接）
    },
    "camera": {
        "mode": "sdk",
        "target_ip": "",
        "index": 0,
        "width": 1280,
        "height": 960,
       
        "dist": [0,0,0,0,0],
        "exposure_us": 1000.0,  # 拍照默认曝光
        "gain_db": 0.0           # 拍照默认增益
    },
    "paths": {
        "save_dir": "./robot_images",
        "save_format": "jpg"
    },
    "commands": {
        "trigger": "C1",
        "error": "E1",
        "success_prefix": "OK",
        "separator": ","
    },
    "calibration": {
        "type": "circles",  # 标定板类型: "circles" (圆点) 或 "chessboard" (棋盘格)
        "rows": 7,
        "cols": 7,
        "spacing": 12.5  # 标定板间距（单位：mm）- 圆点：圆心间距，棋盘格：方格边长
    },
    "hand_eye": {
        "enabled": False,  # 是否启用手眼标定转换
        "T_camera_to_flange": {  # 相机到法兰的变换矩阵
            "translation": [0.0, 0.0, 0.0],  # 平移 [x, y, z] (mm)
            "rotation": [0.0, 0.0, 0.0]      # 旋转 [rx, ry, rz] (deg, xyz顺序)
        }
    }
}

class UniversalVisionServer:
    def __init__(self, root):
        self.root = root
        self.root.title("通用视觉服务器系统 - 拍照参数增强版")
        self.root.geometry("1400x950")
        
        self.config_file = "vision_config.json"
        self.calib_data_file = "calibration_data_backup.json"
        
        self.cfg = self.load_config()  # 加载配置（已修复兼容问题）
        
        self.tcp_running = False
        self.cam_running = False
        
        self.server_socket = None
        self.client_socket = None
        self.cap = None
        self.current_frame = None
        self.lock = threading.Lock()
        
        self.scanner = GigEScanner()
        self.calib_data_list = []
        
        # 手动输入的机械臂位姿（用于tcp_pose无法获取时）
        self.manual_robot_pose = None  # [x, y, z, rx, ry, rz]
        
        self.load_calib_data()
        
        self.show_crosshair = tk.BooleanVar(value=True)
        self.show_focus_score = tk.BooleanVar(value=True)
        self.undistort_view = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
        if not os.path.exists(self.cfg["paths"]["save_dir"]):
            os.makedirs(self.cfg["paths"]["save_dir"])

    # --- 配置管理（核心修复：自动补全缺失配置字段）---
    def load_config(self):
        """加载配置，自动补全新旧配置缺失的字段，避免 KeyError"""
        # 先初始化默认配置
        final_config = DEFAULT_CONFIG.copy()
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    old_data = json.load(f)
                    # 递归补全配置字段（顶层 + 子字典）
                    def merge_config(default, old):
                        for key, val in default.items():
                            if key not in old:
                                old[key] = val
                            else:
                                # 如果是字典，递归补全子字段
                                if isinstance(val, dict) and isinstance(old[key], dict):
                                    merge_config(val, old[key])
                        return old
                    # 合并旧配置和默认配置，缺失字段自动补充
                    final_config = merge_config(DEFAULT_CONFIG, old_data)
                print("[系统] 配置文件加载成功，已自动补全缺失字段")
            except Exception as e:
                print(f"[警告] 加载旧配置文件失败（格式错误/损坏），使用默认配置: {e}")
        else:
            print("[系统] 未找到配置文件，使用默认配置")
        return final_config

    def save_config(self):
        self.update_cfg_from_ui()
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.cfg, f, indent=4)
            messagebox.showinfo("提示", "配置已保存")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def load_calib_data(self):
        if os.path.exists(self.calib_data_file):
            try:
                with open(self.calib_data_file, 'r') as f:
                    data = json.load(f)
                    for d in data:
                        d['corners'] = np.array(d['corners'], dtype=np.float32)
                    self.calib_data_list = data
                    print(f"[系统] 已恢复 {len(data)} 条标定数据")
            except Exception as e:
                print(f"[警告] 加载标定备份数据失败: {e}")

    def save_calib_data(self, show_message=True):
        """保存标定数据到文件
        
        Args:
            show_message: 是否显示成功消息（默认True，删除操作时设为False避免重复弹窗）
        """
        serializable_list = []
        for d in self.calib_data_list:
            item = d.copy()
            item['corners'] = d['corners'].tolist()
            serializable_list.append(item)
        
        try:
            with open(self.calib_data_file, 'w') as f:
                json.dump(serializable_list, f, indent=4)
            if show_message:
                messagebox.showinfo("备份成功", f"已保存 {len(self.calib_data_list)} 条数据到文件")
            else:
                self.log(f"已自动保存 {len(self.calib_data_list)} 条标定数据")
        except Exception as e:
            messagebox.showerror("备份失败", str(e))

    # --- UI 构建 (曝光/增益调节 + 拍照参数显示，已添加默认值兜底) ---
    def setup_ui(self):
        style = ttk.Style()
        style.configure("Bold.TButton", font=("微软雅黑", 10, "bold"))
        
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        frame_run = tk.Frame(notebook)
        notebook.add(frame_run, text="1. 运行监控 (Monitor)")
        self.setup_run_ui(frame_run)
        
        frame_calib = tk.Frame(notebook)
        notebook.add(frame_calib, text="2. 手眼标定集成 (Calibration)")
        self.setup_calib_ui(frame_calib)
        
        frame_cfg = tk.Frame(notebook)
        notebook.add(frame_cfg, text="3. 系统配置 (Config)")
        self.setup_config_ui(frame_cfg)

    def setup_run_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(paned, width=420, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        paned.add(left_frame, minsize=420)
        
        # 1. 独立控制面板
        c_frame = tk.LabelFrame(left_frame, text="独立控制面板", font=("bold", 10), fg="blue")
        c_frame.pack(fill=tk.X, padx=5, pady=5)
        
        row1 = tk.Frame(c_frame)
        row1.pack(fill=tk.X, pady=2)
        self.btn_cam = tk.Button(row1, text="📷 连接相机", command=self.toggle_cam, bg="#e0e0e0", width=15)
        self.btn_cam.pack(side=tk.LEFT, padx=5)
        self.lbl_cam = tk.Label(row1, text="未连接", fg="red")
        self.lbl_cam.pack(side=tk.LEFT)
        
        row2 = tk.Frame(c_frame)
        row2.pack(fill=tk.X, pady=2)
        self.btn_tcp = tk.Button(row2, text="📡 启动监听", command=self.toggle_tcp, bg="#e0e0e0", width=15)
        self.btn_tcp.pack(side=tk.LEFT, padx=5)
        self.lbl_tcp = tk.Label(row2, text="已停止", fg="red")
        self.lbl_tcp.pack(side=tk.LEFT)
        
        # 2. 曝光与增益调节面板（拍照专用，已添加默认值兜底）
        eg_frame = tk.LabelFrame(left_frame, text="拍照曝光 & 增益调节 (强制生效)", font=("bold", 10), fg="darkgreen")
        eg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 曝光时间调节
        exposure_frame = tk.Frame(eg_frame)
        exposure_frame.pack(fill=tk.X, padx=5, pady=3)
        tk.Label(exposure_frame, text="曝光时间 (μs):", width=15, anchor="w").pack(side=tk.LEFT)
        # 兜底：如果配置中缺失字段，使用默认值1000.0
        exposure_default = self.cfg["camera"].get("exposure_us", 1000.0)
        self.var_exposure = tk.DoubleVar(value=exposure_default)
        self.slider_exposure = tk.Scale(exposure_frame, variable=self.var_exposure, from_=100, to=50000, 
                                        orient=tk.HORIZONTAL, length=200, command=self.on_exposure_change)
        self.slider_exposure.pack(side=tk.LEFT, padx=5)
        self.lbl_exposure = tk.Label(exposure_frame, text=f"{self.var_exposure.get():.0f} μs", width=10)
        self.lbl_exposure.pack(side=tk.LEFT)
        self.entry_exposure = tk.Entry(exposure_frame, width=10)
        self.entry_exposure.insert(0, f"{self.var_exposure.get():.0f}")
        self.entry_exposure.pack(side=tk.LEFT, padx=5)
        tk.Button(exposure_frame, text="确认", command=self.set_exposure_by_entry, width=6).pack(side=tk.LEFT)
        
        # 增益调节
        gain_frame = tk.Frame(eg_frame)
        gain_frame.pack(fill=tk.X, padx=5, pady=3)
        tk.Label(gain_frame, text="增益 (dB):", width=15, anchor="w").pack(side=tk.LEFT)
        # 兜底：如果配置中缺失字段，使用默认值0.0
        gain_default = self.cfg["camera"].get("gain_db", 0.0)
        self.var_gain = tk.DoubleVar(value=gain_default)
        self.slider_gain = tk.Scale(gain_frame, variable=self.var_gain, from_=0, to=30, 
                                    orient=tk.HORIZONTAL, length=200, command=self.on_gain_change)
        self.slider_gain.pack(side=tk.LEFT, padx=5)
        self.lbl_gain = tk.Label(gain_frame, text=f"{self.var_gain.get():.1f} dB", width=10)
        self.lbl_gain.pack(side=tk.LEFT)
        self.entry_gain = tk.Entry(gain_frame, width=10)
        self.entry_gain.insert(0, f"{self.var_gain.get():.1f}")
        self.entry_gain.pack(side=tk.LEFT, padx=5)
        tk.Button(gain_frame, text="确认", command=self.set_gain_by_entry, width=6).pack(side=tk.LEFT)
        
        # 3. 手动拍照按钮（强调参数生效）
        m_frame = tk.LabelFrame(left_frame, text="手动调试 (拍照参数强制应用)", font=("bold", 10))
        m_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(m_frame, text="📸 手动拍照 (强制应用当前参数)", command=self.manual_trigger, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=5, pady=5)
        tk.Label(m_frame, text="* 拍照前自动同步曝光/增益参数", fg="red", justify=tk.LEFT).pack(anchor="w", padx=5)

        # 4. 辅助功能开关
        a_frame = tk.LabelFrame(left_frame, text="视觉辅助", font=("bold", 10))
        a_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Checkbutton(a_frame, text="显示中心十字线", variable=self.show_crosshair).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(a_frame, text="显示清晰度评分 (对焦用)", variable=self.show_focus_score).grid(row=0, column=1, sticky="w")
        tk.Checkbutton(a_frame, text="启用实时畸变矫正", variable=self.undistort_view).grid(row=1, column=0, sticky="w", columnspan=2)

        # 5. 工作模式
        w_frame = tk.LabelFrame(left_frame, text="工作模式", font=("bold", 10))
        w_frame.pack(fill=tk.X, padx=5, pady=5)
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(w_frame, text="测试模式 (识别二维码，回传坐标)", variable=self.work_mode, value="TEST", command=self.on_mode_change).pack(anchor="w", padx=5)
        tk.Radiobutton(w_frame, text="标定模式 (识别标定板，仅存图)", variable=self.work_mode, value="CALIB", command=self.on_mode_change).pack(anchor="w", padx=5)
        
        # 手眼标定转换开关
        self.use_hand_eye = tk.BooleanVar(value=self.cfg["hand_eye"].get("enabled", False))
        tk.Checkbutton(w_frame, text="✓ 启用手眼标定转换 (TEST模式下输出基座坐标系)", 
                      variable=self.use_hand_eye, command=self.on_hand_eye_toggle).pack(anchor="w", padx=5)
        if not self.cfg["hand_eye"].get("enabled", False):
            tk.Label(w_frame, text="  [提示] 请先完成手眼标定", fg="orange", font=("Arial", 8)).pack(anchor="w", padx=25)
        
        # 手动输入机械臂位姿按钮
        tk.Button(w_frame, text="✎ 手动输入当前机械臂位姿", 
                 command=self.manual_input_robot_pose, bg="#ff9800", fg="white", 
                 font=("bold", 9)).pack(anchor="w", padx=5, pady=2)
        self.lbl_manual_pose = tk.Label(w_frame, text="", fg="green", font=("Arial", 8))
        self.lbl_manual_pose.pack(anchor="w", padx=25)

        # 6. 日志
        l_frame = tk.LabelFrame(left_frame, text="系统日志 (含拍照参数)", font=("bold", 10))
        l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(l_frame, font=("Consolas", 9), state=tk.NORMAL)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        # 右侧图像区
        self.canvas = tk.Canvas(paned, bg="#222")
        paned.add(self.canvas, stretch="always")
        self.draw_placeholder()

    def setup_calib_ui(self, parent):
        # 标定板类型选择区域
        type_frame = tk.Frame(parent, bg="#f0f0f0", pady=5)
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(type_frame, text="标定板类型:", font=("bold", 10), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        self.var_calib_type = tk.StringVar(value=self.cfg["calibration"].get("type", "circles"))
        tk.Radiobutton(type_frame, text="圆点 (Circles)", variable=self.var_calib_type, 
                      value="circles", command=self.on_calib_type_change, bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(type_frame, text="棋盘格 (Chessboard)", variable=self.var_calib_type, 
                      value="chessboard", command=self.on_calib_type_change, bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        
        # 显示当前配置的标定板参数
        rows = self.cfg["calibration"]["rows"]
        cols = self.cfg["calibration"]["cols"]
        spacing = self.cfg["calibration"]["spacing"]
        calib_type = self.cfg["calibration"].get("type", "circles")
        if calib_type == "chessboard":
            info_text = f"当前配置: {rows}行 x {cols}列 (内角点: {rows-1}x{cols-1}), 方格边长: {spacing}mm"
        else:
            info_text = f"当前配置: {rows}行 x {cols}列, 圆心间距: {spacing}mm"
        tk.Label(type_frame, text=info_text, fg="blue", bg="#f0f0f0", font=("Arial", 9)).pack(side=tk.LEFT, padx=20)
        
        top_frame = tk.Frame(parent, bg="#eeeeee", pady=5)
        top_frame.pack(fill=tk.X)
        
        tk.Button(top_frame, text="📸 立即拍照采集 (强制参数)", command=self.manual_trigger, bg="#4caf50", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="📁 从文件夹加载图片", command=self.load_images_from_folder, bg="#ff9800", fg="white", font=("bold", 10)).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="刷新列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="✎ 录入/修改坐标", command=self.on_edit_pose_btn, bg="#2196f3", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="💾 备份数据", command=self.save_calib_data).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="📷 计算相机内参", command=self.run_camera_calib, bg="#9c27b0", fg="white", font=("bold", 10)).pack(side=tk.RIGHT, padx=10)
        tk.Button(top_frame, text="▶ 计算手眼矩阵", command=self.run_hand_eye_calc, bg="orange", font=("bold", 10)).pack(side=tk.RIGHT, padx=20)
        
        columns = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "status")
        self.tree_calib = ttk.Treeview(parent, columns=columns, show="headings")
        
        self.tree_calib.heading("id", text="ID"); self.tree_calib.column("id", width=40)
        self.tree_calib.heading("img", text="图片文件名"); self.tree_calib.column("img", width=180)
        for c in ["x","y","z","rx","ry","rz"]:
            self.tree_calib.heading(c, text=c.upper()); self.tree_calib.column(c, width=70)
        self.tree_calib.heading("status", text="状态"); self.tree_calib.column("status", width=60)
        
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.view_calib_image)  # 双击查看图片
        
        # 操作区域（在Treeview下方）
        op_frame = tk.Frame(parent, bg="#f0f0f0", pady=5)
        op_frame.pack(fill=tk.X, padx=5)
        tk.Button(op_frame, text="📡 获取选中行的坐标 (发送tcp_pose)", 
                 command=self.get_robot_pose_for_selected, bg="#4caf50", fg="white", 
                 font=("bold", 10)).pack(side=tk.LEFT, padx=10)
        tk.Button(op_frame, text="🗑️ 删除选中行", 
                 command=self.delete_selected_row, bg="#f44336", fg="white", 
                 font=("bold", 10)).pack(side=tk.LEFT, padx=10)
        self.lbl_pose_status = tk.Label(op_frame, text="", fg="blue")
        self.lbl_pose_status.pack(side=tk.LEFT, padx=10)
        
        self.txt_calib_res = tk.Text(parent, height=12, bg="#f8f9fa", font=("Consolas", 10))
        self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        f_left = tk.LabelFrame(paned, text="【机械臂通讯参数】", font=("bold", 11), fg="blue")
        paned.add(f_left, minsize=400)
        
        row = 0
        tk.Label(f_left, text="监听 IP (本机):").grid(row=row, column=0, sticky="e", padx=5)
        self.var_robot_net_bind_ip = tk.StringVar(value=self.cfg["robot_net"].get("bind_ip", "0.0.0.0"))
        tk.Entry(f_left, textvariable=self.var_robot_net_bind_ip, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="(注意：必须填本机 IP 或 0.0.0.0)", fg="red", font=("Arial", 8)).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(f_left, text="监听端口 (Port):").grid(row=row, column=0, sticky="e", padx=5)
        self.var_robot_net_port = tk.StringVar(value=str(self.cfg["robot_net"].get("port", 8000)))
        tk.Entry(f_left, textvariable=self.var_robot_net_port, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="--- 机械臂客户端连接 ---", fg="gray").grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        tk.Label(f_left, text="机械臂 IP:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_robot_net_robot_ip = tk.StringVar(value=self.cfg["robot_net"].get("robot_ip", "192.168.1.100"))
        tk.Entry(f_left, textvariable=self.var_robot_net_robot_ip, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="机械臂端口:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_robot_net_robot_port = tk.StringVar(value=str(self.cfg["robot_net"].get("robot_port", 8080)))
        tk.Entry(f_left, textvariable=self.var_robot_net_robot_port, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="(用于发送tcp_pose命令获取坐标)", fg="gray", font=("Arial", 8)).grid(row=row, column=1, sticky="w")
        row += 1
        
        tk.Label(f_left, text="--- 协议设置 ---", fg="gray").grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        tk.Label(f_left, text="触发拍照指令:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_commands_trigger = tk.StringVar(value=self.cfg["commands"].get("trigger", "C1"))
        tk.Entry(f_left, textvariable=self.var_commands_trigger, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="失败返回:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_commands_error = tk.StringVar(value=self.cfg["commands"].get("error", "E1"))
        tk.Entry(f_left, textvariable=self.var_commands_error, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="成功返回前缀:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_commands_success_prefix = tk.StringVar(value=self.cfg["commands"].get("success_prefix", "OK"))
        tk.Entry(f_left, textvariable=self.var_commands_success_prefix, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="--- 存储设置 ---", fg="gray").grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        tk.Label(f_left, text="保存路径:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_paths_save_dir = tk.StringVar(value=self.cfg["paths"].get("save_dir", "./robot_images"))
        tk.Entry(f_left, textvariable=self.var_paths_save_dir, width=35).grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        f_right = tk.LabelFrame(paned, text="【相机连接参数】", font=("bold", 11), fg="green")
        paned.add(f_right, minsize=400)
        
        r_row = 0
        tk.Label(f_right, text="驱动模式:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_mode = tk.StringVar(value=self.cfg["camera"].get("mode", "sdk"))
        mf = tk.Frame(f_right)
        mf.grid(row=r_row, column=1, sticky="w")
        tk.Radiobutton(mf, text="SDK 直连 (GigE)", variable=self.var_camera_mode, value="sdk").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="IP/RTSP", variable=self.var_camera_mode, value="ip").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="Index USB", variable=self.var_camera_mode, value="index").pack(side=tk.LEFT)
        r_row += 1
        
        if not HAS_HIK_SDK:
            tk.Label(f_right, text="[警告] SDK加载失败，请检查环境", fg="red").grid(row=r_row, column=1, sticky="w")
            r_row += 1
        
        tk.Label(f_right, text="目标 IP:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_target_ip = tk.StringVar(value=self.cfg["camera"].get("target_ip", ""))
        tk.Entry(f_right, textvariable=self.var_camera_target_ip, width=25).grid(row=r_row, column=1, sticky="w", padx=5)
        r_row += 1
        
        scan_f = tk.Frame(f_right)
        scan_f.grid(row=r_row, column=1, sticky="w")
        tk.Button(scan_f, text="扫描局域网 IP", command=self.scan_ip, bg="#b3e5fc").pack(side=tk.LEFT)
        self.lbl_scan_res = tk.Label(scan_f, text="", fg="blue")
        self.lbl_scan_res.pack(side=tk.LEFT, padx=5)
        r_row += 1
        
        tk.Label(f_right, text="相机索引:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_index = tk.StringVar(value=str(self.cfg["camera"].get("index", 0)))
        tk.Entry(f_right, textvariable=self.var_camera_index, width=25).grid(row=r_row, column=1, sticky="w", padx=5)
        r_row += 1
        
        tk.Label(f_right, text="--- Halcon 内参 ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5)
        r_row += 1
        
        for k in ["fx","fy","cx","cy"]:
            tk.Label(f_right, text=f"{k.upper()}:").grid(row=r_row, column=0, sticky="e", padx=5)
            default_val = self.cfg["camera"].get(k, 0)
            setattr(self, f"var_camera_{k}", tk.StringVar(value=str(default_val)))
            tk.Entry(f_right, textvariable=getattr(self, f"var_camera_{k}"), width=25).grid(row=r_row, column=1, sticky="w", padx=5)
            r_row += 1
        
        tk.Button(f_right, text="测试连接 (含曝光增益测试)", command=self.test_camera, bg="#e0e0e0").grid(row=r_row, column=1, pady=10)
        r_row += 1
        
        # 添加标定板尺寸配置区域
        f_calib = tk.LabelFrame(paned, text="【标定板尺寸参数】", font=("bold", 11), fg="purple")
        paned.add(f_calib, minsize=400)
        
        calib_row = 0
        tk.Label(f_calib, text="标定板类型:", font=("bold", 9)).grid(row=calib_row, column=0, sticky="e", padx=5, pady=5)
        self.var_calib_type_config = tk.StringVar(value=self.cfg["calibration"].get("type", "circles"))
        calib_type_frame = tk.Frame(f_calib)
        calib_type_frame.grid(row=calib_row, column=1, sticky="w", padx=5)
        tk.Radiobutton(calib_type_frame, text="圆点", variable=self.var_calib_type_config, value="circles", bg="#f0f0f0").pack(side=tk.LEFT)
        tk.Radiobutton(calib_type_frame, text="棋盘格", variable=self.var_calib_type_config, value="chessboard", bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        calib_row += 1
        
        tk.Label(f_calib, text="行数 (rows):", font=("bold", 9)).grid(row=calib_row, column=0, sticky="e", padx=5, pady=5)
        self.var_calib_rows = tk.StringVar(value=str(self.cfg["calibration"].get("rows", 7)))
        tk.Entry(f_calib, textvariable=self.var_calib_rows, width=25).grid(row=calib_row, column=1, sticky="w", padx=5)
        tk.Label(f_calib, text="(棋盘格: 总行数, 内角点=rows-1)", fg="gray", font=("Arial", 8)).grid(row=calib_row, column=2, sticky="w", padx=5)
        calib_row += 1
        
        tk.Label(f_calib, text="列数 (cols):", font=("bold", 9)).grid(row=calib_row, column=0, sticky="e", padx=5, pady=5)
        self.var_calib_cols = tk.StringVar(value=str(self.cfg["calibration"].get("cols", 7)))
        tk.Entry(f_calib, textvariable=self.var_calib_cols, width=25).grid(row=calib_row, column=1, sticky="w", padx=5)
        tk.Label(f_calib, text="(棋盘格: 总列数, 内角点=cols-1)", fg="gray", font=("Arial", 8)).grid(row=calib_row, column=2, sticky="w", padx=5)
        calib_row += 1
        
        tk.Label(f_calib, text="间距 (spacing):", font=("bold", 9)).grid(row=calib_row, column=0, sticky="e", padx=5, pady=5)
        self.var_calib_spacing = tk.StringVar(value=str(self.cfg["calibration"].get("spacing", 12.5)))
        tk.Entry(f_calib, textvariable=self.var_calib_spacing, width=25).grid(row=calib_row, column=1, sticky="w", padx=5)
        tk.Label(f_calib, text="(单位: mm)", fg="gray", font=("Arial", 8)).grid(row=calib_row, column=2, sticky="w", padx=5)
        calib_row += 1
        
        tk.Label(f_calib, text="说明:", font=("bold", 9), fg="blue").grid(row=calib_row, column=0, sticky="e", padx=5, pady=5)
        info_label = tk.Label(f_calib, 
            text="圆点: 圆心间距 | 棋盘格: 方格边长\n修改后需保存配置并重新采集标定数据", 
            fg="blue", font=("Arial", 8), justify=tk.LEFT)
        info_label.grid(row=calib_row, column=1, columnspan=2, sticky="w", padx=5)
        
        def save_config_and_refresh():
            """保存配置并刷新标定界面显示"""
            self.save_config()
            # 同步标定界面的类型选择（如果已创建）
            if hasattr(self, 'var_calib_type'):
                self.var_calib_type.set(self.cfg["calibration"].get("type", "circles"))
        
        tk.Button(parent, text="💾 保存所有系统配置", command=save_config_and_refresh, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=20, pady=10)

    # --- 曝光/增益调节回调函数 ---
    def on_exposure_change(self, value):
        exposure_val = float(value)
        self.lbl_exposure.config(text=f"{exposure_val:.0f} μs")
        self.entry_exposure.delete(0, tk.END)
        self.entry_exposure.insert(0, f"{exposure_val:.0f}")
        if self.cam_running and hasattr(self.cap, 'set_exposure_time'):
            success = self.cap.set_exposure_time(exposure_val)
            if success:
                self.cfg["camera"]["exposure_us"] = exposure_val
                self.log(f"曝光时间已更新为: {exposure_val:.0f} μs (待拍照应用)")
            else:
                self.log(f"[警告] 曝光时间设置失败")

    def set_exposure_by_entry(self):
        try:
            exposure_val = float(self.entry_exposure.get())
            exposure_val = max(100.0, min(50000.0, exposure_val))
            self.var_exposure.set(exposure_val)
            self.lbl_exposure.config(text=f"{exposure_val:.0f} μs")
            if self.cam_running and hasattr(self.cap, 'set_exposure_time'):
                success = self.cap.set_exposure_time(exposure_val)
                if success:
                    self.cfg["camera"]["exposure_us"] = exposure_val
                    self.log(f"通过输入框设置曝光时间为: {exposure_val:.0f} μs")
                else:
                    self.log(f"[警告] 通过输入框设置曝光时间失败")
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数字作为曝光时间")
            self.entry_exposure.delete(0, tk.END)
            self.entry_exposure.insert(0, f"{self.var_exposure.get():.0f}")

    def on_gain_change(self, value):
        gain_val = float(value)
        self.lbl_gain.config(text=f"{gain_val:.1f} dB")
        self.entry_gain.delete(0, tk.END)
        self.entry_gain.insert(0, f"{gain_val:.1f}")
        if self.cam_running and hasattr(self.cap, 'set_gain'):
            success = self.cap.set_gain(gain_val)
            if success:
                self.cfg["camera"]["gain_db"] = gain_val
                self.log(f"增益已更新为: {gain_val:.1f} dB (待拍照应用)")
            else:
                self.log(f"[警告] 增益设置失败")

    def set_gain_by_entry(self):
        try:
            gain_val = float(self.entry_gain.get())
            gain_val = max(0.0, min(30.0, gain_val))
            self.var_gain.set(gain_val)
            self.lbl_gain.config(text=f"{gain_val:.1f} dB")
            if self.cam_running and hasattr(self.cap, 'set_gain'):
                success = self.cap.set_gain(gain_val)
                if success:
                    self.cfg["camera"]["gain_db"] = gain_val
                    self.log(f"通过输入框设置增益为: {gain_val:.1f} dB")
                else:
                    self.log(f"[警告] 通过输入框设置增益失败")
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数字作为增益值")
            self.entry_gain.delete(0, tk.END)
            self.entry_gain.insert(0, f"{self.var_gain.get():.1f}")

    # --- 核心拍照逻辑（强制应用参数） ---
    def manual_trigger(self):
        if not self.cam_running: 
            if messagebox.askyesno("提示", "相机未连接，是否尝试连接？"):
                self.toggle_cam()
            else:
                return
            if not self.cam_running: return
        
        self.log("===== 手动触发拍照 =====")
        # 拍照前强制同步曝光和增益参数
        # self.log(f"拍照前强制设置曝光: {self.var_exposure.get():.0f} μs")
        # self.log(f"拍照前强制设置增益: {self.var_gain.get():.1f} dB")
        if hasattr(self.cap, 'set_exposure_time'):
            self.cap.set_exposure_time(self.var_exposure.get())
        if hasattr(self.cap, 'set_gain'):
            self.cap.set_gain(self.var_gain.get())
        
        res = self.process_request()
        self.log(f"手动拍照完成，结果: {res}")
     

    def process_request(self):
        # 第一步：强制应用当前曝光和增益参数（关键！）
        if self.cam_running and hasattr(self.cap, 'set_exposure_time'):
            self.cap.set_exposure_time(self.var_exposure.get())
            self.cap.set_gain(self.var_gain.get())
        
        # 获取当前帧
        frame = None
        with self.lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
        
        if frame is None:
            return self.cfg["commands"]["error"] + ",NoImage"
            
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.cfg["paths"]["save_dir"], f"IMG_{ts}.{self.cfg['paths']['save_format']}")
        
        mode = self.work_mode.get()
        result_str = self.cfg["commands"]["error"]
        
        # TEST 模式: 识别二维码
        if mode == "TEST":
            K = np.array([
                [self.cfg["camera"]["fx"], 0, self.cfg["camera"]["cx"]],
                [0, self.cfg["camera"]["fy"], self.cfg["camera"]["cy"]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist = np.array(self.cfg["camera"]["dist"], dtype=np.float64)
            
            det = cv2.QRCodeDetector()
            ret, info, points, _ = det.detectAndDecodeMulti(frame)
            if ret and points is not None:
                pts = points[0].astype(np.float32)
                
                # 确保pts是4个点
                if pts.shape[0] != 4:
                    self.log("识别失败: 二维码角点数量不正确")
                    cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 3)
                else:
                    cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 3)
                    
                    # 二维码尺寸（单位：mm，与相机标定单位一致）
                    qr_size = 26.0  # 假设二维码边长100mm
                    half = qr_size / 2.0
                    
                    # 根据图像坐标确定点的顺序（确保点顺序正确）
                    # OpenCV相机坐标系：X向右，Y向下，Z向前（从相机看向物体）
                    pts_2d = pts.reshape(-1, 2)
                    sum_coords = pts_2d.sum(axis=1)  # x+y
                    diff_coords = pts_2d[:, 0] - pts_2d[:, 1]  # x-y
                    
                    # 找到四个角点
                    # 左上：x+y最小；右下：x+y最大
                    # 右上：x-y最大；左下：x-y最小
                    top_left_idx = np.argmin(sum_coords)
                    bottom_right_idx = np.argmax(sum_coords)
                    top_right_idx = np.argmax(diff_coords)
                    bottom_left_idx = np.argmin(diff_coords)
                    
                    # 重新排序点：左上、右上、右下、左下
                    ordered_pts = np.array([
                        pts_2d[top_left_idx],
                        pts_2d[top_right_idx],
                        pts_2d[bottom_right_idx],
                        pts_2d[bottom_left_idx]
                    ], dtype=np.float32)
                    
                    # 对应的3D坐标（OpenCV坐标系：X右，Y下，Z前）
                    # 【重要】二维码坐标系原点定义在二维码中心（0,0,0）
                    # 因此solvePnP返回的tvec表示二维码中心在相机坐标系中的位置
                    obj_pts = np.array([
                        [-half, -half, 0],  # 左上（X负，Y负）
                        [half, -half, 0],   # 右上（X正，Y负）
                        [half, half, 0],    # 右下（X正，Y正）
                        [-half, half, 0]    # 左下（X负，Y正）
                    ], dtype=np.float32)
                    
                    # 使用solvePnP计算姿态（使用排序后的点）
                    # 【重要说明】solvePnP返回的是：二维码坐标系相对于相机坐标系的变换
                    # tvec: 二维码中心在相机坐标系中的位置（单位：mm）
                    #       表示从相机原点指向二维码中心的向量
                    # rvec: 二维码坐标系相对于相机坐标系的旋转（Rodrigues向量）
                    #       表示如何将相机坐标系的轴旋转到与二维码坐标系的轴对齐
                    # 数学表示：T_camera_to_qr（从相机坐标系到二维码坐标系的变换）
                    succ, rvec, tvec = cv2.solvePnP(obj_pts, ordered_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                    if succ:
                        # 绘制坐标轴（X红，Y绿，Z蓝），原点在二维码中心
                        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 50)
                        
                        # 转换为旋转矩阵和欧拉角
                        rmat, _ = cv2.Rodrigues(rvec)
                        euler = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                        
                        # 检查是否启用手眼标定转换
                        if self.cfg["hand_eye"].get("enabled", False):
                            try:
                                # 转换到基座坐标系
                                # 【输出说明】返回的是二维码中心在基座坐标系中的位置和姿态
                                # 机械臂移动到该位置时，末端应该对准二维码中心
                                tvec_base, euler_base = self.transform_camera_to_base(tvec, rvec)
                                
                                # 返回基座坐标系下的结果
                                pre = self.cfg["commands"]["success_prefix"]
                                sep = self.cfg["commands"]["separator"]
                                result_str = f"{pre}{sep}{tvec_base[0]:.2f}{sep}{tvec_base[1]:.2f}{sep}{tvec_base[2]:.2f}{sep}{euler_base[0]:.2f}{sep}{euler_base[1]:.2f}{sep}{euler_base[2]:.2f}"
                                
                                self.log(f"二维码姿态(相机系): t=({tvec[0][0]:.2f},{tvec[1][0]:.2f},{tvec[2][0]:.2f}) mm, r=({euler[0]:.2f},{euler[1]:.2f},{euler[2]:.2f}) deg")
                                self.log(f"二维码姿态(基座系): t=({tvec_base[0]:.2f},{tvec_base[1]:.2f},{tvec_base[2]:.2f}) mm, r=({euler_base[0]:.2f},{euler_base[1]:.2f},{euler_base[2]:.2f}) deg")
                                self.log(f"[提示] 输出的是二维码中心在基座坐标系中的位置，机械臂移动到该位置时末端应对准二维码中心")
                             
                            except Exception as e:
                                error_msg = str(e)
                                self.log(f"[警告] 手眼标定转换失败: {error_msg}")
                                
                                # 如果是因为无法获取机械臂位姿，提示用户手动输入
                                if "无法获取机械臂位姿" in error_msg or "用户取消了手动输入" in error_msg:
                                    self.log("[提示] 请点击'手动输入当前机械臂位姿'按钮，或确保TCP连接正常")
                                
                                # 使用相机坐标系的结果
                                pre = self.cfg["commands"]["success_prefix"]
                                sep = self.cfg["commands"]["separator"]
                                result_str = f"{pre}{sep}{tvec[0][0]:.2f}{sep}{tvec[1][0]:.2f}{sep}{tvec[2][0]:.2f}{sep}{euler[0]:.2f}{sep}{euler[1]:.2f}{sep}{euler[2]:.2f}"
                        else:
                            # 未启用手眼标定，返回相机坐标系下的结果
                            pre = self.cfg["commands"]["success_prefix"]
                            sep = self.cfg["commands"]["separator"]
                            result_str = f"{pre}{sep}{tvec[0][0]:.2f}{sep}{tvec[1][0]:.2f}{sep}{tvec[2][0]:.2f}{sep}{euler[0]:.2f}{sep}{euler[1]:.2f}{sep}{euler[2]:.2f}"
                            self.log(f"二维码姿态(相机系): t=({tvec[0][0]:.2f},{tvec[1][0]:.2f},{tvec[2][0]:.2f}) mm, r=({euler[0]:.2f},{euler[1]:.2f},{euler[2]:.2f}) deg")
                            self.log(f"[提示] 手眼标定未启用，返回相机坐标系结果。如需基座坐标系，请先完成手眼标定。")
                    else:
                        self.log("识别失败: solvePnP计算失败")
            else:
                self.log("识别失败: 未找到二维码")

        # CALIB 模式: 识别标定板（圆点或棋盘格）
        elif mode == "CALIB":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            calib_type = self.cfg["calibration"].get("type", "circles")
            
            # 根据标定板类型选择识别方法
            if calib_type == "chessboard":
                # 棋盘格标定板：findChessboardCorners
                # 注意：棋盘格的内角点数 = (rows-1) x (cols-1)
                ret, corners = cv2.findChessboardCorners(gray, (cols-1, rows-1), 
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    # 亚像素精度优化
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            else:
                # 圆点标定板：findCirclesGrid
                ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            
            if ret:
                # 绘制识别到的圆点（增强显示）
                # 1. 使用drawChessboardCorners绘制连接线
                cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)
                
                # 2. 绘制每个识别到的圆点（大圆点，更明显）
                for i, corner in enumerate(corners):
                    pt = tuple(map(int, corner.ravel()))
                    # 绘制大圆点（外圈）
                    cv2.circle(frame, pt, 8, (0, 255, 0), 2)  # 绿色外圈
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)  # 红色实心圆
                    # 在第一个点和最后一个点添加编号（便于检查顺序）
                    if i == 0:
                        cv2.putText(frame, "0", (pt[0]+10, pt[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    elif i == len(corners) - 1:
                        cv2.putText(frame, str(len(corners)-1), (pt[0]+10, pt[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 3. 在图片上显示识别信息
                if calib_type == "chessboard":
                    expected_points = (rows-1) * (cols-1)
                else:
                    expected_points = rows * cols
                info_text = f"Detected: {len(corners)}/{expected_points} points ({calib_type})"
                cv2.putText(frame, info_text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                self.calib_data_list.append({
                    "id": len(self.calib_data_list) + 1,
                    "img_path": path,
                    "corners": corners,
                    "robot_pose": None,
                    "exposure_us": self.cap.get_exposure_time(),  # 记录拍照时的曝光
                    "gain_db": self.cap.get_gain()                # 记录拍照时的增益
                })
                self.root.after(0, self.refresh_calib_list)
                result_str = self.cfg["commands"]["success_prefix"]
                self.save_calib_data()
                if calib_type == "chessboard":
                    expected_points = (rows-1) * (cols-1)
                    self.log(f"标定板识别成功: 检测到 {len(corners)}/{expected_points} 个角点 (棋盘格)")
                else:
                    expected_points = rows * cols
                    self.log(f"标定板识别成功: 检测到 {len(corners)}/{expected_points} 个圆点")
            else:
                # 识别失败时，也在图片上显示提示
                cv2.putText(frame, "FAILED: No calibration board detected", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if calib_type == "chessboard":
                    cv2.putText(frame, f"Expected: {rows-1}x{cols-1} corners (chessboard), spacing: {self.cfg['calibration']['spacing']}mm", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"Expected: {rows}x{cols} circles, spacing: {self.cfg['calibration']['spacing']}mm", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.log("识别失败: 未找到标定板")

        # 保存图片
        cv2.imwrite(path, frame)
        # 记录拍照时的实际参数
        actual_expo = self.cap.get_exposure_time() if hasattr(self.cap, 'get_exposure_time') else self.var_exposure.get()
        actual_gain = self.cap.get_gain() if hasattr(self.cap, 'get_gain') else self.var_gain.get()
        # self.log(f"已保存图片: {os.path.basename(path)} | 实际曝光: {actual_expo:.0f} μs | 实际增益: {actual_gain:.1f} dB")
        
        return result_str

    # --- 其他辅助函数 ---
    def on_mode_change(self):
        self.log(f"工作模式切换为: {self.work_mode.get()}")
    
    def on_calib_type_change(self):
        """标定板类型切换"""
        calib_type = self.var_calib_type.get()
        self.cfg["calibration"]["type"] = calib_type
        
        # 同步更新配置界面中的标定板类型（如果已创建）
        if hasattr(self, 'var_calib_type_config'):
            self.var_calib_type_config.set(calib_type)
        
        # 保存配置
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.cfg, f, indent=4)
            self.log(f"标定板类型已切换为: {calib_type}")
        except Exception as e:
            self.log(f"[警告] 保存标定板类型失败: {e}")
    
    def on_hand_eye_toggle(self):
        """手眼标定开关切换"""
        enabled = self.use_hand_eye.get()
        self.cfg["hand_eye"]["enabled"] = enabled
        if enabled:
            if not self.cfg["hand_eye"].get("T_camera_to_flange", {}).get("translation"):
                messagebox.showwarning("警告", "手眼标定结果未找到！\n\n请先完成手眼标定（在标定页面点击'计算手眼矩阵'）")
                self.use_hand_eye.set(False)
                self.cfg["hand_eye"]["enabled"] = False
            else:
                self.log("手眼标定转换已启用")
        else:
            self.log("手眼标定转换已禁用")

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            self.txt_log.insert(tk.END, f"[{timestamp}] {msg}\n")
            self.txt_log.see(tk.END)
        except: pass

    def draw_placeholder(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10: w=800; h=600
        self.canvas.create_text(w//2, h//2, text="等待视频源...\n(请先连接相机)", fill="gray", font=("Arial", 16), justify=tk.CENTER)

    def update_cfg_from_ui(self):
        self.cfg["robot_net"]["bind_ip"] = self.var_robot_net_bind_ip.get()
        try:
            self.cfg["robot_net"]["port"] = int(self.var_robot_net_port.get())
        except:
            self.cfg["robot_net"]["port"] = 8000
        self.cfg["robot_net"]["robot_ip"] = self.var_robot_net_robot_ip.get()
        try:
            self.cfg["robot_net"]["robot_port"] = int(self.var_robot_net_robot_port.get())
        except:
            self.cfg["robot_net"]["robot_port"] = 8080
        self.cfg["paths"]["save_dir"] = self.var_paths_save_dir.get()
        self.cfg["commands"]["trigger"] = self.var_commands_trigger.get()
        self.cfg["commands"]["error"] = self.var_commands_error.get()
        self.cfg["commands"]["success_prefix"] = self.var_commands_success_prefix.get()
        self.cfg["camera"]["mode"] = self.var_camera_mode.get()
        self.cfg["camera"]["target_ip"] = self.var_camera_target_ip.get()
        try:
            self.cfg["camera"]["index"] = int(self.var_camera_index.get())
        except:
            self.cfg["camera"]["index"] = 0
        for k in ["fx","fy","cx","cy"]:
            try:
                self.cfg["camera"][k] = float(getattr(self, f"var_camera_{k}").get())
            except:
                pass
        # 保存拍照参数
        self.cfg["camera"]["exposure_us"] = self.var_exposure.get()
        self.cfg["camera"]["gain_db"] = self.var_gain.get()
        # 保存标定板尺寸参数
        try:
            self.cfg["calibration"]["type"] = self.var_calib_type_config.get()
            self.cfg["calibration"]["rows"] = int(self.var_calib_rows.get())
            self.cfg["calibration"]["cols"] = int(self.var_calib_cols.get())
            self.cfg["calibration"]["spacing"] = float(self.var_calib_spacing.get())
        except (ValueError, AttributeError) as e:
            # 如果配置界面还未创建，忽略错误
            pass

    def scan_ip(self):
        self.lbl_scan_res.config(text="扫描中...")
        self.root.update()
        devices = self.scanner.scan()
        if devices:
            ip = devices[0]['ip']
            self.var_camera_target_ip.set(ip)
            self.lbl_scan_res.config(text=f"发现: {ip}")
            messagebox.showinfo("成功", f"扫描到设备 IP: {ip}\n已自动填入配置。")
        else:
            self.lbl_scan_res.config(text="未发现")
            messagebox.showwarning("提示", "未扫描到 GigE 设备，请确认相机已连接且处于同一网段。")

    def test_camera(self):
        self.update_cfg_from_ui()
        mode = self.cfg["camera"]["mode"]
        try:
            if mode == "sdk":
                if not HAS_HIK_SDK: raise Exception("SDK 库未加载")
                cam = HikCameraWrapper()
                cam.open_by_ip(self.cfg["camera"]["target_ip"])
                # 测试曝光和增益设置
                cam.set_exposure_time(self.cfg["camera"]["exposure_us"])
                cam.set_gain(self.cfg["camera"]["gain_db"])
                ret, frame = cam.read()
                cam.release()
                if ret: 
                    messagebox.showinfo("成功", f"SDK 连接正常!\n分辨率: {frame.shape[1]}x{frame.shape[0]}\n当前曝光: {cam.get_exposure_time():.0f} μs\n当前增益: {cam.get_gain():.1f} dB")
                else: raise Exception("连接成功但无法获取图像")
            else:
                src = int(self.cfg["camera"]["index"]) if mode == "index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src,str) else cv2.VideoCapture(src)
                if not cap.isOpened(): raise Exception("OpenCV 无法打开设备")
                ret, frame = cap.read()
                cap.release()
                if ret: messagebox.showinfo("成功", "连接正常")
                else: raise Exception("无图像数据")
        except Exception as e:
            messagebox.showerror("连接失败", str(e))

    def toggle_cam(self):
        if not self.cam_running:
            self.update_cfg_from_ui()
            mode = self.cfg["camera"]["mode"]
            try:
                if mode == "sdk":
                    if not HAS_HIK_SDK: raise Exception("SDK库未加载")
                    self.cap = HikCameraWrapper()
                    self.cap.open_by_ip(self.cfg["camera"]["target_ip"])
                    # 初始化曝光和增益
                    self.cap.set_exposure_time(self.var_exposure.get())
                    self.cap.set_gain(self.var_gain.get())
                    # 更新显示
                    current_expo = self.cap.get_exposure_time()
                    current_gain = self.cap.get_gain()
                    self.var_exposure.set(current_expo)
                    self.var_gain.set(current_gain)
                    self.lbl_exposure.config(text=f"{current_expo:.0f} μs")
                    self.lbl_gain.config(text=f"{current_gain:.1f} dB")
                    self.entry_exposure.delete(0, tk.END)
                    self.entry_exposure.insert(0, f"{current_expo:.0f}")
                    self.entry_gain.delete(0, tk.END)
                    self.entry_gain.insert(0, f"{current_gain:.1f}")
                else:
                    src = int(self.cfg["camera"]["index"]) if mode=="index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src,str) else cv2.VideoCapture(src)
                    if mode == "index":
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.var_exposure.get() / 1000)
                        self.cap.set(cv2.CAP_PROP_GAIN, self.var_gain.get())
                
                if hasattr(self.cap, 'isOpened') and not self.cap.isOpened():
                    raise Exception("相机打开失败")
                
                self.cam_running = True
                self.btn_cam.config(text="断开相机", bg="#f44336")
                self.lbl_cam.config(text="运行中", fg="green")
                
                threading.Thread(target=self.cam_loop, daemon=True).start()
                self.log("相机已独立连接 | 曝光/增益参数已初始化")
                
            except Exception as e:
                messagebox.showerror("相机错误", str(e))
        else:
            self.cam_running = False
            if self.cap:
                if hasattr(self.cap, 'release'): self.cap.release()
            
            self.btn_cam.config(text="连接相机", bg="#e0e0e0")
            self.lbl_cam.config(text="已断开", fg="red")
            self.draw_placeholder()
            self.log("相机已断开")

    def cam_loop(self):
        while self.cam_running:
            try:
                if self.cap:
                    if hasattr(self.cap, 'read'):
                        ret, frame = self.cap.read()
                    else:
                        ret, frame = False, None
                        
                    if ret:
                        with self.lock:
                            self.current_frame = frame
                        
                        if int(time.time() * 100) % 5 == 0:
                            self.root.after(0, self.update_disp, frame)
            except: pass
            time.sleep(0.01)

    def toggle_tcp(self):
        if not self.tcp_running:
            try:
                ip = self.cfg["robot_net"]["bind_ip"]
                port = int(self.cfg["robot_net"]["port"])
                
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
                try:
                    self.server_socket.bind((ip, port))
                except OSError as e:
                    if e.errno == 10049:
                        if messagebox.askyesno("配置错误", f"无法监听 IP {ip} (本机没有这个IP)。\n是否自动改为 0.0.0.0 (监听所有)?"):
                            ip = "0.0.0.0"
                            self.var_robot_net_bind_ip.set("0.0.0.0")
                            self.server_socket.bind((ip, port))
                        else:
                            return
                    else:
                        raise e
                
                self.server_socket.listen(1)
                self.server_socket.settimeout(0.5)
                
                self.tcp_running = True
                self.btn_tcp.config(text="停止监听", bg="#f44336")
                self.lbl_tcp.config(text=f"监听中 {ip}:{port}", fg="green")
                
                threading.Thread(target=self.net_loop, daemon=True).start()
                self.log("TCP 服务已启动")
                
            except Exception as e:
                messagebox.showerror("TCP错误", str(e))
                if self.server_socket: self.server_socket.close()
        else:
            self.tcp_running = False
            if self.server_socket: self.server_socket.close()
            
            self.btn_tcp.config(text="启动监听", bg="#e0e0e0")
            self.lbl_tcp.config(text="已停止", fg="red")
            self.log("TCP 服务已停止")

    def net_loop(self):
        while self.tcp_running:
            try:
                try:
                    client, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                
                self.client_socket = client
                self.log(f"机械臂已连接: {addr}")
                
                while self.tcp_running:
                    try:
                        data = client.recv(1024)
                        if not data: break
                        
                        msg = data.decode('utf-8').strip()
                        self.log(f"收到指令: {msg}")
                        
                        if msg == self.cfg["commands"]["trigger"]:
                            response = self.process_request()
                            client.send(response.encode('utf-8'))
                            self.log(f"回复: {response}")
                        else:
                            pass
                            
                    except Exception as e:
                        self.log(f"通讯中断: {e}")
                        break
                
                self.client_socket = None
                self.log("机械臂断开连接")
                
            except Exception as e:
                if self.tcp_running: self.log(f"Net Error: {e}")

    def update_disp(self, img):
        if not self.cam_running: return
        
        display_img = img.copy()
        h, w = display_img.shape[:2]
        
        if self.undistort_view.get():
            try:
                K = np.array([[self.cfg["camera"]["fx"],0,self.cfg["camera"]["cx"]],
                              [0,self.cfg["camera"]["fy"],self.cfg["camera"]["cy"]],
                              [0,0,1]], dtype=float)
                D = np.array(self.cfg["camera"]["dist"], dtype=float)
                display_img = cv2.undistort(display_img, K, D)
            except: pass

        if self.show_focus_score.get():
            gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            cv2.putText(display_img, f"Focus: {int(score)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 显示当前曝光和增益（拍照参数预览）
            current_expo = self.var_exposure.get()
            current_gain = self.var_gain.get()
            cv2.putText(display_img, f"拍照曝光: {current_expo:.0f} μs | 拍照增益: {current_gain:.1f} dB", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if self.show_crosshair.get():
            cx, cy = w//2, h//2
            cv2.line(display_img, (cx-50, cy), (cx+50, cy), (0, 0, 255), 2)
            cv2.line(display_img, (cx, cy-50), (cx, cy+50), (0, 0, 255), 2)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10: cw=800
        
        scale = min(cw/w, ch/h) * 0.95
        nh, nw = int(h*scale), int(w*scale)
        
        show = cv2.resize(display_img, (nw, nh))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(show)
        tk_img = ImageTk.PhotoImage(pil)
        
        self.canvas.create_image(cw//2, ch//2, image=tk_img, anchor=tk.CENTER)
        self.canvas.image = tk_img 

    def refresh_calib_list(self):
        for item in self.tree_calib.get_children():
            self.tree_calib.delete(item)
        for d in self.calib_data_list:
            pose = d["robot_pose"]
            status = "已就绪" if pose else "待录入"
            
            vals = [d["id"], os.path.basename(d["img_path"])]
            if pose:
                vals.extend([f"{x:.1f}" for x in pose])
            else:
                vals.extend(["-"]*6)
            vals.append(status)
            
            self.tree_calib.insert("", "end", values=vals)

    def transform_camera_to_base(self, tvec_cam, rvec_cam):
        """将相机坐标系下的位姿转换到基座坐标系
        
        【功能说明】
        将二维码在相机坐标系中的位姿转换为在基座坐标系中的位姿。
        输出的结果是二维码中心在基座坐标系中的位置和姿态。
        机械臂移动到该位置时，末端应该对准二维码中心。
        
        【坐标变换链】
        T_base_to_qr = T_base_to_flange × T_flange_to_camera × T_camera_to_qr
        其中：
        - T_base_to_flange: 当前机械臂位姿（从tcp_pose获取）
        - T_flange_to_camera: 手眼标定结果的逆 (T_camera_to_flange)^(-1)
        - T_camera_to_qr: solvePnP的结果（二维码在相机坐标系中的位姿）
        
        Args:
            tvec_cam: 二维码中心在相机坐标系中的位置 [x, y, z] (mm)
            rvec_cam: 二维码坐标系相对于相机坐标系的旋转 (Rodrigues向量) 或旋转矩阵
        
        Returns:
            tvec_base: 二维码中心在基座坐标系中的位置 [x, y, z] (mm)
            euler_base: 二维码坐标系相对于基座坐标系的旋转 [rx, ry, rz] (deg, xyz顺序)
        """
        if not self.cfg["hand_eye"].get("enabled", False):
            raise Exception("手眼标定未启用或未完成标定")
        
        # 获取当前机械臂位姿（T_base_to_flange）
        robot_pose = None
        
        # 首先尝试通过TCP获取
        success, response = self.send_command_to_robot("tcp_pose", timeout=1.5)
        if success:
            parse_ok, robot_pose = self.parse_robot_pose(response)
            if parse_ok:
                self.log("✓ 已通过TCP获取机械臂当前位姿")
            else:
                self.log(f"[警告] TCP返回数据解析失败: {response}")
        else:
            self.log(f"[警告] 无法通过TCP获取机械臂位姿: {response}")
        
        # 如果TCP获取失败，尝试使用手动输入的位姿
        if robot_pose is None:
            if self.manual_robot_pose is not None:
                robot_pose = self.manual_robot_pose
                
            else:
                # 如果都没有，弹出对话框让用户输入
                robot_pose = self.ask_manual_robot_pose()
                if robot_pose is None:
                    raise Exception("无法获取机械臂位姿，且用户取消了手动输入")
                self.manual_robot_pose = robot_pose
                self.log("✓ 已使用用户输入的机械臂位姿")
        
        # 机械臂当前位姿：robot_pose通常是T_base_to_flange（基座到末端）
        # 直接使用，不需要求逆
        t_base_to_flange = np.array(robot_pose[:3]).reshape(3, 1)  # [x, y, z]
        R_base_to_flange = R.from_euler('xyz', robot_pose[3:], degrees=True).as_matrix()
        
        # 手眼标定结果：T_camera_to_flange
        t_cam_to_flange = np.array(self.cfg["hand_eye"]["T_camera_to_flange"]["translation"]).reshape(3, 1)
        R_cam_to_flange = R.from_euler('xyz', self.cfg["hand_eye"]["T_camera_to_flange"]["rotation"], degrees=True).as_matrix()
        
        # 二维码在相机坐标系下的位姿：T_camera_to_qr
        if isinstance(rvec_cam, np.ndarray) and rvec_cam.shape == (3, 1):
            R_cam_to_qr, _ = cv2.Rodrigues(rvec_cam)
        else:
            R_cam_to_qr = rvec_cam
        t_cam_to_qr = np.array(tvec_cam).reshape(3, 1)
        
        # 计算：T_base_to_qr = T_base_to_flange * T_flange_to_camera * T_camera_to_qr
        # 其中：T_flange_to_camera = T_camera_to_flange^(-1)
        R_flange_to_cam = R_cam_to_flange.T  # 旋转矩阵的转置等于逆
        t_flange_to_cam = -R_flange_to_cam @ t_cam_to_flange
        
        # T_flange_to_qr = T_flange_to_camera * T_camera_to_qr
        R_flange_to_qr = R_flange_to_cam @ R_cam_to_qr
        t_flange_to_qr = R_flange_to_cam @ t_cam_to_qr + t_flange_to_cam
        
        # T_base_to_qr = T_base_to_flange * T_flange_to_qr
        R_base_to_qr = R_base_to_flange @ R_flange_to_qr
        t_base_to_qr = R_base_to_flange @ t_flange_to_qr + t_base_to_flange
        
        # 转换为欧拉角
        euler_base = R.from_matrix(R_base_to_qr).as_euler('xyz', degrees=True)
        
        return t_base_to_qr.flatten(), euler_base

    def ask_manual_robot_pose(self):
        """弹出对话框让用户手动输入机械臂位姿
        
        Returns:
            robot_pose: [x, y, z, rx, ry, rz] 或 None（用户取消）
        """
        win = tk.Toplevel(self.root)
        win.title("手动输入机械臂当前位姿")
        win.geometry("650x250")
        win.transient(self.root)
        win.grab_set()
        
        result = [None]
        
        tk.Label(win, text="请输入当前机械臂位姿（基座坐标系）", font=("bold", 11)).pack(pady=10)
        
        f_main = tk.Frame(win, pady=10)
        f_main.pack()
        
        ents = []
        labels = ["X (mm)", "Y (mm)", "Z (mm)", "Rx (deg)", "Ry (deg)", "Rz (deg)"]
        
        for i, lbl in enumerate(labels):
            f = tk.Frame(f_main)
            f.grid(row=0, column=i, padx=5)
            tk.Label(f, text=lbl, font=("Arial", 9)).pack()
            e = tk.Entry(f, width=10, font=("Arial", 10))
            e.pack()
            # 如果有之前手动输入的位姿，显示出来
            if self.manual_robot_pose:
                e.insert(0, str(self.manual_robot_pose[i]))
            ents.append(e)
        
        def confirm():
            try:
                vals = [float(x.get()) for x in ents]
                result[0] = vals
                self.manual_robot_pose = vals
                # 更新显示
                self.lbl_manual_pose.config(text=f"已保存: X={vals[0]:.1f}, Y={vals[1]:.1f}, Z={vals[2]:.1f}, Rx={vals[3]:.1f}°, Ry={vals[4]:.1f}°, Rz={vals[5]:.1f}°")
                win.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字")
        
        def cancel():
            result[0] = None
            win.destroy()
        
        f_btn = tk.Frame(win)
        f_btn.pack(pady=10)
        tk.Button(f_btn, text="确认", command=confirm, bg="#4caf50", fg="white", width=12, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(f_btn, text="取消", command=cancel, bg="#f44336", fg="white", width=12, height=2).pack(side=tk.LEFT, padx=10)
        
        # 等待窗口关闭
        win.wait_window()
        return result[0]

    def manual_input_robot_pose(self):
        """手动输入机械臂位姿按钮的回调函数"""
        pose = self.ask_manual_robot_pose()
        if pose:
            self.log(f"已手动输入机械臂位姿: X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}, Rx={pose[3]:.2f}°, Ry={pose[4]:.2f}°, Rz={pose[5]:.2f}°")
            messagebox.showinfo("成功", f"已保存机械臂位姿:\nX: {pose[0]:.2f} mm\nY: {pose[1]:.2f} mm\nZ: {pose[2]:.2f} mm\nRx: {pose[3]:.2f}°\nRy: {pose[4]:.2f}°\nRz: {pose[5]:.2f}°")
        else:
            self.log("用户取消了手动输入")

    def send_command_to_robot(self, command, timeout=2.0):
        """向机械臂发送TCP命令并接收响应"""
        self.update_cfg_from_ui()
        robot_ip = self.cfg["robot_net"].get("robot_ip", "192.168.1.100")
        robot_port = self.cfg["robot_net"].get("robot_port", 8080)
        
        try:
            # 创建TCP客户端socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # 连接到机械臂
            sock.connect((robot_ip, robot_port))
            self.log(f"已连接到机械臂 {robot_ip}:{robot_port}")
            
            # 发送命令（机械臂使用\r回车符）
            if command == "tcp_pose":
                cmd = command + '\r'  # 机械臂使用回车符
            else:
                cmd = command + '\n'  # 其他命令使用换行符
            sock.sendall(cmd.encode('utf-8'))
            self.log(f"已发送命令: {command}")
            
            # 接收响应（可能需要多次接收，直到接收到完整数据）
            response = b''
            sock.settimeout(1.0)  # 设置接收超时
            try:
                while True:
                    chunk = sock.recv(1024)
                    if not chunk:
                        break
                    response += chunk
                    # 如果响应中包含换行符或回车符，可能表示数据结束
                    if b'\n' in chunk or b'\r' in chunk:
                        break
            except socket.timeout:
                pass  # 接收完成或超时
            
            response_str = response.decode('utf-8').strip()
            if response_str:
                self.log(f"收到响应: {response_str}")
            else:
                self.log(f"[警告] 未收到有效响应")
            
            sock.close()
            return True, response_str if response_str else "未收到响应"
        except socket.timeout:
            self.log(f"[错误] 连接机械臂超时 ({robot_ip}:{robot_port})")
            return False, "连接超时"
        except ConnectionRefusedError:
            self.log(f"[错误] 机械臂拒绝连接 ({robot_ip}:{robot_port})")
            return False, "连接被拒绝"
        except Exception as e:
            self.log(f"[错误] 连接机械臂失败: {e}")
            return False, str(e)

    def parse_robot_pose(self, response_str):
        """解析机械臂返回的坐标数据
        
        机械臂返回格式：以","分隔，前6个数据对应 x,y,z,a,b,c
        - x,y,z: 平移（mm）
        - a,b,c: 旋转角度（度，可能是欧拉角，顺序需要确认）
        
        支持格式：
        - x,y,z,a,b,c,... (逗号分隔，取前6个)
        - OK,x,y,z,a,b,c,... (带前缀，取前6个数据)
        """
        try:
            # 清理响应字符串（移除换行符、回车符等）
            response_str = response_str.strip().replace('\r', '').replace('\n', '')
            
            if "," in response_str:
                parts = response_str.split(",")
                # 移除空字符串
                parts = [p.strip() for p in parts if p.strip()]
                
                # 如果第一个部分是"OK"或其他前缀，跳过它
                if len(parts) > 0 and parts[0].upper() in ["OK", "SUCCESS"]:
                    parts = parts[1:]
                
                # 取前6个数据：x,y,z,a,b,c
                if len(parts) >= 6:
                    pose = [float(p) for p in parts[:6]]
                    self.log(f"解析成功: x={pose[0]:.4f}, y={pose[1]:.2f}, z={pose[2]:.2f}, a={pose[3]:.2f}, b={pose[4]:.2f}, c={pose[5]:.2f}")
                    return True, pose
                else:
                    self.log(f"[错误] 数据不足6个: {len(parts)}个数据，内容: {response_str}")
                    return False, None
            else:
                self.log(f"[错误] 响应格式错误（无逗号分隔）: {response_str}")
                return False, None
        except ValueError as e:
            self.log(f"[错误] 解析坐标数据失败（数值转换错误）: {e}, 原始数据: {response_str}")
            return False, None
        except Exception as e:
            self.log(f"[错误] 解析坐标数据失败: {e}, 原始数据: {response_str}")
            return False, None

    def get_robot_pose_for_selected(self):
        """获取选中行的机械臂坐标"""
        sel = self.tree_calib.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中一行数据")
            return
        
        idx = self.tree_calib.index(sel[0])
        if idx >= len(self.calib_data_list):
            messagebox.showerror("错误", "数据索引错误")
            return
        
        # 更新状态显示
        self.lbl_pose_status.config(text="正在获取坐标...", fg="blue")
        self.root.update()
        
        # 发送tcp_pose命令
        success, response = self.send_command_to_robot("tcp_pose")
        
        if success:
            # 解析响应
            parse_ok, pose = self.parse_robot_pose(response)
            if parse_ok:
                # 保存到数据记录
                self.calib_data_list[idx]["robot_pose"] = pose
                self.refresh_calib_list()
                self.save_calib_data()
                self.lbl_pose_status.config(text=f"✓ 已获取坐标: {pose}", fg="green")
                messagebox.showinfo("成功", f"已成功获取坐标:\nX: {pose[0]:.2f}\nY: {pose[1]:.2f}\nZ: {pose[2]:.2f}\nRx: {pose[3]:.2f}\nRy: {pose[4]:.2f}\nRz: {pose[5]:.2f}")
            else:
                self.lbl_pose_status.config(text="✗ 解析失败", fg="red")
                messagebox.showerror("错误", f"无法解析机械臂返回的数据:\n{response}\n\n请检查数据格式是否正确（应为：x,y,z,rx,ry,rz）")
        else:
            self.lbl_pose_status.config(text="✗ 连接失败", fg="red")
            messagebox.showerror("错误", f"无法连接到机械臂:\n{response}\n\n请检查配置中的机械臂IP和端口是否正确")

    def delete_selected_row(self):
        """删除选中的标定数据行"""
        sel = self.tree_calib.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的数据行")
            return
        
        idx = self.tree_calib.index(sel[0])
        if idx >= len(self.calib_data_list):
            messagebox.showerror("错误", "数据索引错误")
            return
        
        # 获取要删除的数据信息用于确认
        data = self.calib_data_list[idx]
        img_name = os.path.basename(data.get("img_path", ""))
        
        # 确认删除
        if not messagebox.askyesno("确认删除", f"确定要删除第 {idx+1} 行数据吗？\n\n图片: {img_name}\n\n此操作不可撤销！"):
            return
        
        # 删除数据
        del self.calib_data_list[idx]
        
        # 重新编号ID
        for i, d in enumerate(self.calib_data_list):
            d["id"] = i + 1
        
        # 刷新列表
        self.refresh_calib_list()
        
        # 保存数据（不显示保存消息框，避免重复弹窗）
        self.save_calib_data(show_message=False)
        
        self.log(f"已删除第 {idx+1} 行数据（图片: {img_name}）")
        messagebox.showinfo("删除成功", f"已成功删除第 {idx+1} 行数据\n\n数据已自动保存")

    def load_images_from_folder(self):
        """从文件夹加载标定图片"""
        folder_path = filedialog.askdirectory(title="选择包含标定图片的文件夹")
        if not folder_path:
            return
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))
        
        if not image_files:
            messagebox.showwarning("警告", "所选文件夹中没有找到图片文件")
            return
        
        image_files.sort()  # 按文件名排序
        
        self.log(f"从文件夹加载图片: {len(image_files)} 张")
        
        # 识别标定板类型
        calib_type = self.cfg["calibration"].get("type", "circles")
        rows = self.cfg["calibration"]["rows"]
        cols = self.cfg["calibration"]["cols"]
        
        success_count = 0
        fail_count = 0
        
        for img_path in image_files:
            try:
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    self.log(f"[跳过] 无法读取图片: {os.path.basename(img_path)}")
                    fail_count += 1
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 根据标定板类型识别
                if calib_type == "chessboard":
                    # 棋盘格
                    ret, corners = cv2.findChessboardCorners(gray, (cols-1, rows-1), 
                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                    if ret:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                else:
                    # 圆点
                    ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
                
                if ret:
                    # 绘制识别结果
                    if calib_type == "chessboard":
                        cv2.drawChessboardCorners(img, (cols-1, rows-1), corners, ret)
                    else:
                        cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                    
                    # 保存带标记的图片
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_dir = self.cfg["paths"]["save_dir"]
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{base_name}_calib.jpg")
                    cv2.imwrite(save_path, img)
                    
                    # 添加到标定数据列表
                    self.calib_data_list.append({
                        "id": len(self.calib_data_list) + 1,
                        "img_path": save_path,
                        "corners": corners,
                        "robot_pose": None,
                        "exposure_us": 0,  # 从文件加载，无曝光信息
                        "gain_db": 0
                    })
                    success_count += 1
                    if calib_type == "chessboard":
                        expected_points = (rows-1) * (cols-1)
                        self.log(f"[成功] {os.path.basename(img_path)}: 检测到 {len(corners)}/{expected_points} 个角点 (棋盘格)")
                    else:
                        expected_points = rows * cols
                        self.log(f"[成功] {os.path.basename(img_path)}: 检测到 {len(corners)}/{expected_points} 个圆点")
                else:
                    self.log(f"[失败] {os.path.basename(img_path)}: 未识别到标定板")
                    fail_count += 1
                    
            except Exception as e:
                self.log(f"[错误] 处理图片 {os.path.basename(img_path)} 失败: {e}")
                fail_count += 1
        
        # 刷新列表
        self.root.after(0, self.refresh_calib_list)
        self.save_calib_data()
        
        # 显示结果
        messagebox.showinfo("加载完成", 
            f"从文件夹加载完成！\n\n"
            f"成功: {success_count} 张\n"
            f"失败: {fail_count} 张\n"
            f"总计: {len(image_files)} 张")
        
        self.log(f"从文件夹加载完成: 成功 {success_count} 张, 失败 {fail_count} 张")

    def view_calib_image(self, event):
        """双击查看标定图片，显示识别结果"""
        sel = self.tree_calib.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中一行")
            return
        
        idx = self.tree_calib.index(sel[0])
        data = self.calib_data_list[idx]
        img_path = data["img_path"]
        
        if not os.path.exists(img_path):
            messagebox.showerror("错误", f"图片文件不存在:\n{img_path}")
            return
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("错误", f"无法读取图片:\n{img_path}")
            return
        
        # 如果是标定模式，重新识别并绘制标定板
        if "corners" in data and data["corners"] is not None:
            corners = data["corners"]
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            calib_type = self.cfg["calibration"].get("type", "circles")
            
            # 绘制识别到的标定板（与拍照时相同的绘制方式）
            if calib_type == "chessboard":
                cv2.drawChessboardCorners(img, (cols-1, rows-1), corners, True)
            else:
                cv2.drawChessboardCorners(img, (cols, rows), corners, True)
            
            # 绘制每个识别到的点（圆点标定板才绘制大圆点）
            if calib_type != "chessboard":
                for i, corner in enumerate(corners):
                    pt = tuple(map(int, corner.ravel()))
                    # 绘制大圆点（外圈）
                    cv2.circle(img, pt, 8, (0, 255, 0), 2)  # 绿色外圈
                    cv2.circle(img, pt, 4, (0, 0, 255), -1)  # 红色实心圆
                    # 在第一个点和最后一个点添加编号
                    if i == 0:
                        cv2.putText(img, "0", (pt[0]+10, pt[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    elif i == len(corners) - 1:
                        cv2.putText(img, str(len(corners)-1), (pt[0]+10, pt[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 显示识别信息
            if calib_type == "chessboard":
                expected_points = (rows-1) * (cols-1)
            else:
                expected_points = rows * cols
            info_text = f"Detected: {len(corners)}/{expected_points} points ({calib_type})"
            cv2.putText(img, info_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # 如果有机械臂坐标，也显示
            if data["robot_pose"]:
                pose = data["robot_pose"]
                pose_text = f"Robot: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}"
                cv2.putText(img, pose_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 创建显示窗口
        win = tk.Toplevel(self.root)
        win.title(f"查看标定图片 - ID: {data['id']} - {os.path.basename(img_path)}")
        
        # 计算合适的窗口大小（不超过屏幕的80%）
        h, w = img.shape[:2]
        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)
        
        scale = min(max_width/w, max_height/h, 1.0)
        display_w = int(w * scale)
        display_h = int(h * scale)
        
        win.geometry(f"{display_w}x{display_h}+{int((screen_width-display_w)/2)}+{int((screen_height-display_h)/2)}")
        
        # 创建Canvas和滚动条
        canvas_frame = tk.Frame(win)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(canvas_frame, width=display_w, height=display_h)
        scrollbar_v = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar_h = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_v.grid(row=0, column=1, sticky="ns")
        scrollbar_h.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # 调整图片大小并显示
        img_resized = cv2.resize(img, (display_w, display_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img  # 保持引用
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # 添加信息标签
        info_frame = tk.Frame(win)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        info_text = f"图片路径: {img_path}"
        if data["robot_pose"]:
            pose = data["robot_pose"]
            info_text += f"  |  机械臂坐标: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}, Rx={pose[3]:.1f}, Ry={pose[4]:.1f}, Rz={pose[5]:.1f}"
        else:
            info_text += "  |  状态: 待录入机械臂坐标"
        
        tk.Label(info_frame, text=info_text, anchor="w", justify="left", 
                font=("Arial", 9), wraplength=display_w-20).pack(fill=tk.X)
        
        # 添加关闭按钮
        btn_frame = tk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(btn_frame, text="关闭", command=win.destroy, width=15).pack()

    def on_edit_pose_btn(self):
        self.on_edit_pose(None)

    def on_edit_pose(self, event):
        sel = self.tree_calib.selection()
        if not sel: 
            if event is None: messagebox.showwarning("提示", "请先选中一行")
            return
        
        idx = self.tree_calib.index(sel[0])
        win = tk.Toplevel(self.root)
        win.title(f"录入机械臂坐标 (第 {idx+1} 组)")
        win.geometry("600x180")
        
        ents = []
        labels = ["X (mm)", "Y (mm)", "Z (mm)", "Rx (deg)", "Ry (deg)", "Rz (deg)"]
        
        f_main = tk.Frame(win, pady=20)
        f_main.pack()
        
        for i, lbl in enumerate(labels):
            f = tk.Frame(f_main)
            f.grid(row=0, column=i, padx=5)
            tk.Label(f, text=lbl).pack()
            e = tk.Entry(f, width=8)
            e.pack()
            if self.calib_data_list[idx]["robot_pose"]:
                e.insert(0, str(self.calib_data_list[idx]["robot_pose"][i]))
            ents.append(e)
            
        def confirm():
            try:
                vals = [float(x.get()) for x in ents]
                self.calib_data_list[idx]["robot_pose"] = vals
                self.refresh_calib_list()
                self.save_calib_data()
                win.destroy()
            except:
                messagebox.showerror("错误", "请输入有效数字")
        
        tk.Button(win, text="确认保存", command=confirm, bg="#2196f3", fg="white", width=15).pack(pady=10)

    def run_camera_calib(self):
        """使用标定板数据计算相机内参"""
        # 获取所有有corners的数据（不需要机械臂坐标）
        valid = [d for d in self.calib_data_list if "corners" in d and d["corners"] is not None]
        if len(valid) < 3:
            messagebox.showwarning("警告", f"有效数据不足 3 组，无法计算相机内参\n\n当前有效数据: {len(valid)} 组\n\n请先采集至少3张标定板图片")
            return
        
        try:
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            sp = self.cfg["calibration"]["spacing"]
            calib_type = self.cfg["calibration"].get("type", "circles")
            
            # 根据标定板类型准备3D点（标定板上的点）
            if calib_type == "chessboard":
                # 棋盘格：内角点数 = (rows-1) x (cols-1)
                objp = np.zeros(((rows-1) * (cols-1), 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols-1, 0:rows-1].T.reshape(-1, 2) * sp
            else:
                # 圆点：rows x cols
                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sp
            
            # 准备2D点（图像上的点）
            objpoints = []  # 3D点
            imgpoints = []  # 2D点
            
            for d in valid:
                objpoints.append(objp)
                imgpoints.append(d["corners"])
            
            # 获取图像尺寸
            img = cv2.imread(valid[0]["img_path"])
            if img is None:
                messagebox.showerror("错误", "无法读取图片，无法获取图像尺寸")
                return
            img_size = (img.shape[1], img.shape[0])  # (width, height)
            
            # 执行相机标定
            self.log(f"开始计算相机内参，使用 {len(valid)} 组数据...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None,
                flags=cv2.CALIB_FIX_K3  # 固定K3参数（通常为0）
            )
            
            if not ret:
                messagebox.showerror("错误", "相机标定失败")
                return
            
            # 提取内参
            fx = float(mtx[0, 0])
            fy = float(mtx[1, 1])
            cx = float(mtx[0, 2])
            cy = float(mtx[1, 2])
            dist_coeffs = dist.flatten().tolist()
            
            # 计算重投影误差（评估标定质量）
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            mean_error = total_error / len(objpoints)
            
            # 更新配置
            self.cfg["camera"]["fx"] = fx
            self.cfg["camera"]["fy"] = fy
            self.cfg["camera"]["cx"] = cx
            self.cfg["camera"]["cy"] = cy
            self.cfg["camera"]["dist"] = dist_coeffs
            
            # 更新UI显示
            if hasattr(self, f"var_camera_fx"):
                self.var_camera_fx.set(str(fx))
                self.var_camera_fy.set(str(fy))
                self.var_camera_cx.set(str(cx))
                self.var_camera_cy.set(str(cy))
            
            # 保存配置
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(self.cfg, f, indent=4)
                self.log("相机内参已保存到配置文件")
            except Exception as e:
                self.log(f"[警告] 保存相机内参失败: {e}")
            
            # 显示结果
            result_msg = f"""相机标定完成！

内参矩阵:
  fx = {fx:.2f}
  fy = {fy:.2f}
  cx = {cx:.2f}
  cy = {cy:.2f}

畸变系数:
  k1 = {dist_coeffs[0]:.6f}
  k2 = {dist_coeffs[1]:.6f}
  p1 = {dist_coeffs[2]:.6f}
  p2 = {dist_coeffs[3]:.6f}
  k3 = {dist_coeffs[4]:.6f}

重投影误差: {mean_error:.4f} 像素
（误差越小越好，通常 < 0.5 像素为良好）

已自动保存到配置文件！"""
            
            self.txt_calib_res.delete(1.0, tk.END)
            self.txt_calib_res.insert(tk.END, "====== 相机内参标定结果 ======\n")
            self.txt_calib_res.insert(tk.END, f"fx: {fx:.2f}\n")
            self.txt_calib_res.insert(tk.END, f"fy: {fy:.2f}\n")
            self.txt_calib_res.insert(tk.END, f"cx: {cx:.2f}\n")
            self.txt_calib_res.insert(tk.END, f"cy: {cy:.2f}\n")
            self.txt_calib_res.insert(tk.END, f"\n畸变系数:\n")
            self.txt_calib_res.insert(tk.END, f"k1: {dist_coeffs[0]:.6f}\n")
            self.txt_calib_res.insert(tk.END, f"k2: {dist_coeffs[1]:.6f}\n")
            self.txt_calib_res.insert(tk.END, f"p1: {dist_coeffs[2]:.6f}\n")
            self.txt_calib_res.insert(tk.END, f"p2: {dist_coeffs[3]:.6f}\n")
            self.txt_calib_res.insert(tk.END, f"k3: {dist_coeffs[4]:.6f}\n")
            self.txt_calib_res.insert(tk.END, f"\n重投影误差: {mean_error:.4f} 像素\n")
            self.txt_calib_res.insert(tk.END, f"（误差越小越好，通常 < 0.5 像素为良好）\n")
            
            messagebox.showinfo("相机标定成功", result_msg)
            self.log(f"相机内参标定完成: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, 误差={mean_error:.4f}像素")
            
        except Exception as e:
            error_msg = f"相机标定失败: {str(e)}"
            self.log(f"[错误] {error_msg}")
            messagebox.showerror("错误", error_msg)
            import traceback
            traceback.print_exc()

    def run_hand_eye_calc(self):
        valid = [d for d in self.calib_data_list if d["robot_pose"]]
        if len(valid) < 3:
            self.txt_calib_res.insert(tk.END, "[错误] 有效数据不足 3 组，无法计算\n")
            return
        
        # 检查相机内参是否有效
        fx = self.cfg["camera"].get("fx", 0)
        fy = self.cfg["camera"].get("fy", 0)
        cx = self.cfg["camera"].get("cx", 0)
        cy = self.cfg["camera"].get("cy", 0)
        
        if fx == 0 or fy == 0 or cx == 0 or cy == 0:
            messagebox.showwarning(
                "警告", 
                "相机内参未设置或无效！\n\n"
                "请先进行相机标定：\n"
                "1. 采集至少3张标定板图片（不需要机械臂坐标）\n"
                "2. 点击'计算相机内参'按钮\n"
                "3. 然后再进行手眼标定\n\n"
                "或者手动在配置页面输入相机内参。"
            )
            self.txt_calib_res.insert(tk.END, "[错误] 相机内参未设置，无法计算手眼标定\n")
            self.txt_calib_res.insert(tk.END, "请先点击'计算相机内参'按钮或手动输入内参\n")
            return
        
        try:
            R_gripper2base, t_gripper2base = [], []
            R_target2cam, t_target2cam = [], []
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=float)
            dist = np.array(self.cfg["camera"]["dist"], dtype=float)
            
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            sp = self.cfg["calibration"]["spacing"]
            calib_type = self.cfg["calibration"].get("type", "circles")
            
            # 根据标定板类型生成3D点坐标
            if calib_type == "chessboard":
                # 棋盘格：内角点数 = (rows-1) x (cols-1)
                objp = np.zeros(((rows-1) * (cols-1), 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols-1, 0:rows-1].T.reshape(-1, 2) * sp
            else:
                # 圆点：rows x cols
                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sp
            
            for d in valid:
                ret, rvec, tvec = cv2.solvePnP(objp, d["corners"], K, dist)
                R_target2cam.append(cv2.Rodrigues(rvec)[0])
                t_target2cam.append(tvec)
                
                pose = d["robot_pose"]
                # 【关键修复】robot_pose通常是T_base_to_gripper（基座到末端）
                # 但calibrateHandEye需要T_gripper_to_base（末端到基座），需要求逆
                R_base_to_gripper = R.from_euler('xyz', pose[3:], degrees=True).as_matrix()
                t_base_to_gripper = np.array(pose[:3]).reshape(3, 1)
                
                # 求逆得到T_gripper_to_base
                R_gripper_to_base = R_base_to_gripper.T
                t_gripper_to_base = -R_gripper_to_base @ t_base_to_gripper
                
                R_gripper2base.append(R_gripper_to_base)
                t_gripper2base.append(t_gripper_to_base)
                
            rc, tc = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI)
            
            # 保存手眼标定结果到配置
            euler = R.from_matrix(rc).as_euler('xyz', degrees=True)
            self.cfg["hand_eye"]["T_camera_to_flange"]["translation"] = [float(tc[0][0]), float(tc[1][0]), float(tc[2][0])]
            self.cfg["hand_eye"]["T_camera_to_flange"]["rotation"] = [float(euler[0]), float(euler[1]), float(euler[2])]
            self.cfg["hand_eye"]["enabled"] = True
            
            # 保存配置
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(self.cfg, f, indent=4)
                self.log("手眼标定结果已保存到配置文件")
            except Exception as e:
                self.log(f"[警告] 保存手眼标定结果失败: {e}")
            
            self.txt_calib_res.delete(1.0, tk.END)
            self.txt_calib_res.insert(tk.END, "====== 手眼标定结果 (T_Camera_to_Flange) ======\n\n")
            
            # 1. 显示旋转矩阵（3x3）- 与开源代码格式一致
            self.txt_calib_res.insert(tk.END, "【手眼矩阵分解得到的旋转矩阵】\n")
            self.txt_calib_res.insert(tk.END, "Rotation matrix (3x3):\n")
            for i in range(3):
                row_str = "  ["
                for j in range(3):
                    row_str += f"{rc[i, j]:12.8f}"
                    if j < 2:
                        row_str += ", "
                row_str += "]\n"
                self.txt_calib_res.insert(tk.END, row_str)
            self.txt_calib_res.insert(tk.END, "\n")
            
            # 2. 显示平移矩阵（3x1）- 与开源代码格式一致
            self.txt_calib_res.insert(tk.END, "【手眼矩阵分解得到的平移矩阵】\n")
            self.txt_calib_res.insert(tk.END, "Translation matrix (3x1, mm):\n")
            self.txt_calib_res.insert(tk.END, f"  [[{tc[0][0]:12.8f}]\n")
            self.txt_calib_res.insert(tk.END, f"   [{tc[1][0]:12.8f}]\n")
            self.txt_calib_res.insert(tk.END, f"   [{tc[2][0]:12.8f}]]\n\n")
            
            # 3. 构建并显示完整的4x4齐次变换矩阵 - 与开源代码格式一致
            RT = np.hstack([rc, tc])  # 组合旋转和平移
            RT = np.vstack([RT, np.array([0, 0, 0, 1])])  # 添加最后一行 [0, 0, 0, 1]
            
            self.txt_calib_res.insert(tk.END, "【相机相对于末端的变换矩阵为】\n")
            self.txt_calib_res.insert(tk.END, "Camera to End-Effector Transformation Matrix (4x4):\n")
            for i in range(4):
                row_str = "  ["
                for j in range(4):
                    if j < 3:
                        row_str += f"{RT[i, j]:12.8f}, "
                    else:
                        row_str += f"{RT[i, j]:12.8f}"
                row_str += "]\n"
                self.txt_calib_res.insert(tk.END, row_str)
            self.txt_calib_res.insert(tk.END, "\n")
            
            # 4. 显示欧拉角形式（便于理解）
            self.txt_calib_res.insert(tk.END, "【欧拉角形式 (xyz顺序, 度)】\n")
            self.txt_calib_res.insert(tk.END, f"旋转 Rx: {euler[0]:.4f} deg\n")
            self.txt_calib_res.insert(tk.END, f"旋转 Ry: {euler[1]:.4f} deg\n")
            self.txt_calib_res.insert(tk.END, f"旋转 Rz: {euler[2]:.4f} deg\n\n")
            
            self.txt_calib_res.insert(tk.END, "【平移分量 (mm)】\n")
            self.txt_calib_res.insert(tk.END, f"平移 X: {tc[0][0]:.4f} mm\n")
            self.txt_calib_res.insert(tk.END, f"平移 Y: {tc[1][0]:.4f} mm\n")
            self.txt_calib_res.insert(tk.END, f"平移 Z: {tc[2][0]:.4f} mm\n\n")
            
            # 显示使用的相机内参
            self.txt_calib_res.insert(tk.END, "\n\n====== 使用的相机内参 ======\n")
            self.txt_calib_res.insert(tk.END, f"fx: {fx:.2f}\n")
            self.txt_calib_res.insert(tk.END, f"fy: {fy:.2f}\n")
            self.txt_calib_res.insert(tk.END, f"cx: {cx:.2f}\n")
            self.txt_calib_res.insert(tk.END, f"cy: {cy:.2f}\n")
            dist_coeffs = self.cfg["camera"].get("dist", [0,0,0,0,0])
            self.txt_calib_res.insert(tk.END, f"畸变系数: k1={dist_coeffs[0]:.6f}, k2={dist_coeffs[1]:.6f}, p1={dist_coeffs[2]:.6f}, p2={dist_coeffs[3]:.6f}, k3={dist_coeffs[4]:.6f}\n")
            
            # 显示使用的标定板尺寸
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            spacing = self.cfg["calibration"]["spacing"]
            self.txt_calib_res.insert(tk.END, "\n====== 使用的标定板尺寸 ======\n")
            self.txt_calib_res.insert(tk.END, f"行数 (rows): {rows}\n")
            self.txt_calib_res.insert(tk.END, f"列数 (cols): {cols}\n")
            self.txt_calib_res.insert(tk.END, f"圆心间距 (spacing): {spacing} mm\n")
            self.txt_calib_res.insert(tk.END, f"总点数: {rows * cols}\n")
            self.txt_calib_res.insert(tk.END, f"标定数据组数: {len(valid)}\n")
            
            self.txt_calib_res.insert(tk.END, "\n[提示] 手眼标定结果已保存，可在TEST模式下使用！\n")
            
        except Exception as e:
            self.txt_calib_res.insert(tk.END, f"[计算失败] {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalVisionServer(root)
    root.mainloop()