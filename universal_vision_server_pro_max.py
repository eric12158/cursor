import sys
import os
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
# 全局标志位：是否成功加载了 SDK
HAS_HIK_SDK = False
SDK_ERROR_MSG = ""

try:
    # 1. 尝试加载 MVS Runtime DLL
    # 这是为了解决 Python 3.8+ 找不到 C++ DLL 的问题
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
            # 将 DLL 路径加入环境变量
            os.environ['PATH'] = dll_path + ";" + os.environ['PATH']
            # 针对 Python 3.8+ 的额外处理
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(dll_path)
            print(f"[系统] 成功加载海康驱动 DLL 路径: {dll_path}")
            dll_loaded = True
            break
    
    if not dll_loaded:
        print("[警告] 未找到海康 MVS Runtime 路径，如果连接失败请检查驱动安装。")

    # 2. 尝试导入 Python SDK 包 (MvImport)
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    # 检查当前目录下是否有 MvImport
    if os.path.exists(os.path.join(current_dir, "MvImport")):
        from MvImport.MvCameraControl_class import *
        HAS_HIK_SDK = True
        print("[系统] 海康 SDK Python 接口导入成功！")
    else:
        # 再次尝试去桌面找 (防止用户放在桌面运行但没复制文件夹)
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
# 模块二：海康相机驱动封装 (增强版 - 防丢包配置)
# =============================================================================
class HikCameraWrapper:
    def __init__(self):
        self.handle = None
        self.is_opened = False
        self.data_buf = None
        self.n_payload_size = 0
        
    def open_by_ip(self, ip):
        """通过 IP 地址打开相机 (支持 GigE)"""
        if not HAS_HIK_SDK: 
            raise Exception(f"SDK 环境未就绪: {SDK_ERROR_MSG}")

        # 1. 枚举设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0: 
            raise Exception(f"枚举设备失败，错误码: {hex(ret)}")
        
        if deviceList.nDeviceNum == 0: 
            raise Exception("未发现 GigE 网口相机，请检查网线连接")
            
        # 2. 匹配 IP
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
            
        # 3. 创建句柄
        self.handle = MvCamera()
        ret = self.handle.MV_CC_CreateHandle(target_device)
        if ret != 0: 
            raise Exception(f"创建句柄失败: {hex(ret)}")
            
        # 4. 打开设备
        ret = self.handle.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0: 
            raise Exception(f"打开设备失败: {hex(ret)}")
            
        # 5. 关键配置：防止丢包和超时
        # 设置最佳包大小为 1500 (兼容性最强，防止巨帧丢包)
        ret = self.handle.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        if ret != 0: print(f"[警告] 设置包长失败: {hex(ret)}")
        
        # 强制关闭触发模式 (确保是连续采集，防止相机卡在等待触发状态)
        ret = self.handle.MV_CC_SetEnumValue("TriggerMode", 0) # Off
        if ret != 0: print(f"[警告] 设置触发模式失败: {hex(ret)}")
        
        # 增加包间隔 (减少网络拥堵，防止瞬间流量过大)
        self.handle.MV_CC_SetIntValue("GevSCPD", 2000)
            
        # 6. 分配缓存
        stParam = MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_INTVALUE))
        ret = self.handle.MV_CC_GetIntValue("PayloadSize", stParam)
        self.n_payload_size = stParam.nCurValue
        self.data_buf = (ctypes.c_ubyte * self.n_payload_size)()
        
        # 7. 开始取流
        ret = self.handle.MV_CC_StartGrabbing()
        if ret != 0: 
            raise Exception(f"开始取流失败: {hex(ret)}")
            
        self.is_opened = True
        print(f"[SDK] 相机 {ip} 打开成功")

    def read(self):
        """读取图像"""
        if not self.is_opened: return False, None
        
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ctypes.memset(ctypes.byref(stFrameInfo), 0, ctypes.sizeof(MV_FRAME_OUT_INFO_EX))
        
        # 超时设为 2000ms，给大图传输留足时间
        ret = self.handle.MV_CC_GetOneFrameTimeout(ctypes.byref(self.data_buf), self.n_payload_size, stFrameInfo, 2000)
        
        if ret == 0:
            h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
            data = np.frombuffer(self.data_buf, count=int(self.n_payload_size), dtype=np.uint8)
            pt = stFrameInfo.enPixelType
            
            # 兼容多种像素格式
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
            
            # 默认尝试 Mono8 解析
            try:
                img = data.reshape((h, w))
                return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except:
                return False, None
        else:
            # 只有真的是错误时才打印，避免刷屏
            # 0x80000007 是超时，偶尔超时可以忽略
            if ret != 0x80000007: 
                print(f"[SDK Warning] GetFrame 异常: {hex(ret)}")
            return False, None

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
            # 创建 UDP 套接字
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)
            
            # GigE Discovery Packet (标准发现包)
            # Key(0x42) | Flag(0x11) | Cmd(0x0002) | Length(0x0000) | ReqID(0x0001)
            msg = struct.pack('>BBHHH', 0x42, 0x11, 0x0002, 0x0000, 0x0001)
            # 广播到标准端口 3956
            sock.sendto(msg, ('255.255.255.255', 3956))
            
            start = time.time()
            while time.time() - start < timeout:
                try:
                    data, addr = sock.recvfrom(1024)
                    if len(data) > 0:
                        # 简单去重
                        if addr[0] not in [d['ip'] for d in devices]:
                            devices.append({'ip': addr[0]})
                except socket.timeout:
                    break
            sock.close()
        except Exception as e:
            print(f"扫描异常: {e}")
        return devices


# =============================================================================
# 模块四：主程序界面与逻辑 (UniversalVisionServer Pro Max)
# =============================================================================

# 默认配置
DEFAULT_CONFIG = {
    "robot_net": {
        "bind_ip": "0.0.0.0",
        "port": 8000
    },
    "camera": {
        "mode": "sdk",
        "target_ip": "",
        "index": 0,
        "width": 1280,
        "height": 960,
        "fx": 3247.61, "fy": 3244.23, "cx": 2684.71, "cy": 1979.3,
        "dist": [0,0,0,0,0]
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
        "rows": 7,
        "cols": 7,
        "spacing": 5.0
    }
}

class UniversalVisionServer:
    def __init__(self, root):
        self.root = root
        self.root.title("通用视觉服务器系统 - 专业分控版 (PRO MAX)")
        self.root.geometry("1400x950")
        
        self.config_file = "vision_config.json"
        self.calib_data_file = "calibration_data_backup.json"
        
        self.cfg = self.load_config()
        
        # === 核心状态标志 (解耦的关键) ===
        self.tcp_running = False
        self.cam_running = False
        
        self.server_socket = None
        self.client_socket = None
        self.cap = None
        self.current_frame = None
        self.lock = threading.Lock()
        
        self.scanner = GigEScanner()
        self.calib_data_list = []
        
        # 加载之前的标定数据
        self.load_calib_data()
        
        # 辅助功能开关
        self.show_crosshair = tk.BooleanVar(value=True)
        self.show_focus_score = tk.BooleanVar(value=True)
        self.undistort_view = tk.BooleanVar(value=False)
        
        # 初始化 UI
        self.setup_ui()
        
        # 确保目录
        if not os.path.exists(self.cfg["paths"]["save_dir"]):
            os.makedirs(self.cfg["paths"]["save_dir"])

    # --- 配置管理 ---
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for k, v in DEFAULT_CONFIG.items():
                        if k not in data: data[k] = v
                    return data
            except: pass
        return DEFAULT_CONFIG.copy()

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
                    # 恢复 numpy array
                    for d in data:
                        d['corners'] = np.array(d['corners'], dtype=np.float32)
                    self.calib_data_list = data
                    print(f"[系统] 已恢复 {len(data)} 条标定数据")
            except: pass

    def save_calib_data(self):
        # 序列化 numpy
        serializable_list = []
        for d in self.calib_data_list:
            item = d.copy()
            item['corners'] = d['corners'].tolist()
            serializable_list.append(item)
        
        with open(self.calib_data_file, 'w') as f:
            json.dump(serializable_list, f, indent=4)
        messagebox.showinfo("备份成功", f"已保存 {len(self.calib_data_list)} 条数据到文件")

    # --- UI 构建 (完全展开版) ---
    def setup_ui(self):
        # 样式美化
        style = ttk.Style()
        style.configure("Bold.TButton", font=("微软雅黑", 10, "bold"))
        
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: 监控与操作
        frame_run = tk.Frame(notebook)
        notebook.add(frame_run, text="1. 运行监控 (Monitor)")
        self.setup_run_ui(frame_run)
        
        # Tab 2: 标定工具
        frame_calib = tk.Frame(notebook)
        notebook.add(frame_calib, text="2. 手眼标定集成 (Calibration)")
        self.setup_calib_ui(frame_calib)
        
        # Tab 3: 配置
        frame_cfg = tk.Frame(notebook)
        notebook.add(frame_cfg, text="3. 系统配置 (Config)")
        self.setup_config_ui(frame_cfg)

    def setup_run_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制栏
        left_frame = tk.Frame(paned, width=420, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        paned.add(left_frame, minsize=420)
        
        # 1. 独立控制面板 (分控核心)
        c_frame = tk.LabelFrame(left_frame, text="独立控制面板", font=("bold", 10), fg="blue")
        c_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 相机控制行
        row1 = tk.Frame(c_frame)
        row1.pack(fill=tk.X, pady=2)
        self.btn_cam = tk.Button(row1, text="📷 连接相机", command=self.toggle_cam, bg="#e0e0e0", width=15)
        self.btn_cam.pack(side=tk.LEFT, padx=5)
        self.lbl_cam = tk.Label(row1, text="未连接", fg="red")
        self.lbl_cam.pack(side=tk.LEFT)
        
        # TCP 控制行
        row2 = tk.Frame(c_frame)
        row2.pack(fill=tk.X, pady=2)
        self.btn_tcp = tk.Button(row2, text="📡 启动监听", command=self.toggle_tcp, bg="#e0e0e0", width=15)
        self.btn_tcp.pack(side=tk.LEFT, padx=5)
        self.lbl_tcp = tk.Label(row2, text="已停止", fg="red")
        self.lbl_tcp.pack(side=tk.LEFT)
        
        # 2. 手动调试区 (独立于 TCP)
        m_frame = tk.LabelFrame(left_frame, text="手动调试 (Manual Debug)", font=("bold", 10))
        m_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(m_frame, text="📸 手动拍照 (无需TCP指令)", command=self.manual_trigger, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=5, pady=5)
        tk.Label(m_frame, text="* 即使机械臂未连接，也可以点击此按钮测试流程", fg="gray", justify=tk.LEFT).pack(anchor="w", padx=5)

        # 3. 辅助功能开关
        a_frame = tk.LabelFrame(left_frame, text="视觉辅助", font=("bold", 10))
        a_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Checkbutton(a_frame, text="显示中心十字线", variable=self.show_crosshair).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(a_frame, text="显示清晰度评分 (对焦用)", variable=self.show_focus_score).grid(row=0, column=1, sticky="w")
        tk.Checkbutton(a_frame, text="启用实时畸变矫正", variable=self.undistort_view).grid(row=1, column=0, sticky="w", columnspan=2)

        # 4. 工作模式
        w_frame = tk.LabelFrame(left_frame, text="工作模式", font=("bold", 10))
        w_frame.pack(fill=tk.X, padx=5, pady=5)
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(w_frame, text="测试模式 (识别二维码，回传坐标)", variable=self.work_mode, value="TEST", command=self.on_mode_change).pack(anchor="w", padx=5)
        tk.Radiobutton(w_frame, text="标定模式 (识别标定板，仅存图)", variable=self.work_mode, value="CALIB", command=self.on_mode_change).pack(anchor="w", padx=5)

        # 5. 日志
        l_frame = tk.LabelFrame(left_frame, text="系统日志", font=("bold", 10))
        l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(l_frame, font=("Consolas", 9), state=tk.NORMAL)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        # 右侧图像区
        self.canvas = tk.Canvas(paned, bg="#222")
        paned.add(self.canvas, stretch="always")
        self.draw_placeholder()

    def setup_calib_ui(self, parent):
        # 顶部工具栏
        top_frame = tk.Frame(parent, bg="#eeeeee", pady=5)
        top_frame.pack(fill=tk.X)
        
        # 快捷拍照
        tk.Button(top_frame, text="📸 立即拍照采集", command=self.manual_trigger, bg="#4caf50", fg="white").pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="刷新列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        
        # 显式的大按钮
        tk.Button(top_frame, text="✎ 录入/修改坐标", command=self.on_edit_pose_btn, bg="#2196f3", fg="white").pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="💾 备份数据", command=self.save_calib_data).pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="▶ 计算手眼矩阵", command=self.run_hand_eye_calc, bg="orange", font=("bold", 10)).pack(side=tk.RIGHT, padx=20)
        
        # 数据列表
        columns = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "status")
        self.tree_calib = ttk.Treeview(parent, columns=columns, show="headings")
        
        self.tree_calib.heading("id", text="ID"); self.tree_calib.column("id", width=40)
        self.tree_calib.heading("img", text="图片文件名"); self.tree_calib.column("img", width=180)
        for c in ["x","y","z","rx","ry","rz"]:
            self.tree_calib.heading(c, text=c.upper()); self.tree_calib.column(c, width=70)
        self.tree_calib.heading("status", text="状态"); self.tree_calib.column("status", width=60)
        
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.on_edit_pose)
        
        # 结果输出区
        self.txt_calib_res = tk.Text(parent, height=12, bg="#f8f9fa", font=("Consolas", 10))
        self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左：机械臂
        f_left = tk.LabelFrame(paned, text="【机械臂通讯参数】", font=("bold", 11), fg="blue")
        paned.add(f_left, minsize=400)
        
        row = 0
        
        # 本机监听设置
        tk.Label(f_left, text="监听 IP (本机):").grid(row=row, column=0, sticky="e", padx=5)
        self.var_robot_net_bind_ip = tk.StringVar(value=self.cfg["robot_net"].get("bind_ip", "0.0.0.0"))
        tk.Entry(f_left, textvariable=self.var_robot_net_bind_ip, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        # 智能提示
        tk.Label(f_left, text="(注意：必须填本机 IP 或 0.0.0.0)", fg="red", font=("Arial", 8)).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(f_left, text="监听端口 (Port):").grid(row=row, column=0, sticky="e", padx=5)
        self.var_robot_net_port = tk.StringVar(value=str(self.cfg["robot_net"].get("port", 8000)))
        tk.Entry(f_left, textvariable=self.var_robot_net_port, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="--- 协议设置 ---", fg="gray").grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # 触发指令
        tk.Label(f_left, text="触发拍照指令:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_commands_trigger = tk.StringVar(value=self.cfg["commands"].get("trigger", "C1"))
        tk.Entry(f_left, textvariable=self.var_commands_trigger, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        # 失败返回
        tk.Label(f_left, text="失败返回:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_commands_error = tk.StringVar(value=self.cfg["commands"].get("error", "E1"))
        tk.Entry(f_left, textvariable=self.var_commands_error, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        # 成功前缀
        tk.Label(f_left, text="成功返回前缀:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_commands_success_prefix = tk.StringVar(value=self.cfg["commands"].get("success_prefix", "OK"))
        tk.Entry(f_left, textvariable=self.var_commands_success_prefix, width=25).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        tk.Label(f_left, text="--- 存储设置 ---", fg="gray").grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # 保存路径
        tk.Label(f_left, text="保存路径:").grid(row=row, column=0, sticky="e", padx=5)
        self.var_paths_save_dir = tk.StringVar(value=self.cfg["paths"].get("save_dir", "./robot_images"))
        tk.Entry(f_left, textvariable=self.var_paths_save_dir, width=35).grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        # 右：相机
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
        
        # IP 设置
        tk.Label(f_right, text="目标 IP:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_target_ip = tk.StringVar(value=self.cfg["camera"].get("target_ip", ""))
        tk.Entry(f_right, textvariable=self.var_camera_target_ip, width=25).grid(row=r_row, column=1, sticky="w", padx=5)
        r_row += 1
        
        # 扫描按钮
        scan_f = tk.Frame(f_right)
        scan_f.grid(row=r_row, column=1, sticky="w")
        tk.Button(scan_f, text="扫描局域网 IP", command=self.scan_ip, bg="#b3e5fc").pack(side=tk.LEFT)
        self.lbl_scan_res = tk.Label(scan_f, text="", fg="blue")
        self.lbl_scan_res.pack(side=tk.LEFT, padx=5)
        r_row += 1
        
        # Index
        tk.Label(f_right, text="相机索引:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_index = tk.StringVar(value=str(self.cfg["camera"].get("index", 0)))
        tk.Entry(f_right, textvariable=self.var_camera_index, width=25).grid(row=r_row, column=1, sticky="w", padx=5)
        r_row += 1
        
        tk.Label(f_right, text="--- Halcon 内参 ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5)
        r_row += 1
        
        # 内参 Fx
        tk.Label(f_right, text="Fx:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_fx = tk.StringVar(value=str(self.cfg["camera"].get("fx", 0)))
        tk.Entry(f_right, textvariable=self.var_camera_fx, width=25).grid(row=r_row, column=1, sticky="w", padx=5)
        r_row += 1
        
        # 内参 Fy
        tk.Label(f_right, text="Fy:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_fy = tk.StringVar(value=str(self.cfg["camera"].get("fy", 0)))
        tk.Entry(f_right, textvariable=self.var_camera_fy, width=25).grid(row=r_row, column=1, sticky="w", padx=5)
        r_row += 1
        
        # 内参 Cx
        tk.Label(f_right, text="Cx:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_cx = tk.StringVar(value=str(self.cfg["camera"].get("cx", 0)))
        tk.Entry(f_right, textvariable=self.var_camera_cx, width=25).grid(row=r_row, column=1, sticky="w", padx=5)
        r_row += 1
        
        # 内参 Cy
        tk.Label(f_right, text="Cy:").grid(row=r_row, column=0, sticky="e", padx=5)
        self.var_camera_cy = tk.StringVar(value=str(self.cfg["camera"].get("cy", 0)))
        tk.Entry(f_right, textvariable=self.var_camera_cy, width=25).grid(row=r_row, column=1, sticky="w", padx=5)
        r_row += 1
        
        tk.Button(f_right, text="测试连接", command=self.test_camera, bg="#e0e0e0").grid(row=r_row, column=1, pady=10)
        
        tk.Button(parent, text="💾 保存所有系统配置", command=self.save_config, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=20, pady=10)

    # -------------------------------------------------------------------------
    # 核心业务逻辑
    # -------------------------------------------------------------------------
    def on_mode_change(self):
        self.log(f"工作模式切换为: {self.work_mode.get()}")

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
        # 机械臂
        self.cfg["robot_net"]["bind_ip"] = self.var_robot_net_bind_ip.get()
        self.cfg["robot_net"]["port"] = int(self.var_robot_net_port.get())
        # 路径
        self.cfg["paths"]["save_dir"] = self.var_paths_save_dir.get()
        self.cfg["commands"]["trigger"] = self.var_commands_trigger.get()
        self.cfg["commands"]["error"] = self.var_commands_error.get()
        self.cfg["commands"]["success_prefix"] = self.var_commands_success_prefix.get()
        # 相机
        self.cfg["camera"]["mode"] = self.var_camera_mode.get()
        self.cfg["camera"]["target_ip"] = self.var_camera_target_ip.get()
        try: self.cfg["camera"]["index"] = int(self.var_camera_index.get())
        except: pass
        # 内参
        for k in ["fx","fy","cx","cy"]:
            try: self.cfg["camera"][k] = float(getattr(self, f"var_camera_{k}").get())
            except: pass

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
                ret, frame = cam.read()
                cam.release()
                if ret: messagebox.showinfo("成功", f"SDK 连接正常!\n分辨率: {frame.shape[1]}x{frame.shape[0]}")
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

    # --- 独立控制：相机 ---
    def toggle_cam(self):
        if not self.cam_running:
            self.update_cfg_from_ui()
            mode = self.cfg["camera"]["mode"]
            try:
                if mode == "sdk":
                    if not HAS_HIK_SDK: raise Exception("SDK库未加载")
                    self.cap = HikCameraWrapper()
                    self.cap.open_by_ip(self.cfg["camera"]["target_ip"])
                else:
                    src = int(self.cfg["camera"]["index"]) if mode=="index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src,str) else cv2.VideoCapture(src)
                
                # 检查是否成功
                if hasattr(self.cap, 'isOpened') and not self.cap.isOpened():
                    raise Exception("相机打开失败")
                
                self.cam_running = True
                self.btn_cam.config(text="断开相机", bg="#f44336")
                self.lbl_cam.config(text="运行中", fg="green")
                
                threading.Thread(target=self.cam_loop, daemon=True).start()
                self.log("相机已独立连接")
                
            except Exception as e:
                messagebox.showerror("相机错误", str(e))
        else:
            # 停止
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
                        
                        # UI 刷新 (降频)
                        if int(time.time() * 100) % 5 == 0:
                            self.root.after(0, self.update_disp, frame)
            except: pass
            time.sleep(0.01)

    # --- 独立控制：TCP ---
    def toggle_tcp(self):
        if not self.tcp_running:
            try:
                ip = self.cfg["robot_net"]["bind_ip"]
                port = int(self.cfg["robot_net"]["port"])
                
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
                # 智能纠错：防止填错 IP 导致无法启动
                try:
                    self.server_socket.bind((ip, port))
                except OSError as e:
                    if e.errno == 10049: # 地址无效
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
                            # 处理拍照请求
                            response = self.process_request()
                            client.send(response.encode('utf-8'))
                            self.log(f"回复: {response}")
                        else:
                            # 扩展指令
                            pass
                            
                    except Exception as e:
                        self.log(f"通讯中断: {e}")
                        break
                
                self.client_socket = None
                self.log("机械臂断开连接")
                
            except Exception as e:
                if self.tcp_running: self.log(f"Net Error: {e}")

    # --- 通用处理 ---
    def manual_trigger(self):
        """手动触发拍照逻辑"""
        if not self.cam_running: 
            if messagebox.askyesno("提示", "相机未连接，是否尝试连接？"):
                self.toggle_cam()
            else:
                return
            
            if not self.cam_running: return # 还是连接失败
            
        self.log("用户手动触发拍照...")
        res = self.process_request()
        self.log(f"手动处理结果: {res}")
        messagebox.showinfo("结果", f"处理完成: {res}")

    def process_request(self):
        # 核心处理
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
                pts = points[0]
                cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 3)
                
                # 假设 100mm 标定
                qr_size = 100.0
                half = qr_size / 2.0
                obj_pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
                
                succ, rvec, tvec = cv2.solvePnP(obj_pts, pts, K, dist)
                if succ:
                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 50)
                    rmat, _ = cv2.Rodrigues(rvec)
                    euler = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    
                    pre = self.cfg["commands"]["success_prefix"]
                    sep = self.cfg["commands"]["separator"]
                    result_str = f"{pre}{sep}{tvec[0][0]:.2f}{sep}{tvec[1][0]:.2f}{sep}{tvec[2][0]:.2f}{sep}{euler[0]:.2f}{sep}{euler[1]:.2f}{sep}{euler[2]:.2f}"
            else:
                self.log("识别失败: 未找到二维码")

        # CALIB 模式: 识别圆点标定板
        elif mode == "CALIB":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            
            if ret:
                cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)
                # 记录到列表
                self.calib_data_list.append({
                    "id": len(self.calib_data_list) + 1,
                    "img_path": path,
                    "corners": corners,
                    "robot_pose": None
                })
                self.root.after(0, self.refresh_calib_list)
                result_str = self.cfg["commands"]["success_prefix"]
                # 自动保存一份标定数据
                self.save_calib_data()
            else:
                self.log("识别失败: 未找到标定板")

        cv2.imwrite(path, frame)
        self.log(f"已保存: {os.path.basename(path)}")
        return result_str

    def update_disp(self, img):
        if not self.cam_running: return
        
        display_img = img.copy()
        h, w = display_img.shape[:2]
        
        # 1. 畸变矫正预览
        if self.undistort_view.get():
            try:
                K = np.array([[self.cfg["camera"]["fx"],0,self.cfg["camera"]["cx"]],
                              [0,self.cfg["camera"]["fy"],self.cfg["camera"]["cy"]],
                              [0,0,1]], dtype=float)
                D = np.array(self.cfg["camera"]["dist"], dtype=float)
                display_img = cv2.undistort(display_img, K, D)
            except: pass

        # 2. 清晰度评分
        if self.show_focus_score.get():
            gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            cv2.putText(display_img, f"Focus: {int(score)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 3. 中心十字线
        if self.show_crosshair.get():
            cx, cy = w//2, h//2
            cv2.line(display_img, (cx-50, cy), (cx+50, cy), (0, 0, 255), 2)
            cv2.line(display_img, (cx, cy-50), (cx, cy+50), (0, 0, 255), 2)

        # 缩放显示
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

    def run_hand_eye_calc(self):
        valid = [d for d in self.calib_data_list if d["robot_pose"]]
        if len(valid) < 3:
            self.txt_calib_res.insert(tk.END, "[错误] 有效数据不足 3 组，无法计算\n")
            return
        
        try:
            R_gripper2base, t_gripper2base = [], []
            R_target2cam, t_target2cam = [], []
            
            K = np.array([
                [self.cfg["camera"]["fx"], 0, self.cfg["camera"]["cx"]],
                [0, self.cfg["camera"]["fy"], self.cfg["camera"]["cy"]],
                [0, 0, 1]
            ], dtype=float)
            dist = np.array(self.cfg["camera"]["dist"], dtype=float)
            
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            sp = self.cfg["calibration"]["spacing"]
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sp
            
            for d in valid:
                ret, rvec, tvec = cv2.solvePnP(objp, d["corners"], K, dist)
                R_target2cam.append(cv2.Rodrigues(rvec)[0])
                t_target2cam.append(tvec)
                
                pose = d["robot_pose"]
                t_gripper2base.append(np.array(pose[:3]).reshape(3, 1))
                # 欧拉角转旋转矩阵 (假设 XYZ 顺序)
                r_mat_g = R.from_euler('xyz', pose[3:], degrees=True).as_matrix()
                
                R_gripper2base.append(r_mat_g)
                
            rc, tc = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI)
            
            self.txt_calib_res.delete(1.0, tk.END)
            self.txt_calib_res.insert(tk.END, "====== 标定结果 (T_Camera_to_Flange) ======\n")
            self.txt_calib_res.insert(tk.END, f"平移 X: {tc[0][0]:.4f} mm\n")
            self.txt_calib_res.insert(tk.END, f"平移 Y: {tc[1][0]:.4f} mm\n")
            self.txt_calib_res.insert(tk.END, f"平移 Z: {tc[2][0]:.4f} mm\n\n")
            
            euler = R.from_matrix(rc).as_euler('xyz', degrees=True)
            self.txt_calib_res.insert(tk.END, f"旋转 Rx: {euler[0]:.4f} deg\n")
            self.txt_calib_res.insert(tk.END, f"旋转 Ry: {euler[1]:.4f} deg\n")
            self.txt_calib_res.insert(tk.END, f"旋转 Rz: {euler[2]:.4f} deg\n")
            
            self.txt_calib_res.insert(tk.END, "\n[矩阵形式]\n")
            self.txt_calib_res.insert(tk.END, str(rc))
            
        except Exception as e:
            self.txt_calib_res.insert(tk.END, f"[计算失败] {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalVisionServer(root)
    root.mainloop()
