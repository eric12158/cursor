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
from PIL import Image, ImageTk
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# =============================================================================
# 模块一：海康 SDK 自动加载与驱动封装
# =============================================================================

# 全局标志位：是否成功加载了 SDK
HAS_HIK_SDK = False

try:
    # 1. 尝试自动搜索海康 SDK 的安装路径
    possible_sdk_paths = [
        os.path.join(os.getcwd(), "MvImport"), # 优先搜索当前目录
        r"C:\Program Files (x86)\MVS\Development\Samples\Python",
        r"C:\Program Files\MVS\Development\Samples\Python",
        r"D:\Program Files (x86)\MVS\Development\Samples\Python",
        r"D:\MVS\Development\Samples\Python"
    ]
    
    found_path = None
    for p in possible_sdk_paths:
        if os.path.exists(os.path.join(p, "MvImport")):
            if p not in sys.path:
                sys.path.append(p)
            found_path = p
            break
            
    # 2. 尝试导入 SDK 核心类
    from MvImport.MvCameraControl_class import *
    HAS_HIK_SDK = True
    print(f"[系统信息] 海康 SDK 加载成功，路径: {found_path}")

except ImportError:
    print("[系统警告] 未找到海康 MVS SDK，SDK直连模式将不可用 (请安装 MVS 或检查路径)")
    HAS_HIK_SDK = False

# --- 海康相机驱动类 ---
class HikCameraWrapper:
    def __init__(self):
        self.handle = None
        self.is_opened = False
        self.data_buf = None
        self.n_payload_size = 0
        
    def open_by_ip(self, ip):
        """通过 IP 地址打开相机"""
        if not HAS_HIK_SDK:
            raise Exception("SDK 库未加载，无法使用此模式")

        # 1. 枚举设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            raise Exception(f"枚举设备失败，错误码: {ret}")
        if deviceList.nDeviceNum == 0:
            raise Exception("未发现 GigE 网口相机")
            
        # 2. 寻找匹配 IP 的设备
        target_device = None
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = ctypes.cast(deviceList.pDeviceInfo[i], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                # 解析 IP 地址
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                str_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
                
                if str_ip == ip:
                    target_device = mvcc_dev_info
                    break
        
        if target_device is None:
            raise Exception(f"未找到 IP 为 {ip} 的相机，请检查连接")
            
        # 3. 创建句柄
        self.handle = MvCamera()
        ret = self.handle.MV_CC_CreateHandle(target_device)
        if ret != 0:
            raise Exception(f"创建句柄失败: {ret}")
            
        # 4. 打开设备
        ret = self.handle.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception(f"打开设备失败: {ret}")
            
        # 5. 设置优化包大小 (GigE 必须步骤)
        nPacketSize = self.handle.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = self.handle.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            
        # 6. 获取 Payload Size 并分配内存
        stParam = MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_INTVALUE))
        ret = self.handle.MV_CC_GetIntValue("PayloadSize", stParam)
        self.n_payload_size = stParam.nCurValue
        self.data_buf = (ctypes.c_ubyte * self.n_payload_size)()
        
        # 7. 开始取流
        ret = self.handle.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception(f"开始取流失败: {ret}")
            
        self.is_opened = True
        return True

    def read(self):
        """读取一帧图像，返回兼容 OpenCV 的 (ret, img)"""
        if not self.is_opened:
            return False, None
            
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ctypes.memset(ctypes.byref(stFrameInfo), 0, ctypes.sizeof(MV_FRAME_OUT_INFO_EX))
        
        # 超时时间 1000ms
        ret = self.handle.MV_CC_GetOneFrameTimeout(ctypes.byref(self.data_buf), self.n_payload_size, stFrameInfo, 1000)
        
        if ret == 0:
            # 转换图像格式
            h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
            pixel_type = stFrameInfo.enPixelType
            
            # 将 ctypes 数组转为 numpy
            data = np.frombuffer(self.data_buf, count=int(self.n_payload_size), dtype=np.uint8)
            
            # 格式转换逻辑
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
            
            # 如果是其他彩色格式，尝试默认转换
            try:
                # 这是一个简化的假设，实际可能需要更复杂的解码
                img = data.reshape((h, w))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
                return True, img
            except:
                return False, None
        
        return False, None

    def release(self):
        """释放资源"""
        if self.handle:
            self.handle.MV_CC_StopGrabbing()
            self.handle.MV_CC_CloseDevice()
            self.handle.MV_CC_DestroyHandle()
        self.is_opened = False

# =============================================================================
# 模块二：GigE 设备扫描器
# =============================================================================
class GigEScanner:
    def scan(self, timeout=1.0):
        devices = []
        try:
            # 创建 UDP 套接字
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)
            
            # GigE Discovery Packet (标准发现包)
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
            print(f"扫描出错: {e}")
        return devices

# =============================================================================
# 模块三：主程序 (UniversalVisionServer)
# =============================================================================

# 默认配置文件结构
DEFAULT_CONFIG = {
    "robot_net": {
        "bind_ip": "0.0.0.0",
        "port": 8000
    },
    "camera": {
        "mode": "sdk",     # sdk, ip, index
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
        self.root.title("通用视觉服务器系统 - 完整功能版")
        self.root.geometry("1400x950")
        
        self.config_file = "vision_config.json"
        self.cfg = self.load_config()
        
        # 运行时变量
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.cap = None
        self.current_frame = None
        self.lock = threading.Lock()
        
        self.scanner = GigEScanner()
        self.calib_data_list = []
        
        # 初始化 UI
        self.setup_ui()
        
        # 确保目录存在
        if not os.path.exists(self.cfg["paths"]["save_dir"]):
            try:
                os.makedirs(self.cfg["paths"]["save_dir"])
            except: pass

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    # 合并默认配置，防止新版本缺字段
                    for k, v in DEFAULT_CONFIG.items():
                        if k not in data:
                            data[k] = v
                        elif isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                if sub_k not in data[k]:
                                    data[k][sub_k] = sub_v
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

    # -------------------------------------------------------------------------
    # UI 构建 (完整展开)
    # -------------------------------------------------------------------------
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 页面 1: 运行监控
        frame_run = tk.Frame(notebook)
        notebook.add(frame_run, text="1. 运行监控")
        self.setup_run_ui(frame_run)
        
        # 页面 2: 手眼标定
        frame_calib = tk.Frame(notebook)
        notebook.add(frame_calib, text="2. 手眼标定集成")
        self.setup_calib_ui(frame_calib)
        
        # 页面 3: 系统配置
        frame_cfg = tk.Frame(notebook)
        notebook.add(frame_cfg, text="3. 系统配置")
        self.setup_config_ui(frame_cfg)

    def setup_run_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(paned, width=420, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        paned.add(left_frame, minsize=420)
        
        # 1. 服务器控制
        s_frame = tk.LabelFrame(left_frame, text="服务器控制", font=("bold", 10))
        s_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_start = tk.Button(s_frame, text="启动服务 (Start Server)", command=self.toggle_server, bg="#4caf50", fg="white", height=2, font=("bold", 11))
        self.btn_start.pack(fill=tk.X, padx=5, pady=5)
        
        status_frame = tk.Frame(s_frame)
        status_frame.pack(fill=tk.X, padx=5)
        self.lbl_status = tk.Label(status_frame, text="状态: 已停止", fg="red", font=("Arial", 10))
        self.lbl_status.pack(side=tk.LEFT)
        self.lbl_client = tk.Label(status_frame, text="客户端: 无连接", fg="gray", font=("Arial", 10))
        self.lbl_client.pack(side=tk.RIGHT)
        
        # 2. 模式选择
        m_frame = tk.LabelFrame(left_frame, text="工作模式 (Work Mode)", font=("bold", 10))
        m_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(m_frame, text="测试模式 (识别二维码，回传坐标)", variable=self.work_mode, value="TEST", command=self.on_mode_change).pack(anchor="w", padx=5)
        tk.Radiobutton(m_frame, text="标定模式 (识别标定板，仅存图)", variable=self.work_mode, value="CALIB", command=self.on_mode_change).pack(anchor="w", padx=5)
        
        # 3. 日志
        l_frame = tk.LabelFrame(left_frame, text="通讯日志 (Logs)", font=("bold", 10))
        l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(l_frame, font=("Consolas", 9), state=tk.NORMAL)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        # 右侧图像
        self.canvas = tk.Canvas(paned, bg="#222")
        paned.add(self.canvas, stretch="always")
        self.draw_placeholder()

    def setup_calib_ui(self, parent):
        top_frame = tk.Frame(parent, pady=5)
        top_frame.pack(fill=tk.X)
        
        tk.Button(top_frame, text="刷新数据列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        
        # 显式的大按钮
        tk.Button(top_frame, text="✎ 手动录入/修改坐标", command=self.on_edit_pose_btn, bg="#2196f3", fg="white").pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="▶ 计算手眼矩阵", command=self.run_hand_eye_calc, bg="orange").pack(side=tk.LEFT, padx=10)
        
        columns = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "ok")
        self.tree_calib = ttk.Treeview(parent, columns=columns, show="headings")
        for col in columns: 
            self.tree_calib.heading(col, text=col)
            self.tree_calib.column(col, width=60)
        self.tree_calib.column("img", width=150)
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.on_edit_pose)
        
        self.txt_calib_res = tk.Text(parent, height=10, bg="#e0e0e0", font=("Consolas", 10))
        self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：机械臂
        f_left = tk.LabelFrame(paned, text="【机械臂通讯 (TCP Server)】", font=("bold", 11), fg="blue")
        paned.add(f_left, minsize=450)
        
        row = 0
        def add_entry(p, label, key_group, key_item, width=25):
            nonlocal row
            tk.Label(p, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=5)
            var = tk.StringVar(value=str(self.cfg[key_group].get(key_item, "")))
            entry = tk.Entry(p, textvariable=var, width=width)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=5)
            setattr(self, f"var_{key_group}_{key_item}", var)
            row += 1

        add_entry(f_left, "监听 IP (本机):", "robot_net", "bind_ip")
        add_entry(f_left, "监听端口 (Port):", "robot_net", "port")
        
        tk.Label(f_left, text="--- 指令协议 ---", fg="gray").grid(row=row, column=0, columnspan=2, pady=10); row+=1
        add_entry(f_left, "触发拍照指令:", "commands", "trigger")
        add_entry(f_left, "失败返回:", "commands", "error")
        add_entry(f_left, "成功返回前缀:", "commands", "success_prefix")
        
        tk.Label(f_left, text="--- 文件存储 ---", fg="gray").grid(row=row, column=0, columnspan=2, pady=10); row+=1
        add_entry(f_left, "保存路径:", "paths", "save_dir", width=35)

        # 右侧：相机
        f_right = tk.LabelFrame(paned, text="【相机连接 (支持海康SDK)】", font=("bold", 11), fg="green")
        paned.add(f_right, minsize=450)
        
        r_row = 0
        def add_cam_entry(label, key_item, width=25):
            nonlocal r_row
            tk.Label(f_right, text=label).grid(row=r_row, column=0, sticky="e", padx=5, pady=5)
            var = tk.StringVar(value=str(self.cfg["camera"].get(key_item, "")))
            entry = tk.Entry(f_right, textvariable=var, width=width)
            entry.grid(row=r_row, column=1, sticky="w", padx=5, pady=5)
            setattr(self, f"var_camera_{key_item}", var)
            r_row += 1

        tk.Label(f_right, text="连接模式:").grid(row=r_row, column=0, sticky="e", padx=5, pady=5)
        self.var_camera_mode = tk.StringVar(value=self.cfg["camera"].get("mode", "sdk"))
        mf = tk.Frame(f_right)
        mf.grid(row=r_row, column=1, sticky="w")
        tk.Radiobutton(mf, text="海康 SDK 直连 (推荐)", variable=self.var_camera_mode, value="sdk").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="IP/RTSP 模式", variable=self.var_camera_mode, value="ip").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="Index 索引", variable=self.var_camera_mode, value="index").pack(side=tk.LEFT)
        r_row += 1
        
        if not HAS_HIK_SDK:
            tk.Label(f_right, text="[警告] 未检测到 SDK 库文件，SDK 模式将不可用", fg="red").grid(row=r_row, column=0, columnspan=2); r_row+=1
        
        add_cam_entry("相机 IP:", "target_ip")
        
        scan_f = tk.Frame(f_right)
        scan_f.grid(row=r_row, column=1, sticky="w")
        tk.Button(scan_f, text="扫描 IP", command=self.scan_ip, bg="#b3e5fc").pack(side=tk.LEFT)
        self.lbl_scan_res = tk.Label(scan_f, text="", fg="blue")
        self.lbl_scan_res.pack(side=tk.LEFT, padx=5)
        r_row += 1
        
        add_cam_entry("相机 Index:", "index")
        
        tk.Label(f_right, text="--- 内参 (Halcon) ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5); r_row+=1
        add_cam_entry("Fx:", "fx")
        add_cam_entry("Fy:", "fy")
        add_cam_entry("Cx:", "cx")
        add_cam_entry("Cy:", "cy")
        
        tk.Button(f_right, text="测试相机连接", command=self.test_camera, bg="#e0e0e0").grid(row=r_row, column=1, pady=10); r_row+=1

        tk.Button(parent, text="保存所有配置", command=self.save_config, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=20, pady=10)

    # -------------------------------------------------------------------------
    # 逻辑功能
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
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        if w < 10: w=800; h=600
        self.canvas.create_text(w//2, h//2, text="等待视频源...", fill="gray", font=("Arial", 16))

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
        devs = self.scanner.scan()
        if devs: 
            ip = devs[0]['ip']
            self.var_camera_target_ip.set(ip)
            self.lbl_scan_res.config(text=f"发现: {ip}")
            messagebox.showinfo("成功", f"扫描到设备 IP: {ip}\n已自动填入。")
        else:
            self.lbl_scan_res.config(text="未发现")
            messagebox.showwarning("失败", "未扫描到 GigE 设备。请确认相机已上电且在同一网段。")

    def test_camera(self):
        self.update_cfg_from_ui()
        mode = self.cfg["camera"]["mode"]
        
        try:
            if mode == "sdk":
                if not HAS_HIK_SDK:
                    raise Exception("SDK 库未加载，请检查 MVS 是否安装或库文件是否存在")
                
                cam = HikCameraWrapper()
                cam.open_by_ip(self.cfg["camera"]["target_ip"])
                ret, frame = cam.read()
                cam.release()
                
                if ret:
                    messagebox.showinfo("成功", f"SDK 连接成功!\n分辨率: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    raise Exception("SDK 打开成功但读取图像失败")
            else:
                # 兼容旧逻辑
                src = int(self.cfg["camera"]["index"]) if mode == "index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                if isinstance(src, str):
                    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(src)
                    
                if not cap.isOpened():
                    raise Exception("OpenCV 无法打开视频源")
                    
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    messagebox.showinfo("成功", "连接成功，画面正常")
                else:
                    raise Exception("连接建立但无图像数据")
                    
        except Exception as e:
            messagebox.showerror("连接失败", f"错误详情: {str(e)}")

    def toggle_server(self):
        if not self.is_running:
            # 启动
            try:
                # 1. 网络监听
                ip = self.cfg["robot_net"]["bind_ip"]
                port = int(self.cfg["robot_net"]["port"])
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((ip, port))
                self.server_socket.listen(1)
                self.server_socket.settimeout(0.5)
                
                # 2. 相机连接
                self.update_cfg_from_ui()
                mode = self.cfg["camera"]["mode"]
                
                if mode == "sdk":
                    if not HAS_HIK_SDK: raise Exception("SDK 未安装或未加载")
                    self.cap = HikCameraWrapper()
                    self.cap.open_by_ip(self.cfg["camera"]["target_ip"])
                else:
                    src = int(self.cfg["camera"]["index"]) if mode=="index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                    if isinstance(src, str):
                        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                    else:
                        self.cap = cv2.VideoCapture(src)
                
                # 检查相机是否开启 (SDK 模式在 init 时已抛出异常，这里主要查 OpenCV)
                if hasattr(self.cap, 'isOpened') and not self.cap.isOpened():
                    self.log("警告: 相机未连接，服务将以无图像模式运行")
                
                self.is_running = True
                self.btn_start.config(text="停止服务 (Stop)", bg="#f44336")
                self.lbl_status.config(text=f"监听中: {ip}:{port}", fg="green")
                
                # 启动线程
                threading.Thread(target=self.net_loop, daemon=True).start()
                threading.Thread(target=self.cam_loop, daemon=True).start()
                self.log("服务已启动，等待机械臂连接...")
                
            except Exception as e:
                messagebox.showerror("启动错误", str(e))
                if self.server_socket: self.server_socket.close()
        else:
            # 停止
            self.is_running = False
            if self.server_socket: self.server_socket.close()
            # 释放相机
            if self.cap:
                if hasattr(self.cap, 'release'): self.cap.release()
            
            self.btn_start.config(text="启动服务 (Start)", bg="#4caf50")
            self.lbl_status.config(text="已停止", fg="red")
            self.draw_placeholder()
            self.log("服务已停止")

    def net_loop(self):
        """网络监听线程"""
        while self.is_running:
            try:
                try:
                    client, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                
                self.client_socket = client
                self.log(f"机械臂已连接: {addr}")
                self.root.after(0, lambda: self.lbl_client.config(text=f"连接来自: {addr}", fg="blue"))
                
                while self.is_running:
                    try:
                        data = client.recv(1024)
                        if not data: break
                        
                        msg = data.decode('utf-8').strip()
                        self.log(f"收到指令: {msg}")
                        
                        trig = self.cfg["commands"]["trigger"]
                        if msg == trig:
                            # 触发核心处理逻辑
                            response = self.process_request()
                            client.send(response.encode('utf-8'))
                            self.log(f"回复: {response}")
                        else:
                            # 心跳或其他指令
                            pass
                            
                    except Exception as e:
                        self.log(f"通讯中断: {e}")
                        break
                
                self.client_socket = None
                self.log("机械臂断开连接")
                self.root.after(0, lambda: self.lbl_client.config(text="无连接", fg="gray"))
                
            except Exception as e:
                if self.is_running: self.log(f"Net Error: {e}")

    def cam_loop(self):
        """相机采集线程"""
        while self.is_running:
            try:
                if self.cap:
                    # 兼容 SDK Wrapper 和 OpenCV VideoCapture
                    if hasattr(self.cap, 'read'):
                        ret, frame = self.cap.read()
                    else:
                        ret, frame = False, None
                        
                    if ret:
                        with self.lock:
                            self.current_frame = frame
                        # 降频刷新 UI
                        if int(time.time() * 100) % 5 == 0:
                            self.root.after(0, self.update_disp, frame)
            except: pass
            time.sleep(0.01)

    def update_disp(self, img):
        if not self.is_running: return
        h, w = img.shape[:2]
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw<10: cw=800
        scale = min(cw/w, ch/h) * 0.95
        nh, nw = int(h*scale), int(w*scale)
        
        show = cv2.resize(img, (nw, nh))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(show)
        tk_img = ImageTk.PhotoImage(pil)
        self.canvas.create_image(cw//2, ch//2, image=tk_img, anchor=tk.CENTER)
        self.canvas.image = tk_img

    def process_request(self):
        """处理拍照请求"""
        frame = None
        with self.lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
        
        if frame is None:
            return self.cfg["commands"]["error"] + ",NoImage"
            
        # 生成保存路径
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.cfg["paths"]["save_dir"], f"IMG_{ts}.{self.cfg['paths']['save_format']}")
        
        mode = self.work_mode.get()
        result_str = self.cfg["commands"]["error"]
        
        # 模式 1: 测试模式 (二维码识别 + PnP)
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
                # 绘制
                cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 3)
                
                # 假设二维码 100mm
                qr_size = 100.0
                half = qr_size / 2.0
                obj_pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
                
                succ, rvec, tvec = cv2.solvePnP(obj_pts, pts, K, dist)
                if succ:
                    # 绘制坐标轴
                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 50)
                    
                    # 转欧拉角
                    rmat, _ = cv2.Rodrigues(rvec)
                    euler = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    
                    prefix = self.cfg["commands"]["success_prefix"]
                    sep = self.cfg["commands"]["separator"]
                    # 格式: OK, X, Y, Z, Rx, Ry, Rz
                    result_str = f"{prefix}{sep}{tvec[0][0]:.2f}{sep}{tvec[1][0]:.2f}{sep}{tvec[2][0]:.2f}{sep}{euler[0]:.2f}{sep}{euler[1]:.2f}{sep}{euler[2]:.2f}"
            else:
                self.log("未识别到二维码")

        # 模式 2: 标定模式 (存图 + 识别角点)
        elif mode == "CALIB":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            
            if ret:
                cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)
                # 存入列表
                self.calib_data_list.append({
                    "id": len(self.calib_data_list) + 1,
                    "img_path": path,
                    "corners": corners,
                    "robot_pose": None # 等待手动录入
                })
                self.root.after(0, self.refresh_calib_list)
                result_str = self.cfg["commands"]["success_prefix"]
            else:
                self.log("未识别到标定板")

        # 保存图片
        cv2.imwrite(path, frame)
        self.log(f"已保存: {os.path.basename(path)}")
        return result_str

    # -------------------------------------------------------------------------
    # 标定辅助
    # -------------------------------------------------------------------------
    def refresh_calib_list(self):
        for item in self.tree_calib.get_children():
            self.tree_calib.delete(item)
        for d in self.calib_data_list:
            pose = d["robot_pose"]
            self.tree_calib.insert("", "end", values=(
                d["id"], os.path.basename(d["img_path"]), 
                *(pose if pose else ["-"]*6),
                "Yes"
            ))

    def on_edit_pose_btn(self):
        self.on_edit_pose(None)

    def on_edit_pose(self, event):
        sel = self.tree_calib.selection()
        if not sel: return
        item = sel[0]
        idx = self.tree_calib.index(item)
        
        win = tk.Toplevel(self.root)
        win.title(f"输入 Pose (ID: {idx+1})")
        ents = []
        for i, l in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            tk.Label(win, text=l).grid(row=0, column=i, padx=5)
            e = tk.Entry(win, width=8)
            e.grid(row=1, column=i, padx=5)
            # 回填已有数据
            if self.calib_data_list[idx]["robot_pose"]:
                e.insert(0, str(self.calib_data_list[idx]["robot_pose"][i]))
            ents.append(e)
            
        def confirm():
            try:
                vals = [float(x.get()) for x in ents]
                self.calib_data_list[idx]["robot_pose"] = vals
                self.refresh_calib_list()
                win.destroy()
            except:
                messagebox.showerror("错误", "请输入有效数字")
        tk.Button(win, text="确认", command=confirm).grid(row=2, column=0, columnspan=6, pady=10)

    def run_hand_eye_calc(self):
        valid = [d for d in self.calib_data_list if d["robot_pose"]]
        if len(valid) < 3:
            self.txt_calib_res.insert(tk.END, "错误: 至少需要 3 组完整数据\n")
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
            
            # 生成 Object Points
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            sp = self.cfg["calibration"]["spacing"]
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sp
            
            for d in valid:
                # 图像姿态
                ret, rvec, tvec = cv2.solvePnP(objp, d["corners"], K, dist)
                rmat, _ = cv2.Rodrigues(rvec)
                R_target2cam.append(rmat)
                t_target2cam.append(tvec)
                
                # 机械臂姿态
                pose = d["robot_pose"]
                t_g2b = np.array(pose[:3]).reshape(3, 1)
                # 欧拉角转旋转矩阵 (假设 XYZ 顺序)
                r_mat_g = R.from_euler('xyz', pose[3:], degrees=True).as_matrix()
                
                R_gripper2base.append(r_mat_g)
                t_gripper2base.append(t_g2b)
                
            # 计算 (默认 Eye-in-Hand)
            rc, tc = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI)
            
            self.txt_calib_res.delete(1.0, tk.END)
            self.txt_calib_res.insert(tk.END, "=== 标定结果 (Camera -> Gripper) ===\n")
            self.txt_calib_res.insert(tk.END, f"平移 XYZ (mm):\n{tc.flatten()}\n")
            self.txt_calib_res.insert(tk.END, f"\n旋转矩阵 R:\n{rc}\n")
            
            euler = R.from_matrix(rc).as_euler('xyz', degrees=True)
            self.txt_calib_res.insert(tk.END, f"\n欧拉角 (Rx, Ry, Rz):\n{euler}\n")
            
        except Exception as e:
            self.txt_calib_res.insert(tk.END, f"计算失败: {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalVisionServer(root)
    root.mainloop()
