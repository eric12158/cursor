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
# 模块一：海康 SDK 核心加载器
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
    for dll_path in possible_dll_paths:
        if os.path.exists(dll_path):
            os.environ['PATH'] = dll_path + ";" + os.environ['PATH']
            if hasattr(os, 'add_dll_directory'): os.add_dll_directory(dll_path)
            break

    current_dir = os.getcwd()
    if current_dir not in sys.path: sys.path.append(current_dir)

    if os.path.exists(os.path.join(current_dir, "MvImport")):
        from MvImport.MvCameraControl_class import *
        HAS_HIK_SDK = True
    else:
        # 尝试桌面
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.exists(os.path.join(desktop, "MvImport")):
            sys.path.append(desktop)
            from MvImport.MvCameraControl_class import *
            HAS_HIK_SDK = True
except Exception as e:
    HAS_HIK_SDK = False
    SDK_ERROR_MSG = str(e)

# =============================================================================
# 模块二：海康相机驱动
# =============================================================================
class HikCameraWrapper:
    def __init__(self):
        self.handle = None
        self.is_opened = False
        self.data_buf = None
        self.n_payload_size = 0
        
    def open_by_ip(self, ip):
        if not HAS_HIK_SDK: raise Exception(f"SDK不可用: {SDK_ERROR_MSG}")
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, deviceList)
        if ret != 0: raise Exception(f"枚举失败: {hex(ret)}")
        if deviceList.nDeviceNum == 0: raise Exception("未发现GigE相机")
            
        target_device = None
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = ctypes.cast(deviceList.pDeviceInfo[i], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                if f"{nip1}.{nip2}.{nip3}.{nip4}" == ip:
                    target_device = mvcc_dev_info
                    break
        
        if target_device is None: raise Exception(f"未找到IP为 {ip} 的相机")
        self.handle = MvCamera()
        if self.handle.MV_CC_CreateHandle(target_device) != 0: raise Exception("创建句柄失败")
        if self.handle.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0: raise Exception("打开设备失败")
        
        self.handle.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        self.handle.MV_CC_SetEnumValue("TriggerMode", 0) # 强制连续模式
        self.handle.MV_CC_SetIntValue("GevSCPD", 1000) # 包间隔
            
        stParam = MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_INTVALUE))
        self.handle.MV_CC_GetIntValue("PayloadSize", stParam)
        self.n_payload_size = stParam.nCurValue
        self.data_buf = (ctypes.c_ubyte * self.n_payload_size)()
        
        if self.handle.MV_CC_StartGrabbing() != 0: raise Exception("取流失败")
        self.is_opened = True

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
                return True, cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2BGR)
            elif PixelType_Gvsp_BayerGR8 == pt:
                return True, cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_BayerGR2BGR)
            elif PixelType_Gvsp_BayerRG8 == pt:
                return True, cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_BayerRG2BGR)
            elif PixelType_Gvsp_RGB8_Packed == pt:
                return True, cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
            try: return True, cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2BGR)
            except: return False, None
        return False, None

    def release(self):
        if self.handle:
            self.handle.MV_CC_StopGrabbing()
            self.handle.MV_CC_CloseDevice()
            self.handle.MV_CC_DestroyHandle()
        self.is_opened = False

# =============================================================================
# 模块三：GigE 扫描器
# =============================================================================
class GigEScanner:
    def scan(self, timeout=1.0):
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
                    if len(data) > 0 and addr[0] not in [d['ip'] for d in devices]: devices.append({'ip': addr[0]})
                except socket.timeout: break
            sock.close()
        except: pass
        return devices

# =============================================================================
# 模块四：主程序 (分控专业版)
# =============================================================================
DEFAULT_CONFIG = {
    "robot_net": {"bind_ip": "0.0.0.0", "port": 8000},
    "camera": {"mode": "sdk", "target_ip": "", "index": 0, "width": 1280, "height": 960, "fx": 3247.61, "fy": 3244.23, "cx": 2684.71, "cy": 1979.3, "dist": [0,0,0,0,0]},
    "paths": {"save_dir": "./robot_images", "save_format": "jpg"},
    "commands": {"trigger": "C1", "error": "E1", "success_prefix": "OK", "separator": ","},
    "calibration": {"rows": 7, "cols": 7, "spacing": 5.0}
}

class UniversalVisionServer:
    def __init__(self, root):
        self.root = root
        self.root.title("通用视觉服务器系统 - 专业分控版 (PRO)")
        self.root.geometry("1400x950")
        
        self.config_file = "vision_config.json"
        self.calib_data_file = "calibration_data_backup.json"
        self.cfg = self.load_config()
        
        # 状态标志 (解耦的关键)
        self.tcp_running = False
        self.cam_running = False
        
        self.server_socket = None
        self.client_socket = None
        self.cap = None
        self.current_frame = None
        self.lock = threading.Lock()
        
        self.scanner = GigEScanner()
        self.calib_data_list = []
        self.load_calib_data()
        
        # 辅助变量
        self.show_crosshair = tk.BooleanVar(value=True)
        self.show_focus_score = tk.BooleanVar(value=True)
        self.undistort_view = tk.BooleanVar(value=False)
        
        self.setup_ui()
        if not os.path.exists(self.cfg["paths"]["save_dir"]): os.makedirs(self.cfg["paths"]["save_dir"])

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
            with open(self.config_file, 'w') as f: json.dump(self.cfg, f, indent=4)
            messagebox.showinfo("提示", "配置已保存")
        except: pass

    def load_calib_data(self):
        if os.path.exists(self.calib_data_file):
            try:
                with open(self.calib_data_file, 'r') as f:
                    data = json.load(f)
                    for d in data: d['corners'] = np.array(d['corners'], dtype=np.float32)
                    self.calib_data_list = data
            except: pass

    def save_calib_data(self):
        sl = []
        for d in self.calib_data_list:
            i = d.copy(); i['corners'] = d['corners'].tolist(); sl.append(i)
        with open(self.calib_data_file, 'w') as f: json.dump(sl, f, indent=4)
        messagebox.showinfo("提示", "数据已备份")

    def setup_ui(self):
        nb = ttk.Notebook(self.root); nb.pack(fill=tk.BOTH, expand=True)
        fr = tk.Frame(nb); nb.add(fr, text="1. 运行监控 (Monitor)"); self.setup_run_ui(fr)
        fc = tk.Frame(nb); nb.add(fc, text="2. 手眼标定 (Calibration)"); self.setup_calib_ui(fc)
        fg = tk.Frame(nb); nb.add(fg, text="3. 系统配置 (Config)"); self.setup_config_ui(fg)

    def setup_run_ui(self, p):
        paned = tk.PanedWindow(p, orient=tk.HORIZONTAL); paned.pack(fill=tk.BOTH, expand=True)
        left = tk.Frame(paned, width=420, bg="#f0f0f0"); left.pack_propagate(False); paned.add(left, minsize=420)
        
        # 分离控制区
        c_frame = tk.LabelFrame(left, text="独立控制面板", font=("bold", 10), fg="blue")
        c_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 相机控制
        f1 = tk.Frame(c_frame); f1.pack(fill=tk.X, pady=2)
        self.btn_cam = tk.Button(f1, text="📷 连接相机", command=self.toggle_cam, bg="#e0e0e0", width=15)
        self.btn_cam.pack(side=tk.LEFT, padx=5)
        self.lbl_cam = tk.Label(f1, text="未连接", fg="red"); self.lbl_cam.pack(side=tk.LEFT)
        
        # TCP控制
        f2 = tk.Frame(c_frame); f2.pack(fill=tk.X, pady=2)
        self.btn_tcp = tk.Button(f2, text="📡 启动监听", command=self.toggle_tcp, bg="#e0e0e0", width=15)
        self.btn_tcp.pack(side=tk.LEFT, padx=5)
        self.lbl_tcp = tk.Label(f2, text="已停止", fg="red"); self.lbl_tcp.pack(side=tk.LEFT)
        
        # 手动调试
        m_frame = tk.LabelFrame(left, text="手动调试", font=("bold", 10))
        m_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(m_frame, text="📸 手动拍照 (无需TCP)", command=self.manual_trigger, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=5, pady=5)
        
        # 辅助功能
        a_frame = tk.LabelFrame(left, text="视觉辅助", font=("bold", 10))
        a_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Checkbutton(a_frame, text="中心十字线", variable=self.show_crosshair).grid(row=0, column=0)
        tk.Checkbutton(a_frame, text="对焦评分", variable=self.show_focus_score).grid(row=0, column=1)
        tk.Checkbutton(a_frame, text="畸变矫正", variable=self.undistort_view).grid(row=1, column=0)

        # 模式
        w_frame = tk.LabelFrame(left, text="模式", font=("bold", 10))
        w_frame.pack(fill=tk.X, padx=5, pady=5)
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(w_frame, text="测试模式 (回传坐标)", variable=self.work_mode, value="TEST", command=lambda:self.log(f"模式: {self.work_mode.get()}")).pack(anchor="w", padx=5)
        tk.Radiobutton(w_frame, text="标定模式 (仅存图)", variable=self.work_mode, value="CALIB", command=lambda:self.log(f"模式: {self.work_mode.get()}")).pack(anchor="w", padx=5)

        l_frame = tk.LabelFrame(left, text="日志", font=("bold", 10)); l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(l_frame, font=("Consolas", 9)); self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(paned, bg="#222"); paned.add(self.canvas, stretch="always")
        self.draw_placeholder()

    def setup_calib_ui(self, p):
        top = tk.Frame(p, pady=5); top.pack(fill=tk.X)
        tk.Button(top, text="📸 立即拍照", command=self.manual_trigger, bg="#4caf50", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="刷新列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="✎ 录入坐标", command=self.on_edit_pose_btn, bg="#2196f3", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="💾 备份数据", command=self.save_calib_data).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="▶ 计算矩阵", command=self.run_hand_eye_calc, bg="orange").pack(side=tk.RIGHT, padx=20)
        
        self.tree_calib = ttk.Treeview(p, columns=("id","img","x","y","z","rx","ry","rz","stat"), show="headings")
        self.tree_calib.heading("id", text="ID"); self.tree_calib.column("id", width=40)
        self.tree_calib.heading("img", text="Img"); self.tree_calib.column("img", width=150)
        for c in ["x","y","z","rx","ry","rz"]: self.tree_calib.heading(c, text=c.upper()); self.tree_calib.column(c, width=60)
        self.tree_calib.heading("stat", text="Stat"); self.tree_calib.column("stat", width=60)
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.on_edit_pose)
        
        self.txt_calib_res = tk.Text(p, height=10, bg="#e0e0e0"); self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, p):
        paned = tk.PanedWindow(p, orient=tk.HORIZONTAL); paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        f_left = tk.LabelFrame(paned, text="机械臂通讯", fg="blue"); paned.add(f_left, minsize=400)
        row=0
        def ae(par, l, g, k, w=25):
            nonlocal row; tk.Label(par, text=l).grid(row=row, column=0, sticky="e", padx=5)
            v=tk.StringVar(value=str(self.cfg[g].get(k,""))); tk.Entry(par, textvariable=v, width=w).grid(row=row, column=1, sticky="w", padx=5)
            setattr(self, f"var_{g}_{k}", v); row+=1
        ae(f_left, "监听 IP:", "robot_net", "bind_ip"); tk.Label(f_left, text="(务必填本机IP 或 0.0.0.0)", fg="red").grid(row=row-1, column=2, sticky="w")
        ae(f_left, "监听端口:", "robot_net", "port")
        row+=1; ae(f_left, "触发指令:", "commands", "trigger"); ae(f_left, "失败返回:", "commands", "error"); ae(f_left, "成功前缀:", "commands", "success_prefix")
        row+=1; ae(f_left, "保存路径:", "paths", "save_dir", w=35)
        
        f_right = tk.LabelFrame(paned, text="相机连接", fg="green"); paned.add(f_right, minsize=400)
        r_row=0
        def ac(l, k, w=25):
            nonlocal r_row; tk.Label(f_right, text=l).grid(row=r_row, column=0, sticky="e", padx=5)
            v=tk.StringVar(value=str(self.cfg["camera"].get(k,""))); tk.Entry(f_right, textvariable=v, width=w).grid(row=r_row, column=1, sticky="w", padx=5)
            setattr(self, f"var_camera_{k}", v); r_row+=1
        tk.Label(f_right, text="模式:").grid(row=r_row, column=0, sticky="e"); self.var_camera_mode=tk.StringVar(value=self.cfg["camera"].get("mode","sdk"))
        mf = tk.Frame(f_right); mf.grid(row=r_row, column=1, sticky="w")
        tk.Radiobutton(mf, text="SDK直连", variable=self.var_camera_mode, value="sdk").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="IP/RTSP", variable=self.var_camera_mode, value="ip").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="Index", variable=self.var_camera_mode, value="index").pack(side=tk.LEFT)
        r_row+=1
        if not HAS_HIK_SDK: tk.Label(f_right, text="[SDK未就绪]", fg="red").grid(row=r_row, column=1); r_row+=1
        ac("相机 IP:", "target_ip")
        tk.Button(f_right, text="扫描 IP", command=self.scan_ip, bg="#b3e5fc").grid(row=r_row, column=1, sticky="w"); r_row+=1
        ac("相机 Index:", "index"); r_row+=1; ac("Fx:", "fx"); ac("Fy:", "fy"); ac("Cx:", "cx"); ac("Cy:", "cy")
        tk.Button(p, text="💾 保存配置", command=self.save_config, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=20, pady=10)

    # --- 核心逻辑 ---
    def log(self, m): 
        try: self.txt_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {m}\n"); self.txt_log.see(tk.END)
        except: pass
    def draw_placeholder(self): self.canvas.delete("all"); self.canvas.create_text(400,300,text="等待相机连接...",fill="gray", font=("Arial", 16))

    def update_cfg_from_ui(self):
        # 自动更新配置
        self.cfg["robot_net"]["bind_ip"] = self.var_robot_net_bind_ip.get()
        self.cfg["robot_net"]["port"] = int(self.var_robot_net_port.get())
        self.cfg["paths"]["save_dir"] = self.var_paths_save_dir.get()
        self.cfg["commands"]["trigger"] = self.var_commands_trigger.get()
        self.cfg["commands"]["error"] = self.var_commands_error.get()
        self.cfg["commands"]["success_prefix"] = self.var_commands_success_prefix.get()
        self.cfg["camera"]["mode"] = self.var_camera_mode.get()
        self.cfg["camera"]["target_ip"] = self.var_camera_target_ip.get()
        try: self.cfg["camera"]["index"] = int(self.var_camera_index.get())
        except: pass
        for k in ["fx","fy","cx","cy"]:
            try: self.cfg["camera"][k] = float(getattr(self, f"var_camera_{k}").get())
            except: pass

    def scan_ip(self):
        d = self.scanner.scan()
        if d: self.var_camera_target_ip.set(d[0]['ip']); messagebox.showinfo("成功", f"发现 IP: {d[0]['ip']}")
        else: messagebox.showwarning("失败", "未发现设备")

    # --- 独立控制：相机 ---
    def toggle_cam(self):
        if not self.cam_running:
            self.update_cfg_from_ui()
            mode = self.cfg["camera"]["mode"]
            try:
                if mode == "sdk":
                    if not HAS_HIK_SDK: raise Exception("SDK库未加载")
                    self.cap = HikCameraWrapper(); self.cap.open_by_ip(self.cfg["camera"]["target_ip"])
                else:
                    src = int(self.cfg["camera"]["index"]) if mode=="index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src,str) else cv2.VideoCapture(src)
                
                if hasattr(self.cap, 'isOpened') and not self.cap.isOpened(): raise Exception("打开失败")
                self.cam_running = True
                self.btn_cam.config(text="断开相机", bg="#f44336")
                self.lbl_cam.config(text="运行中", fg="green")
                threading.Thread(target=self.cam_loop, daemon=True).start()
                self.log("相机已连接")
            except Exception as e: messagebox.showerror("相机错误", str(e))
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
                    if hasattr(self.cap, 'read'): ret, f = self.cap.read()
                    else: ret, f = False, None
                    if ret:
                        with self.lock: self.current_frame = f
                        if int(time.time()*100)%5==0: self.root.after(0, self.update_disp, f)
            except: pass
            time.sleep(0.01)

    # --- 独立控制：TCP ---
    def toggle_tcp(self):
        if not self.tcp_running:
            try:
                ip = self.cfg["robot_net"]["bind_ip"]; port = int(self.cfg["robot_net"]["port"])
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    self.server_socket.bind((ip, port))
                except OSError as e:
                    if e.errno == 10049: # 地址无效
                        if messagebox.askyesno("配置错误", f"无法监听 IP {ip} (本机没有这个IP)。\n是否自动改为 0.0.0.0 (监听所有)?"):
                            ip = "0.0.0.0"; self.var_robot_net_bind_ip.set("0.0.0.0"); self.server_socket.bind((ip, port))
                        else: return
                    else: raise e
                
                self.server_socket.listen(1); self.server_socket.settimeout(0.5)
                self.tcp_running = True
                self.btn_tcp.config(text="停止监听", bg="#f44336")
                self.lbl_tcp.config(text=f"监听中 {ip}:{port}", fg="green")
                threading.Thread(target=self.net_loop, daemon=True).start()
                self.log("TCP 服务已启动")
            except Exception as e: messagebox.showerror("TCP错误", str(e)); if self.server_socket: self.server_socket.close()
        else:
            self.tcp_running = False
            if self.server_socket: self.server_socket.close()
            self.btn_tcp.config(text="启动监听", bg="#e0e0e0")
            self.lbl_tcp.config(text="已停止", fg="red")
            self.log("TCP 服务已停止")

    def net_loop(self):
        while self.tcp_running:
            try:
                try: cl, ad = self.server_socket.accept()
                except socket.timeout: continue
                self.client_socket = cl; self.log(f"机械臂连接: {ad}")
                while self.tcp_running:
                    try:
                        d = cl.recv(1024)
                        if not d: break
                        msg = d.decode().strip(); self.log(f"指令: {msg}")
                        if msg == self.cfg["commands"]["trigger"]:
                            res = self.process()
                            cl.send(res.encode()); self.log(f"回复: {res}")
                    except: break
                self.client_socket = None; self.log("机械臂断开")
            except: pass

    # --- 通用处理 ---
    def manual_trigger(self):
        if not self.cam_running: 
            if messagebox.askyesno("提示", "相机未连接，是否尝试连接？"): self.toggle_cam()
            else: return
            if not self.cam_running: return # 连接失败
        self.log("手动触发拍照...")
        res = self.process()
        self.log(f"手动结果: {res}")
        messagebox.showinfo("结果", f"处理完成: {res}")

    def process(self):
        f = None
        with self.lock: f = self.current_frame.copy() if self.current_frame is not None else None
        if f is None: return self.cfg["commands"]["error"] + ",NoImage"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f"); p = os.path.join(self.cfg["paths"]["save_dir"], f"IMG_{ts}.jpg")
        mode = self.work_mode.get(); res = self.cfg["commands"]["error"]
        
        if mode == "TEST":
            K = np.array([[self.cfg["camera"]["fx"],0,self.cfg["camera"]["cx"]],[0,self.cfg["camera"]["fy"],self.cfg["camera"]["cy"]],[0,0,1]], float)
            D = np.array(self.cfg["camera"]["dist"], float)
            ret, _, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(f)
            if ret and pts is not None:
                op = np.array([[-50,50,0],[50,50,0],[50,-50,0],[-50,-50,0]], float)
                succ, rvec, tvec = cv2.solvePnP(op, pts[0], K, D)
                if succ:
                    eu = R.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler('xyz', degrees=True)
                    cv2.drawFrameAxes(f, K, D, rvec, tvec, 50)
                    pre = self.cfg["commands"]["success_prefix"]; sep = self.cfg["commands"]["separator"]
                    res = f"{pre}{sep}{tvec[0][0]:.2f}{sep}{tvec[1][0]:.2f}{sep}{tvec[2][0]:.2f}{sep}{eu[0]:.2f}{sep}{eu[1]:.2f}{sep}{eu[2]:.2f}"
        elif mode == "CALIB":
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            r,c = self.cfg["calibration"]["rows"], self.cfg["calibration"]["cols"]
            ret, crn = cv2.findCirclesGrid(g, (c,r), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            if ret:
                cv2.drawChessboardCorners(f, (c,r), crn, ret)
                self.calib_data_list.append({"id":len(self.calib_data_list)+1, "img_path":p, "corners":crn, "robot_pose":None})
                self.root.after(0, self.refresh_calib_list); self.save_calib_data()
                res = self.cfg["commands"]["success_prefix"]
        
        cv2.imwrite(p, f); self.log(f"已存图: {os.path.basename(p)}")
        return res

    def update_disp(self, img):
        if not self.cam_running: return
        di = img.copy()
        if self.undistort_view.get():
            try:
                K = np.array([[self.cfg["camera"]["fx"],0,self.cfg["camera"]["cx"]],[0,self.cfg["camera"]["fy"],self.cfg["camera"]["cy"]],[0,0,1]], float)
                D = np.array(self.cfg["camera"]["dist"], float); di = cv2.undistort(di, K, D)
            except: pass
        if self.show_focus_score.get():
            s = cv2.Laplacian(cv2.cvtColor(di, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            cv2.putText(di, f"Focus: {int(s)}", (20,40), 0, 1, (0,255,0), 2)
        if self.show_crosshair.get():
            h,w = di.shape[:2]; cx,cy = w//2,h//2
            cv2.line(di, (cx-50,cy), (cx+50,cy), (0,0,255), 2); cv2.line(di, (cx,cy-50), (cx,cy+50), (0,0,255), 2)
        
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height(); sc = min(cw/di.shape[1], ch/di.shape[0])*0.95
        di = cv2.resize(di, (int(di.shape[1]*sc), int(di.shape[0]*sc))); di = cv2.cvtColor(di, cv2.COLOR_BGR2RGB)
        p = ImageTk.PhotoImage(Image.fromarray(di))
        self.canvas.create_image(cw//2, ch//2, image=p, anchor=tk.CENTER); self.canvas.image=p

    def refresh_calib_list(self):
        for i in self.tree_calib.get_children(): self.tree_calib.delete(i)
        for d in self.calib_data_list:
            p = d["robot_pose"]; v = [d["id"], os.path.basename(d["img_path"])]
            if p: v.extend([f"{x:.1f}" for x in p]); v.append("OK")
            else: v.extend(["-"]*6); v.append("Wait")
            self.tree_calib.insert("", "end", values=v)

    def on_edit_pose_btn(self): self.on_edit_pose(None)
    def on_edit_pose(self, e):
        s = self.tree_calib.selection()
        if not s: return
        idx = self.tree_calib.index(s[0]); w = tk.Toplevel(self.root); w.title("Pose")
        ents = []
        for i, l in enumerate(["X","Y","Z","Rx","Ry","Rz"]):
            tk.Label(w, text=l).grid(row=0, column=i); e = tk.Entry(w, width=8); e.grid(row=1, column=i); ents.append(e)
            if self.calib_data_list[idx]["robot_pose"]: e.insert(0, str(self.calib_data_list[idx]["robot_pose"][i]))
        def cf():
            try: self.calib_data_list[idx]["robot_pose"] = [float(x.get()) for x in ents]; self.refresh_calib_list(); self.save_calib_data(); w.destroy()
            except: pass
        tk.Button(w, text="Save", command=cf).grid(row=2, columnspan=6)

    def run_hand_eye_calc(self):
        v = [d for d in self.calib_data_list if d["robot_pose"]]
        if len(v)<3: return
        try:
            Rg, Tg, Rc, Tc = [], [], [], []
            K = np.array([[self.cfg["camera"]["fx"],0,self.cfg["camera"]["cx"]],[0,self.cfg["camera"]["fy"],self.cfg["camera"]["cy"]],[0,0,1]], float)
            op = np.zeros((self.cfg["calibration"]["rows"]*self.cfg["calibration"]["cols"],3), np.float32)
            op[:,:2] = np.mgrid[0:self.cfg["calibration"]["cols"],0:self.cfg["calibration"]["rows"]].T.reshape(-1,2)*self.cfg["calibration"]["spacing"]
            for d in v:
                _, rv, tv = cv2.solvePnP(op, d["corners"], K, np.zeros(5))
                Rc.append(cv2.Rodrigues(rv)[0]); Tc.append(tv)
                Tg.append(np.array(d["robot_pose"][:3]).reshape(3,1)); Rg.append(R.from_euler('xyz', d["robot_pose"][3:], degrees=True).as_matrix())
            rc, tc = cv2.calibrateHandEye(Rg, Tg, Rc, Tc, method=cv2.CALIB_HAND_EYE_TSAI)
            self.txt_calib_res.delete(1.0, tk.END); self.txt_calib_res.insert(tk.END, f"Result:\nXYZ: {tc.flatten()}\nEuler: {R.from_matrix(rc).as_euler('xyz', degrees=True)}")
        except Exception as e: self.txt_calib_res.insert(tk.END, str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalVisionServer(root)
    root.mainloop()
