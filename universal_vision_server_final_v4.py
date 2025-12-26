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
# 模块一：海康 SDK 强力加载器 (Auto SDK Loader with DLL Fix)
# =============================================================================
HAS_HIK_SDK = False
SDK_ERROR_MSG = ""

try:
    # 1. 核心修复：主动寻找并加载 MVS Runtime DLL
    # Python 3.8+ 需要使用 os.add_dll_directory 才能加载非系统目录的 DLL
    possible_dll_paths = [
        r"C:\Program Files (x86)\MVS\Runtime\Win64_x64",
        r"C:\Program Files (x86)\MVS\Runtime\Win32_x86",
        r"D:\MVS\Runtime\Win64_x64",
        r"D:\MVS\Runtime\Win32_x86",
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_x86"
    ]
    
    dll_found = False
    for dll_path in possible_dll_paths:
        if os.path.exists(dll_path):
            # 将 DLL 路径加入环境变量
            os.environ['PATH'] = dll_path + ";" + os.environ['PATH']
            # 针对 Python 3.8+ 的额外处理
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(dll_path)
            print(f"[系统] 已加载海康驱动路径: {dll_path}")
            dll_found = True
            break
            
    if not dll_found:
        print("[警告] 未在标准路径找到 MVS Runtime，如果报错请检查 MVS 是否安装正确")

    # 2. 确保当前目录在 sys.path 中，以便能 import MvImport
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    # 3. 尝试导入 SDK
    # 注意：文件夹结构必须是 当前目录/MvImport/MvCameraControl_class.py
    if os.path.exists(os.path.join(current_dir, "MvImport")):
        from MvImport.MvCameraControl_class import *
        HAS_HIK_SDK = True
        print("[系统] 海康 SDK Python 接口加载成功！")
    else:
        raise ImportError("当前目录下未找到 'MvImport' 文件夹")

except Exception as e:
    SDK_ERROR_MSG = str(e)
    print(f"[系统错误] SDK 加载失败，详细原因: {e}")
    print("建议：请确保 'MvImport' 文件夹在当前脚本旁边，且电脑已安装海康 MVS 客户端。")
    HAS_HIK_SDK = False

# =============================================================================
# 模块二：海康相机驱动封装
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
        tlayerType = MV_GIGE_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0: raise Exception(f"枚举设备失败: {ret}")
        if deviceList.nDeviceNum == 0: raise Exception("未发现 GigE 相机 (请检查网线/防火墙)")
            
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
        
        if target_device is None: raise Exception(f"未找到 IP 为 {ip} 的相机")
            
        self.handle = MvCamera()
        ret = self.handle.MV_CC_CreateHandle(target_device)
        if ret != 0: raise Exception(f"创建句柄失败: {ret}")
            
        ret = self.handle.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0: raise Exception(f"打开设备失败: {ret}")
            
        nPacketSize = self.handle.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0: self.handle.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            
        stParam = MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_INTVALUE))
        ret = self.handle.MV_CC_GetIntValue("PayloadSize", stParam)
        self.n_payload_size = stParam.nCurValue
        self.data_buf = (ctypes.c_ubyte * self.n_payload_size)()
        
        ret = self.handle.MV_CC_StartGrabbing()
        if ret != 0: raise Exception(f"取流失败: {ret}")
        self.is_opened = True

    def read(self):
        if not self.is_opened: return False, None
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ctypes.memset(ctypes.byref(stFrameInfo), 0, ctypes.sizeof(MV_FRAME_OUT_INFO_EX))
        
        ret = self.handle.MV_CC_GetOneFrameTimeout(ctypes.byref(self.data_buf), self.n_payload_size, stFrameInfo, 1000)
        if ret == 0:
            h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
            data = np.frombuffer(self.data_buf, count=int(self.n_payload_size), dtype=np.uint8)
            
            if PixelType_Gvsp_Mono8 == stFrameInfo.enPixelType:
                return True, cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2BGR)
            elif PixelType_Gvsp_BayerGR8 == stFrameInfo.enPixelType:
                return True, cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_BayerGR2BGR)
            elif PixelType_Gvsp_BayerRG8 == stFrameInfo.enPixelType:
                return True, cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_BayerRG2BGR)
            
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
                    if len(data) > 0 and addr[0] not in [d['ip'] for d in devices]:
                        devices.append({'ip': addr[0]})
                except socket.timeout: break
            sock.close()
        except: pass
        return devices

# =============================================================================
# 模块四：主程序逻辑
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
        self.root.title("通用视觉服务器系统 - 终极修复版 V4")
        self.root.geometry("1400x950")
        
        self.config_file = "vision_config.json"
        self.cfg = self.load_config()
        
        self.server_socket = None; self.client_socket = None
        self.is_running = False; self.cap = None; self.current_frame = None
        self.lock = threading.Lock()
        
        self.scanner = GigEScanner()
        self.calib_data_list = []
        
        self.setup_ui()
        if not os.path.exists(self.cfg["paths"]["save_dir"]): os.makedirs(self.cfg["paths"]["save_dir"])

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for k,v in DEFAULT_CONFIG.items():
                        if k not in data: data[k] = v
                    return data
            except: pass
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        self.update_cfg_from_ui()
        with open(self.config_file, 'w') as f: json.dump(self.cfg, f, indent=4)
        messagebox.showinfo("提示", "配置已保存")

    def setup_ui(self):
        nb = ttk.Notebook(self.root); nb.pack(fill=tk.BOTH, expand=True)
        fr = tk.Frame(nb); nb.add(fr, text="1. 运行监控"); self.setup_run_ui(fr)
        fc = tk.Frame(nb); nb.add(fc, text="2. 手眼标定集成"); self.setup_calib_ui(fc)
        fg = tk.Frame(nb); nb.add(fg, text="3. 系统配置"); self.setup_config_ui(fg)

    def setup_run_ui(self, p):
        paned = tk.PanedWindow(p, orient=tk.HORIZONTAL); paned.pack(fill=tk.BOTH, expand=True)
        left = tk.Frame(paned, width=420, bg="#f0f0f0"); left.pack_propagate(False); paned.add(left, minsize=420)
        
        sf = tk.LabelFrame(left, text="服务器控制", font=("bold", 10)); sf.pack(fill=tk.X, padx=5, pady=5)
        self.btn_start = tk.Button(sf, text="启动服务 (Start)", command=self.toggle_server, bg="#4caf50", fg="white", height=2, font=("bold", 11)); self.btn_start.pack(fill=tk.X, padx=5, pady=5)
        stf = tk.Frame(sf); stf.pack(fill=tk.X, padx=5)
        self.lbl_status = tk.Label(stf, text="状态: 已停止", fg="red"); self.lbl_status.pack(side=tk.LEFT)
        self.lbl_client = tk.Label(stf, text="客户端: 无连接", fg="gray"); self.lbl_client.pack(side=tk.RIGHT)
        
        mf = tk.LabelFrame(left, text="工作模式", font=("bold", 10)); mf.pack(fill=tk.X, padx=5, pady=5)
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(mf, text="测试模式 (回传坐标)", variable=self.work_mode, value="TEST", command=self.on_mode_change).pack(anchor="w", padx=5)
        tk.Radiobutton(mf, text="标定模式 (仅存图)", variable=self.work_mode, value="CALIB", command=self.on_mode_change).pack(anchor="w", padx=5)
        
        lf = tk.LabelFrame(left, text="日志", font=("bold", 10)); lf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(lf, font=("Consolas", 9)); self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(paned, bg="#222"); paned.add(self.canvas, stretch="always")
        self.draw_placeholder()

    def setup_calib_ui(self, p):
        top = tk.Frame(p, pady=5); top.pack(fill=tk.X)
        tk.Button(top, text="刷新列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="✎ 手动录入/修改坐标", command=self.on_edit_pose_btn, bg="#2196f3", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="▶ 计算手眼矩阵", command=self.run_hand_eye_calc, bg="orange").pack(side=tk.LEFT, padx=10)
        
        cols = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "ok")
        self.tree_calib = ttk.Treeview(p, columns=cols, show="headings")
        for c in cols: self.tree_calib.heading(c, text=c); self.tree_calib.column(c, width=60)
        self.tree_calib.column("img", width=150); self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.on_edit_pose)
        
        self.txt_calib_res = tk.Text(p, height=10, bg="#e0e0e0", font=("Consolas", 10)); self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, p):
        paned = tk.PanedWindow(p, orient=tk.HORIZONTAL); paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        f_left = tk.LabelFrame(paned, text="【机械臂通讯】", fg="blue"); paned.add(f_left, minsize=400)
        row=0
        def ae(par, l, g, k, w=25):
            nonlocal row; tk.Label(par, text=l).grid(row=row, column=0, sticky="e", padx=5)
            v=tk.StringVar(value=str(self.cfg[g].get(k,""))); tk.Entry(par, textvariable=v, width=w).grid(row=row, column=1, sticky="w", padx=5)
            setattr(self, f"var_{g}_{k}", v); row+=1
        ae(f_left, "监听 IP:", "robot_net", "bind_ip"); ae(f_left, "监听端口:", "robot_net", "port")
        row+=1; ae(f_left, "触发拍照:", "commands", "trigger"); ae(f_left, "失败返回:", "commands", "error"); ae(f_left, "成功前缀:", "commands", "success_prefix")
        row+=1; ae(f_left, "保存路径:", "paths", "save_dir", w=35)
        
        f_right = tk.LabelFrame(paned, text="【相机连接】", fg="green"); paned.add(f_right, minsize=400)
        r_row=0
        def ac(l, k, w=25):
            nonlocal r_row; tk.Label(f_right, text=l).grid(row=r_row, column=0, sticky="e", padx=5)
            v=tk.StringVar(value=str(self.cfg["camera"].get(k,""))); tk.Entry(f_right, textvariable=v, width=w).grid(row=r_row, column=1, sticky="w", padx=5)
            setattr(self, f"var_camera_{k}", v); r_row+=1
        tk.Label(f_right, text="模式:").grid(row=r_row, column=0, sticky="e"); self.var_camera_mode=tk.StringVar(value=self.cfg["camera"].get("mode","sdk"))
        mf = tk.Frame(f_right); mf.grid(row=r_row, column=1, sticky="w")
        tk.Radiobutton(mf, text="海康 SDK 直连", variable=self.var_camera_mode, value="sdk").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="IP/RTSP", variable=self.var_camera_mode, value="ip").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="Index", variable=self.var_camera_mode, value="index").pack(side=tk.LEFT)
        r_row+=1
        
        if not HAS_HIK_SDK: tk.Label(f_right, text=f"[警告] SDK不可用: {SDK_ERROR_MSG[:20]}...", fg="red").grid(row=r_row, column=1, sticky="w"); r_row+=1
        
        ac("相机 IP:", "target_ip")
        sf = tk.Frame(f_right); sf.grid(row=r_row, column=1, sticky="w")
        tk.Button(sf, text="扫描 IP", command=self.scan_ip, bg="#b3e5fc").pack(side=tk.LEFT)
        self.lbl_scan_res = tk.Label(sf, text="", fg="blue"); self.lbl_scan_res.pack(side=tk.LEFT, padx=5)
        r_row+=1
        ac("相机 Index:", "index"); r_row+=1
        ac("Fx:", "fx"); ac("Fy:", "fy"); ac("Cx:", "cx"); ac("Cy:", "cy")
        tk.Button(f_right, text="测试相机", command=self.test_camera, bg="#e0e0e0").grid(row=r_row, column=1, pady=10)
        tk.Button(p, text="保存全部配置", command=self.save_config, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=20, pady=10)

    # --- 逻辑 ---
    def on_mode_change(self): self.log(f"模式: {self.work_mode.get()}")
    def log(self, m): 
        try: self.txt_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {m}\n"); self.txt_log.see(tk.END)
        except: pass
    def draw_placeholder(self): self.canvas.delete("all"); self.canvas.create_text(400,300,text="等待视频源...",fill="gray", font=("Arial", 16))

    def update_cfg_from_ui(self):
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
        if d: self.var_camera_target_ip.set(d[0]['ip']); self.lbl_scan_res.config(text=f"发现: {d[0]['ip']}")
        else: messagebox.showwarning("提示", "未扫描到设备")

    def test_camera(self):
        self.update_cfg_from_ui(); mode = self.cfg["camera"]["mode"]
        try:
            if mode == "sdk":
                if not HAS_HIK_SDK: raise Exception(f"SDK未加载: {SDK_ERROR_MSG}")
                c = HikCameraWrapper(); c.open_by_ip(self.cfg["camera"]["target_ip"])
                ret, f = c.read(); c.release()
                if ret: messagebox.showinfo("成功", f"SDK 连接成功!\n{f.shape}")
                else: raise Exception("SDK 读取失败")
            else:
                src = int(self.cfg["camera"]["index"]) if mode=="index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src,str) else cv2.VideoCapture(src)
                if not cap.isOpened(): raise Exception("打开失败")
                ret, f = cap.read(); cap.release()
                if ret: messagebox.showinfo("成功", "连接成功")
                else: raise Exception("无图像")
        except Exception as e: messagebox.showerror("失败", str(e))

    def toggle_server(self):
        if not self.is_running:
            try:
                ip=self.cfg["robot_net"]["bind_ip"]; port=int(self.cfg["robot_net"]["port"])
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((ip, port)); self.server_socket.listen(1); self.server_socket.settimeout(0.5)
                
                self.update_cfg_from_ui(); mode = self.cfg["camera"]["mode"]
                if mode == "sdk":
                    if not HAS_HIK_SDK: raise Exception("SDK未加载")
                    self.cap = HikCameraWrapper(); self.cap.open_by_ip(self.cfg["camera"]["target_ip"])
                else:
                    src = int(self.cfg["camera"]["index"]) if mode=="index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src,str) else cv2.VideoCapture(src)
                
                self.is_running = True
                self.btn_start.config(text="停止服务", bg="#f44336"); self.lbl_status.config(text="监听中", fg="green")
                threading.Thread(target=self.net_loop, daemon=True).start()
                threading.Thread(target=self.cam_loop, daemon=True).start()
                self.log("服务已启动")
            except Exception as e: messagebox.showerror("启动错误", str(e))
        else:
            self.is_running = False
            if self.server_socket: self.server_socket.close()
            if self.cap: self.cap.release()
            self.btn_start.config(text="启动服务", bg="#4caf50"); self.lbl_status.config(text="停止", fg="red")
            self.draw_placeholder(); self.log("已停止")

    def net_loop(self):
        while self.is_running:
            try:
                try: cl, ad = self.server_socket.accept()
                except socket.timeout: continue
                self.client_socket = cl; self.log(f"连接: {ad}"); self.root.after(0, lambda:self.lbl_client.config(text=f"{ad}", fg="blue"))
                while self.is_running:
                    try:
                        d = cl.recv(1024)
                        if not d: break
                        msg = d.decode().strip(); self.log(f"指令: {msg}")
                        if msg == self.cfg["commands"]["trigger"]:
                            res = self.process()
                            cl.send(res.encode()); self.log(f"回复: {res}")
                    except: break
                self.client_socket = None; self.root.after(0, lambda:self.lbl_client.config(text="无连接", fg="gray"))
            except: pass

    def cam_loop(self):
        while self.is_running:
            try:
                if self.cap:
                    if hasattr(self.cap, 'read'): ret, f = self.cap.read()
                    else: ret, f = False, None
                    if ret:
                        with self.lock: self.current_frame = f
                        if int(time.time()*100)%5==0: self.root.after(0, self.update_disp, f)
            except: pass
            time.sleep(0.01)

    def update_disp(self, img):
        if not self.is_running: return
        h,w = img.shape[:2]; cw=self.canvas.winfo_width(); ch=self.canvas.winfo_height()
        if cw<10: cw=800
        sc = min(cw/w, ch/h)*0.95; nh,nw = int(h*sc), int(w*sc)
        i = cv2.resize(img, (nw,nh)); i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        p = ImageTk.PhotoImage(Image.fromarray(i))
        self.canvas.create_image(cw//2, ch//2, image=p, anchor=tk.CENTER); self.canvas.image=p

    def process(self):
        f = None
        with self.lock: f = self.current_frame.copy() if self.current_frame is not None else None
        if f is None: return self.cfg["commands"]["error"]
        
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
                self.root.after(0, self.refresh_calib_list)
                res = self.cfg["commands"]["success_prefix"]

        cv2.imwrite(p, f); self.log(f"保存: {os.path.basename(p)}")
        return res

    def refresh_calib_list(self):
        for i in self.tree_calib.get_children(): self.tree_calib.delete(i)
        for d in self.calib_data_list: self.tree_calib.insert("", "end", values=(d["id"], os.path.basename(d["img_path"]), *(d["robot_pose"] if d["robot_pose"] else ["-"]*6), "Yes"))

    def on_edit_pose_btn(self): self.on_edit_pose(None)
    def on_edit_pose(self, e):
        sel = self.tree_calib.selection()
        if not sel: return
        idx = self.tree_calib.index(sel[0]); w = tk.Toplevel(self.root); w.title("Pose")
        ents = []
        for i, l in enumerate(["X","Y","Z","Rx","Ry","Rz"]):
            tk.Label(w, text=l).grid(row=0, column=i); e = tk.Entry(w, width=8); e.grid(row=1, column=i); ents.append(e)
            if self.calib_data_list[idx]["robot_pose"]: e.insert(0, str(self.calib_data_list[idx]["robot_pose"][i]))
        def cf():
            try: self.calib_data_list[idx]["robot_pose"] = [float(x.get()) for x in ents]; self.refresh_calib_list(); w.destroy()
            except: pass
        tk.Button(w, text="OK", command=cf).grid(row=2, columnspan=6)

    def run_hand_eye_calc(self):
        valid = [d for d in self.calib_data_list if d["robot_pose"]]
        if len(valid)<3: return
        try:
            Rg, Tg, Rc, Tc = [], [], [], []
            K = np.array([[self.cfg["camera"]["fx"],0,self.cfg["camera"]["cx"]],[0,self.cfg["camera"]["fy"],self.cfg["camera"]["cy"]],[0,0,1]], float)
            op = np.zeros((self.cfg["calibration"]["rows"]*self.cfg["calibration"]["cols"],3), np.float32)
            op[:,:2] = np.mgrid[0:self.cfg["calibration"]["cols"],0:self.cfg["calibration"]["rows"]].T.reshape(-1,2)*self.cfg["calibration"]["spacing"]
            for d in valid:
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
