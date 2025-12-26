import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import json
import os
import socket
import struct
import threading
import time
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# --- 默认配置 ---
DEFAULT_CONFIG = {
    "robot_net": {
        "bind_ip": "0.0.0.0",
        "port": 8000
    },
    "camera": {
        "mode": "ip",      # ip 或 index
        "target_ip": "",
        "index": 0,
        "width": 1280,
        "height": 960,
        "fx": 2000.0, "fy": 2000.0, "cx": 640.0, "cy": 480.0,
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

class GigEScanner:
    """简易的 GigE Vision 设备发现工具"""
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
                except socket.timeout:
                    break
            sock.close()
        except Exception as e:
            print(f"Scan error: {e}")
        return devices

class UniversalVisionServer:
    def __init__(self, root):
        self.root = root
        self.root.title("通用视觉服务器系统 (Universal Vision Server) - 最终修正版")
        self.root.geometry("1400x950")
        
        self.config_file = "vision_config.json"
        self.cfg = self.load_config()
        
        # 运行时状态
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.cap = None
        self.current_frame = None
        self.lock = threading.Lock()
        
        self.scanner = GigEScanner()
        self.calib_data_list = []
        
        # UI 构建
        self.setup_ui()
        
        # 初始化目录
        try:
            if not os.path.exists(self.cfg["paths"]["save_dir"]):
                os.makedirs(self.cfg["paths"]["save_dir"])
        except:
            pass

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    # 简单合并默认配置，防止缺字段
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
            messagebox.showerror("错误", f"保存失败: {e}")

    # -------------------------------------------------------------------------
    # UI 构建
    # -------------------------------------------------------------------------
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        frame_run = tk.Frame(notebook)
        notebook.add(frame_run, text="1. 运行监控")
        self.setup_run_ui(frame_run)
        
        frame_calib = tk.Frame(notebook)
        notebook.add(frame_calib, text="2. 手眼标定集成")
        self.setup_calib_ui(frame_calib)
        
        frame_cfg = tk.Frame(notebook)
        notebook.add(frame_cfg, text="3. 系统配置")
        self.setup_config_ui(frame_cfg)

    def setup_run_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        left = tk.Frame(paned, width=420, bg="#f0f0f0")
        left.pack_propagate(False)
        paned.add(left, minsize=420)
        
        # 服务器控制
        s_frame = tk.LabelFrame(left, text="服务器控制", font=("bold", 10))
        s_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_start = tk.Button(s_frame, text="启动服务 (Start Server)", command=self.toggle_server, bg="#4caf50", fg="white", height=2, font=("bold", 11))
        self.btn_start.pack(fill=tk.X, padx=5, pady=5)
        
        status_frame = tk.Frame(s_frame)
        status_frame.pack(fill=tk.X, padx=5)
        self.lbl_status = tk.Label(status_frame, text="状态: 已停止", fg="red", font=("Arial", 10))
        self.lbl_status.pack(side=tk.LEFT)
        self.lbl_client = tk.Label(status_frame, text="客户端: 无连接", fg="gray", font=("Arial", 10))
        self.lbl_client.pack(side=tk.RIGHT)
        
        # 模式选择
        m_frame = tk.LabelFrame(left, text="工作模式 (Work Mode)", font=("bold", 10))
        m_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(m_frame, text="测试模式 (识别二维码，回传坐标)", variable=self.work_mode, value="TEST", command=self.on_mode_change).pack(anchor="w", padx=5)
        tk.Radiobutton(m_frame, text="标定模式 (识别标定板，仅存图)", variable=self.work_mode, value="CALIB", command=self.on_mode_change).pack(anchor="w", padx=5)
        
        # 日志
        l_frame = tk.LabelFrame(left, text="通讯日志 (Logs)", font=("bold", 10))
        l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(l_frame, font=("Consolas", 9), state=tk.NORMAL)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        # 右侧图像
        self.canvas = tk.Canvas(paned, bg="#222")
        paned.add(self.canvas, stretch="always")
        self.draw_placeholder()

    def setup_calib_ui(self, parent):
        top = tk.Frame(parent)
        top.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(top, text="刷新数据列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="计算手眼矩阵 (Calculate)", command=self.run_hand_eye_calc, bg="orange").pack(side=tk.LEFT, padx=10)
        
        columns = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "ok")
        self.tree_calib = ttk.Treeview(parent, columns=columns, show="headings")
        for col in columns: 
            self.tree_calib.heading(col, text=col)
            self.tree_calib.column(col, width=60)
        self.tree_calib.column("img", width=200)
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.on_edit_pose)
        
        self.txt_calib_res = tk.Text(parent, height=10, bg="#e0e0e0", font=("Consolas", 10))
        self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === 左侧：机械臂通讯 ===
        f_left = tk.LabelFrame(paned, text="【机械臂通讯 (TCP Server)】", font=("bold", 11), fg="blue")
        paned.add(f_left, minsize=450)
        
        row = 0
        def add_entry(p, label, key_group, key_item, w=25):
            nonlocal row
            tk.Label(p, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=5)
            var = tk.StringVar(value=str(self.cfg[key_group].get(key_item, "")))
            entry = tk.Entry(p, textvariable=var, width=w)
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
        add_entry(f_left, "保存路径:", "paths", "save_dir", w=35)

        # === 右侧：相机配置 ===
        f_right = tk.LabelFrame(paned, text="【海康相机连接 (Camera)】", font=("bold", 11), fg="green")
        paned.add(f_right, minsize=450)
        
        r_row = 0
        def add_cam_entry(label, key_item, w=25):
            nonlocal r_row
            tk.Label(f_right, text=label).grid(row=r_row, column=0, sticky="e", padx=5, pady=5)
            var = tk.StringVar(value=str(self.cfg["camera"].get(key_item, "")))
            entry = tk.Entry(f_right, textvariable=var, width=w)
            entry.grid(row=r_row, column=1, sticky="w", padx=5, pady=5)
            setattr(self, f"var_camera_{key_item}", var)
            r_row += 1

        tk.Label(f_right, text="连接模式:").grid(row=r_row, column=0, sticky="e", padx=5, pady=5)
        self.var_camera_mode = tk.StringVar(value=self.cfg["camera"].get("mode", "ip"))
        mf = tk.Frame(f_right)
        mf.grid(row=r_row, column=1, sticky="w")
        tk.Radiobutton(mf, text="IP 连接 (推荐)", variable=self.var_camera_mode, value="ip").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="索引 Index", variable=self.var_camera_mode, value="index").pack(side=tk.LEFT)
        r_row += 1

        # IP
        tk.Label(f_right, text="--- IP 模式设置 ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5); r_row+=1
        add_cam_entry("相机 IP:", "target_ip")
        
        scan_f = tk.Frame(f_right)
        scan_f.grid(row=r_row, column=1, sticky="w")
        tk.Button(scan_f, text="扫描局域网 IP", command=self.scan_ip, bg="#b3e5fc").pack(side=tk.LEFT)
        self.lbl_scan_res = tk.Label(scan_f, text="", fg="blue")
        self.lbl_scan_res.pack(side=tk.LEFT, padx=5)
        r_row += 1

        # Index
        tk.Label(f_right, text="--- Index 模式设置 ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5); r_row+=1
        add_cam_entry("相机 Index:", "index")

        # 内参
        tk.Label(f_right, text="--- 相机内参 (Halcon) ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5); r_row+=1
        add_cam_entry("Fx:", "fx")
        add_cam_entry("Fy:", "fy")
        add_cam_entry("Cx:", "cx")
        add_cam_entry("Cy:", "cy")
        
        tk.Button(f_right, text="测试相机连接", command=self.test_camera, bg="#e0e0e0").grid(row=r_row, column=1, sticky="w", pady=10); r_row+=1

        # 底部保存
        tk.Button(parent, text="保存所有配置", command=self.save_config, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=20, pady=10)

    # -------------------------------------------------------------------------
    # 逻辑功能
    # -------------------------------------------------------------------------
    def on_mode_change(self):
        # 修复了这里：之前漏掉的方法
        mode = self.work_mode.get()
        self.log(f"工作模式切换为: {mode}")

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
        except: self.cfg["camera"]["index"] = 0
        # 内参
        for k in ["fx", "fy", "cx", "cy"]:
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
            messagebox.showinfo("成功", f"扫描到设备 IP: {ip}\n已自动填入。")
        else:
            self.lbl_scan_res.config(text="未发现")
            messagebox.showwarning("提示", "未扫描到 GigE 设备。请确认在同一网段。")

    def get_video_source(self):
        mode = self.var_camera_mode.get()
        if mode == "index":
            try: return int(self.cfg["camera"]["index"])
            except: return 0
        else:
            ip = self.cfg["camera"]["target_ip"]
            if not ip: return None
            # 海康通用 RTSP 格式
            return f"rtsp://{ip}:554/Streaming/Channels/101"

    def test_camera(self):
        self.update_cfg_from_ui()
        src = self.get_video_source()
        try:
            messagebox.showinfo("提示", f"正在尝试连接: {src}\n请稍候...")
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise Exception("无法打开视频流")
            ret, frame = cap.read()
            cap.release()
            if ret:
                messagebox.showinfo("成功", f"画面读取正常!\n分辨率: {frame.shape[1]}x{frame.shape[0]}")
            else:
                raise Exception("无画面数据")
        except Exception as e:
            messagebox.showerror("连接失败", f"错误: {e}")

    # -------------------------------------------------------------------------
    # 核心服务逻辑
    # -------------------------------------------------------------------------
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
                src = self.get_video_source()
                # 针对 IP 相机使用 FFMPEG 可能会更稳定
                if isinstance(src, str):
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                else:
                    self.cap = cv2.VideoCapture(src)
                
                if not self.cap.isOpened():
                    # 允许相机连接失败也启动服务，但在日志里报错
                    self.log(f"警告: 相机 {src} 连接失败，服务将以无相机模式启动")
                
                self.is_running = True
                self.btn_start.config(text="停止服务 (Stop)", bg="#f44336")
                self.lbl_status.config(text=f"监听中: {ip}:{port}", fg="green")
                
                # 启动线程
                threading.Thread(target=self.network_thread, daemon=True).start()
                threading.Thread(target=self.camera_thread, daemon=True).start()
                self.log("服务已启动")
                
            except Exception as e:
                messagebox.showerror("启动错误", str(e))
                if self.server_socket: self.server_socket.close()
        else:
            # 停止
            self.is_running = False
            if self.server_socket: self.server_socket.close()
            if self.cap: self.cap.release()
            self.btn_start.config(text="启动服务 (Start)", bg="#4caf50")
            self.lbl_status.config(text="已停止", fg="red")
            self.draw_placeholder()
            self.log("服务已停止")

    def network_thread(self):
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
                            resp = self.process_request()
                            client.send(resp.encode('utf-8'))
                            self.log(f"回复: {resp}")
                    except Exception as e:
                        self.log(f"通讯中断: {e}")
                        break
                
                self.client_socket = None
                self.log("机械臂断开连接")
                self.root.after(0, lambda: self.lbl_client.config(text="无连接", fg="gray"))
                
            except Exception as e:
                if self.is_running: self.log(f"Net Error: {e}")

    def camera_thread(self):
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.current_frame = frame
                    # 降频显示
                    if int(time.time() * 100) % 5 == 0:
                        self.root.after(0, self.update_display, frame)
                else:
                    # 尝试重连? 
                    pass
            time.sleep(0.01)

    def update_display(self, img):
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
        # 核心业务逻辑
        frame = None
        with self.lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
        
        if frame is None:
            return self.cfg["commands"]["error"] + ",NoImage"
            
        # 保存图片
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.cfg["paths"]["save_dir"], f"IMG_{ts}.{self.cfg['paths']['save_format']}")
        
        mode = self.work_mode.get()
        result_str = self.cfg["commands"]["error"]
        
        # 1. TEST 模式
        if mode == "TEST":
            # 读取内参
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
                # 简单的姿态计算
                qr_size = 100.0
                half = qr_size/2.0
                obj_pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
                succ, rvec, tvec = cv2.solvePnP(obj_pts, pts, K, dist)
                
                if succ:
                    # 绘制
                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 50)
                    cv2.polylines(frame, [pts.astype(int)], True, (0,255,0), 3)
                    
                    # 格式化
                    rmat, _ = cv2.Rodrigues(rvec)
                    euler = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    prefix = self.cfg["commands"]["success_prefix"]
                    sep = self.cfg["commands"]["separator"]
                    # 格式: OK,x,y,z,rx,ry,rz
                    result_str = f"{prefix}{sep}{tvec[0][0]:.2f}{sep}{tvec[1][0]:.2f}{sep}{tvec[2][0]:.2f}{sep}{euler[0]:.2f}{sep}{euler[1]:.2f}{sep}{euler[2]:.2f}"
            else:
                self.log("未识别到二维码")

        # 2. CALIB 模式
        elif mode == "CALIB":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            if ret:
                cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)
                # 记录
                self.calib_data_list.append({
                    "id": len(self.calib_data_list)+1,
                    "img_path": path,
                    "corners": corners,
                    "robot_pose": None
                })
                self.root.after(0, self.refresh_calib_list)
                result_str = self.cfg["commands"]["success_prefix"]
            else:
                self.log("未识别到标定板")

        # 写入磁盘
        cv2.imwrite(path, frame)
        self.log(f"已保存: {os.path.basename(path)}")
        return result_str

    # -------------------------------------------------------------------------
    # 标定辅助
    # -------------------------------------------------------------------------
    def refresh_calib_list(self):
        for item in self.tree_calib.get_children(): self.tree_calib.delete(item)
        for d in self.calib_data_list:
            pose = d["robot_pose"]
            self.tree_calib.insert("", "end", values=(
                d["id"], os.path.basename(d["img_path"]), 
                *(pose if pose else ["-"]*6),
                "Yes"
            ))

    def on_edit_pose(self, event):
        sel = self.tree_calib.selection()
        if not sel: return
        item = sel[0]
        idx = self.tree_calib.index(item)
        
        win = tk.Toplevel(self.root)
        win.title(f"输入 Pose (ID: {idx+1})")
        ents = []
        labels = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        for i, lbl in enumerate(labels):
            tk.Label(win, text=lbl).grid(row=0, column=i)
            e = tk.Entry(win, width=8)
            e.grid(row=1, column=i)
            ents.append(e)
        def confirm():
            try:
                vals = [float(e.get()) for e in ents]
                self.calib_data_list[idx]["robot_pose"] = vals
                self.refresh_calib_list()
                win.destroy()
            except: messagebox.showerror("错误", "请输入数字")
        tk.Button(win, text="确定", command=confirm).grid(row=2, column=0, columnspan=6)

    def run_hand_eye_calc(self):
        valid = [d for d in self.calib_data_list if d["robot_pose"]]
        if len(valid) < 3:
            self.txt_calib_res.insert(tk.END, "错误: 至少需要3组带坐标的数据\n")
            return
        
        try:
            R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
            K = np.array([[self.cfg["camera"]["fx"],0,self.cfg["camera"]["cx"]],
                          [0,self.cfg["camera"]["fy"],self.cfg["camera"]["cy"]],
                          [0,0,1]], dtype=float)
            dist = np.array(self.cfg["camera"]["dist"], dtype=float)
            
            # 生成 Object Points
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            sp = self.cfg["calibration"]["spacing"]
            objp = np.zeros((rows*cols,3), np.float32)
            objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2) * sp
            
            for d in valid:
                # 图像姿态
                ret, rvec, tvec = cv2.solvePnP(objp, d["corners"], K, dist)
                rmat, _ = cv2.Rodrigues(rvec)
                R_t2c.append(rmat); t_t2c.append(tvec)
                
                # 机械臂姿态
                pose = d["robot_pose"]
                t_g2b.append(np.array(pose[:3]).reshape(3,1))
                r_mat_g = R.from_euler('xyz', pose[3:], degrees=True).as_matrix()
                R_g2b.append(r_mat_g)
                
            # 计算 (Eye-in-Hand)
            rc, tc = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
            
            self.txt_calib_res.delete(1.0, tk.END)
            self.txt_calib_res.insert(tk.END, f"标定成功!\n平移 XYZ: {tc.flatten()}\n")
            self.txt_calib_res.insert(tk.END, f"旋转矩阵:\n{rc}\n")
            
        except Exception as e:
            self.txt_calib_res.insert(tk.END, f"计算失败: {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalVisionServer(root)
    root.mainloop()
