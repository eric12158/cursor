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
        "bind_ip": "0.0.0.0",  # 机械臂连接的本机IP
        "port": 8000
    },
    "camera": {
        "mode": "ip",      # ip 或 index
        "target_ip": "",   # 相机IP
        "index": 0,        # 索引
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
            # GigE Discovery Packet
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
        self.root.title("通用视觉服务器系统 (Universal Vision Server) - 工业修正版")
        self.root.geometry("1400x900")
        
        self.config_file = "vision_config.json"
        self.cfg = self.load_config()
        
        # 运行时状态
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.cap = None
        self.lock = threading.Lock()
        self.current_frame = None
        
        self.scanner = GigEScanner()
        self.calib_data_list = []
        
        self.setup_ui()
        
        if not os.path.exists(self.cfg["paths"]["save_dir"]):
            os.makedirs(self.cfg["paths"]["save_dir"])

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    # 兼容旧配置
                    data = json.load(f)
                    if "network" in data and "robot_net" not in data:
                        data["robot_net"] = {"bind_ip": data["network"]["ip"], "port": data["network"]["port"]}
                    return data
            except: pass
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        self.update_cfg_from_ui()
        with open(self.config_file, 'w') as f:
            json.dump(self.cfg, f, indent=4)
        messagebox.showinfo("提示", "配置已保存")

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        frame_run = tk.Frame(notebook); notebook.add(frame_run, text="1. 运行监控")
        self.setup_run_ui(frame_run)
        
        frame_calib = tk.Frame(notebook); notebook.add(frame_calib, text="2. 手眼标定集成")
        self.setup_calib_ui(frame_calib)
        
        frame_cfg = tk.Frame(notebook); notebook.add(frame_cfg, text="3. 系统配置")
        self.setup_config_ui(frame_cfg)

    def setup_run_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        left = tk.Frame(paned, width=400, bg="#f0f0f0")
        left.pack_propagate(False)
        paned.add(left, minsize=400)
        
        s_frame = tk.LabelFrame(left, text="服务器控制 (与机械臂)", font=("bold", 10))
        s_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_start = tk.Button(s_frame, text="启动服务", command=self.toggle_server, bg="#4caf50", fg="white", height=2)
        self.btn_start.pack(fill=tk.X, padx=5, pady=5)
        self.lbl_status = tk.Label(s_frame, text="状态: 已停止", fg="red", font=("Arial", 12))
        self.lbl_status.pack(pady=5)
        self.lbl_client = tk.Label(s_frame, text="客户端: 无连接", fg="gray")
        self.lbl_client.pack(pady=2)
        
        m_frame = tk.LabelFrame(left, text="工作模式", font=("bold", 10))
        m_frame.pack(fill=tk.X, padx=5, pady=5)
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(m_frame, text="测试模式 (回传坐标)", variable=self.work_mode, value="TEST", command=self.on_mode_change).pack(anchor="w")
        tk.Radiobutton(m_frame, text="标定模式 (仅存图)", variable=self.work_mode, value="CALIB", command=self.on_mode_change).pack(anchor="w")
        
        l_frame = tk.LabelFrame(left, text="日志", font=("bold", 10))
        l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(l_frame, font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(paned, bg="#333")
        paned.add(self.canvas, stretch="always")

    def setup_calib_ui(self, parent):
        top = tk.Frame(parent)
        top.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(top, text="刷新列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="计算手眼矩阵", command=self.run_hand_eye_calc, bg="orange").pack(side=tk.LEFT, padx=10)
        
        columns = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "ok")
        self.tree_calib = ttk.Treeview(parent, columns=columns, show="headings")
        for col in columns: 
            self.tree_calib.heading(col, text=col)
            self.tree_calib.column(col, width=60)
        self.tree_calib.column("img", width=150)
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.on_edit_pose)
        
        self.txt_calib_res = tk.Text(parent, height=8, bg="#e0e0e0")
        self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, parent):
        # 使用 PanedWindow 分割左右两块 IP 设置
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === 左侧：机械臂通讯设置 ===
        f_left = tk.LabelFrame(paned, text="【机械臂通讯设置】 (作为服务端)", font=("bold", 12), fg="blue")
        paned.add(f_left, minsize=400)
        
        row = 0
        def add_entry(p, label, key_group, key_item, width=25):
            nonlocal row
            tk.Label(p, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=5)
            var = tk.StringVar(value=str(self.cfg[key_group].get(key_item, "")))
            entry = tk.Entry(p, textvariable=var, width=width)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=5)
            setattr(self, f"var_{key_group}_{key_item}", var)
            row += 1

        add_entry(f_left, "本机监听 IP:", "robot_net", "bind_ip")
        tk.Label(f_left, text="(0.0.0.0 代表监听所有网卡)", fg="gray").grid(row=row, column=1, sticky="w"); row+=1
        add_entry(f_left, "监听端口:", "robot_net", "port")
        
        tk.Label(f_left, text="--- 指令协议 ---", font=("bold", 10)).grid(row=row, column=0, columnspan=2, pady=10); row+=1
        add_entry(f_left, "触发拍照:", "commands", "trigger")
        add_entry(f_left, "失败返回:", "commands", "error")
        add_entry(f_left, "成功前缀:", "commands", "success_prefix")
        
        tk.Label(f_left, text="--- 存储 ---", font=("bold", 10)).grid(row=row, column=0, columnspan=2, pady=10); row+=1
        add_entry(f_left, "保存路径:", "paths", "save_dir")

        # === 右侧：相机连接设置 ===
        f_right = tk.LabelFrame(paned, text="【海康相机连接设置】", font=("bold", 12), fg="green")
        paned.add(f_right, minsize=400)
        
        # 重置 row 给右侧用
        r_row = 0
        def add_cam_entry(label, key_item):
            nonlocal r_row
            tk.Label(f_right, text=label).grid(row=r_row, column=0, sticky="e", padx=5, pady=5)
            var = tk.StringVar(value=str(self.cfg["camera"].get(key_item, "")))
            entry = tk.Entry(f_right, textvariable=var, width=25)
            entry.grid(row=r_row, column=1, sticky="w", padx=5, pady=5)
            setattr(self, f"var_camera_{key_item}", var)
            r_row += 1

        # 模式选择
        tk.Label(f_right, text="连接模式:").grid(row=r_row, column=0, sticky="e", padx=5, pady=5)
        self.var_camera_mode = tk.StringVar(value=self.cfg["camera"].get("mode", "ip"))
        mf = tk.Frame(f_right)
        mf.grid(row=r_row, column=1, sticky="w")
        tk.Radiobutton(mf, text="使用 IP 直连 (无密/RTSP)", variable=self.var_camera_mode, value="ip").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="使用 MVS 索引 (Index)", variable=self.var_camera_mode, value="index").pack(side=tk.LEFT)
        r_row += 1

        # IP 区域
        tk.Label(f_right, text="--- 选项 A: IP 连接 ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5); r_row+=1
        add_cam_entry("相机 IP:", "target_ip")
        
        scan_f = tk.Frame(f_right)
        scan_f.grid(row=r_row, column=1, sticky="w")
        tk.Button(scan_f, text="扫描局域网 IP", command=self.scan_ip, bg="#b3e5fc").pack(side=tk.LEFT)
        self.lbl_scan_res = tk.Label(scan_f, text="", fg="blue")
        self.lbl_scan_res.pack(side=tk.LEFT, padx=5)
        r_row += 1

        # Index 区域
        tk.Label(f_right, text="--- 选项 B: 索引连接 ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5); r_row+=1
        add_cam_entry("相机索引:", "index")
        tk.Label(f_right, text="(需先用MVS软件映射)", fg="gray", font=("Arial", 8)).grid(row=r_row, column=1, sticky="w"); r_row+=1
        
        # 内参
        tk.Label(f_right, text="--- 内参 (Halcon) ---", fg="gray").grid(row=r_row, column=0, columnspan=2, pady=5); r_row+=1
        add_cam_entry("Fx:", "fx")
        add_cam_entry("Fy:", "fy")
        add_cam_entry("Cx:", "cx")
        add_cam_entry("Cy:", "cy")

        # 测试按钮
        tk.Button(f_right, text="测试连接相机", command=self.test_camera, bg="#e0e0e0").grid(row=r_row, column=1, sticky="w", pady=10); r_row+=1

        # 底部保存
        tk.Button(parent, text="=== 保存全部配置 ===", command=self.save_config, bg="#2196f3", fg="white", height=2, font=("bold", 12)).pack(fill=tk.X, padx=20, pady=10)

    def update_cfg_from_ui(self):
        # 机械臂配置
        self.cfg["robot_net"]["bind_ip"] = self.var_robot_net_bind_ip.get()
        self.cfg["robot_net"]["port"] = int(self.var_robot_net_port.get())
        
        # 路径与指令
        self.cfg["paths"]["save_dir"] = self.var_paths_save_dir.get()
        self.cfg["commands"]["trigger"] = self.var_commands_trigger.get()
        self.cfg["commands"]["error"] = self.var_commands_error.get()
        self.cfg["commands"]["success_prefix"] = self.var_commands_success_prefix.get()
        
        # 相机配置
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
            self.lbl_scan_res.config(text=f"已发现: {ip}")
            messagebox.showinfo("成功", f"扫描到设备 IP: {ip}\n已自动填入。")
        else:
            self.lbl_scan_res.config(text="未发现")
            messagebox.showwarning("失败", "未扫描到 GigE 设备。\n请确认相机已上电且在同一网段。")

    def get_video_source(self):
        mode = self.var_camera_mode.get()
        if mode == "index":
            try: return int(self.cfg["camera"]["index"])
            except: return 0
        else:
            ip = self.cfg["camera"]["target_ip"]
            if not ip: return None
            # 海康/大华 通用无密 RTSP 或 IP 直接尝试
            # 优先尝试标准 RTSP 端口
            return f"rtsp://{ip}:554/Streaming/Channels/101"

    def test_camera(self):
        self.update_cfg_from_ui()
        src = self.get_video_source()
        try:
            messagebox.showinfo("提示", f"正在尝试连接: {src}")
            # 针对 IP 连接，强制使用 FFMPEG 后端通常更稳
            if isinstance(src, str):
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(src)
                
            if not cap.isOpened():
                raise Exception("无法打开视频流")
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                messagebox.showinfo("成功", f"画面读取正常！\n分辨率: {frame.shape[1]}x{frame.shape[0]}")
            else:
                raise Exception("无画面数据")
        except Exception as e:
            messagebox.showerror("连接失败", f"错误: {e}\n\n排查建议:\n1. IP 是否正确 (ping一下)\n2. 是否被其他软件(MVS)占用\n3. 尝试切换连接模式")

    # -------------------------------------------------------------------------
    # 核心逻辑 (保持不变)
    # -------------------------------------------------------------------------
    def toggle_server(self):
        if not self.is_running:
            try:
                # 绑定机械臂监听
                ip = self.cfg["robot_net"]["bind_ip"]
                port = int(self.cfg["robot_net"]["port"])
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind((ip, port))
                self.server_socket.listen(1)
                self.server_socket.settimeout(1.0)
                
                # 连接相机
                self.update_cfg_from_ui()
                src = self.get_video_source()
                if isinstance(src, str):
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                else:
                    self.cap = cv2.VideoCapture(src)
                
                if not self.cap.isOpened():
                    raise Exception(f"无法打开相机源: {src}")

                self.is_running = True
                self.btn_start.config(text="停止服务", bg="#f44336")
                self.lbl_status.config(text=f"监听中: {ip}:{port}", fg="green")
                
                threading.Thread(target=self.network_loop, daemon=True).start()
                threading.Thread(target=self.camera_loop, daemon=True).start()
                self.log("服务已启动...")
                
            except Exception as e:
                messagebox.showerror("错误", f"启动失败: {e}")
                if self.server_socket: self.server_socket.close()
        else:
            self.is_running = False
            if self.server_socket: self.server_socket.close()
            if self.cap: self.cap.release()
            self.btn_start.config(text="启动服务", bg="#4caf50")
            self.lbl_status.config(text="已停止", fg="red")
            self.log("服务已停止")

    def network_loop(self):
        while self.is_running:
            try:
                try:
                    client, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                
                self.client_socket = client
                self.root.after(0, lambda: self.lbl_client.config(text=f"连接来自: {addr}", fg="blue"))
                self.log(f"机械臂已连接: {addr}")
                
                while self.is_running:
                    try:
                        data = client.recv(1024)
                        if not data: break
                        msg = data.decode('utf-8').strip()
                        self.log(f"指令: {msg}")
                        
                        if msg == self.cfg["commands"]["trigger"]:
                            response = self.handle_trigger()
                            client.send(response.encode('utf-8'))
                            self.log(f"回复: {response}")
                        else:
                            pass
                    except Exception as e:
                        self.log(f"通讯异常: {e}")
                        break
                self.client_socket = None
                self.root.after(0, lambda: self.lbl_client.config(text="无连接", fg="gray"))
                self.log("机械臂断开")
            except Exception as e:
                if self.is_running: self.log(f"Server Error: {e}")

    def camera_loop(self):
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
                    if int(time.time() * 10) % 2 == 0:
                        self.root.after(0, self.update_display, frame)
            time.sleep(0.01)

    def update_display(self, img):
        h, w = img.shape[:2]
        scale = 0.4
        nh, nw = int(h*scale), int(w*scale)
        show_img = cv2.resize(img, (nw, nh))
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(show_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(nw//2, nh//2, image=tk_img)
        self.canvas.image = tk_img

    def handle_trigger(self):
        if self.current_frame is None: return self.cfg["commands"]["error"]
        frame = self.current_frame.copy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(self.cfg["paths"]["save_dir"], f"IMG_{timestamp}.{self.cfg['paths']['save_format']}")
        
        mode = self.work_mode.get()
        result_str = self.cfg["commands"]["error"]
        
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
                cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
                qr_size = 100.0 
                half = qr_size / 2.0
                obj_pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
                succ, rvec, tvec = cv2.solvePnP(obj_pts, pts, K, dist)
                if succ:
                    rmat, _ = cv2.Rodrigues(rvec)
                    euler = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    prefix = self.cfg["commands"]["success_prefix"]
                    sep = self.cfg["commands"]["separator"]
                    result_str = f"{prefix}{sep}{tvec[0][0]:.2f}{sep}{tvec[1][0]:.2f}{sep}{tvec[2][0]:.2f}{sep}{euler[0]:.2f}{sep}{euler[1]:.2f}{sep}{euler[2]:.2f}"
            else:
                self.log("无二维码")

        elif mode == "CALIB":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            if ret:
                cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)
                self.calib_data_list.append({
                    "id": len(self.calib_data_list) + 1,
                    "img_path": save_path,
                    "corners": corners,
                    "robot_pose": None 
                })
                self.root.after(0, self.refresh_calib_list)
                result_str = self.cfg["commands"]["success_prefix"]
            else:
                self.log("无标定板")

        cv2.imwrite(save_path, frame)
        self.log(f"保存: {os.path.basename(save_path)}")
        return result_str

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda: self.txt_log.insert(tk.END, f"[{timestamp}] {msg}\n"))
        self.root.after(0, lambda: self.txt_log.see(tk.END))

    def refresh_calib_list(self):
        for item in self.tree_calib.get_children(): self.tree_calib.delete(item)
        for d in self.calib_data_list:
            self.tree_calib.insert("", "end", values=(
                d["id"], os.path.basename(d["img_path"]), 
                *(d["robot_pose"] if d["robot_pose"] else ["-"]*6),
                "Yes"
            ))

    def on_edit_pose(self, event):
        item = self.tree_calib.selection()[0]
        idx = self.tree_calib.index(item)
        win = tk.Toplevel(self.root)
        win.title(f"输入第 {idx+1} 组机械臂坐标")
        entries = []
        labels = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        for i, lbl in enumerate(labels):
            tk.Label(win, text=lbl).grid(row=0, column=i)
            e = tk.Entry(win, width=8)
            e.grid(row=1, column=i)
            entries.append(e)
        def confirm():
            try:
                vals = [float(e.get()) for e in entries]
                self.calib_data_list[idx]["robot_pose"] = vals
                self.refresh_calib_list()
                win.destroy()
            except:
                messagebox.showerror("错误", "请输入有效数字")
        tk.Button(win, text="确定", command=confirm).grid(row=2, column=0, columnspan=6)

    def run_hand_eye_calc(self):
        valid_data = [d for d in self.calib_data_list if d["robot_pose"] is not None]
        if len(valid_data) < 3:
            self.txt_calib_res.insert(tk.END, "错误: 有效数据不足3组\n")
            return
        try:
            R_gripper2base = []
            t_gripper2base = []
            R_target2cam = []
            t_target2cam = []
            K = np.array([
                [self.cfg["camera"]["fx"], 0, self.cfg["camera"]["cx"]],
                [0, self.cfg["camera"]["fy"], self.cfg["camera"]["cy"]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist = np.array(self.cfg["camera"]["dist"], dtype=np.float64)
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            spacing = self.cfg["calibration"]["spacing"]
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            objp = objp * float(spacing)
            
            for d in valid_data:
                ret, rvec, tvec = cv2.solvePnP(objp, d["corners"], K, dist)
                rmat, _ = cv2.Rodrigues(rvec)
                R_target2cam.append(rmat)
                t_target2cam.append(tvec)
                pose = d["robot_pose"]
                t_g2b = np.array(pose[:3]).reshape(3,1)
                r_g2b = R.from_euler('xyz', pose[3:], degrees=True).as_matrix()
                R_gripper2base.append(r_g2b)
                t_gripper2base.append(t_g2b)
                
            R_cam2g, t_cam2g = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base, 
                R_target2cam, t_target2cam, 
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            self.txt_calib_res.delete(1.0, tk.END)
            self.txt_calib_res.insert(tk.END, "计算成功!\n平移(XYZ): " + str(t_cam2g.flatten()) + "\n")
        except Exception as e:
            self.txt_calib_res.insert(tk.END, f"计算失败: {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalVisionServer(root)
    root.mainloop()
