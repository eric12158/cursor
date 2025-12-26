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
    "network": {
        "ip": "0.0.0.0",
        "port": 8000,
        "buffer_size": 1024
    },
    "camera": {
        "mode": "usb",     # usb 或 rtsp
        "id": 0,           # USB 索引
        "rtsp_url": "",    # RTSP 地址模板
        "ip": "",          # 目标 IP
        "user": "admin",
        "pwd": "",
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
            # 创建 UDP 套接字
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)
            
            # GVCP Discovery Packet (标准 GigE 发现包)
            # Key(0x42) | Flag(0x11) | Cmd(0x0002 - DISCOVERY) | Length(0x0000) | ReqID(0x0001)
            msg = struct.pack('>BBHHH', 0x42, 0x11, 0x0002, 0x0000, 0x0001)
            
            # 广播到标准 GigE 端口 3956
            sock.sendto(msg, ('255.255.255.255', 3956))
            
            start = time.time()
            while time.time() - start < timeout:
                try:
                    data, addr = sock.recvfrom(1024)
                    if len(data) > 0:
                        # 解析简单的响应
                        # 工业相机的响应通常包含厂商信息、序列号、IP等
                        # 这里我们主要提取 IP (addr[0])
                        # 尝试解析 Model Name (通常在偏移量较大的位置，具体取决于厂商实现，这里简化处理)
                        
                        # 简单的去重
                        if addr[0] not in [d['ip'] for d in devices]:
                            devices.append({
                                'ip': addr[0],
                                'port': addr[1],
                                'raw_len': len(data)
                            })
                except socket.timeout:
                    break
            sock.close()
        except Exception as e:
            print(f"Scan error: {e}")
        return devices

class UniversalVisionServer:
    def __init__(self, root):
        self.root = root
        self.root.title("通用视觉服务器系统 (Universal Vision Server) - 工业版")
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
                    return json.load(f)
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
        
        s_frame = tk.LabelFrame(left, text="服务器控制", font=("bold", 10))
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
        row = 0
        def add_entry(p, label, key_group, key_item, width=30):
            nonlocal row
            tk.Label(p, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
            var = tk.StringVar(value=str(self.cfg[key_group].get(key_item, "")))
            entry = tk.Entry(p, textvariable=var, width=width)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            setattr(self, f"var_{key_group}_{key_item}", var)
            row += 1

        frame = tk.Frame(parent)
        frame.pack(padx=20, pady=20)
        
        # --- 相机配置 ---
        tk.Label(frame, text="--- 相机连接 ---", font=("bold", 10), fg="blue").grid(row=row, column=0, columnspan=2, pady=(10,5)); row+=1
        
        # 模式选择
        tk.Label(frame, text="连接方式:").grid(row=row, column=0, sticky="e")
        self.var_camera_mode = tk.StringVar(value=self.cfg["camera"].get("mode", "usb"))
        mode_frame = tk.Frame(frame)
        mode_frame.grid(row=row, column=1, sticky="w")
        tk.Radiobutton(mode_frame, text="USB/虚拟映射(Index)", variable=self.var_camera_mode, value="usb").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="网络/RTSP(IP)", variable=self.var_camera_mode, value="rtsp").pack(side=tk.LEFT)
        row += 1
        
        add_entry(frame, "USB 索引 ID:", "camera", "id")
        
        # 扫描功能
        tk.Label(frame, text="IP 扫描:").grid(row=row, column=0, sticky="e")
        scan_frame = tk.Frame(frame)
        scan_frame.grid(row=row, column=1, sticky="w")
        tk.Button(scan_frame, text="扫描局域网 GigE 相机", command=self.scan_ip, bg="#81d4fa").pack(side=tk.LEFT)
        self.lbl_scan_res = tk.Label(scan_frame, text="未扫描", fg="gray")
        self.lbl_scan_res.pack(side=tk.LEFT, padx=5)
        row += 1
        
        add_entry(frame, "目标 IP:", "camera", "ip")
        add_entry(frame, "RTSP 账号:", "camera", "user")
        add_entry(frame, "RTSP 密码:", "camera", "pwd")
        
        tk.Button(frame, text="测试连接", command=self.test_camera, bg="#e0e0e0").grid(row=row, column=1, sticky="w", pady=5); row+=1

        # --- 其他配置 ---
        tk.Label(frame, text="--- 网络/存储/内参 ---", font=("bold", 10)).grid(row=row, column=0, columnspan=2, pady=(10,5)); row+=1
        add_entry(frame, "监听端口:", "network", "port")
        add_entry(frame, "图片保存路径:", "paths", "save_dir")
        add_entry(frame, "触发指令:", "commands", "trigger")
        add_entry(frame, "内参 Fx:", "camera", "fx", width=10)
        
        tk.Button(frame, text="保存所有配置", command=self.save_config, bg="#2196f3", fg="white", height=2).grid(row=row, column=0, columnspan=2, pady=20, sticky="ew")

    def update_cfg_from_ui(self):
        # 简化版更新逻辑
        self.cfg["camera"]["mode"] = self.var_camera_mode.get()
        # 更新其他所有绑定的 var_...
        for key_group in self.cfg:
            for key_item in self.cfg[key_group]:
                var_name = f"var_{key_group}_{key_item}"
                if hasattr(self, var_name):
                    val = getattr(self, var_name).get()
                    try:
                        if isinstance(self.cfg[key_group][key_item], int): val = int(val)
                        elif isinstance(self.cfg[key_group][key_item], float): val = float(val)
                    except: pass
                    self.cfg[key_group][key_item] = val

    def scan_ip(self):
        self.lbl_scan_res.config(text="扫描中...", fg="orange")
        self.root.update()
        devices = self.scanner.scan()
        if devices:
            ip_list = [d['ip'] for d in devices]
            res_text = f"发现: {', '.join(ip_list)}"
            self.lbl_scan_res.config(text=res_text, fg="green")
            # 自动填入第一个
            self.var_camera_ip.set(devices[0]['ip'])
            messagebox.showinfo("扫描结果", f"发现 {len(devices)} 个设备:\n{ip_list}\n已自动填入第一个 IP")
        else:
            self.lbl_scan_res.config(text="未发现 GigE 设备", fg="red")
            messagebox.showwarning("提示", "未通过 UDP 广播发现设备。\n1. 请检查防火墙是否允许 UDP 3956。\n2. 确保相机和电脑在同一网段。\n3. 若无法发现，请手动输入 IP。")

    def get_video_source(self):
        """根据配置生成 OpenCV 可以接受的 source 参数"""
        mode = self.var_camera_mode.get() # 从 UI 实时获取
        if mode == "usb":
            return int(self.cfg["camera"]["id"])
        else:
            # 构建 RTSP 链接
            # 海康常见格式: rtsp://user:pwd@ip:554/Streaming/Channels/101
            # 或者是 OpenRTSP, 或者其他 HTTP 流
            # 这里提供一个通用模板构建
            ip = self.cfg["camera"]["ip"]
            user = self.cfg["camera"]["user"]
            pwd = self.cfg["camera"]["pwd"]
            if not ip: return None
            
            # 如果 IP 字段里直接填了完整的 rtsp://... 则直接用
            if ip.startswith("rtsp://") or ip.startswith("http://"):
                return ip
            
            # 否则尝试构建海康标准格式
            if user and pwd:
                return f"rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/101"
            else:
                return f"rtsp://{ip}:554/Streaming/Channels/101"

    def test_camera(self):
        self.update_cfg_from_ui()
        src = self.get_video_source()
        if src is None:
            messagebox.showerror("错误", "无效的相机配置")
            return
            
        try:
            messagebox.showinfo("提示", f"正在尝试连接:\n{src}\n(可能需要几秒钟)")
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise Exception("无法打开视频流")
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                messagebox.showinfo("成功", f"连接成功!\n分辨率: {frame.shape[1]}x{frame.shape[0]}")
            else:
                raise Exception("连接建立但无图像数据")
        except Exception as e:
            messagebox.showerror("连接失败", f"错误信息:\n{e}\n\n建议:\n1. 检查 IP/账号密码\n2. 若是工业相机，建议使用官方软件映射为 USB 设备模式(Index)\n3. 检查 RTSP 功能是否开启")

    # -------------------------------------------------------------------------
    # 核心逻辑
    # -------------------------------------------------------------------------
    def toggle_server(self):
        if not self.is_running:
            try:
                ip = self.cfg["network"]["ip"]
                port = int(self.cfg["network"]["port"])
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind((ip, port))
                self.server_socket.listen(1)
                self.server_socket.settimeout(1.0)
                
                # 打开相机
                self.update_cfg_from_ui()
                src = self.get_video_source()
                self.cap = cv2.VideoCapture(src)
                
                # 针对网络流，设置缓冲区大小可能有助于减少延迟
                if isinstance(src, str):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not self.cap.isOpened():
                    raise Exception(f"无法连接相机: {src}")

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
                        data = client.recv(self.cfg["network"]["buffer_size"])
                        if not data: break
                        
                        msg = data.decode('utf-8').strip()
                        self.log(f"收到指令: {msg}")
                        
                        trigger_cmd = self.cfg["commands"]["trigger"]
                        
                        if msg == trigger_cmd:
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
                self.log("机械臂断开连接")
                
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
                else:
                    # 断线重连尝试
                    pass 
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
            # 测试模式
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
                self.log("二维码检测失败")

        elif mode == "CALIB":
            # 标定模式
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
                self.log("标定板检测失败")

        cv2.imwrite(save_path, frame)
        self.log(f"已保存: {os.path.basename(save_path)}")
        return result_str

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda: self.txt_log.insert(tk.END, f"[{timestamp}] {msg}\n"))
        self.root.after(0, lambda: self.txt_log.see(tk.END))

    def on_mode_change(self):
        self.log(f"模式切换为: {self.work_mode.get()}")

    def refresh_calib_list(self):
        for item in self.tree_calib.get_children(): self.tree_calib.delete(item)
        for d in self.calib_data_list:
            pose_str = str(d["robot_pose"]) if d["robot_pose"] else "双击填入坐标"
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
