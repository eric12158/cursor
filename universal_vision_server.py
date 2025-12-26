import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import json
import os
import socket
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
        "id": 0,           # 相机索引: 0, 1, 2...
        "width": 1280,     # 分辨率宽
        "height": 960,     # 分辨率高
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

class UniversalVisionServer:
    def __init__(self, root):
        self.root = root
        self.root.title("通用视觉服务器系统 (Universal Vision Server)")
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
        
        # 手眼标定数据缓存
        self.calib_data_list = []
        
        self.setup_ui()
        
        # 确保保存目录存在
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
        # 从UI获取最新值更新cfg
        self.update_cfg_from_ui()
        with open(self.config_file, 'w') as f:
            json.dump(self.cfg, f, indent=4)
        messagebox.showinfo("提示", "配置已保存")

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # === 页面1: 运行监控 ===
        frame_run = tk.Frame(notebook)
        notebook.add(frame_run, text="1. 运行监控")
        self.setup_run_ui(frame_run)
        
        # === 页面2: 手眼标定集成 ===
        frame_calib = tk.Frame(notebook)
        notebook.add(frame_calib, text="2. 手眼标定集成")
        self.setup_calib_ui(frame_calib)
        
        # === 页面3: 系统配置 ===
        frame_cfg = tk.Frame(notebook)
        notebook.add(frame_cfg, text="3. 系统配置")
        self.setup_config_ui(frame_cfg)

    # -------------------------------------------------------------------------
    # UI 构建部分
    # -------------------------------------------------------------------------
    def setup_run_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制
        left = tk.Frame(paned, width=400, bg="#f0f0f0")
        left.pack_propagate(False)
        paned.add(left, minsize=400)
        
        # 服务器控制
        s_frame = tk.LabelFrame(left, text="服务器控制", font=("bold", 10))
        s_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_start = tk.Button(s_frame, text="启动服务 (监听机械臂)", command=self.toggle_server, bg="#4caf50", fg="white", height=2)
        self.btn_start.pack(fill=tk.X, padx=5, pady=5)
        
        self.lbl_status = tk.Label(s_frame, text="状态: 已停止", fg="red", font=("Arial", 12))
        self.lbl_status.pack(pady=5)
        self.lbl_client = tk.Label(s_frame, text="客户端: 无连接", fg="gray")
        self.lbl_client.pack(pady=2)
        
        # 模式选择
        m_frame = tk.LabelFrame(left, text="工作模式", font=("bold", 10))
        m_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(m_frame, text="测试模式 (回传二维码坐标)", variable=self.work_mode, value="TEST", command=self.on_mode_change).pack(anchor="w")
        tk.Radiobutton(m_frame, text="标定模式 (仅存图，不回传坐标)", variable=self.work_mode, value="CALIB", command=self.on_mode_change).pack(anchor="w")
        
        # 日志
        l_frame = tk.LabelFrame(left, text="通讯日志", font=("bold", 10))
        l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(l_frame, font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        # 右侧图像
        self.canvas = tk.Canvas(paned, bg="#333")
        paned.add(self.canvas, stretch="always")

    def setup_calib_ui(self, parent):
        # 简化版的手眼标定界面，专注于数据管理
        top = tk.Frame(parent)
        top.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(top, text="在此处完善机械臂坐标，然后进行计算").pack(side=tk.LEFT)
        tk.Button(top, text="刷新数据列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="计算手眼矩阵", command=self.run_hand_eye_calc, bg="orange").pack(side=tk.LEFT, padx=10)
        
        # 列表
        columns = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "corners_ok")
        self.tree_calib = ttk.Treeview(parent, columns=columns, show="headings")
        for col in columns: 
            self.tree_calib.heading(col, text=col)
            self.tree_calib.column(col, width=80)
        self.tree_calib.column("img", width=200)
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.on_edit_pose)
        
        # 结果
        self.txt_calib_res = tk.Text(parent, height=8, bg="#e0e0e0")
        self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, parent):
        # 使用简单的 Grid 布局配置参数
        row = 0
        def add_entry(p, label, key_group, key_item, width=30):
            nonlocal row
            tk.Label(p, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
            var = tk.StringVar(value=str(self.cfg[key_group][key_item]))
            entry = tk.Entry(p, textvariable=var, width=width)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            # 动态绑定保存
            setattr(self, f"var_{key_group}_{key_item}", var)
            row += 1

        frame = tk.Frame(parent)
        frame.pack(padx=20, pady=20)
        
        # --- 相机配置 (新增) ---
        tk.Label(frame, text="--- 相机硬件连接 ---", font=("bold", 10), fg="blue").grid(row=row, column=0, columnspan=2, pady=(10,5)); row+=1
        add_entry(frame, "相机索引 ID (0/1/2):", "camera", "id")
        add_entry(frame, "分辨率 宽:", "camera", "width")
        add_entry(frame, "分辨率 高:", "camera", "height")
        
        # 测试按钮
        tk.Button(frame, text="测试打开相机", command=self.test_camera, bg="#e0e0e0").grid(row=row, column=1, sticky="w", pady=5); row+=1
        
        # --- 网络配置 ---
        tk.Label(frame, text="--- 网络配置 ---", font=("bold", 10)).grid(row=row, column=0, columnspan=2, pady=(10,5)); row+=1
        add_entry(frame, "监听 IP:", "network", "ip")
        add_entry(frame, "监听端口:", "network", "port")
        
        # --- 机械臂指令 ---
        tk.Label(frame, text="--- 机械臂指令 ---", font=("bold", 10)).grid(row=row, column=0, columnspan=2, pady=(10,5)); row+=1
        add_entry(frame, "触发拍照指令:", "commands", "trigger")
        add_entry(frame, "失败返回指令:", "commands", "error")
        add_entry(frame, "成功返回前缀:", "commands", "success_prefix")
        
        # --- 存储配置 ---
        tk.Label(frame, text="--- 存储配置 ---", font=("bold", 10)).grid(row=row, column=0, columnspan=2, pady=(10,5)); row+=1
        add_entry(frame, "图片保存路径:", "paths", "save_dir")
        
        # --- 内参配置 ---
        tk.Label(frame, text="--- 相机内参 (Halcon) ---", font=("bold", 10)).grid(row=row, column=0, columnspan=2, pady=(10,5)); row+=1
        add_entry(frame, "Fx:", "camera", "fx")
        add_entry(frame, "Fy:", "camera", "fy")
        add_entry(frame, "Cx:", "camera", "cx")
        add_entry(frame, "Cy:", "camera", "cy")

        tk.Button(frame, text="保存所有配置", command=self.save_config, bg="#2196f3", fg="white", height=2).grid(row=row, column=0, columnspan=2, pady=20, sticky="ew")

    def update_cfg_from_ui(self):
        # 这是一个简单的反射更新，实际项目可以更严谨
        for key_group in self.cfg:
            for key_item in self.cfg[key_group]:
                var_name = f"var_{key_group}_{key_item}"
                if hasattr(self, var_name):
                    val = getattr(self, var_name).get()
                    # 尝试转换类型
                    try:
                        if isinstance(self.cfg[key_group][key_item], int): val = int(val)
                        elif isinstance(self.cfg[key_group][key_item], float): val = float(val)
                    except: pass
                    self.cfg[key_group][key_item] = val

    def test_camera(self):
        self.update_cfg_from_ui() # 先获取当前输入
        try:
            cam_id = int(self.cfg["camera"]["id"])
            w = int(self.cfg["camera"]["width"])
            h = int(self.cfg["camera"]["height"])
            
            cap = cv2.VideoCapture(cam_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            
            if not cap.isOpened():
                messagebox.showerror("失败", f"无法打开相机 ID: {cam_id}")
                return
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                messagebox.showinfo("成功", f"相机连接正常！\n获取分辨率: {frame.shape[1]}x{frame.shape[0]}")
            else:
                messagebox.showerror("失败", "相机已打开但无法读取画面")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # -------------------------------------------------------------------------
    # 核心逻辑
    # -------------------------------------------------------------------------
    def toggle_server(self):
        if not self.is_running:
            # 启动
            try:
                ip = self.cfg["network"]["ip"]
                port = int(self.cfg["network"]["port"])
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind((ip, port))
                self.server_socket.listen(1)
                self.server_socket.settimeout(1.0) # 非阻塞
                
                # 打开相机 (在这里连接)
                cam_id = int(self.cfg["camera"]["id"])
                self.cap = cv2.VideoCapture(cam_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["camera"]["width"])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["camera"]["height"])
                
                if not self.cap.isOpened():
                    raise Exception(f"无法打开相机 ID {cam_id}")

                self.is_running = True
                self.btn_start.config(text="停止服务", bg="#f44336")
                self.lbl_status.config(text=f"监听中: {ip}:{port}", fg="green")
                
                # 开启线程
                threading.Thread(target=self.network_loop, daemon=True).start()
                threading.Thread(target=self.camera_loop, daemon=True).start()
                self.log("服务已启动...")
                
            except Exception as e:
                messagebox.showerror("错误", f"启动失败: {e}")
                if self.server_socket: self.server_socket.close()
        else:
            # 停止
            self.is_running = False
            if self.server_socket: self.server_socket.close()
            if self.cap: self.cap.release()
            self.btn_start.config(text="启动服务", bg="#4caf50")
            self.lbl_status.config(text="已停止", fg="red")
            self.log("服务已停止")

    def network_loop(self):
        while self.is_running:
            try:
                # 等待连接
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
                            # 触发拍照逻辑
                            response = self.handle_trigger()
                            client.send(response.encode('utf-8'))
                            self.log(f"回复: {response}")
                        else:
                            # 简单的握手或心跳
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
                    # 显示 (降频显示以节省资源)
                    if int(time.time() * 10) % 2 == 0:
                        self.root.after(0, self.update_display, frame)
            time.sleep(0.03)

    def update_display(self, img):
        # 简单缩放显示
        h, w = img.shape[:2]
        scale = 0.4
        nh, nw = int(h*scale), int(w*scale)
        show_img = cv2.resize(img, (nw, nh))
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(show_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(nw//2, nh//2, image=tk_img)
        self.canvas.image = tk_img # 引用保持

    def handle_trigger(self):
        """核心处理逻辑：拍照 -> 识别 -> 存图 -> 返回"""
        if self.current_frame is None: return self.cfg["commands"]["error"]
        
        # 1. 抓取当前帧
        frame = self.current_frame.copy()
        
        # 2. 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(self.cfg["paths"]["save_dir"], f"IMG_{timestamp}.{self.cfg['paths']['save_format']}")
        
        # 3. 根据模式处理
        mode = self.work_mode.get()
        result_str = self.cfg["commands"]["error"]
        
        if mode == "TEST":
            # 测试模式：识别二维码并返回坐标
            K = np.array([
                [self.cfg["camera"]["fx"], 0, self.cfg["camera"]["cx"]],
                [0, self.cfg["camera"]["fy"], self.cfg["camera"]["cy"]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist = np.array(self.cfg["camera"]["dist"], dtype=np.float64)
            
            det = cv2.QRCodeDetector()
            ret, info, points, _ = det.detectAndDecodeMulti(frame)
            
            if ret and points is not None:
                # 简单的姿态解算
                pts = points[0]
                # 绘制
                cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
                
                # 假设二维码 100mm (这里需要一个标准值，或者在配置里加)
                qr_size = 100.0 
                half = qr_size / 2.0
                obj_pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
                
                succ, rvec, tvec = cv2.solvePnP(obj_pts, pts, K, dist)
                if succ:
                    # 构建返回字符串: OK,x,y,z,rx,ry,rz
                    # 将 rvec 转欧拉角
                    rmat, _ = cv2.Rodrigues(rvec)
                    euler = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    
                    prefix = self.cfg["commands"]["success_prefix"]
                    sep = self.cfg["commands"]["separator"]
                    # 格式: OK, X, Y, Z, Rx, Ry, Rz (保留2位小数)
                    result_str = f"{prefix}{sep}{tvec[0][0]:.2f}{sep}{tvec[1][0]:.2f}{sep}{tvec[2][0]:.2f}{sep}{euler[0]:.2f}{sep}{euler[1]:.2f}{sep}{euler[2]:.2f}"
            else:
                self.log("二维码检测失败")

        elif mode == "CALIB":
            # 标定模式：检测标定板，记录结果，不返回坐标，只返回 OK
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows = self.cfg["calibration"]["rows"]
            cols = self.cfg["calibration"]["cols"]
            ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            
            if ret:
                cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)
                # 缓存数据
                self.calib_data_list.append({
                    "id": len(self.calib_data_list) + 1,
                    "img_path": save_path,
                    "corners": corners,
                    "robot_pose": None # 等待用户填入
                })
                self.root.after(0, self.refresh_calib_list)
                result_str = self.cfg["commands"]["success_prefix"] # 告诉机械臂拍好了
            else:
                self.log("标定板检测失败")
                result_str = self.cfg["commands"]["error"]

        # 4. 保存图片 (带绘制结果)
        cv2.imwrite(save_path, frame)
        self.log(f"已保存: {os.path.basename(save_path)}")
        
        return result_str

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda: self.txt_log.insert(tk.END, f"[{timestamp}] {msg}\n"))
        self.root.after(0, lambda: self.txt_log.see(tk.END))

    def on_mode_change(self):
        self.log(f"模式切换为: {self.work_mode.get()}")

    # -------------------------------------------------------------------------
    # 手眼标定辅助
    # -------------------------------------------------------------------------
    def refresh_calib_list(self):
        # 刷新 Treeview
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
        
        # 弹窗输入坐标
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
        # 简化的计算逻辑，直接复用之前的方法
        # 检查所有数据是否有坐标
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
                # Target->Cam
                ret, rvec, tvec = cv2.solvePnP(objp, d["corners"], K, dist)
                rmat, _ = cv2.Rodrigues(rvec)
                R_target2cam.append(rmat)
                t_target2cam.append(tvec)
                
                # Gripper->Base (假设 XYZ+EulerXYZ)
                pose = d["robot_pose"]
                t_g2b = np.array(pose[:3]).reshape(3,1)
                r_g2b = R.from_euler('xyz', pose[3:], degrees=True).as_matrix()
                R_gripper2base.append(r_g2b)
                t_gripper2base.append(t_g2b)
                
            # 计算 (默认 Eye-in-Hand)
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
