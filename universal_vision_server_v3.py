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

# 尝试导入我们写的 SDK 驱动
try:
    from hik_driver import HikCamera
    HAS_HIK_SDK = True
except:
    HAS_HIK_SDK = False

# --- 默认配置 ---
DEFAULT_CONFIG = {
    "robot_net": {
        "bind_ip": "0.0.0.0",
        "port": 8000
    },
    "camera": {
        "mode": "sdk",     # ip, index, sdk
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

class UniversalVisionServer:
    def __init__(self, root):
        self.root = root
        self.root.title("通用视觉服务器系统 V3 (含海康SDK支持)")
        self.root.geometry("1400x950")
        
        self.config_file = "vision_config.json"
        self.cfg = self.load_config()
        
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.cap = None
        self.current_frame = None
        self.lock = threading.Lock()
        
        self.scanner = GigEScanner()
        self.calib_data_list = []
        
        self.setup_ui()
        if not os.path.exists(self.cfg["paths"]["save_dir"]):
            os.makedirs(self.cfg["paths"]["save_dir"])

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
        left = tk.Frame(paned, width=420, bg="#f0f0f0"); left.pack_propagate(False)
        paned.add(left, minsize=420)
        
        s_frame = tk.LabelFrame(left, text="服务器控制", font=("bold", 10))
        s_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_start = tk.Button(s_frame, text="启动服务 (Start)", command=self.toggle_server, bg="#4caf50", fg="white", height=2, font=("bold", 11))
        self.btn_start.pack(fill=tk.X, padx=5, pady=5)
        status_frame = tk.Frame(s_frame); status_frame.pack(fill=tk.X, padx=5)
        self.lbl_status = tk.Label(status_frame, text="状态: 已停止", fg="red"); self.lbl_status.pack(side=tk.LEFT)
        self.lbl_client = tk.Label(status_frame, text="客户端: 无连接", fg="gray"); self.lbl_client.pack(side=tk.RIGHT)
        
        m_frame = tk.LabelFrame(left, text="工作模式", font=("bold", 10))
        m_frame.pack(fill=tk.X, padx=5, pady=5)
        self.work_mode = tk.StringVar(value="TEST")
        tk.Radiobutton(m_frame, text="测试模式 (回传坐标)", variable=self.work_mode, value="TEST", command=self.on_mode_change).pack(anchor="w", padx=5)
        tk.Radiobutton(m_frame, text="标定模式 (仅存图)", variable=self.work_mode, value="CALIB", command=self.on_mode_change).pack(anchor="w", padx=5)
        
        l_frame = tk.LabelFrame(left, text="日志", font=("bold", 10))
        l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(l_frame, font=("Consolas", 9)); self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(paned, bg="#222"); paned.add(self.canvas, stretch="always")
        self.draw_placeholder()

    def setup_calib_ui(self, parent):
        top = tk.Frame(parent, pady=5); top.pack(fill=tk.X)
        tk.Button(top, text="刷新列表", command=self.refresh_calib_list).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="✎ 手动录入/修改坐标", command=self.on_edit_pose_btn, bg="#2196f3", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="▶ 计算手眼矩阵", command=self.run_hand_eye_calc, bg="orange").pack(side=tk.LEFT, padx=10)
        
        columns = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "ok")
        self.tree_calib = ttk.Treeview(parent, columns=columns, show="headings")
        for c in columns: self.tree_calib.heading(c, text=c); self.tree_calib.column(c, width=60)
        self.tree_calib.column("img", width=150)
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_calib.bind("<Double-1>", self.on_edit_pose)
        
        self.txt_calib_res = tk.Text(parent, height=10, bg="#e0e0e0", font=("Consolas", 10))
        self.txt_calib_res.pack(fill=tk.X, padx=5, pady=5)

    def setup_config_ui(self, parent):
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        f_left = tk.LabelFrame(paned, text="【机械臂通讯】", fg="blue"); paned.add(f_left, minsize=400)
        
        row = 0
        def add_e(p, l, g, k, w=25):
            nonlocal row
            tk.Label(p, text=l).grid(row=row, column=0, sticky="e", padx=5); var=tk.StringVar(value=str(self.cfg[g].get(k,"")))
            tk.Entry(p, textvariable=var, width=w).grid(row=row, column=1, sticky="w", padx=5); setattr(self, f"var_{g}_{k}", var); row+=1
        
        add_e(f_left, "监听 IP:", "robot_net", "bind_ip")
        add_e(f_left, "监听端口:", "robot_net", "port")
        row+=1
        add_e(f_left, "触发拍照:", "commands", "trigger")
        add_e(f_left, "失败返回:", "commands", "error")
        add_e(f_left, "成功前缀:", "commands", "success_prefix")
        row+=1
        add_e(f_left, "保存路径:", "paths", "save_dir", w=35)
        
        f_right = tk.LabelFrame(paned, text="【相机连接 (支持海康SDK)】", fg="green"); paned.add(f_right, minsize=400)
        r_row = 0
        def add_c(l, k, w=25):
            nonlocal r_row
            tk.Label(f_right, text=l).grid(row=r_row, column=0, sticky="e", padx=5); var=tk.StringVar(value=str(self.cfg["camera"].get(k,"")))
            tk.Entry(f_right, textvariable=var, width=w).grid(row=r_row, column=1, sticky="w", padx=5); setattr(self, f"var_camera_{k}", var); r_row+=1
            
        tk.Label(f_right, text="连接模式:").grid(row=r_row, column=0, sticky="e"); self.var_camera_mode=tk.StringVar(value=self.cfg["camera"].get("mode","sdk"))
        mf = tk.Frame(f_right); mf.grid(row=r_row, column=1, sticky="w")
        tk.Radiobutton(mf, text="海康 SDK 直连 (最佳)", variable=self.var_camera_mode, value="sdk").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="IP 兼容模式", variable=self.var_camera_mode, value="ip").pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="Index 索引", variable=self.var_camera_mode, value="index").pack(side=tk.LEFT)
        r_row+=1
        
        if not HAS_HIK_SDK:
            tk.Label(f_right, text="[警告] 未检测到 SDK 库文件，SDK 模式不可用", fg="red").grid(row=r_row, column=0, columnspan=2); r_row+=1
        
        add_c("相机 IP:", "target_ip")
        scan_f = tk.Frame(f_right); scan_f.grid(row=r_row, column=1, sticky="w")
        tk.Button(scan_f, text="扫描 IP", command=self.scan_ip, bg="#b3e5fc").pack(side=tk.LEFT)
        self.lbl_scan_res = tk.Label(scan_f, text="", fg="blue"); self.lbl_scan_res.pack(side=tk.LEFT, padx=5)
        r_row+=1
        
        add_c("相机 Index:", "index")
        r_row+=1
        add_c("Fx:", "fx"); add_c("Fy:", "fy"); add_c("Cx:", "cx"); add_c("Cy:", "cy")
        
        tk.Button(f_right, text="测试连接", command=self.test_camera, bg="#e0e0e0").grid(row=r_row, column=1, pady=10)
        
        tk.Button(parent, text="保存配置", command=self.save_config, bg="#2196f3", fg="white", height=2).pack(fill=tk.X, padx=20, pady=10)

    # --- 逻辑 ---
    def on_mode_change(self): self.log(f"模式切换: {self.work_mode.get()}")
    def log(self, m):
        ts = datetime.now().strftime("%H:%M:%S"); 
        try: self.txt_log.insert(tk.END, f"[{ts}] {m}\n"); self.txt_log.see(tk.END)
        except: pass
    def draw_placeholder(self): self.canvas.delete("all"); self.canvas.create_text(400,300,text="等待视频源...",fill="gray")
    
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
        devs = self.scanner.scan()
        if devs: 
            ip = devs[0]['ip']; self.var_camera_target_ip.set(ip); self.lbl_scan_res.config(text=f"发现: {ip}")
            messagebox.showinfo("成功", f"发现 IP: {ip}")
        else: messagebox.showwarning("失败", "未发现设备")

    def test_camera(self):
        self.update_cfg_from_ui()
        mode = self.cfg["camera"]["mode"]
        
        try:
            if mode == "sdk":
                if not HAS_HIK_SDK: raise Exception("缺少 SDK 库文件")
                cam = HikCamera()
                cam.open_by_ip(self.cfg["camera"]["target_ip"])
                ret, frame = cam.read()
                cam.release()
                if ret: messagebox.showinfo("成功", f"SDK 连接成功!\n{frame.shape}")
                else: raise Exception("SDK 读取失败")
            else:
                # 兼容旧逻辑
                src = int(self.cfg["camera"]["index"]) if mode=="index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src,str) else cv2.VideoCapture(src)
                if not cap.isOpened(): raise Exception("无法打开")
                ret, frame = cap.read(); cap.release()
                if ret: messagebox.showinfo("成功", "连接成功")
                else: raise Exception("无图像")
        except Exception as e: messagebox.showerror("失败", str(e))

    def toggle_server(self):
        if not self.is_running:
            try:
                ip = self.cfg["robot_net"]["bind_ip"]; port = int(self.cfg["robot_net"]["port"])
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((ip, port)); self.server_socket.listen(1); self.server_socket.settimeout(0.5)
                
                self.update_cfg_from_ui()
                mode = self.cfg["camera"]["mode"]
                if mode == "sdk":
                    if not HAS_HIK_SDK: raise Exception("SDK 未安装")
                    self.cap = HikCamera()
                    self.cap.open_by_ip(self.cfg["camera"]["target_ip"])
                else:
                    src = int(self.cfg["camera"]["index"]) if mode=="index" else f"rtsp://{self.cfg['camera']['target_ip']}:554/Streaming/Channels/101"
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src,str) else cv2.VideoCapture(src)
                
                if not self.cap.isOpened(): self.log("警告: 相机未连接")
                
                self.is_running = True
                self.btn_start.config(text="停止服务", bg="#f44336"); self.lbl_status.config(text="监听中", fg="green")
                threading.Thread(target=self.net_loop, daemon=True).start()
                threading.Thread(target=self.cam_loop, daemon=True).start()
                self.log("服务已启动")
            except Exception as e: messagebox.showerror("错误", str(e))
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
                        msg = d.decode().strip(); self.log(f"收到: {msg}")
                        if msg == self.cfg["commands"]["trigger"]:
                            res = self.process()
                            cl.send(res.encode()); self.log(f"回复: {res}")
                    except: break
                self.client_socket = None; self.root.after(0, lambda:self.lbl_client.config(text="无连接", fg="gray"))
            except: pass

    def cam_loop(self):
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock: self.current_frame = frame
                    if int(time.time()*100)%5==0: self.root.after(0, self.update_disp, frame)
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
        with self.lock: 
            if self.current_frame is not None: f = self.current_frame.copy()
        if f is None: return self.cfg["commands"]["error"]
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        p = os.path.join(self.cfg["paths"]["save_dir"], f"IMG_{ts}.jpg")
        
        mode = self.work_mode.get()
        res = self.cfg["commands"]["error"]
        
        if mode == "TEST":
            K = np.array([[self.cfg["camera"]["fx"],0,self.cfg["camera"]["cx"]],[0,self.cfg["camera"]["fy"],self.cfg["camera"]["cy"]],[0,0,1]], float)
            D = np.array(self.cfg["camera"]["dist"], float)
            ret, _, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(f)
            if ret and pts is not None:
                op = np.array([[-50,50,0],[50,50,0],[50,-50,0],[-50,-50,0]], float) # 假设100mm
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
        idx = self.tree_calib.index(sel[0])
        w = tk.Toplevel(self.root); w.title("输入 Pose")
        ents = []
        for i, l in enumerate(["X","Y","Z","Rx","Ry","Rz"]):
            tk.Label(w, text=l).grid(row=0, column=i); e = tk.Entry(w, width=8); e.grid(row=1, column=i); ents.append(e)
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
                Tg.append(np.array(d["robot_pose"][:3]).reshape(3,1))
                Rg.append(R.from_euler('xyz', d["robot_pose"][3:], degrees=True).as_matrix())
            
            rc, tc = cv2.calibrateHandEye(Rg, Tg, Rc, Tc, method=cv2.CALIB_HAND_EYE_TSAI)
            self.txt_calib_res.delete(1.0, tk.END); self.txt_calib_res.insert(tk.END, f"Result:\nXYZ: {tc.flatten()}\nEuler: {R.from_matrix(rc).as_euler('xyz', degrees=True)}")
        except Exception as e: self.txt_calib_res.insert(tk.END, str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalVisionServer(root)
    root.mainloop()
