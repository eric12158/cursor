import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import json

class GraspLogicTool:
    def __init__(self, root):
        self.root = root
        self.root.title("机械臂视觉抓取配置工具 - 偏移量调试")
        self.root.geometry("1400x900")
        
        # --- 核心变量 ---
        # 默认 Halcon 内参 (示例值，请在界面修改)
        self.fx = tk.DoubleVar(value=2000.0)
        self.fy = tk.DoubleVar(value=2000.0)
        self.cx = tk.DoubleVar(value=1280.0)
        self.cy = tk.DoubleVar(value=960.0)
        
        # 畸变系数 (k1, k2, p1, p2, k3)
        self.k1 = tk.DoubleVar(value=0.0)
        self.k2 = tk.DoubleVar(value=0.0)
        self.p1 = tk.DoubleVar(value=0.0)
        self.p2 = tk.DoubleVar(value=0.0)
        self.k3 = tk.DoubleVar(value=0.0)
        
        # 二维码参数
        self.qr_size = tk.DoubleVar(value=100.0) # mm
        
        # 抓取偏移量 (相对于二维码中心)
        self.offset_x = tk.DoubleVar(value=50.0)
        self.offset_y = tk.DoubleVar(value=0.0)
        self.offset_z = tk.DoubleVar(value=0.0)
        self.offset_angle = tk.DoubleVar(value=0.0) # 绕Z轴旋转
        
        # 图像处理
        self.current_img = None
        self.tk_image = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # === 左侧控制区 ===
        left_frame = tk.Frame(main_paned, width=400, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        main_paned.add(left_frame, minsize=400)
        
        # 1. 相机内参设置 (Halcon 导入)
        p_frame = tk.LabelFrame(left_frame, text="1. 相机内参 (来自 Halcon)", font=("bold", 10))
        p_frame.pack(fill=tk.X, padx=5, pady=5)
        
        grid_opts = {'padx':2, 'pady':2, 'sticky':'w'}
        tk.Label(p_frame, text="Fx:").grid(row=0, column=0, **grid_opts)
        tk.Entry(p_frame, textvariable=self.fx, width=8).grid(row=0, column=1, **grid_opts)
        tk.Label(p_frame, text="Fy:").grid(row=0, column=2, **grid_opts)
        tk.Entry(p_frame, textvariable=self.fy, width=8).grid(row=0, column=3, **grid_opts)
        
        tk.Label(p_frame, text="Cx:").grid(row=1, column=0, **grid_opts)
        tk.Entry(p_frame, textvariable=self.cx, width=8).grid(row=1, column=1, **grid_opts)
        tk.Label(p_frame, text="Cy:").grid(row=1, column=2, **grid_opts)
        tk.Entry(p_frame, textvariable=self.cy, width=8).grid(row=1, column=3, **grid_opts)
        
        tk.Label(p_frame, text="畸变 (k1,k2,p1,p2,k3):").grid(row=2, column=0, columnspan=4, **grid_opts)
        d_frame = tk.Frame(p_frame)
        d_frame.grid(row=3, column=0, columnspan=4)
        for i, var in enumerate([self.k1, self.k2, self.p1, self.p2, self.k3]):
            tk.Entry(d_frame, textvariable=var, width=6).pack(side=tk.LEFT, padx=1)

        # 2. 抓取偏移调节 (核心功能)
        o_frame = tk.LabelFrame(left_frame, text="2. 二维码 -> 物体 偏移调节", font=("bold", 10), fg="blue")
        o_frame.pack(fill=tk.X, padx=5, pady=5)
        
        def make_slider(parent, label, var, range_min, range_max):
            f = tk.Frame(parent)
            f.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(f, text=label, width=10, anchor='w').pack(side=tk.LEFT)
            tk.Scale(f, from_=range_min, to=range_max, variable=var, orient=tk.HORIZONTAL, command=lambda x: self.process_image()).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Entry(f, textvariable=var, width=6).pack(side=tk.RIGHT)

        tk.Label(o_frame, text="二维码边长 (mm):").pack(anchor='w', padx=5)
        tk.Entry(o_frame, textvariable=self.qr_size).pack(fill=tk.X, padx=5)

        make_slider(o_frame, "X 偏移(mm)", self.offset_x, -200, 200)
        make_slider(o_frame, "Y 偏移(mm)", self.offset_y, -200, 200)
        make_slider(o_frame, "Z 偏移(mm)", self.offset_z, -100, 100)
        make_slider(o_frame, "旋转(deg)", self.offset_angle, -180, 180)
        
        # 3. 图像操作
        i_frame = tk.LabelFrame(left_frame, text="3. 图像加载", font=("bold", 10))
        i_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(i_frame, text="加载测试图片", bg="#add8e6", command=self.load_image).pack(fill=tk.X, padx=5, pady=5)
        
        # 4. 实时数据输出
        r_frame = tk.LabelFrame(left_frame, text="4. 实时坐标计算 (相对于相机)", font=("bold", 10))
        r_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(r_frame, height=10, font=("Consolas", 10), bg="#000", fg="#0f0")
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # === 右侧图像区 ===
        right_frame = tk.Frame(main_paned, bg="#333")
        main_paned.add(right_frame, stretch="always")
        
        self.canvas = tk.Canvas(right_frame, bg="#333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        
    # --- 核心逻辑 ---
    
    def get_camera_matrix(self):
        return np.array([
            [self.fx.get(), 0, self.cx.get()],
            [0, self.fy.get(), self.cy.get()],
            [0, 0, 1]
        ], dtype=np.float64)
        
    def get_dist_coeffs(self):
        return np.array([self.k1.get(), self.k2.get(), self.p1.get(), self.p2.get(), self.k3.get()], dtype=np.float64)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        try:
            # 兼容中文路径
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.current_img = img
            self.process_image()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def process_image(self):
        if self.current_img is None: return
        
        img_disp = self.current_img.copy()
        K = self.get_camera_matrix()
        D = self.get_dist_coeffs()
        
        # 1. 检测二维码
        det = cv2.QRCodeDetector()
        ret, decoded_info, points, _ = det.detectAndDecodeMulti(img_disp)
        
        if not ret or points is None:
            self.log("未检测到二维码")
            self.update_canvas(img_disp)
            return
            
        pts = points[0]
        # 画二维码角点
        for i, p in enumerate(pts):
            cv2.circle(img_disp, tuple(map(int, p)), 5, (0,0,255), -1)
            
        # 2. PnP 求解二维码位姿 (T_cam_qr)
        half_sz = self.qr_size.get() / 2.0
        obj_pts = np.array([
            [-half_sz, half_sz, 0],
            [ half_sz, half_sz, 0],
            [ half_sz,-half_sz, 0],
            [-half_sz,-half_sz, 0]
        ], dtype=np.float64)
        
        success, rvec, tvec = cv2.solvePnP(obj_pts, pts, K, D)
        if not success: return
        
        # 绘制二维码坐标系 (RGB)
        cv2.drawFrameAxes(img_disp, K, D, rvec, tvec, half_sz)
        
        # 3. 计算抓取点位姿 (T_cam_obj = T_cam_qr * T_offset)
        # 将 rvec, tvec 转为 4x4 矩阵
        R_qr, _ = cv2.Rodrigues(rvec)
        T_cam_qr = np.eye(4)
        T_cam_qr[:3, :3] = R_qr
        T_cam_qr[:3, 3] = tvec.flatten()
        
        # 构建偏移矩阵 T_offset (QR -> Object)
        dx, dy, dz = self.offset_x.get(), self.offset_y.get(), self.offset_z.get()
        da = np.radians(self.offset_angle.get())
        
        T_offset = np.eye(4)
        # 旋转 (绕Z轴)
        R_off = np.array([
            [np.cos(da), -np.sin(da), 0],
            [np.sin(da),  np.cos(da), 0],
            [0,           0,          1]
        ])
        T_offset[:3, :3] = R_off
        T_offset[:3, 3] = [dx, dy, dz]
        
        # 级联: 相机下的物体位姿
        T_cam_obj = T_cam_qr @ T_offset
        
        # 4. 可视化抓取点 (黄色坐标系)
        # 提取物体 rvec, tvec 用于绘制
        rvec_obj, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
        tvec_obj = T_cam_obj[:3, 3]
        
        cv2.drawFrameAxes(img_disp, K, D, rvec_obj, tvec_obj, half_sz * 0.8)
        
        # 投射一个黄色圆圈代表抓取中心
        center_point_2d, _ = cv2.projectPoints(np.array([[0,0,0]], dtype=np.float64), rvec_obj, tvec_obj, K, D)
        center = tuple(map(int, center_point_2d[0].ravel()))
        cv2.circle(img_disp, center, 10, (0, 255, 255), 2) # 黄色空心圆
        cv2.putText(img_disp, "Grasp Point", (center[0]+15, center[1]), 0, 0.8, (0, 255, 255), 2)
        
        # 5. 输出数据
        pos = tvec_obj
        # 简单的欧拉角 (XYZ固定角) - 仅作参考，实际给机器人需要根据机器人定义转换
        # 这里输出的是相机坐标系下的坐标
        self.log("-" * 30)
        self.log(f"[抓取点 - 相机坐标系]")
        self.log(f"X: {pos[0]:.2f} mm")
        self.log(f"Y: {pos[1]:.2f} mm")
        self.log(f"Z: {pos[2]:.2f} mm")
        self.log(f"距离: {np.linalg.norm(pos):.2f} mm")
        self.log(f"\n[调试说明]")
        self.log("调节左侧滑块，使黄色坐标系")
        self.log("准确落在物体抓取中心")
        
        self.update_canvas(img_disp)

    def log(self, msg):
        self.txt_log.delete(1.0, tk.END) # 刷新模式
        self.txt_log.insert(tk.END, msg)

    # --- 图像显示辅助 ---
    def update_canvas(self, img):
        h, w = img.shape[:2]
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw<10: cw=800
        scale = min(cw/w, ch/h) * self.zoom
        nw, nh = int(w*scale), int(h*scale)
        if nw>0:
            pil = Image.fromarray(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=0))
            self.tk_image = ImageTk.PhotoImage(pil)
            self.canvas.delete("all")
            self.canvas.create_image(cw//2+self.pan_x, ch//2+self.pan_y, image=self.tk_image, anchor=tk.CENTER)

    def on_zoom(self, e):
        self.zoom *= (1.1 if e.delta>0 else 0.9); self.process_image()
    def on_drag_start(self, e):
        self.drag_start = (e.x, e.y)
    def on_drag_move(self, e):
        if self.drag_start:
            self.pan_x += e.x - self.drag_start[0]; self.pan_y += e.y - self.drag_start[1]
            self.drag_start = (e.x, e.y); self.process_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = GraspLogicTool(root)
    root.mainloop()
