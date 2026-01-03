import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import os
import math

class QRCodePoseEstimatorPro:
    def __init__(self, root):
        self.root = root
        self.root.title("二维码姿态解算工具（专业稳定版）")
        self.root.geometry("1400x900")
        
        # 相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calib_img_size = None  # 标定时的分辨率 (w, h)
        
        # 图像状态
        self.current_cv_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        
        # 姿态缓存（平滑用）
        self.pose_history = []
        self.smooth_window = 5
        
        # 初始化UI
        self.setup_ui()
        
    def setup_ui(self):
        # 1. 相机参数加载区
        top_frame = tk.LabelFrame(self.root, text="第一步：加载相机参数", padx=10, pady=5, font=("Arial", 10, "bold"))
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.param_status = tk.Label(top_frame, text="⚠ 未加载参数", fg="red", font=("Arial", 10, "bold"))
        self.param_status.pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="加载JSON参数", command=self.load_camera_params, bg="#e3f2fd").pack(side=tk.LEFT, padx=5)
        self.calib_info_label = tk.Label(top_frame, text="", fg="gray")
        self.calib_info_label.pack(side=tk.LEFT, padx=20)
        
        # 手动覆盖分辨率（防止JSON缺少分辨率信息）
        tk.Label(top_frame, text="标定分辨率(宽x高):").pack(side=tk.LEFT, padx=5)
        self.calib_w_var = tk.IntVar(value=0)
        self.calib_h_var = tk.IntVar(value=0)
        tk.Entry(top_frame, textvariable=self.calib_w_var, width=6).pack(side=tk.LEFT)
        tk.Label(top_frame, text="x").pack(side=tk.LEFT)
        tk.Entry(top_frame, textvariable=self.calib_h_var, width=6).pack(side=tk.LEFT)
        tk.Button(top_frame, text="强制更新缩放", command=self.update_scale_factor, bg="#ffebee").pack(side=tk.LEFT, padx=5)

        # 2. 操作区
        mid_frame = tk.LabelFrame(self.root, text="第二步：检测设置", padx=10, pady=5, font=("Arial", 10, "bold"))
        mid_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(mid_frame, text="打开图片", command=self.load_image, bg="#e8f5e9").pack(side=tk.LEFT, padx=5)
        self.curr_res_label = tk.Label(mid_frame, text="当前图片: 无", fg="blue")
        self.curr_res_label.pack(side=tk.LEFT, padx=5)
        
        tk.Label(mid_frame, text="|  二维码物理边长(mm):").pack(side=tk.LEFT, padx=(20, 5))
        self.qr_size_var = tk.DoubleVar(value=26.0)
        tk.Entry(mid_frame, textvariable=self.qr_size_var, width=8).pack(side=tk.LEFT)
        
        tk.Button(mid_frame, text="▶ 执行高精度解算", command=self.run_pose_estimation, bg="#fff3e0", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=20)
        
        # 3. 图像显示+结果区
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 图像画布
        canvas_frame = tk.Frame(main_paned, bg="#404040")
        self.canvas = tk.Canvas(canvas_frame, bg="#404040")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # 鼠标交互
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas_msg = self.canvas.create_text(400, 300, text="请按照步骤加载参数并打开图片", fill="gray", font=("Arial", 14))
        main_paned.add(canvas_frame, stretch="always")
        
        # 结果显示区
        result_frame = tk.Frame(main_paned, width=400)
        tk.Label(result_frame, text="解算日志 & 结果", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.result_text = tk.Text(result_frame, width=55, font=("Consolas", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        main_paned.add(result_frame, minsize=350)

    def log(self, msg, tag=None):
        self.result_text.insert(tk.END, msg + "\n", tag)
        self.result_text.see(tk.END)
        # 简单的颜色标签
        if tag == "error":
            self.result_text.tag_config("error", foreground="red")
        elif tag == "success":
            self.result_text.tag_config("success", foreground="green")
        elif tag == "warning":
            self.result_text.tag_config("warning", foreground="orange")

    def load_camera_params(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON文件", "*.json")])
        if not fpath:
            return
        
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            
            # 尝试读取分辨率
            if "image_size" in data:
                self.calib_img_size = tuple(data["image_size"])
                self.calib_w_var.set(self.calib_img_size[0])
                self.calib_h_var.set(self.calib_img_size[1])
                self.calib_info_label.config(text=f"标定分辨率: {self.calib_img_size[0]}x{self.calib_img_size[1]}")
            else:
                self.calib_img_size = None
                self.calib_info_label.config(text="⚠ 警告: JSON未包含分辨率信息", fg="orange")
                self.log("警告: JSON文件中缺少 'image_size' 字段。请在上方手动输入标定时使用的分辨率！", "warning")
            
            self.param_status.config(text=f"已加载: {os.path.basename(fpath)}", fg="green")
            self.log(f"参数加载成功: {fpath}", "success")
            
        except Exception as e:
            messagebox.showerror("加载失败", f"解析参数出错：{str(e)}")
            self.param_status.config(text="加载失败", fg="red")

    def update_scale_factor(self):
        """手动更新标定分辨率"""
        w = self.calib_w_var.get()
        h = self.calib_h_var.get()
        if w > 0 and h > 0:
            self.calib_img_size = (w, h)
            self.calib_info_label.config(text=f"标定分辨率(手动): {w}x{h}", fg="blue")
            self.log(f"已手动更新标定分辨率为: {w}x{h}")
        else:
            messagebox.showwarning("输入错误", "请输入有效的宽和高")

    def load_image(self):
        fpath = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg *.png *.bmp *.tif")])
        if not fpath:
            return
        
        try:
            # 读取图片
            self.current_cv_img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
            if self.current_cv_img is None:
                raise Exception("解码失败")
            
            # 统一转为BGR
            if len(self.current_cv_img.shape) == 2:
                self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_GRAY2BGR)
            elif self.current_cv_img.shape[2] == 4:
                self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGRA2BGR)
            
            h, w = self.current_cv_img.shape[:2]
            self.curr_res_label.config(text=f"当前图片: {w}x{h}")
            self.log(f"加载图片: {os.path.basename(fpath)} ({w}x{h})")
            
            # 检查分辨率匹配情况
            if self.calib_img_size:
                cw, ch = self.calib_img_size
                if cw != w or ch != h:
                    scale = w / cw
                    self.log(f"⚠ 注意: 图片分辨率 ({w}x{h}) 与标定分辨率 ({cw}x{ch}) 不一致", "warning")
                    self.log(f"  -> 将自动应用缩放系数: {scale:.4f}")
            else:
                self.log("⚠ 警告: 未知标定分辨率，假设为 1.0 (可能导致Z轴误差严重!)", "warning")

            self.scale = 1.0
            self.offset_x = 0
            self.offset_y = 0
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def get_scaled_camera_matrix(self, current_w, current_h):
        """获取适配当前分辨率的内参"""
        if self.camera_matrix is None:
            return None, 1.0
            
        if self.calib_img_size is None:
            # 如果没有标定分辨率，假设不需要缩放 (危险，但没办法)
            return self.camera_matrix, 1.0
            
        calib_w, _ = self.calib_img_size
        scale_factor = current_w / float(calib_w)
        
        if abs(scale_factor - 1.0) < 0.001:
            return self.camera_matrix, 1.0
            
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= scale_factor
        new_matrix[1, 1] *= scale_factor
        new_matrix[0, 2] *= scale_factor
        new_matrix[1, 2] *= scale_factor
        
        return new_matrix, scale_factor

    def run_pose_estimation(self):
        if self.camera_matrix is None:
            messagebox.showwarning("警告", "请先加载相机参数")
            return
        if self.current_cv_img is None:
            messagebox.showwarning("警告", "请先打开图片")
            return

        qr_size = self.qr_size_var.get()
        img_display = self.current_cv_img.copy()
        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
        
        # 1. 检测二维码
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(gray)
        
        if not retval or len(points) == 0:
            self.log("未检测到二维码", "error")
            return
            
        # 只取第一个检测到的
        pts_raw = points[0]  # shape (4, 2)
        
        # 2. 动态亚像素优化
        # 计算二维码在图像上的大致尺寸(像素)，以此决定搜索窗口大小
        edge_len = np.linalg.norm(pts_raw[0] - pts_raw[1])
        win_size = int(max(3, min(edge_len / 15, 30)))  # 动态窗口: 边长的1/15
        
        self.log(f"检测到二维码，像素边长约: {edge_len:.1f}px")
        self.log(f"亚像素搜索窗口: {win_size}x{win_size}")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        img_points = np.expand_dims(pts_raw, axis=1).astype(np.float32) # (4, 1, 2)
        
        img_points_sub = cv2.cornerSubPix(gray, img_points, (win_size, win_size), (-1,-1), criteria)
        
        # 3. 准备PnP参数
        h, w = img_display.shape[:2]
        mtx_scaled, scale_factor = self.get_scaled_camera_matrix(w, h)
        
        # 定义世界坐标系 (中心为原点，Z轴垂直向外?) 
        # OpenCV标准: X右, Y下, Z前(进屏幕). 
        # 为了让Z轴朝向相机，通常物体坐标系Z=0，且相机在Z<0的位置? 不，相机在原点，物体在Z>0.
        # 无论如何，我们定义物体四个角在平面 Z=0 上
        # 顺序: TL, TR, BR, BL (对应 detector 的输出顺序)
        half_s = qr_size / 2.0
        obj_points = np.array([
            [-half_s, -half_s, 0],  # TL (Top-Left)
            [ half_s, -half_s, 0],  # TR (Top-Right)
            [ half_s,  half_s, 0],  # BR (Bottom-Right)
            [-half_s,  half_s, 0]   # BL (Bottom-Left)
        ], dtype=np.float64)
        
        # 4. 解算 PnP (IPPE_SQUARE 最适合平面4点)
        success, rvec, tvec = cv2.solvePnP(
            obj_points, 
            img_points_sub, 
            mtx_scaled, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if not success:
            self.log("PnP解算失败", "error")
            return
            
        # 5. LM优化
        rvec, tvec = cv2.solvePnPRefineLM(obj_points, img_points_sub, mtx_scaled, self.dist_coeffs, rvec, tvec)
        
        # 6. 计算距离和欧拉角
        x, y, z = tvec.flatten()
        distance = np.linalg.norm(tvec)
        
        # 旋转矩阵 -> 欧拉角
        rmat, _ = cv2.Rodrigues(rvec)
        # 转换为 RPY (Roll=X, Pitch=Y, Yaw=Z)
        # 注意: 这里计算的是 物体相对于相机 的姿态
        sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(rmat[2,1], rmat[2,2])
            pitch = math.atan2(-rmat[2,0], sy)
            yaw = math.atan2(rmat[1,0], rmat[0,0])
        else:
            roll = math.atan2(-rmat[1,2], rmat[1,1])
            pitch = math.atan2(-rmat[2,0], sy)
            yaw = 0
            
        r_deg = np.degrees(roll)
        p_deg = np.degrees(pitch)
        y_deg = np.degrees(yaw)
        
        # 7. 绘制与显示
        # 限制轴长度，防止画到相机后面 (Z < 0)
        # 坐标系原点在物体中心，物体在 Z_camera = z 处.
        # 我们想画 Z轴(蓝色) 代表物体法向量.
        # 在物体坐标系中，法向量通常是 (0,0,1) 或 (0,0,-1).
        # 根据右手定则 (X右Y下), Z是垂直屏幕向里的.
        # 如果要画指向相机的轴，应该是 (0,0,-len).
        axis_len = min(qr_size, z * 0.5) # 动态长度，不超过距离的一半
        
        self.draw_overlay(img_display, img_points_sub, rvec, tvec, mtx_scaled, axis_len)
        self.current_cv_img = img_display
        self.display_image()
        
        # 8. 输出
        self.log("-" * 30)
        self.log(f"【高精度解算结果】")
        self.log(f"缩放系数: {scale_factor:.3f} (基准: {self.calib_img_size})")
        self.log(f"距离(Distance): {distance:.2f} mm")
        self.log(f"平移(X Y Z): {x:.2f}, {y:.2f}, {z:.2f} mm")
        self.log(f"旋转(R P Y): {r_deg:.2f}°, {p_deg:.2f}°, {y_deg:.2f}°")
        
        if z < qr_size:
            self.log("⚠ 警告: 计算出的Z距离极小，极可能是标定分辨率与图像不匹配！", "error")
            self.log("  -> 请检查 '标定分辨率' 是否正确输入。", "error")

    def draw_overlay(self, img, pts, rvec, tvec, mtx, length):
        pts = pts.astype(np.int32)
        
        # 1. 绘制边框
        for i in range(4):
            p1 = tuple(pts[i][0])
            p2 = tuple(pts[(i+1)%4][0])
            cv2.line(img, p1, p2, (0, 255, 0), 2)
            
        # 2. 绘制角点顺序 (红-绿-蓝-黄) 对应 0-1-2-3
        colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]
        for i, pt in enumerate(pts):
            cv2.circle(img, tuple(pt[0]), 6, colors[i], -1)
            cv2.putText(img, str(i), (pt[0][0]+10, pt[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

        # 3. 绘制中心坐标轴
        # 物体坐标系原点在中心 (0,0,0)
        axis_pts = np.float32([
            [0,0,0],      # Origin
            [length,0,0], # X
            [0,length,0], # Y
            [0,0,-length] # Z (反向，指向相机)
        ]).reshape(-1,3)
        
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, mtx, self.dist_coeffs)
        imgpts = imgpts.astype(np.int32)
        
        o = tuple(imgpts[0].ravel())
        x = tuple(imgpts[1].ravel())
        y = tuple(imgpts[2].ravel())
        z = tuple(imgpts[3].ravel())
        
        cv2.line(img, o, x, (0,0,255), 3) # X - Red
        cv2.line(img, o, y, (0,255,0), 3) # Y - Green
        cv2.line(img, o, z, (255,0,0), 3) # Z - Blue
        
        cv2.putText(img, "X", x, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(img, "Y", y, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(img, "Z", z, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    def display_image(self):
        if self.current_cv_img is None:
            return
        
        img_rgb = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # 画布尺寸
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        
        # 计算适合的缩放
        scale_fit = min(cw/w, ch/h)
        final_scale = scale_fit * self.scale
        
        new_w, new_h = int(w*final_scale), int(h*final_scale)
        if new_w < 1 or new_h < 1: return
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        self.pil_img = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        self.canvas.delete("all")
        cx = cw // 2 + self.offset_x
        cy = ch // 2 + self.offset_y
        self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.tk_img)

    def on_mouse_wheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.scale *= 0.9
        else:
            self.scale *= 1.1
        self.display_image()

    def on_mouse_press(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def on_mouse_drag(self, event):
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.offset_x += dx
        self.offset_y += dy
        self.last_x = event.x
        self.last_y = event.y
        self.display_image()

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    root = tk.Tk()
    app = QRCodePoseEstimatorPro(root)
    root.mainloop()
