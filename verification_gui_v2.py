import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import json
import os
import math

class VerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("相机测距验证工具 V2.1 (含姿态角度)")
        self.root.geometry("1400x900")
        
        # 默认参数
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 图像状态
        self.current_cv_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # 1. 顶部：参数加载区
        top_frame = tk.LabelFrame(self.root, text="第一步：加载相机参数", padx=10, pady=5)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_param_status = tk.Label(top_frame, text="未加载参数", fg="red", font=("Arial", 10, "bold"))
        self.lbl_param_status.pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="从 JSON 文件加载参数", command=self.load_params_from_file, bg="#e3f2fd").pack(side=tk.LEFT, padx=5)
        
        # 增加手动修正系数
        tk.Label(top_frame, text="   |   临时修正系数 (默认1.0):").pack(side=tk.LEFT)
        self.scale_factor_var = tk.DoubleVar(value=1.0)
        tk.Entry(top_frame, textvariable=self.scale_factor_var, width=6).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="应用系数并重算", command=self.apply_scale_factor, bg="#fff9c4").pack(side=tk.LEFT)
        tk.Label(top_frame, text="(如果结果大3倍，请输入0.333)").pack(side=tk.LEFT, padx=5, fg="gray")

        # 2. 中部：检测操作区
        mid_frame = tk.LabelFrame(self.root, text="第二步：检测与验证", padx=10, pady=5)
        mid_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(mid_frame, text="打开图片", command=self.load_image, bg="#e8f5e9", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Label(mid_frame, text="二维码真实边长(mm):").pack(side=tk.LEFT, padx=(20, 5))
        self.qr_size_var = tk.DoubleVar(value=26.0)
        tk.Entry(mid_frame, textvariable=self.qr_size_var, width=8).pack(side=tk.LEFT)
        
        tk.Button(mid_frame, text="执行测距与姿态解算", command=self.run_measurement, bg="#fff3e0", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=20)
        
        # 3. 主视图
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 图像显示
        center_frame = tk.Frame(main_paned, bg="#404040")
        self.canvas = tk.Canvas(center_frame, bg="#404040")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        
        self.canvas_msg = self.canvas.create_text(400, 300, text="请先加载参数，然后打开图片", fill="gray", font=("Arial", 14), anchor=tk.CENTER)
        
        main_paned.add(center_frame, stretch="always")
        
        # 侧边栏：结果信息
        side_frame = tk.Frame(main_paned, width=350)
        tk.Label(side_frame, text="测量结果 (包含旋转角度):").pack(anchor=tk.W)
        self.txt_result = tk.Text(side_frame, width=50, font=("Consolas", 10))
        self.txt_result.pack(fill=tk.BOTH, expand=True)
        main_paned.add(side_frame, minsize=300)
        
    def load_params_from_file(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not fpath: return
        
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 保存原始参数，方便反复重置
            self.original_camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.camera_matrix = self.original_camera_matrix.copy()
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            
            self.lbl_param_status.config(text=f"参数已加载: {os.path.basename(fpath)}", fg="green")
            self.log(f"成功加载参数文件: {fpath}")
            self.log(f"Fx={self.camera_matrix[0,0]:.2f}, Fy={self.camera_matrix[1,1]:.2f}")
            
        except Exception as e:
            messagebox.showerror("加载失败", f"无法解析参数文件: {e}")

    def apply_scale_factor(self):
        if not hasattr(self, 'original_camera_matrix'):
            return
            
        try:
            factor = self.scale_factor_var.get()
        except:
            return
            
        # 修正内参：fx, fy 乘以系数
        self.camera_matrix = self.original_camera_matrix.copy()
        self.camera_matrix[0, 0] *= factor
        self.camera_matrix[1, 1] *= factor
        
        self.log("-" * 30)
        self.log(f"应用修正系数: {factor}")
        self.log(f"新 Fx={self.camera_matrix[0,0]:.2f}, 新 Fy={self.camera_matrix[1,1]:.2f}")
        
        # 如果有图，直接重算
        if self.current_cv_img is not None:
            self.run_measurement()

    def load_image(self):
        fpath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp *.tif")])
        if not fpath: return
        
        try:
            self.current_cv_img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
            # 自动处理通道
            if self.current_cv_img is not None:
                if len(self.current_cv_img.shape) == 2:
                    self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_GRAY2BGR)
                elif self.current_cv_img.shape[2] == 4:
                    self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGRA2BGR)
            
            if self.current_cv_img is None: raise Exception("Decode failed")
        except Exception as e:
            messagebox.showerror("错误", f"无法读取图片: {e}")
            return
            
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_image()
        self.log(f"已加载图片: {fpath}")

    def run_measurement(self):
        if self.camera_matrix is None:
            messagebox.showwarning("警告", "请先加载相机参数！")
            return
        if self.current_cv_img is None:
            messagebox.showwarning("警告", "请先打开一张图片！")
            return
            
        try:
            qr_real_size = self.qr_size_var.get()
        except:
            messagebox.showerror("错误", "请输入有效的二维码尺寸！")
            return

        # 1. 检测二维码
        img_temp = self.current_cv_img.copy()
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(img_temp)
        
        if not retval:
            self.log("未检测到二维码。")
            return
            
        self.log(f"检测到 {len(points)} 个二维码。")
        
        # 准备 3D 坐标
        obj_points = np.array([
            [0, 0, 0],              # TL
            [qr_real_size, 0, 0],   # TR
            [qr_real_size, qr_real_size, 0], # BR
            [0, qr_real_size, 0]    # BL
        ], dtype=np.float64)
        
        for i in range(len(points)):
            img_points = points[i].reshape(4, 2).astype(np.float64)
            
            # PnP 解算
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs)
            
            if success:
                dist_mm = np.linalg.norm(tvec)
                
                # --- 计算欧拉角 (旋转角度) ---
                # Rodrigues 变换: 旋转向量 -> 旋转矩阵
                rmat, _ = cv2.Rodrigues(rvec)
                
                # 计算欧拉角 (Pitch, Yaw, Roll) - 这里的顺序取决于你的坐标系定义
                # 一般习惯用 XYZ 顺序 (Pitch, Yaw, Roll)
                # pitch = atan2(R[2,1], R[2,2])
                # yaw = atan2(-R[2,0], sqrt(R[2,1]^2 + R[2,2]^2))
                # roll = atan2(R[1,0], R[0,0])
                
                sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
                singular = sy < 1e-6

                if not singular:
                    x = math.atan2(rmat[2,1] , rmat[2,2])
                    y = math.atan2(-rmat[2,0], sy)
                    z = math.atan2(rmat[1,0], rmat[0,0])
                else:
                    x = math.atan2(-rmat[1,2], rmat[1,1])
                    y = math.atan2(-rmat[2,0], sy)
                    z = 0

                rx = math.degrees(x)
                ry = math.degrees(y)
                rz = math.degrees(z)
                
                # 绘制结果
                self._draw_overlay(img_temp, img_points, rvec, tvec, dist_mm, (rx, ry, rz), i)
                
                # 详细日志
                self.log(f"\n--- QR Code #{i+1} ---")
                self.log(f"内容: {decoded_info[i]}")
                self.log(f"计算距离: {dist_mm:.2f} mm")
                self.log(f"平移 (X, Y, Z):")
                self.log(f"  X={tvec[0][0]:.2f}, Y={tvec[1][0]:.2f}, Z={tvec[2][0]:.2f}")
                self.log(f"旋转角度 (欧拉角):")
                self.log(f"  Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
            else:
                self.log(f"QR #{i+1} PnP解算失败")

        self.current_cv_img = img_temp
        self.display_image()

    def _draw_overlay(self, img, pts, rvec, tvec, dist, angles, idx):
        pts = pts.astype(np.int32)
        rx, ry, rz = angles
        
        # 1. 画绿色边框
        for j in range(4):
            cv2.line(img, tuple(pts[j]), tuple(pts[(j+1)%4]), (0, 255, 0), 2)
            
        # 2. 标记角点顺序
        labels = ["TL", "TR", "BR", "BL"]
        for j, pt in enumerate(pts):
            cv2.circle(img, tuple(pt), 4, (0, 0, 255), -1)
            cv2.putText(img, labels[j], (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 3. 绘制坐标轴
        axis_len = 15.0 
        axis_pts = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        imgpts = imgpts.astype(np.int32)
        
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3) # X - Red
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3) # Y - Green
        cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3) # Z - Blue
        
        # 4. 显示文字
        center_x = int(np.mean(pts[:,0]))
        center_y = int(np.mean(pts[:,1]))
        
        info_dist = f"Dist: {dist:.1f}mm"
        info_rot = f"Rot: {rx:.1f}, {ry:.1f}, {rz:.1f}"
        
        cv2.putText(img, info_dist, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, info_rot, (center_x, center_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def display_image(self):
        if self.current_cv_img is None: return
        
        img_rgb = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10: canvas_w = 800
        if canvas_h < 10: canvas_h = 600
        
        scale_fit = min(canvas_w / w, canvas_h / h)
        final_scale = scale_fit * self.scale
        
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        if new_w <= 0 or new_h <= 0: return
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        self.pil_img = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        self.canvas.delete("all")
        cx = canvas_w // 2 + self.offset_x
        cy = canvas_h // 2 + self.offset_y
        self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.tk_img)

    def log(self, msg):
        self.txt_result.insert(tk.END, msg + "\n")
        self.txt_result.see(tk.END)

    # 鼠标交互
    def on_mouse_wheel(self, event):
        if self.current_cv_img is None: return
        if event.num == 5 or event.delta < 0:
            self.scale *= 0.9
        else:
            self.scale *= 1.1
        self.scale = max(0.1, min(self.scale, 50.0))
        self.display_image()

    def on_mouse_press(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def on_mouse_drag(self, event):
        if self.current_cv_img is None: return
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.offset_x += dx
        self.offset_y += dy
        self.last_x = event.x
        self.last_y = event.y
        self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    app = VerificationApp(root)
    root.mainloop()
