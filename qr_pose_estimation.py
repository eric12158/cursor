import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import os
import math

class QRCodePoseEstimator:
    def __init__(self, root):
        self.root = root
        self.root.title("二维码姿态解算工具（高精度版）")
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
        self.smooth_window = 3
        
        # 初始化UI
        self.setup_ui()
        
    def setup_ui(self):
        # 1. 相机参数加载区
        top_frame = tk.LabelFrame(self.root, text="相机参数加载", padx=10, pady=5)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.param_status = tk.Label(top_frame, text="未加载参数", fg="red", font=("Arial", 10, "bold"))
        self.param_status.pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="加载JSON参数", command=self.load_camera_params, bg="#e3f2fd").pack(side=tk.LEFT, padx=5)
        self.calib_res_label = tk.Label(top_frame, text="", fg="gray")
        self.calib_res_label.pack(side=tk.LEFT, padx=20)

        # 2. 操作区
        mid_frame = tk.LabelFrame(self.root, text="检测设置", padx=10, pady=5)
        mid_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(mid_frame, text="打开图片", command=self.load_image, bg="#e8f5e9").pack(side=tk.LEFT, padx=5)
        self.curr_res_label = tk.Label(mid_frame, text="", fg="blue")
        self.curr_res_label.pack(side=tk.LEFT, padx=5)
        
        tk.Label(mid_frame, text="二维码边长(mm):").pack(side=tk.LEFT, padx=(20, 5))
        self.qr_size_var = tk.DoubleVar(value=26.0)
        tk.Entry(mid_frame, textvariable=self.qr_size_var, width=8).pack(side=tk.LEFT)
        
        tk.Button(mid_frame, text="解算位姿（高精度）", command=self.run_pose_estimation, bg="#fff3e0", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=20)

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
        self.canvas_msg = self.canvas.create_text(400, 300, text="请先加载参数和图片", fill="gray", font=("Arial", 14))
        main_paned.add(canvas_frame, stretch="always")
        
        # 结果显示区
        result_frame = tk.Frame(main_paned, width=350)
        tk.Label(result_frame, text="位姿解算结果（机械臂可用）", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.result_text = tk.Text(result_frame, width=50, font=("Consolas", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        main_paned.add(result_frame, minsize=300)

    def load_camera_params(self):
        """加载相机内参和畸变系数（JSON格式）"""
        fpath = filedialog.askopenfilename(filetypes=[("JSON文件", "*.json")])
        if not fpath:
            return
        
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            
            # 读取标定时的分辨率
            if "image_size" in data:
                self.calib_img_size = tuple(data["image_size"])
                self.calib_res_label.config(text=f"标定分辨率: {self.calib_img_size[0]}x{self.calib_img_size[1]}")
            
            self.param_status.config(text=f"参数已加载: {os.path.basename(fpath)}", fg="green")
            self.log(f"成功加载相机参数：{fpath}")
            
        except Exception as e:
            messagebox.showerror("加载失败", f"解析参数出错：{str(e)}")
            self.param_status.config(text="参数加载失败", fg="red")

    def load_image(self):
        """加载待检测图片"""
        fpath = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg *.png *.bmp *.tif")])
        if not fpath:
            return
        
        try:
            # 读取图片（兼容中文路径）
            self.current_cv_img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
            if self.current_cv_img is None:
                raise Exception("图片解码失败")
            
            # 格式转换（灰度/透明通道）
            if len(self.current_cv_img.shape) == 2:
                self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_GRAY2BGR)
            elif self.current_cv_img.shape[2] == 4:
                self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGRA2BGR)
            
            h, w = self.current_cv_img.shape[:2]
            self.curr_res_label.config(text=f"当前分辨率: {w}x{h}")
            self.log(f"加载图片：{fpath} ({w}x{h})")
            
            # 重置缩放和平移
            self.scale = 1.0
            self.offset_x = 0
            self.offset_y = 0
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("加载失败", f"读取图片出错：{str(e)}")

    def get_scaled_camera_matrix(self, current_w, current_h):
        """根据分辨率差异缩放相机内参"""
        if self.calib_img_size is None or self.camera_matrix is None:
            return self.camera_matrix, 1.0
        
        calib_w, calib_h = self.calib_img_size
        scale_factor = current_w / float(calib_w)
        
        # 差异过小则不缩放
        if abs(scale_factor - 1.0) < 0.01:
            return self.camera_matrix, 1.0
        
        self.log(f"分辨率不匹配，自动缩放内参（缩放系数：{scale_factor:.4f}）")
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= scale_factor  # fx
        new_matrix[1, 1] *= scale_factor  # fy
        new_matrix[0, 2] *= scale_factor  # cx
        new_matrix[1, 2] *= scale_factor  # cy
        
        return new_matrix, scale_factor

    def run_pose_estimation(self):
        """核心：高精度二维码位姿解算"""
        # 前置校验
        if self.camera_matrix is None:
            messagebox.showwarning("警告", "请先加载相机参数！")
            return
        if self.current_cv_img is None:
            messagebox.showwarning("警告", "请先打开图片！")
            return
        
        try:
            qr_size = self.qr_size_var.get()
            if qr_size <= 0:
                raise Exception("二维码边长必须大于0")
        except:
            messagebox.showerror("错误", "请输入有效的二维码边长！")
            return

    
        img_temp = self.current_cv_img.copy()
        gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(gray)    
        if not retval:
            self.log("未检测到二维码！")
            return
        
        self.log(f"检测到 {len(points)} 个二维码，处理第1个...")
        img_points = np.float32(points[0]).reshape(-1, 1, 2)  # 转换为(4,1,2)格式
        img = self.current_cv_img.copy()
        h, w = img.shape[:2]
        camera_matrix_scaled, _ = self.get_scaled_camera_matrix(w, h)
        # 3. 亚像素角点细化（核心：解决img_points_sub未定义问题）
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        img_points_sub = cv2.cornerSubPix(
            gray,          # 灰度图（用于梯度计算）
            img_points,    # 原始角点
            (15, 15),      # 搜索窗口
            (-1, -1),      # 死区
            criteria       # 迭代条件
        )
        
        # 格式校验
        if img_points_sub.shape != (4, 1, 2):
            self.log("亚像素角点格式异常，降级使用原始角点")
            img_points_sub = img_points

        # 4. 定义二维码3D物理坐标 (Centered at origin)
        half_size = qr_size / 2.0
        obj_points = np.array([
            [-half_size, -half_size, 0],    # Top-Left (Index 0)
            [ half_size, -half_size, 0],    # Top-Right (Index 1)
            [ half_size,  half_size, 0],    # Bottom-Right (Index 2)
            [-half_size,  half_size, 0]     # Bottom-Left (Index 3)
        ], dtype=np.float64)

        # 5. PnP初始解（IPPE_SQUARE：正方形目标专用，精度更高）
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points_sub,
            camera_matrix_scaled,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            self.log("PnP初始解算失败！")
            return

        # 6. LM迭代优化（细化姿态）
        try:
            rvec_refine, tvec_refine = cv2.solvePnPRefineLM(
                obj_points,
                img_points_sub,
                camera_matrix_scaled,
                self.dist_coeffs,
                rvec,
                tvec
            )
            self.log("LM迭代细化成功")
        except Exception as e:
            self.log(f"LM迭代失败，使用初始解：{str(e)}")
            rvec_refine, tvec_refine = rvec, tvec

        # 7. 转换为机械臂可用的RPY欧拉角（X/Y/Z旋转，角度制）
        rmat, _ = cv2.Rodrigues(rvec_refine)
        sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(rmat[2,1], rmat[2,2])    # X轴（横滚）
            pitch = math.atan2(-rmat[2,0], sy)         # Y轴（俯仰）
            yaw = math.atan2(rmat[1,0], rmat[0,0])      # Z轴（偏航）
        else:
            roll = math.atan2(-rmat[1,2], rmat[1,1])
            pitch = math.atan2(-rmat[2,0], sy)
            yaw = 0.0

        # 弧度转角度
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        # 8. 平移坐标（mm）
        x = tvec_refine[0][0]
        y = tvec_refine[1][0]
        z = tvec_refine[2][0]
        distance = np.linalg.norm(tvec_refine)

        # 9. 平滑处理（抑制误差波动）
        current_pose = (x, y, z, roll_deg, pitch_deg, yaw_deg)
        self.pose_history.append(current_pose)
        if len(self.pose_history) > self.smooth_window:
            self.pose_history.pop(0)
        
        # 计算平滑后的值
        smooth_x = np.mean([p[0] for p in self.pose_history])
        smooth_y = np.mean([p[1] for p in self.pose_history])
        smooth_z = np.mean([p[2] for p in self.pose_history])
        smooth_roll = np.mean([p[3] for p in self.pose_history])
        smooth_pitch = np.mean([p[4] for p in self.pose_history])
        smooth_yaw = np.mean([p[5] for p in self.pose_history])

        # 10. 绘制结果到图像
        self.draw_overlay(img, img_points_sub, rvec_refine, tvec_refine, 
                          (smooth_roll, smooth_pitch, smooth_yaw), distance)
        self.current_cv_img = img
        self.display_image()

        # 11. 输出结果（机械臂可直接复制）
        self.log("="*40)
        self.log("位姿解算结果")
        self.log(f"距离: {distance:.2f} mm (Z轴: {smooth_z:.2f} mm)")
        self.log(f"平移坐标 (X Y Z): {smooth_x:.2f}  {smooth_y:.2f}  {smooth_z:.2f} mm")
        self.log(f"旋转姿态 (Roll Pitch Yaw): {smooth_roll:.2f}°  {smooth_pitch:.2f}°  {smooth_yaw:.2f}°")
       
        self.log("="*40)

    def draw_overlay(self, img, pts, rvec, tvec, angles, distance):
        """在图像上绘制二维码框、坐标轴、位姿信息"""
        pts = pts.astype(np.int32)
        roll, pitch, yaw = angles

        # 绘制二维码边框
        for i in range(4):
            cv2.line(img, tuple(pts[i][0]), tuple(pts[(i+1)%4][0]), (0, 255, 0), 2)
        
        # 标记角点
        labels = ["TL", "TR", "BR", "BL"]
        for i, pt in enumerate(pts):
            cv2.circle(img, tuple(pt[0]), 4, (0, 0, 255), -1)
            cv2.putText(img, labels[i], (pt[0][0]+10, pt[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 2)

        # 绘制3D坐标轴
        axis_len = 15.0
        axis_pts = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        imgpts = imgpts.astype(np.int32)
        
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 3)  # X轴（红）
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 3)  # Y轴（绿）
        cv2.line(img, origin, tuple(imgpts[3].ravel()), (255,0,0), 3)  # Z轴（蓝）

        # 绘制位姿信息
        center_x = int(np.mean(pts[:,0,0]))
        center_y = int(np.mean(pts[:,0,1]))
        cv2.putText(img, f"距离: {distance:.1f}mm", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(img, f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}", (center_x, center_y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    def display_image(self):
        """显示图像到画布"""
        if self.current_cv_img is None:
            return
        
        img_rgb = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # 适配画布大小
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        scale_fit = min(canvas_w / w, canvas_h / h)
        final_scale = scale_fit * self.scale
        
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        if new_w <=0 or new_h <=0:
            return
        
        # 缩放图像
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.pil_img = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        # 绘制到画布
        self.canvas.delete("all")
        cx = canvas_w // 2 + self.offset_x
        cy = canvas_h // 2 + self.offset_y
        self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.tk_img)

    def log(self, msg):
        """输出日志到结果文本框"""
        self.result_text.insert(tk.END, msg + "\n")
        self.result_text.see(tk.END)

    # 鼠标交互函数
    def on_mouse_wheel(self, event):
        """滚轮缩放"""
        if self.current_cv_img is None:
            return
        if event.num == 5 or event.delta < 0:
            self.scale *= 0.9
        else:
            self.scale *= 1.1
        self.scale = max(0.1, min(self.scale, 50.0))
        self.display_image()

    def on_mouse_press(self, event):
        """鼠标按下"""
        self.last_x = event.x
        self.last_y = event.y

    def on_mouse_drag(self, event):
        """鼠标拖动"""
        if self.current_cv_img is None:
            return
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.offset_x += dx
        self.offset_y += dy
        self.last_x = event.x
        self.last_y = event.y
        self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    # 适配高DPI屏幕
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = QRCodePoseEstimator(root)
    root.mainloop()