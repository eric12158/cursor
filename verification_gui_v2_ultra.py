import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import json
import os
import math
from collections import deque

class VerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("相机测距验证工具 V3.0 (超稳定增强版)")
        self.root.geometry("1500x950")
        
        # 默认参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calib_img_size = None
        
        # 图像状态
        self.current_cv_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        
        # 历史数据（用于时间序列滤波）
        self.history_size = 5  # 保留最近5帧
        self.pose_history = deque(maxlen=self.history_size)
        
        self.setup_ui()
        
    def setup_ui(self):
        # 1. 顶部：参数加载区
        top_frame = tk.LabelFrame(self.root, text="第一步：加载相机参数", padx=10, pady=5)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_param_status = tk.Label(top_frame, text="未加载参数", fg="red", font=("Arial", 10, "bold"))
        self.lbl_param_status.pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="从 JSON 文件加载参数", command=self.load_params_from_file, bg="#e3f2fd").pack(side=tk.LEFT, padx=5)
        
        self.lbl_calib_res = tk.Label(top_frame, text="", fg="gray")
        self.lbl_calib_res.pack(side=tk.LEFT, padx=20)

        # 2. 中部：检测操作区
        mid_frame = tk.LabelFrame(self.root, text="第二步：检测与验证", padx=10, pady=5)
        mid_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(mid_frame, text="打开图片", command=self.load_image, bg="#e8f5e9", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.lbl_curr_res = tk.Label(mid_frame, text="", fg="blue")
        self.lbl_curr_res.pack(side=tk.LEFT, padx=5)
        
        tk.Label(mid_frame, text="二维码真实边长(mm):").pack(side=tk.LEFT, padx=(20, 5))
        self.qr_size_var = tk.DoubleVar(value=26.0)
        tk.Entry(mid_frame, textvariable=self.qr_size_var, width=8).pack(side=tk.LEFT)
        
        tk.Button(mid_frame, text="执行测距与姿态解算", command=self.run_measurement, bg="#fff3e0", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=20)
        
        # 高级选项
        adv_frame = tk.LabelFrame(self.root, text="高级优化选项", padx=10, pady=5)
        adv_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.use_multi_solve = tk.BooleanVar(value=True)
        tk.Checkbutton(adv_frame, text="多次求解平均(推荐)", variable=self.use_multi_solve).pack(side=tk.LEFT, padx=5)
        
        self.use_temporal_filter = tk.BooleanVar(value=False)
        tk.Checkbutton(adv_frame, text="时间序列滤波(多图)", variable=self.use_temporal_filter).pack(side=tk.LEFT, padx=5)
        
        self.use_edge_enhance = tk.BooleanVar(value=True)
        tk.Checkbutton(adv_frame, text="边缘增强", variable=self.use_edge_enhance).pack(side=tk.LEFT, padx=5)
        
        tk.Button(adv_frame, text="清除历史数据", command=self.clear_history, bg="#ffebee").pack(side=tk.LEFT, padx=20)
        
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
        side_frame = tk.Frame(main_paned, width=400)
        tk.Label(side_frame, text="测量结果 (超稳定版):").pack(anchor=tk.W)
        self.txt_result = tk.Text(side_frame, width=55, font=("Consolas", 9))
        self.txt_result.pack(fill=tk.BOTH, expand=True)
        main_paned.add(side_frame, minsize=350)
        
    def clear_history(self):
        self.pose_history.clear()
        self.log("已清除历史数据")
        
    def load_params_from_file(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not fpath: return
        
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            
            if "image_size" in data:
                self.calib_img_size = tuple(data["image_size"])
                self.lbl_calib_res.config(text=f"标定分辨率: {self.calib_img_size[0]}x{self.calib_img_size[1]}")
            else:
                self.calib_img_size = None
                self.lbl_calib_res.config(text="标定分辨率: 未知 (旧版数据)")

            self.lbl_param_status.config(text=f"参数已加载: {os.path.basename(fpath)}", fg="green")
            self.log(f"成功加载参数文件: {fpath}")
            if self.calib_img_size:
                self.log(f"标定分辨率: {self.calib_img_size}")
            
        except Exception as e:
            messagebox.showerror("加载失败", f"无法解析参数文件: {e}")

    def load_image(self):
        fpath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp *.tif")])
        if not fpath: return
        
        try:
            self.current_cv_img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
            if self.current_cv_img is not None:
                if len(self.current_cv_img.shape) == 2:
                    self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_GRAY2BGR)
                elif self.current_cv_img.shape[2] == 4:
                    self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGRA2BGR)
            
            if self.current_cv_img is None: raise Exception("Decode failed")
        except Exception as e:
            messagebox.showerror("错误", f"无法读取图片: {e}")
            return
            
        h, w = self.current_cv_img.shape[:2]
        self.lbl_curr_res.config(text=f"当前图片分辨率: {w}x{h}")
        self.log(f"已加载图片: {fpath} ({w}x{h})")
        
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_image()

    def get_scaled_camera_matrix(self, current_w, current_h):
        """根据分辨率差异自动缩放内参"""
        if self.calib_img_size is None or self.camera_matrix is None:
            return self.camera_matrix, 1.0
            
        calib_w, calib_h = self.calib_img_size
        scale_factor = current_w / float(calib_w)
        
        if abs(scale_factor - 1.0) < 0.01:
            return self.camera_matrix, 1.0
            
        self.log("-" * 30)
        self.log(f"检测到分辨率不匹配！")
        self.log(f"标定: {calib_w}x{calib_h} -> 当前: {current_w}x{current_h}")
        self.log(f"自动应用缩放系数: {scale_factor:.4f}")
        
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= scale_factor
        new_matrix[1, 1] *= scale_factor
        new_matrix[0, 2] *= scale_factor
        new_matrix[1, 2] *= scale_factor
        
        return new_matrix, scale_factor

    def enhance_image(self, gray):
        """增强版图像预处理"""
        # 1. CLAHE 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. 可选：边缘增强
        if self.use_edge_enhance.get():
            # 锐化内核
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]], dtype=np.float32) / 1.0
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # 3. 双边滤波（保边降噪）
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        return enhanced

    def refine_corners_ultra(self, gray, corners):
        """超精细亚像素优化"""
        # 第一次优化：大窗口
        criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        refined1 = cv2.cornerSubPix(
            gray, 
            corners.copy(), 
            winSize=(15, 15),  # 更大的窗口 31x31
            zeroZone=(-1, -1),
            criteria=criteria1
        )
        
        # 第二次优化：小窗口精修
        criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.00001)
        refined2 = cv2.cornerSubPix(
            gray, 
            refined1.copy(), 
            winSize=(5, 5),  # 小窗口微调
            zeroZone=(-1, -1),
            criteria=criteria2
        )
        
        return refined2

    def solve_pnp_multiple_times(self, obj_points, img_points, camera_matrix, dist_coeffs, n_trials=5):
        """多次求解取最优结果"""
        results = []
        
        for trial in range(n_trials):
            # 添加微小扰动（模拟不同初始值）
            if trial > 0:
                noise = np.random.randn(*img_points.shape) * 0.01  # 0.01像素的噪声
                img_points_noisy = img_points + noise
            else:
                img_points_noisy = img_points.copy()
            
            # 方法1: IPPE_SQUARE (仅4点)
            if len(obj_points) >= 4:
                try:
                    success1, rvec1, tvec1 = cv2.solvePnP(
                        obj_points[:4], 
                        img_points_noisy[:4], 
                        camera_matrix, 
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if success1:
                        # LM优化
                        rvec1, tvec1 = cv2.solvePnPRefineLM(
                            obj_points[:4], img_points[:4], camera_matrix, dist_coeffs,
                            rvec1, tvec1
                        )
                        # 计算重投影误差
                        reproj, _ = cv2.projectPoints(obj_points[:4], rvec1, tvec1, camera_matrix, dist_coeffs)
                        error1 = np.mean(np.linalg.norm(img_points[:4] - reproj.reshape(-1, 2), axis=1))
                        results.append((rvec1, tvec1, error1, 'IPPE_SQUARE'))
                except:
                    pass
            
            # 方法2: 迭代法 (全部点)
            try:
                success2, rvec2, tvec2 = cv2.solvePnP(
                    obj_points, 
                    img_points_noisy, 
                    camera_matrix, 
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success2:
                    # LM优化
                    rvec2, tvec2 = cv2.solvePnPRefineLM(
                        obj_points, img_points, camera_matrix, dist_coeffs,
                        rvec2, tvec2
                    )
                    reproj, _ = cv2.projectPoints(obj_points, rvec2, tvec2, camera_matrix, dist_coeffs)
                    error2 = np.mean(np.linalg.norm(img_points - reproj.reshape(-1, 2), axis=1))
                    results.append((rvec2, tvec2, error2, 'ITERATIVE'))
            except:
                pass
        
        if not results:
            return None, None, 999.0, 'FAILED'
        
        # 选择误差最小的结果
        best = min(results, key=lambda x: x[2])
        
        # 如果有多个结果，计算中位数（更鲁棒）
        if len(results) >= 3:
            rvecs = np.array([r[0].ravel() for r in results])
            tvecs = np.array([r[1].ravel() for r in results])
            
            rvec_median = np.median(rvecs, axis=0).reshape(3, 1)
            tvec_median = np.median(tvecs, axis=0).reshape(3, 1)
            
            return rvec_median, tvec_median, best[2], best[3] + '_MEDIAN'
        
        return best[0], best[1], best[2], best[3]

    def rotation_matrix_to_quaternion(self, R):
        """旋转矩阵转四元数（避免万向节死锁）"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])

    def quaternion_to_euler(self, q):
        """四元数转欧拉角（更稳定）"""
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def apply_temporal_filter(self, rvec, tvec):
        """时间序列滤波（多帧平均）"""
        if not self.use_temporal_filter.get():
            return rvec, tvec
        
        # 添加到历史
        self.pose_history.append((rvec.copy(), tvec.copy()))
        
        if len(self.pose_history) < 2:
            return rvec, tvec
        
        # 使用移动平均
        rvecs = np.array([r.ravel() for r, t in self.pose_history])
        tvecs = np.array([t.ravel() for r, t in self.pose_history])
        
        # 加权平均（最近的权重更大）
        weights = np.exp(np.linspace(-1, 0, len(self.pose_history)))
        weights /= weights.sum()
        
        rvec_filtered = np.average(rvecs, axis=0, weights=weights).reshape(3, 1)
        tvec_filtered = np.average(tvecs, axis=0, weights=weights).reshape(3, 1)
        
        return rvec_filtered, tvec_filtered

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

        # 1. 自动适配相机参数
        h, w = self.current_cv_img.shape[:2]
        current_matrix, scale_applied = self.get_scaled_camera_matrix(w, h)

        # 2. 增强预处理
        img_temp = self.current_cv_img.copy()
        gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        gray_enhanced = self.enhance_image(gray)
        
        # 3. 检测二维码
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(gray_enhanced)
 
        if not retval:
            self.log("未检测到二维码。")
            return
            
        self.log(f"检测到 {len(points)} 个二维码。")
        self.log("=" * 50)
        
        # 准备 3D 坐标（5个点）
        half_size = qr_real_size / 2.0
        obj_points = np.array([
            [0, 0, 0],
            [qr_real_size, 0, 0],
            [qr_real_size, qr_real_size, 0],
            [0, qr_real_size, 0],
            [half_size, half_size, 0]
        ], dtype=np.float64)
        
        for i in range(len(points)):
            self.log(f"\n--- QR Code #{i+1} ---")
            
            # 获取角点
            img_points_4 = points[i].reshape(4, 2).astype(np.float32)
            center_point = np.mean(img_points_4, axis=0, keepdims=True).astype(np.float32)
            img_points_5 = np.vstack([img_points_4, center_point])
            
            # 超精细亚像素优化
            img_points_refined = self.refine_corners_ultra(gray_enhanced, img_points_5)
            img_points_refined = img_points_refined.astype(np.float64)
            
            pixel_shift = np.linalg.norm(img_points_5 - img_points_refined, axis=1)
            self.log(f"亚像素优化 - 平均: {np.mean(pixel_shift):.4f}px, 最大: {np.max(pixel_shift):.4f}px")
            
            # 多次求解
            if self.use_multi_solve.get():
                rvec, tvec, error, method = self.solve_pnp_multiple_times(
                    obj_points, img_points_refined, current_matrix, self.dist_coeffs, n_trials=5
                )
                self.log(f"采用方法: {method} (重投影误差: {error:.4f}px)")
            else:
                # 单次求解
                success, rvec, tvec = cv2.solvePnP(
                    obj_points[:4], img_points_refined[:4], 
                    current_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if success:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        obj_points[:4], img_points_refined[:4], 
                        current_matrix, self.dist_coeffs, rvec, tvec
                    )
                    reproj, _ = cv2.projectPoints(obj_points[:4], rvec, tvec, current_matrix, self.dist_coeffs)
                    error = np.mean(np.linalg.norm(img_points_refined[:4] - reproj.reshape(-1, 2), axis=1))
                    self.log(f"采用方法: IPPE_SQUARE (重投影误差: {error:.4f}px)")
                else:
                    self.log("PnP求解失败")
                    continue
            
            if rvec is None:
                continue
            
            # 时间序列滤波
            if self.use_temporal_filter.get():
                rvec_raw = rvec.copy()
                tvec_raw = tvec.copy()
                rvec, tvec = self.apply_temporal_filter(rvec, tvec)
                self.log(f"时间滤波: 已使用 {len(self.pose_history)} 帧数据")
            
            # 计算距离
            dist_mm = np.linalg.norm(tvec)
            
            # 使用四元数计算欧拉角（更稳定）
            rmat, _ = cv2.Rodrigues(rvec)
            quat = self.rotation_matrix_to_quaternion(rmat)
            roll, pitch, yaw = self.quaternion_to_euler(quat)
            
            rx = math.degrees(roll)
            ry = math.degrees(pitch)
            rz = math.degrees(yaw)
            
            # 绘制结果
            self._draw_overlay(img_temp, img_points_refined, rvec, tvec, 
                              dist_mm, (rx, ry, rz), i, current_matrix, error)
            
            # 详细日志
            self.log(f"✓ 距离: {dist_mm:.2f} mm")
            self.log(f"✓ 旋转角度: Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
            self.log(f"✓ 平移向量: X={tvec[0][0]:.2f}, Y={tvec[1][0]:.2f}, Z={tvec[2][0]:.2f}")
            self.log(f"✓ 四元数: w={quat[0]:.4f}, x={quat[1]:.4f}, y={quat[2]:.4f}, z={quat[3]:.4f}")

        self.current_cv_img = img_temp
        self.display_image()

    def _draw_overlay(self, img, pts, rvec, tvec, dist, angles, idx, camera_mtx, error):
        pts = pts.astype(np.int32)
        rx, ry, rz = angles
        
        corner_pts = pts[:4]
        center_pt = pts[4]
        
        # 1. 绘制边框（根据误差着色）
        if error < 0.3:
            color = (0, 255, 0)  # 绿色：优秀
        elif error < 0.6:
            color = (0, 255, 255)  # 黄色：良好
        else:
            color = (0, 165, 255)  # 橙色：一般
            
        for j in range(4):
            cv2.line(img, tuple(corner_pts[j]), tuple(corner_pts[(j+1)%4]), color, 3)
            
        # 2. 标记角点
        labels = ["TL", "TR", "BR", "BL"]
        for j, pt in enumerate(corner_pts):
            cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(img, labels[j], (pt[0]+10, pt[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 3. 标记中心点
        cv2.circle(img, tuple(center_pt), 7, (255, 0, 255), -1)
        cv2.circle(img, tuple(center_pt), 10, (255, 0, 255), 2)
        cv2.putText(img, "C", (center_pt[0]+12, center_pt[1]-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 4. 绘制坐标轴
        axis_len = 20.0 
        axis_pts = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_mtx, self.dist_coeffs)
        imgpts = imgpts.astype(np.int32)
        
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 4)  # X - 红
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 4)  # Y - 绿
        cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 4)  # Z - 蓝
        
        # 5. 显示信息
        center_x = center_pt[0]
        center_y = center_pt[1]
        
        info_lines = [
            f"D: {dist:.1f}mm",
            f"Rx: {rx:.2f}°",
            f"Ry: {ry:.2f}°",
            f"Rz: {rz:.2f}°",
            f"E: {error:.3f}px"
        ]
        
        for idx, line in enumerate(info_lines):
            y_offset = 40 + idx * 25
            cv2.putText(img, line, (center_x + 15, center_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.pil_img = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        self.canvas.delete("all")
        cx = canvas_w // 2 + self.offset_x
        cy = canvas_h // 2 + self.offset_y
        self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.tk_img)

    def log(self, msg):
        self.txt_result.insert(tk.END, msg + "\n")
        self.txt_result.see(tk.END)

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
