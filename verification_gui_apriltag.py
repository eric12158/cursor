import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import math
from collections import deque

try:
    from pupil_apriltags import Detector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False
    print("警告: 未安装 pupil-apriltags, 请运行: pip install pupil-apriltags")

class VerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AprilTag测距验证工具 V3.1 (超稳定版)")
        self.root.geometry("1500x950")
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calib_img_size = None
        
        self.current_cv_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        
        self.pose_history = deque(maxlen=5)
        
        if APRILTAG_AVAILABLE:
            self.detector = None  # 将在加载参数后初始化
        
        self.setup_ui()
        
    def setup_ui(self):
        top_frame = tk.LabelFrame(self.root, text="第一步：加载相机参数", padx=10, pady=5)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_param_status = tk.Label(top_frame, text="未加载参数", fg="red", font=("Arial", 10, "bold"))
        self.lbl_param_status.pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_frame, text="从 JSON 文件加载参数", command=self.load_params_from_file, bg="#e3f2fd").pack(side=tk.LEFT, padx=5)
        
        self.lbl_calib_res = tk.Label(top_frame, text="", fg="gray")
        self.lbl_calib_res.pack(side=tk.LEFT, padx=20)

        mid_frame = tk.LabelFrame(self.root, text="第二步：检测与验证", padx=10, pady=5)
        mid_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(mid_frame, text="打开图片", command=self.load_image, bg="#e8f5e9", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.lbl_curr_res = tk.Label(mid_frame, text="", fg="blue")
        self.lbl_curr_res.pack(side=tk.LEFT, padx=5)
        
        tk.Label(mid_frame, text="AprilTag真实边长(mm):").pack(side=tk.LEFT, padx=(20, 5))
        self.tag_size_var = tk.DoubleVar(value=26.0)
        tk.Entry(mid_frame, textvariable=self.tag_size_var, width=8).pack(side=tk.LEFT)
        
        tk.Button(mid_frame, text="执行测距与姿态解算", command=self.run_measurement, bg="#fff3e0", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=20)
        
        adv_frame = tk.LabelFrame(self.root, text="高级优化选项", padx=10, pady=5)
        adv_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.use_multi_solve = tk.BooleanVar(value=True)
        tk.Checkbutton(adv_frame, text="多次求解平均", variable=self.use_multi_solve).pack(side=tk.LEFT, padx=5)
        
        self.use_temporal_filter = tk.BooleanVar(value=False)
        tk.Checkbutton(adv_frame, text="时间序列滤波", variable=self.use_temporal_filter).pack(side=tk.LEFT, padx=5)
        
        self.use_subpix = tk.BooleanVar(value=True)
        tk.Checkbutton(adv_frame, text="超精细亚像素", variable=self.use_subpix).pack(side=tk.LEFT, padx=5)
        
        tk.Button(adv_frame, text="清除历史", command=self.clear_history, bg="#ffebee").pack(side=tk.LEFT, padx=20)
        
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        center_frame = tk.Frame(main_paned, bg="#404040")
        self.canvas = tk.Canvas(center_frame, bg="#404040")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        
        main_paned.add(center_frame, stretch="always")
        
        side_frame = tk.Frame(main_paned, width=400)
        tk.Label(side_frame, text="测量结果 (AprilTag超稳定版):").pack(anchor=tk.W)
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
                self.lbl_calib_res.config(text="标定分辨率: 未知")

            # 初始化AprilTag检测器
            if APRILTAG_AVAILABLE:
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]
                self.detector = Detector(
                    families='tag36h11',
                    nthreads=4,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25
                )
                self.log(f"AprilTag检测器已初始化 (tag36h11)")

            self.lbl_param_status.config(text=f"参数已加载: {os.path.basename(fpath)}", fg="green")
            self.log(f"成功加载参数文件: {fpath}")
            
        except Exception as e:
            messagebox.showerror("加载失败", f"{e}")

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
        self.log(f"已加载图片: {os.path.basename(fpath)} ({w}x{h})")
        
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_image()

    def get_scaled_camera_matrix(self, current_w, current_h):
        if self.calib_img_size is None or self.camera_matrix is None:
            return self.camera_matrix, 1.0
            
        calib_w, calib_h = self.calib_img_size
        scale_factor = current_w / float(calib_w)
        
        if abs(scale_factor - 1.0) < 0.01:
            return self.camera_matrix, 1.0
            
        self.log(f"分辨率缩放: {calib_w}x{calib_h} -> {current_w}x{current_h} (x{scale_factor:.4f})")
        
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= scale_factor
        new_matrix[1, 1] *= scale_factor
        new_matrix[0, 2] *= scale_factor
        new_matrix[1, 2] *= scale_factor
        
        return new_matrix, scale_factor

    def refine_corners_ultra(self, gray, corners):
        """超精细亚像素优化"""
        if not self.use_subpix.get():
            return corners
            
        # 第一次：大窗口
        criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        refined1 = cv2.cornerSubPix(gray, corners.copy(), winSize=(15, 15), zeroZone=(-1, -1), criteria=criteria1)
        
        # 第二次：小窗口精修
        criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.00001)
        refined2 = cv2.cornerSubPix(gray, refined1.copy(), winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria2)
        
        return refined2

    def solve_pnp_multiple(self, obj_pts, img_pts, cam_mtx, dist, n_trials=5):
        """多次求解取最优"""
        if not self.use_multi_solve.get():
            # 单次求解
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if success:
                rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, cam_mtx, dist, rvec, tvec)
                reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist)
                error = np.mean(np.linalg.norm(img_pts - reproj.reshape(-1, 2), axis=1))
                return rvec, tvec, error, 'IPPE_SQUARE'
            return None, None, 999.0, 'FAILED'
        
        results = []
        
        for trial in range(n_trials):
            if trial > 0:
                noise = np.random.randn(*img_pts.shape) * 0.01
                img_pts_noisy = img_pts + noise
            else:
                img_pts_noisy = img_pts.copy()
            
            # IPPE_SQUARE
            try:
                s1, rv1, tv1 = cv2.solvePnP(obj_pts, img_pts_noisy, cam_mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if s1:
                    rv1, tv1 = cv2.solvePnPRefineLM(obj_pts, img_pts, cam_mtx, dist, rv1, tv1)
                    reproj, _ = cv2.projectPoints(obj_pts, rv1, tv1, cam_mtx, dist)
                    err1 = np.mean(np.linalg.norm(img_pts - reproj.reshape(-1, 2), axis=1))
                    results.append((rv1, tv1, err1, 'IPPE'))
            except: pass
            
            # ITERATIVE
            try:
                s2, rv2, tv2 = cv2.solvePnP(obj_pts, img_pts_noisy, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if s2:
                    rv2, tv2 = cv2.solvePnPRefineLM(obj_pts, img_pts, cam_mtx, dist, rv2, tv2)
                    reproj, _ = cv2.projectPoints(obj_pts, rv2, tv2, cam_mtx, dist)
                    err2 = np.mean(np.linalg.norm(img_pts - reproj.reshape(-1, 2), axis=1))
                    results.append((rv2, tv2, err2, 'ITER'))
            except: pass
        
        if not results:
            return None, None, 999.0, 'FAILED'
        
        best = min(results, key=lambda x: x[2])
        
        if len(results) >= 3:
            rvecs = np.array([r[0].ravel() for r in results])
            tvecs = np.array([r[1].ravel() for r in results])
            rvec_med = np.median(rvecs, axis=0).reshape(3, 1)
            tvec_med = np.median(tvecs, axis=0).reshape(3, 1)
            return rvec_med, tvec_med, best[2], best[3] + '_MEDIAN'
        
        return best[0], best[1], best[2], best[3]

    def rotation_to_quaternion(self, R):
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
        w, x, y, z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = 2 * (w*y - z*x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw

    def apply_temporal_filter(self, rvec, tvec):
        if not self.use_temporal_filter.get():
            return rvec, tvec
        
        self.pose_history.append((rvec.copy(), tvec.copy()))
        
        if len(self.pose_history) < 2:
            return rvec, tvec
        
        rvecs = np.array([r.ravel() for r, t in self.pose_history])
        tvecs = np.array([t.ravel() for r, t in self.pose_history])
        
        weights = np.exp(np.linspace(-1, 0, len(self.pose_history)))
        weights /= weights.sum()
        
        rvec_filt = np.average(rvecs, axis=0, weights=weights).reshape(3, 1)
        tvec_filt = np.average(tvecs, axis=0, weights=weights).reshape(3, 1)
        
        return rvec_filt, tvec_filt

    def run_measurement(self):
        if not APRILTAG_AVAILABLE:
            messagebox.showerror("错误", "请先安装: pip install pupil-apriltags")
            return
            
        if self.camera_matrix is None or self.detector is None:
            messagebox.showwarning("警告", "请先加载相机参数！")
            return
        if self.current_cv_img is None:
            messagebox.showwarning("警告", "请先打开图片！")
            return
            
        try:
            tag_size = self.tag_size_var.get()
        except:
            messagebox.showerror("错误", "请输入有效的标签尺寸！")
            return

        h, w = self.current_cv_img.shape[:2]
        current_matrix, scale_applied = self.get_scaled_camera_matrix(w, h)

        img_temp = self.current_cv_img.copy()
        gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        
        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # 检测AprilTag
        fx = current_matrix[0, 0]
        fy = current_matrix[1, 1]
        cx = current_matrix[0, 2]
        cy = current_matrix[1, 2]
        
        tags = self.detector.detect(gray_enhanced, estimate_tag_pose=False, camera_params=[fx, fy, cx, cy], tag_size=tag_size/1000.0)
        
        if not tags:
            self.log("未检测到AprilTag。")
            return
            
        self.log(f"检测到 {len(tags)} 个AprilTag。")
        self.log("=" * 50)
        
        # 3D坐标（5点）
        half = tag_size / 2.0
        obj_points = np.array([
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
            [0, 0, 0]  # 中心
        ], dtype=np.float64)
        
        for i, tag in enumerate(tags):
            self.log(f"\n--- AprilTag ID={tag.tag_id} ---")
            
            # 获取角点 (AprilTag格式: 左下,右下,右上,左上)
            corners = tag.corners.astype(np.float32)
            # 转换为: 左上,右上,右下,左下
            corners_reorder = np.array([corners[3], corners[2], corners[1], corners[0]], dtype=np.float32)
            
            # 添加中心点
            center = np.mean(corners_reorder, axis=0, keepdims=True).astype(np.float32)
            corners_5 = np.vstack([corners_reorder, center])
            
            # 超精细亚像素
            corners_refined = self.refine_corners_ultra(gray_enhanced, corners_5)
            corners_refined = corners_refined.astype(np.float64)
            
            shift = np.linalg.norm(corners_5 - corners_refined, axis=1)
            self.log(f"亚像素优化: 平均={np.mean(shift):.4f}px, 最大={np.max(shift):.4f}px")
            
            # 多次求解
            rvec, tvec, error, method = self.solve_pnp_multiple(
                obj_points, corners_refined, current_matrix, self.dist_coeffs, n_trials=5
            )
            
            if rvec is None:
                self.log("PnP求解失败")
                continue
                
            self.log(f"方法: {method}, 重投影误差: {error:.4f}px")
            
            # 时间滤波
            if self.use_temporal_filter.get():
                rvec, tvec = self.apply_temporal_filter(rvec, tvec)
                self.log(f"时间滤波: 使用 {len(self.pose_history)} 帧")
            
            dist_mm = np.linalg.norm(tvec)
            
            # 四元数->欧拉角
            rmat, _ = cv2.Rodrigues(rvec)
            quat = self.rotation_to_quaternion(rmat)
            roll, pitch, yaw = self.quaternion_to_euler(quat)
            
            rx = math.degrees(roll)
            ry = math.degrees(pitch)
            rz = math.degrees(yaw)
            
            self._draw_overlay(img_temp, corners_refined, rvec, tvec, dist_mm, (rx, ry, rz), tag.tag_id, current_matrix, error)
            
            self.log(f"✓ 距离: {dist_mm:.2f} mm")
            self.log(f"✓ 角度: Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
            self.log(f"✓ 平移: X={tvec[0][0]:.2f}, Y={tvec[1][0]:.2f}, Z={tvec[2][0]:.2f}")

        self.current_cv_img = img_temp
        self.display_image()

    def _draw_overlay(self, img, pts, rvec, tvec, dist, angles, tag_id, cam_mtx, error):
        pts = pts.astype(np.int32)
        rx, ry, rz = angles
        
        corners = pts[:4]
        center = pts[4]
        
        # 颜色编码
        color = (0, 255, 0) if error < 0.3 else (0, 255, 255) if error < 0.6 else (0, 165, 255)
            
        for j in range(4):
            cv2.line(img, tuple(corners[j]), tuple(corners[(j+1)%4]), color, 3)
            
        # 角点
        labels = ["TL", "TR", "BR", "BL"]
        for j, pt in enumerate(corners):
            cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(img, labels[j], (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 中心
        cv2.circle(img, tuple(center), 7, (255, 0, 255), -1)
        cv2.circle(img, tuple(center), 10, (255, 0, 255), 2)
        
        # 坐标轴
        axis_len = 20.0 
        axis_pts = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, cam_mtx, self.dist_coeffs)
        imgpts = imgpts.astype(np.int32)
        
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 4)
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 4)
        cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 4)
        
        # 信息
        cx, cy = center[0], center[1]
        info = [
            f"ID:{tag_id}",
            f"D:{dist:.1f}mm",
            f"Rx:{rx:.2f}°",
            f"Ry:{ry:.2f}°",
            f"Rz:{rz:.2f}°",
            f"E:{error:.3f}px"
        ]
        
        for idx, line in enumerate(info):
            cv2.putText(img, line, (cx + 15, cy + 25 + idx*22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

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
