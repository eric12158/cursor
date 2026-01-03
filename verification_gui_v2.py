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
        self.root.title("相机测距验证工具 V2.4 (姿态稳定优化版)")
        self.root.geometry("1400x900")
        
        # 默认参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calib_img_size = None # (width, height) from json
        
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
        
        self.check_reproject = tk.BooleanVar(value=True)
        tk.Checkbutton(mid_frame, text="显示重投影验证点(黄色)", variable=self.check_reproject).pack(side=tk.LEFT, padx=10)
        
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
            
            self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            
            if "image_size" in data:
                self.calib_img_size = tuple(data["image_size"]) # (w, h)
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
        """核心算法：根据分辨率差异自动缩放内参"""
        if self.calib_img_size is None or self.camera_matrix is None:
            return self.camera_matrix, (1.0, 1.0)
            
        calib_w, calib_h = self.calib_img_size
        # 注意：如果当前分辨率和标定分辨率的宽高缩放不一致（例如非等比缩放/裁剪），
        # 只按宽度缩放会导致 fx/fy、cx/cy 不匹配，从而显著放大 Rx/Ry 误差。
        sx = current_w / float(calib_w)
        sy = current_h / float(calib_h)
        
        if abs(sx - 1.0) < 0.01 and abs(sy - 1.0) < 0.01:
            return self.camera_matrix, (1.0, 1.0)
            
        self.log("-" * 30)
        self.log(f"检测到分辨率不匹配！")
        self.log(f"标定: {calib_w}x{calib_h} -> 当前: {current_w}x{current_h}")
        self.log(f"自动应用缩放系数: sx={sx:.6f}, sy={sy:.6f}")
        if abs(sx - sy) > 0.01:
            self.log("警告：sx != sy，当前图像可能经过非等比缩放或裁剪，姿态角误差可能增大。")
        
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= sx  # fx
        new_matrix[1, 1] *= sy  # fy
        new_matrix[0, 2] *= sx  # cx
        new_matrix[1, 2] *= sy  # cy
        
        return new_matrix, (sx, sy)

    @staticmethod
    def _order_quad_points(pts_4x2: np.ndarray) -> np.ndarray:
        """
        将四边形角点重排为 TL, TR, BR, BL（与 obj_points 一致）。
        QRCodeDetector 的点序在不同 OpenCV 版本/不同姿态下可能不稳定，不重排会直接导致 Rx/Ry 抖动或翻转。
        """
        pts = np.asarray(pts_4x2, dtype=np.float64).reshape(4, 2)
        s = pts.sum(axis=1)              # x+y
        d = (pts[:, 0] - pts[:, 1])      # x-y

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmax(d)]
        bl = pts[np.argmin(d)]
        return np.stack([tl, tr, br, bl], axis=0)

    @staticmethod
    def _mean_reproj_error(obj_pts: np.ndarray, img_pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                           camera_mtx: np.ndarray, dist_coeffs: np.ndarray) -> float:
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_mtx, dist_coeffs)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - img_pts.reshape(-1, 2), axis=1)
        return float(np.mean(err))

    def _solve_pose_square(self, obj_points: np.ndarray, img_points: np.ndarray,
                           camera_mtx: np.ndarray, dist_coeffs: np.ndarray):
        """
        针对平面正方形：优先用 IPPE_SQUARE（会给出双解），用重投影误差 + Z>0 选最优解，再用 LM 精修。
        若当前 OpenCV 不支持 solvePnPGeneric/IPPE，则回退到 ITERATIVE + LM 精修。
        """
        img_points = np.asarray(img_points, dtype=np.float64).reshape(4, 2)

        best_rvec = None
        best_tvec = None
        best_err = None

        # 1) 首选：IPPE 正方形双解
        try:
            ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                obj_points, img_points, camera_mtx, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ret and rvecs is not None and tvecs is not None and len(rvecs) > 0:
                for rv, tv in zip(rvecs, tvecs):
                    tv = tv.reshape(3, 1)
                    rv = rv.reshape(3, 1)
                    err = self._mean_reproj_error(obj_points, img_points, rv, tv, camera_mtx, dist_coeffs)
                    z_ok = float(tv[2, 0]) > 0.0
                    # 优先选择 Z>0 的解；在同类解里选误差更小的
                    score = (0 if z_ok else 1, err)
                    if best_err is None or score < best_err:
                        best_err = score
                        best_rvec, best_tvec = rv, tv
        except Exception:
            best_rvec = None
            best_tvec = None

        # 2) 回退：ITERATIVE
        if best_rvec is None or best_tvec is None:
            ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_mtx, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return False, None, None, None
            best_rvec, best_tvec = rvec, tvec

        # 3) LM 精修（对角度抖动很关键）
        try:
            rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                obj_points, img_points, camera_mtx, dist_coeffs, best_rvec, best_tvec
            )
        except Exception:
            rvec_ref, tvec_ref = best_rvec, best_tvec

        reproj_err = self._mean_reproj_error(obj_points, img_points, rvec_ref, tvec_ref, camera_mtx, dist_coeffs)
        return True, rvec_ref, tvec_ref, reproj_err

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

        # 2. 检测二维码
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
            # 强制角点顺序对齐：TL,TR,BR,BL
            img_points = self._order_quad_points(points[i].reshape(4, 2))
            
            # PnP 解算（针对正方形：IPPE 双解选解 + LM 精修，显著降低 Rx/Ry 抖动）
            success, rvec, tvec, reproj_err = self._solve_pose_square(
                obj_points, img_points, current_matrix, self.dist_coeffs
            )
            
            if success:
                dist_mm = float(np.linalg.norm(tvec))
                
                # --- 计算欧拉角 ---
                rmat, _ = cv2.Rodrigues(rvec)
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
                self._draw_overlay(img_temp, img_points, rvec, tvec, dist_mm, (rx, ry, rz), i, current_matrix, obj_points)
                
                # 详细日志
                self.log(f"\n--- QR Code #{i+1} ---")
                self.log(f"计算距离: {dist_mm:.2f} mm")
                self.log(f"旋转角度 (欧拉角): Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
                self.log(f"平移 (X,Y,Z): {tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f}")
                if reproj_err is not None:
                    self.log(f"平均重投影误差: {reproj_err:.3f} px")
                
            else:
                self.log(f"QR #{i+1} PnP解算失败")

        self.current_cv_img = img_temp
        self.display_image()

    def _draw_overlay(self, img, pts, rvec, tvec, dist, angles, idx, camera_mtx, obj_points):
        pts = pts.astype(np.int32)
        rx, ry, rz = angles
        
        # 1. 画绿色边框
        for j in range(4):
            cv2.line(img, tuple(pts[j]), tuple(pts[(j+1)%4]), (0, 255, 0), 2)
            
        # 2. 标记角点
        labels = ["TL", "TR", "BR", "BL"]
        for j, pt in enumerate(pts):
            cv2.circle(img, tuple(pt), 4, (0, 0, 255), -1)
            cv2.putText(img, labels[j], (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 3. 绘制坐标轴 (传入适配后的 camera_mtx)
        axis_len = 15.0 
        axis_pts = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_mtx, self.dist_coeffs)
        imgpts = imgpts.astype(np.int32)
        
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3) # X
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3) # Y
        cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3) # Z
        
        # 4. 重投影验证 (Yellow Circles)
        if self.check_reproject.get():
            reproj_pts, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_mtx, self.dist_coeffs)
            reproj_pts = reproj_pts.reshape(-1, 2).astype(np.int32)
            for pt in reproj_pts:
                # 画空心黄圈，套在红色实心点外面
                cv2.circle(img, tuple(pt), 8, (0, 255, 255), 2)
        
        # 5. 显示文字
        center_x = int(np.mean(pts[:,0]))
        center_y = int(np.mean(pts[:,1]))
        
        info_dist = f"Dist: {dist:.1f}mm"
        info_rot = f"R: {rx:.1f}, {ry:.1f}, {rz:.1f}"
        
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
