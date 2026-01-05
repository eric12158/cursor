import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import os
import math
from collections import deque

try:
    from pupil_apriltags import Detector
    APRILTAG_AVAILABLE = True
except:
    APRILTAG_AVAILABLE = False

class RobotVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("机器人视觉引导系统 V1.0")
        self.root.geometry("1600x1000")
        
        # 相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calib_img_size = None
        
        # 手眼标定矩阵
        self.T_cam_to_flange = None
        
        # 图像
        self.current_cv_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 检测结果
        self.current_qr_pose = None  # (tx, ty, tz, rx, ry, rz)
        
        if APRILTAG_AVAILABLE:
            self.detector = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # ========== 左侧控制面板 ==========
        left_panel = tk.Frame(self.root, width=450)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 1. 相机参数
        cam_frame = tk.LabelFrame(left_panel, text="1️⃣ 相机参数", padx=10, pady=5)
        cam_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_cam_status = tk.Label(cam_frame, text="未加载", fg="red")
        self.lbl_cam_status.pack(anchor=tk.W)
        tk.Button(cam_frame, text="加载相机标定文件", command=self.load_camera, bg="#e3f2fd").pack(fill=tk.X, pady=2)
        
        # 2. 手眼标定
        handeye_frame = tk.LabelFrame(left_panel, text="2️⃣ 手眼标定矩阵", padx=10, pady=5)
        handeye_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_handeye_status = tk.Label(handeye_frame, text="未加载", fg="red")
        self.lbl_handeye_status.pack(anchor=tk.W)
        tk.Button(handeye_frame, text="加载手眼标定结果", command=self.load_handeye, bg="#e3f2fd").pack(fill=tk.X, pady=2)
        
        # 3. 检测设置
        detect_frame = tk.LabelFrame(left_panel, text="3️⃣ 检测设置", padx=10, pady=5)
        detect_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(detect_frame, text="标签家族:").pack(anchor=tk.W)
        self.tag_family_var = tk.StringVar(value="自动检测")
        families = ["自动检测", "tag16h5", "tag25h9", "tag36h11"]
        ttk.Combobox(detect_frame, textvariable=self.tag_family_var, values=families, state="readonly").pack(fill=tk.X, pady=2)
        
        tk.Label(detect_frame, text="标签边长 (mm):").pack(anchor=tk.W)
        self.tag_size_var = tk.DoubleVar(value=26.0)
        tk.Entry(detect_frame, textvariable=self.tag_size_var).pack(fill=tk.X, pady=2)
        
        tk.Button(detect_frame, text="📷 打开图片", command=self.load_image, bg="#e8f5e9").pack(fill=tk.X, pady=2)
        tk.Button(detect_frame, text="🎯 执行检测", command=self.run_detection, bg="#fff3e0", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=2)
        
        # 4. 机械臂位姿输入
        robot_frame = tk.LabelFrame(left_panel, text="4️⃣ 当前机械臂法兰位姿", padx=10, pady=5)
        robot_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(robot_frame, text="⚠️ 从机械臂控制器读取实时位姿", fg="orange").pack(anchor=tk.W)
        
        pose_grid = tk.Frame(robot_frame)
        pose_grid.pack(fill=tk.X, pady=5)
        
        self.flange_tx = tk.DoubleVar(value=300.0)
        self.flange_ty = tk.DoubleVar(value=100.0)
        self.flange_tz = tk.DoubleVar(value=400.0)
        self.flange_rx = tk.DoubleVar(value=180.0)
        self.flange_ry = tk.DoubleVar(value=0.0)
        self.flange_rz = tk.DoubleVar(value=90.0)
        
        labels = ["X(mm):", "Y(mm):", "Z(mm):", "Rx(°):", "Ry(°):", "Rz(°):"]
        vars = [self.flange_tx, self.flange_ty, self.flange_tz, self.flange_rx, self.flange_ry, self.flange_rz]
        
        for i, (lbl, var) in enumerate(zip(labels, vars)):
            row = i // 2
            col = (i % 2) * 2
            tk.Label(pose_grid, text=lbl, width=8).grid(row=row, column=col, sticky=tk.W, padx=2)
            tk.Entry(pose_grid, textvariable=var, width=10).grid(row=row, column=col+1, padx=2, pady=2)
        
        tk.Button(robot_frame, text="🤖 计算目标在基座系位姿", command=self.calc_target_in_base, bg="#c5e1a5", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=5)
        
        # 5. 抓取位姿生成
        grasp_frame = tk.LabelFrame(left_panel, text="5️⃣ 生成抓取位姿", padx=10, pady=5)
        grasp_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(grasp_frame, text="接近高度偏移 (mm):").pack(anchor=tk.W)
        self.offset_z_var = tk.DoubleVar(value=50.0)
        tk.Entry(grasp_frame, textvariable=self.offset_z_var).pack(fill=tk.X, pady=2)
        
        tk.Button(grasp_frame, text="✨ 生成抓取位姿", command=self.generate_grasp, bg="#b39ddb", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=5)
        
        # ========== 右侧显示区 ==========
        right_panel = tk.Frame(self.root)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 图像显示
        img_frame = tk.LabelFrame(right_panel, text="图像显示")
        img_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = tk.Canvas(img_frame, bg="#404040")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 结果显示
        result_frame = tk.LabelFrame(right_panel, text="计算结果")
        result_frame.pack(fill=tk.BOTH, pady=5)
        
        self.txt_result = tk.Text(result_frame, height=12, font=("Consolas", 9))
        self.txt_result.pack(fill=tk.BOTH, expand=True)
        
    def load_camera(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not fpath: return
        
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            
            if "image_size" in data:
                self.calib_img_size = tuple(data["image_size"])
            
            self.lbl_cam_status.config(text=f"✅ 已加载: {os.path.basename(fpath)}", fg="green")
            self.log(f"相机参数已加载: {fpath}")
            
            # 初始化检测器
            if APRILTAG_AVAILABLE:
                self.init_detector()
            
        except Exception as e:
            messagebox.showerror("错误", str(e))
    
    def load_handeye(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("TXT", "*.txt")])
        if not fpath: return
        
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # 从JSON读取手眼标定
            tx = data["translation"]["x"]
            ty = data["translation"]["y"]
            tz = data["translation"]["z"]
            rx = data["rotation"]["rx"]
            ry = data["rotation"]["ry"]
            rz = data["rotation"]["rz"]
            
            self.T_cam_to_flange = self.make_transform(tx, ty, tz, rx, ry, rz)
            
            self.lbl_handeye_status.config(text=f"✅ 已加载: {os.path.basename(fpath)}", fg="green")
            self.log(f"手眼标定已加载:")
            self.log(f"  平移: X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f}")
            self.log(f"  旋转: Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
            
        except Exception as e:
            # 手动输入
            self.manual_input_handeye()
    
    def manual_input_handeye(self):
        """手动输入手眼标定结果"""
        dialog = tk.Toplevel(self.root)
        dialog.title("手动输入手眼标定")
        dialog.geometry("400x350")
        
        tk.Label(dialog, text="输入手眼标定矩阵 (相机->法兰)", font=("Arial", 12, "bold")).pack(pady=10)
        
        frame = tk.Frame(dialog)
        frame.pack(pady=10)
        
        vars_dict = {}
        labels = ["X(mm)", "Y(mm)", "Z(mm)", "Rx(°)", "Ry(°)", "Rz(°)"]
        defaults = [-21.0093, -7.6736, 576.7124, 20.8621, 11.3074, 117.7986]
        
        for i, (lbl, default) in enumerate(zip(labels, defaults)):
            tk.Label(frame, text=lbl, width=10).grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            var = tk.DoubleVar(value=default)
            tk.Entry(frame, textvariable=var, width=15).grid(row=i, column=1, padx=5, pady=5)
            vars_dict[lbl] = var
        
        def confirm():
            try:
                tx = vars_dict["X(mm)"].get()
                ty = vars_dict["Y(mm)"].get()
                tz = vars_dict["Z(mm)"].get()
                rx = vars_dict["Rx(°)"].get()
                ry = vars_dict["Ry(°)"].get()
                rz = vars_dict["Rz(°)"].get()
                
                self.T_cam_to_flange = self.make_transform(tx, ty, tz, rx, ry, rz)
                self.lbl_handeye_status.config(text="✅ 已手动输入", fg="green")
                self.log(f"手眼标定已加载: T={tx:.1f},{ty:.1f},{tz:.1f}, R={rx:.1f},{ry:.1f},{rz:.1f}")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("错误", str(e))
        
        tk.Button(dialog, text="确定", command=confirm, bg="#4caf50", fg="white").pack(pady=10)
    
    def init_detector(self):
        if not APRILTAG_AVAILABLE:
            return
        
        family = self.tag_family_var.get()
        if family == "自动检测":
            family = "tag16h5 tag25h9 tag36h11"
        
        self.detector = Detector(families=family, nthreads=4, quad_decimate=1.0, 
                                 quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
    
    def load_image(self):
        fpath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if not fpath: return
        
        self.current_cv_img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
        if self.current_cv_img.shape[2] == 4:
            self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGRA2BGR)
        
        self.display_image()
        self.log(f"图片已加载: {os.path.basename(fpath)}")
    
    def run_detection(self):
        if not APRILTAG_AVAILABLE:
            messagebox.showerror("错误", "请安装: pip install pupil-apriltags")
            return
        
        if self.camera_matrix is None:
            messagebox.showwarning("警告", "请先加载相机参数")
            return
        
        if self.current_cv_img is None:
            messagebox.showwarning("警告", "请先打开图片")
            return
        
        self.init_detector()
        
        gray = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        tag_size = self.tag_size_var.get() / 1000.0
        
        tags = self.detector.detect(gray, estimate_tag_pose=False, camera_params=[fx, fy, cx, cy], tag_size=tag_size)
        
        if not tags:
            self.log("❌ 未检测到AprilTag")
            return
        
        self.log(f"✅ 检测到 {len(tags)} 个AprilTag")
        
        # 使用第一个标签
        tag = tags[0]
        
        # PnP求解
        half = self.tag_size_var.get() / 2.0
        obj_pts = np.array([[-half,-half,0], [half,-half,0], [half,half,0], [-half,half,0]], dtype=np.float64)
        
        corners = tag.corners.astype(np.float32)
        corners_reorder = np.array([corners[3], corners[2], corners[1], corners[0]], dtype=np.float32)
        
        # 亚像素
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        corners_refined = cv2.cornerSubPix(gray, corners_reorder, winSize=(10,10), zeroZone=(-1,-1), criteria=criteria)
        
        success, rvec, tvec = cv2.solvePnP(obj_pts, corners_refined.astype(np.float64), 
                                           self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        
        if success:
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, corners_refined.astype(np.float64), 
                                             self.camera_matrix, self.dist_coeffs, rvec, tvec)
            
            # 提取姿态
            rmat, _ = cv2.Rodrigues(rvec)
            rx, ry, rz = self.rotation_to_euler(rmat)
            
            self.current_qr_pose = (tvec[0][0], tvec[1][0], tvec[2][0], rx, ry, rz)
            
            self.log(f"\n📍 标签在相机坐标系的位姿:")
            self.log(f"   平移: X={tvec[0][0]:.2f}, Y={tvec[1][0]:.2f}, Z={tvec[2][0]:.2f} mm")
            self.log(f"   旋转: Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
            
            # 绘制
            img_show = self.current_cv_img.copy()
            for j in range(4):
                pt1 = tuple(corners_refined[j].astype(np.int32))
                pt2 = tuple(corners_refined[(j+1)%4].astype(np.int32))
                cv2.line(img_show, pt1, pt2, (0, 255, 0), 3)
            
            # 坐标轴
            axis = np.float32([[20,0,0], [0,20,0], [0,0,-20], [0,0,0]]).reshape(-1,3)
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            imgpts = imgpts.astype(np.int32)
            origin = tuple(imgpts[3].ravel())
            cv2.line(img_show, origin, tuple(imgpts[0].ravel()), (0,0,255), 3)
            cv2.line(img_show, origin, tuple(imgpts[1].ravel()), (0,255,0), 3)
            cv2.line(img_show, origin, tuple(imgpts[2].ravel()), (255,0,0), 3)
            
            self.current_cv_img = img_show
            self.display_image()
    
    def calc_target_in_base(self):
        if self.T_cam_to_flange is None:
            messagebox.showwarning("警告", "请先加载手眼标定")
            return
        
        if self.current_qr_pose is None:
            messagebox.showwarning("警告", "请先执行检测")
            return
        
        # 二维码在相机系
        qr_tx, qr_ty, qr_tz, qr_rx, qr_ry, qr_rz = self.current_qr_pose
        T_qr_to_cam = self.make_transform(qr_tx, qr_ty, qr_tz, qr_rx, qr_ry, qr_rz)
        
        # 法兰在基座系
        T_flange_to_base = self.make_transform(
            self.flange_tx.get(), self.flange_ty.get(), self.flange_tz.get(),
            self.flange_rx.get(), self.flange_ry.get(), self.flange_rz.get()
        )
        
        # 变换链: Base <- Flange <- Camera <- QR
        T_qr_to_base = T_flange_to_base @ self.T_cam_to_flange @ T_qr_to_cam
        
        tx = T_qr_to_base[0, 3]
        ty = T_qr_to_base[1, 3]
        tz = T_qr_to_base[2, 3]
        rx, ry, rz = self.rotation_to_euler(T_qr_to_base[:3, :3])
        
        self.log(f"\n🎯 目标在基座坐标系的位姿:")
        self.log(f"   平移: X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f} mm")
        self.log(f"   旋转: Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
        self.log(f"\n💡 可以将此位姿发送给机械臂执行移动！")
        
        self.target_in_base = (tx, ty, tz, rx, ry, rz)
    
    def generate_grasp(self):
        if not hasattr(self, 'target_in_base'):
            messagebox.showwarning("警告", "请先计算目标在基座系位姿")
            return
        
        tx, ty, tz, rx, ry, rz = self.target_in_base
        offset_z = self.offset_z_var.get()
        
        grasp_pose = (tx, ty, tz + offset_z, rx, ry, rz)
        
        self.log(f"\n🤖 推荐抓取位姿 (上方{offset_z}mm):")
        self.log(f"   平移: X={grasp_pose[0]:.2f}, Y={grasp_pose[1]:.2f}, Z={grasp_pose[2]:.2f} mm")
        self.log(f"   旋转: Rx={grasp_pose[3]:.2f}°, Ry={grasp_pose[4]:.2f}°, Rz={grasp_pose[5]:.2f}°")
        self.log(f"\n📋 抓取流程:")
        self.log(f"   1. 移动到此位姿")
        self.log(f"   2. 下降Z轴接近物体")
        self.log(f"   3. 闭合夹爪")
        self.log(f"   4. 提升并移动到目标")
    
    def make_transform(self, tx, ty, tz, rx, ry, rz):
        """构造4x4变换矩阵"""
        rx_rad = math.radians(rx)
        ry_rad = math.radians(ry)
        rz_rad = math.radians(rz)
        
        Rx = np.array([[1, 0, 0], [0, math.cos(rx_rad), -math.sin(rx_rad)], [0, math.sin(rx_rad), math.cos(rx_rad)]])
        Ry = np.array([[math.cos(ry_rad), 0, math.sin(ry_rad)], [0, 1, 0], [-math.sin(ry_rad), 0, math.cos(ry_rad)]])
        Rz = np.array([[math.cos(rz_rad), -math.sin(rz_rad), 0], [math.sin(rz_rad), math.cos(rz_rad), 0], [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T
    
    def rotation_to_euler(self, R):
        """旋转矩阵转欧拉角"""
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return math.degrees(x), math.degrees(y), math.degrees(z)
    
    def display_image(self):
        if self.current_cv_img is None: return
        
        img_rgb = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10: canvas_w = 800
        if canvas_h < 10: canvas_h = 600
        
        scale_fit = min(canvas_w/w, canvas_h/h) * 0.95
        new_w = int(w * scale_fit)
        new_h = int(h * scale_fit)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        self.pil_img = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=self.tk_img)
    
    def log(self, msg):
        self.txt_result.insert(tk.END, msg + "\n")
        self.txt_result.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotVisionApp(root)
    root.mainloop()
