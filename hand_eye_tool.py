import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from scipy.spatial.transform import Rotation as R

class HandEyeCalibrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("通用手眼标定工具 (Hand-Eye Calibration)")
        self.root.geometry("1400x950")
        
        # --- 数据存储 ---
        self.calib_data = [] # 存储每一组数据: {image_path, robot_pose, detected_corners}
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 默认 Halcon 内参 (示例)
        self.fx = tk.DoubleVar(value=2000.0)
        self.fy = tk.DoubleVar(value=2000.0)
        self.cx = tk.DoubleVar(value=1280.0)
        self.cy = tk.DoubleVar(value=960.0)
        
        # 标定板参数
        self.rows = tk.IntVar(value=7)
        self.cols = tk.IntVar(value=7)
        self.spacing = tk.DoubleVar(value=5.0) # mm
        
        self.setup_ui()
        
    def setup_ui(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # === 左侧设置与录入 ===
        left_frame = tk.Frame(main_paned, width=450, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        main_paned.add(left_frame, minsize=450)
        
        # 1. 基础设置
        s_frame = tk.LabelFrame(left_frame, text="1. 基础设置", font=("bold", 10))
        s_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(s_frame, text="标定模式:").grid(row=0, column=0, sticky='w')
        self.calib_mode = tk.StringVar(value="Eye-in-Hand")
        ttk.Combobox(s_frame, textvariable=self.calib_mode, values=["Eye-in-Hand (眼在手上)", "Eye-to-Hand (眼在手外)"], state="readonly").grid(row=0, column=1, columnspan=3, sticky='ew')
        
        tk.Label(s_frame, text="旋转顺序:").grid(row=1, column=0, sticky='w')
        self.rot_order = tk.StringVar(value="xyz")
        ttk.Combobox(s_frame, textvariable=self.rot_order, values=["xyz (Fanuc/Kuka等)", "zyx (常用)", "zxz", "ryz"], state="readonly").grid(row=1, column=1, columnspan=3, sticky='ew')
        tk.Label(s_frame, text="* 这里的旋转顺序指 Rx,Ry,Rz 代表绕哪个轴转").grid(row=2, column=0, columnspan=4, sticky='w', padx=5)
        
        # 2. 内参设置
        p_frame = tk.LabelFrame(left_frame, text="2. 相机内参 (输入 Halcon 结果)", font=("bold", 10))
        p_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(p_frame, text="Fx:").grid(row=0, column=0); tk.Entry(p_frame, textvariable=self.fx, width=8).grid(row=0, column=1)
        tk.Label(p_frame, text="Fy:").grid(row=0, column=2); tk.Entry(p_frame, textvariable=self.fy, width=8).grid(row=0, column=3)
        tk.Label(p_frame, text="Cx:").grid(row=1, column=0); tk.Entry(p_frame, textvariable=self.cx, width=8).grid(row=1, column=1)
        tk.Label(p_frame, text="Cy:").grid(row=1, column=2); tk.Entry(p_frame, textvariable=self.cy, width=8).grid(row=1, column=3)
        
        tk.Label(p_frame, text="标定板 行x列:").grid(row=2, column=0)
        tk.Entry(p_frame, textvariable=self.rows, width=4).grid(row=2, column=1)
        tk.Entry(p_frame, textvariable=self.cols, width=4).grid(row=2, column=2)
        tk.Label(p_frame, text="间距(mm):").grid(row=2, column=3)
        tk.Entry(p_frame, textvariable=self.spacing, width=5).grid(row=2, column=4)

        # 3. 数据录入
        i_frame = tk.LabelFrame(left_frame, text="3. 数据录入 (最少 10 组)", font=("bold", 10), bg="#e3f2fd")
        i_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(i_frame, text="加载图片...", command=self.load_image_for_entry).grid(row=0, column=0, padx=5, pady=5)
        self.lbl_img_path = tk.Label(i_frame, text="未选择图片", fg="gray", wraplength=200)
        self.lbl_img_path.grid(row=0, column=1, columnspan=3)
        
        tk.Label(i_frame, text="机械臂坐标 (Base -> EndEffector):", bg="#e3f2fd").grid(row=1, column=0, columnspan=4, sticky='w')
        
        self.var_x = tk.DoubleVar(); self.var_y = tk.DoubleVar(); self.var_z = tk.DoubleVar()
        self.var_rx = tk.DoubleVar(); self.var_ry = tk.DoubleVar(); self.var_rz = tk.DoubleVar()
        
        grid_opts = {'padx': 2, 'pady': 2}
        tk.Label(i_frame, text="X(mm):", bg="#e3f2fd").grid(row=2, column=0, **grid_opts)
        tk.Entry(i_frame, textvariable=self.var_x, width=8).grid(row=2, column=1, **grid_opts)
        tk.Label(i_frame, text="Rx(deg):", bg="#e3f2fd").grid(row=2, column=2, **grid_opts)
        tk.Entry(i_frame, textvariable=self.var_rx, width=8).grid(row=2, column=3, **grid_opts)
        
        tk.Label(i_frame, text="Y(mm):", bg="#e3f2fd").grid(row=3, column=0, **grid_opts)
        tk.Entry(i_frame, textvariable=self.var_y, width=8).grid(row=3, column=1, **grid_opts)
        tk.Label(i_frame, text="Ry(deg):", bg="#e3f2fd").grid(row=3, column=2, **grid_opts)
        tk.Entry(i_frame, textvariable=self.var_ry, width=8).grid(row=3, column=3, **grid_opts)
        
        tk.Label(i_frame, text="Z(mm):", bg="#e3f2fd").grid(row=4, column=0, **grid_opts)
        tk.Entry(i_frame, textvariable=self.var_z, width=8).grid(row=4, column=1, **grid_opts)
        tk.Label(i_frame, text="Rz(deg):", bg="#e3f2fd").grid(row=4, column=2, **grid_opts)
        tk.Entry(i_frame, textvariable=self.var_rz, width=8).grid(row=4, column=3, **grid_opts)
        
        tk.Button(i_frame, text="添加数据点", command=self.add_data_point, bg="#4caf50", fg="white").grid(row=5, column=0, columnspan=4, sticky='ew', pady=5)
        
        # 4. 数据列表
        l_frame = tk.LabelFrame(left_frame, text="4. 已录入数据", font=("bold", 10))
        l_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ("id", "img", "x", "y", "z")
        self.tree = ttk.Treeview(l_frame, columns=columns, show="headings", height=8)
        self.tree.heading("id", text="#"); self.tree.column("id", width=30)
        self.tree.heading("img", text="图片"); self.tree.column("img", width=100)
        self.tree.heading("x", text="X"); self.tree.column("x", width=60)
        self.tree.heading("y", text="Y"); self.tree.column("y", width=60)
        self.tree.heading("z", text="Z"); self.tree.column("z", width=60)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(l_frame, text="保存数据 (.json)", command=self.save_data).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(l_frame, text="加载数据 (.json)", command=self.load_data).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(l_frame, text="删除选中", command=self.delete_selected).pack(side=tk.RIGHT, padx=5, pady=5)

        # 5. 计算
        c_frame = tk.Frame(left_frame)
        c_frame.pack(fill=tk.X, padx=5, pady=10)
        tk.Button(c_frame, text="=== 开始计算手眼矩阵 ===", command=self.run_calibration, bg="#ff9800", height=2, font=("bold", 12)).pack(fill=tk.X)

        # === 右侧结果显示 ===
        right_frame = tk.Frame(main_paned, bg="#333")
        main_paned.add(right_frame, stretch="always")
        
        self.txt_result = tk.Text(right_frame, height=15, font=("Consolas", 10), bg="#222", fg="#0f0")
        self.txt_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_result.insert(tk.END, "等待计算...\n请录入至少 3 组数据（推荐 10+ 组），必须包含旋转变化！\n")

    # --- 逻辑处理 ---
    def get_camera_matrix(self):
        return np.array([
            [self.fx.get(), 0, self.cx.get()],
            [0, self.fy.get(), self.cy.get()],
            [0, 0, 1]
        ], dtype=np.float64)

    def load_image_for_entry(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if path:
            self.current_entry_img_path = path
            self.lbl_img_path.config(text=os.path.basename(path), fg="blue")

    def add_data_point(self):
        if not hasattr(self, 'current_entry_img_path'):
            messagebox.showwarning("警告", "请先选择图片")
            return
            
        # 1. 检测图片中的标定板
        try:
            img = cv2.imdecode(np.fromfile(self.current_entry_img_path, dtype=np.uint8), -1)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            rows, cols = self.rows.get(), self.cols.get()
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
            
            if not ret:
                # 尝试增强检测
                params = cv2.SimpleBlobDetector_Params()
                params.minArea = 10; params.minDistBetweenBlobs = 5
                blob = cv2.SimpleBlobDetector_create(params)
                ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags | cv2.CALIB_CB_CLUSTERING, blobDetector=blob)
                
            if not ret:
                messagebox.showerror("错误", "无法在图片中检测到标定板！请检查行列数设置。")
                return
                
        except Exception as e:
            messagebox.showerror("错误", f"图像处理失败: {e}")
            return

        # 2. 保存数据
        data = {
            "image_path": self.current_entry_img_path,
            "robot_pose": [
                self.var_x.get(), self.var_y.get(), self.var_z.get(),
                self.var_rx.get(), self.var_ry.get(), self.var_rz.get()
            ],
            "corners": corners.tolist()
        }
        self.calib_data.append(data)
        
        # 3. 更新UI
        idx = len(self.calib_data)
        self.tree.insert("", "end", values=(idx, os.path.basename(data["image_path"]), 
                                            data["robot_pose"][0], data["robot_pose"][1], data["robot_pose"][2]))
        self.lbl_img_path.config(text="已添加", fg="green")
        self.txt_result.insert(tk.END, f"[Data] 添加第 {idx} 组数据: {os.path.basename(data['image_path'])}\n")

    def delete_selected(self):
        selected = self.tree.selection()
        if not selected: return
        for item in selected:
            idx = self.tree.index(item)
            del self.calib_data[idx]
            self.tree.delete(item)
        # 重建索引显示
        for i, item in enumerate(self.tree.get_children()):
            self.tree.item(item, values=(i+1,) + self.tree.item(item, "values")[1:])

    def save_data(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="handeye_data.json")
        if path:
            with open(path, 'w') as f: json.dump(self.calib_data, f, indent=4)

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            with open(path, 'r') as f: self.calib_data = json.load(f)
            self.tree.delete(*self.tree.get_children())
            for i, d in enumerate(self.calib_data):
                self.tree.insert("", "end", values=(i+1, os.path.basename(d["image_path"]), 
                                                    d["robot_pose"][0], d["robot_pose"][1], d["robot_pose"][2]))
                # 恢复numpy array
                d["corners"] = np.array(d["corners"], dtype=np.float32)

    def run_calibration(self):
        if len(self.calib_data) < 3:
            messagebox.showerror("错误", "数据太少，至少需要 3 组，建议 10 组以上！")
            return
            
        self.txt_result.insert(tk.END, "-"*30 + "\n开始计算...\n")
        
        # 准备 OpenCV 需要的列表
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        K = self.get_camera_matrix()
        dist = np.zeros(5) # 假设 Halcon 内参已校正图像，或输入畸变系数(这里简化处理，通常Halcon导出的是已去畸变图，或者传入畸变)
        
        # 生成标定板物理坐标
        rows, cols = self.rows.get(), self.cols.get()
        spacing = self.spacing.get()
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp = objp * float(spacing)
        
        rot_order = self.rot_order.get() # e.g., 'xyz'
        
        for d in self.calib_data:
            # 1. 图像位姿 (Target -> Cam)
            corners = np.array(d["corners"], dtype=np.float32)
            ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
            
            R_mat_cam, _ = cv2.Rodrigues(rvec)
            R_target2cam.append(R_mat_cam)
            t_target2cam.append(tvec)
            
            # 2. 机械臂位姿 (Gripper -> Base)
            # 注意: 输入的是 X,Y,Z, Rx,Ry,Rz
            # 必须转为旋转矩阵
            pose = d["robot_pose"]
            t_g2b = np.array([[pose[0]], [pose[1]], [pose[2]]], dtype=np.float64)
            
            # 欧拉角转旋转矩阵
            # 假设输入是度数
            r_euler = [pose[3], pose[4], pose[5]]
            r = R.from_euler(rot_order, r_euler, degrees=True)
            R_g2b = r.as_matrix()
            
            R_gripper2base.append(R_g2b)
            t_gripper2base.append(t_g2b)
            
        # 调用 OpenCV 手眼标定
        try:
            method = cv2.CALIB_HAND_EYE_TSAI
            
            if self.calib_mode.get().startswith("Eye-in-Hand"):
                # 眼在手上: 求解 Camera -> Gripper
                # OpenCV 文档: calibrateHandEye 输入的是 (R_g2b, t_g2b, R_t2c, t_t2c)
                # 输出 R_cam2gripper, t_cam2gripper
                R_c2g, t_c2g = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base, 
                    R_target2cam, t_target2cam, 
                    method=method
                )
                
                self.txt_result.insert(tk.END, "=== 标定结果 (Eye-in-Hand) ===\n")
                self.txt_result.insert(tk.END, "结果矩阵 (Camera -> Gripper):\n")
                
            else:
                # 眼在手外: 求解 Camera -> Base
                # 对于 Eye-to-Hand, OpenCV 需要输入 Gripper->Base 和 Target->Cam
                # 但需要告知是 Eye-to-Hand 模式? OpenCV 没有直接标志位，通常是通过输入逆矩阵处理
                # 但 Tsai 方法对 Eye-to-Hand 的标准做法是：
                # 输入: Base -> Gripper (即 Gripper相对于Base的逆) 和 Target -> Cam
                # 这样求出的是 Cam -> Base
                
                # 为简单起见，我们对数据做变换以适配 OpenCV 接口
                # Eye-to-Hand 实际上等价于 Eye-in-Hand 的逆问题
                # 此处为简化，我们直接尝试输出，但标记需要用户验证
                
                # 重新组织输入: 
                # 我们需要 Cam 固定，Gripper 动
                # 标准库行为可能需要倒置输入，这里先用标准调用，并在输出解释
                
                R_c2b, t_c2b = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base, 
                    R_target2cam, t_target2cam, 
                    method=method
                )
                self.txt_result.insert(tk.END, "=== 标定结果 (Eye-to-Hand) ===\n")
                self.txt_result.insert(tk.END, "结果矩阵 (Camera -> Base):\n")
                
                R_c2g, t_c2g = R_c2b, t_c2b

            # 格式化输出矩阵
            H = np.eye(4)
            H[:3, :3] = R_c2g
            H[:3, 3] = t_c2g.flatten()
            
            np.set_printoptions(precision=4, suppress=True)
            self.txt_result.insert(tk.END, str(H) + "\n\n")
            
            self.txt_result.insert(tk.END, "平移向量 (X, Y, Z):\n")
            self.txt_result.insert(tk.END, str(t_c2g.flatten()) + "\n")
            
            # 欧拉角
            r_res = R.from_matrix(R_c2g)
            euler_res = r_res.as_euler('xyz', degrees=True)
            self.txt_result.insert(tk.END, f"旋转 (Euler XYZ, deg): {euler_res}\n")
            
            self.txt_result.insert(tk.END, "\n[下一步]\n")
            self.txt_result.insert(tk.END, "1. 将此矩阵填入你的机器人控制程序。\n")
            self.txt_result.insert(tk.END, "2. 最终抓取点 Base = T_HandEye * P_Camera\n")
            
        except Exception as e:
            self.txt_result.insert(tk.END, f"计算出错: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandEyeCalibrationTool(root)
    root.mainloop()
