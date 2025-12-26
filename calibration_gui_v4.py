import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import json
from datetime import datetime

# --- 默认配置 ---
DEFAULT_ROWS = 7
DEFAULT_COLS = 7
DEFAULT_SPACING = 5.0  # mm

class CalibrationSystemV4:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV 通用标定系统 V4.0 - 专业完整版")
        self.root.geometry("1600x950")
        
        # 核心数据
        self.calib_images = [] 
        self.camera_matrix = None
        self.dist_coeffs = None
        self.reproj_err = 0.0
        self.rvecs = []
        self.tvecs = []
        
        # 图像显示状态
        self.current_img_idx = -1
        self.tk_image = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(main_paned, width=420, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        main_paned.add(left_frame, minsize=400)
        
        # --- 1. 标定板设置 ---
        p_frame = tk.LabelFrame(left_frame, text="1. 标定板参数", bg="#f0f0f0", font=("微软雅黑", 10, "bold"))
        p_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(p_frame, text="行数:", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.var_rows = tk.IntVar(value=DEFAULT_ROWS)
        tk.Entry(p_frame, textvariable=self.var_rows, width=5).grid(row=0, column=1)
        
        tk.Label(p_frame, text="列数:", bg="#f0f0f0").grid(row=0, column=2, padx=5)
        self.var_cols = tk.IntVar(value=DEFAULT_COLS)
        tk.Entry(p_frame, textvariable=self.var_cols, width=5).grid(row=0, column=3)
        
        tk.Label(p_frame, text="间距(mm):", bg="#f0f0f0", fg="blue").grid(row=1, column=0, padx=5, pady=5)
        self.var_spacing = tk.DoubleVar(value=DEFAULT_SPACING)
        tk.Entry(p_frame, textvariable=self.var_spacing, width=10).grid(row=1, column=1, columnspan=2, sticky="w")

        # --- 2. 初始猜测 (解决焦距算不准的关键) ---
        g_frame = tk.LabelFrame(left_frame, text="2. 初始参数猜测 (可选)", bg="#f0f0f0", font=("微软雅黑", 10, "bold"))
        g_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.use_phys_guess = tk.BooleanVar(value=False)
        tk.Checkbutton(g_frame, text="启用物理参数初始猜测 (解决平面退化)", variable=self.use_phys_guess, bg="#f0f0f0", command=self.toggle_guess_inputs).grid(row=0, column=0, columnspan=4, sticky="w")
        
        tk.Label(g_frame, text="像元尺寸(um):", bg="#f0f0f0").grid(row=1, column=0, sticky="w", padx=5)
        self.var_pixel_size = tk.DoubleVar(value=3.45) # 常见工业相机
        self.ent_pixel = tk.Entry(g_frame, textvariable=self.var_pixel_size, width=8, state=tk.DISABLED)
        self.ent_pixel.grid(row=1, column=1)
        
        tk.Label(g_frame, text="镜头焦距(mm):", bg="#f0f0f0").grid(row=1, column=2, sticky="w", padx=5)
        self.var_focal_len = tk.DoubleVar(value=16.0)
        self.ent_focal = tk.Entry(g_frame, textvariable=self.var_focal_len, width=8, state=tk.DISABLED)
        self.ent_focal.grid(row=1, column=3)

        # --- 3. 图像加载 ---
        l_frame = tk.LabelFrame(left_frame, text="3. 图像处理", bg="#f0f0f0", font=("微软雅黑", 10, "bold"))
        l_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(l_frame, text="选择文件夹并检测", command=self.load_and_detect, bg="#e3f2fd", height=2).pack(fill=tk.X, padx=5, pady=5)
        
        self.lst_images = tk.Listbox(l_frame, height=10, selectmode=tk.SINGLE, font=("Consolas", 9))
        self.lst_images.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lst_images.bind("<<ListboxSelect>>", self.on_select_image)
        
        # --- 4. 标定执行 ---
        c_frame = tk.LabelFrame(left_frame, text="4. 标定计算", bg="#f0f0f0", font=("微软雅黑", 10, "bold"))
        c_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.btn_calib = tk.Button(c_frame, text="执行标定计算", command=self.run_calibration, bg="#c8e6c9", height=2, state=tk.DISABLED)
        self.btn_calib.pack(fill=tk.X, padx=5, pady=5)
        
        self.txt_log = tk.Text(c_frame, height=8, font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Button(c_frame, text="保存结果 (JSON)", command=self.save_result, bg="#ffcc80").pack(fill=tk.X, padx=5, pady=5)

        # 右侧显示区域
        right_frame = tk.Frame(main_paned, bg="#333333")
        main_paned.add(right_frame, stretch="always")
        
        self.canvas = tk.Canvas(right_frame, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        
        self.draw_center_text("通用标定工具 V4.0\n等待加载图片...")

    def log(self, msg):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)

    def draw_center_text(self, text):
        self.canvas.delete("all")
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        if w < 10: w=800
        if h < 10: h=600
        self.canvas.create_text(w//2, h//2, text=text, fill="white", font=("Arial", 16), justify=tk.CENTER)

    def toggle_guess_inputs(self):
        state = tk.NORMAL if self.use_phys_guess.get() else tk.DISABLED
        self.ent_pixel.config(state=state)
        self.ent_focal.config(state=state)

    # --- 图像处理 ---
    def load_and_detect(self):
        folder = filedialog.askdirectory()
        if not folder: return
        self.calib_images = []
        self.lst_images.delete(0, tk.END)
        self.txt_log.delete(1.0, tk.END)
        self.btn_calib.config(state=tk.DISABLED)
        
        exts = ['*.jpg', '*.png', '*.bmp', '*.jpeg', '*.tif']
        files = []
        for ext in exts: files.extend(glob.glob(os.path.join(folder, ext)))
        files.sort()
        if not files: return
        
        rows, cols = self.var_rows.get(), self.var_cols.get()
        self.log(f"扫描 {len(files)} 张图片 ({rows}x{cols})...")
        threading.Thread(target=self._process_images, args=(files, rows, cols), daemon=True).start()

    def _process_images(self, files, rows, cols):
        count_ok = 0
        for i, fpath in enumerate(files):
            try:
                img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
                if img is None: continue
                
                # 兼容通道
                if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flags = cv2.CALIB_CB_SYMMETRIC_GRID
                ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
                
                # 增强检测
                if not ret:
                    params = cv2.SimpleBlobDetector_Params()
                    params.minArea = 10; params.minDistBetweenBlobs = 5
                    blob = cv2.SimpleBlobDetector_create(params)
                    ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags | cv2.CALIB_CB_CLUSTERING, blobDetector=blob)
                
                self.calib_images.append({
                    "path": fpath, "name": os.path.basename(fpath),
                    "img": img, "gray_shape": gray.shape[::-1],
                    "corners": corners, "found": ret, "pose_idx": -1
                })
                
                tag = "[OK]" if ret else "[--]"
                if ret: count_ok += 1
                self.root.after(0, self._update_list, f"{tag} {os.path.basename(fpath)}", i)
            except: pass
        
        self.root.after(0, self._finish_detection, count_ok)

    def _update_list(self, text, idx):
        self.lst_images.insert(tk.END, text)
        if "[OK]" in text: self.lst_images.itemconfig(idx, {'bg': '#e8f5e9'})
        else: self.lst_images.itemconfig(idx, {'fg': '#999999'})
        self.lst_images.see(tk.END)

    def _finish_detection(self, count):
        self.log(f"检测完成，有效: {count}")
        if count >= 3: self.btn_calib.config(state=tk.NORMAL)

    # --- 标定核心逻辑 ---
    def run_calibration(self):
        try:
            rows = self.var_rows.get()
            cols = self.var_cols.get()
            spacing = self.var_spacing.get()
        except: return
        
        valid_data = [d for d in self.calib_images if d['found']]
        if not valid_data:
            messagebox.showwarning("警告", "没有有效的标定数据，请检查标定板参数或图片检测结果")
            return
        
        # 1. 动态生成物理坐标 (确保每次点击都重新生成)
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp = objp * float(spacing)
        
        objpoints = [objp.copy() for _ in valid_data]
        imgpoints = [d['corners'].copy() for d in valid_data]
        img_size = valid_data[0]['gray_shape']
        
        self.log("="*30)
        self.log(f"开始计算 (间距: {spacing}mm)")
        self.root.update()
        
        try:
            # 2. 初始内参策略
            init_camera_matrix = None
            flags = 0
            
            if self.use_phys_guess.get():
                # 物理估算模式 (Halcon 风格)
                try:
                    px_size_mm = self.var_pixel_size.get() / 1000.0 # um -> mm
                    focal_mm = self.var_focal_len.get()
                    f_pix = focal_mm / px_size_mm
                    cx, cy = img_size[0]/2, img_size[1]/2
                    
                    init_camera_matrix = np.array([
                        [f_pix, 0, cx],
                        [0, f_pix, cy],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    
                    # 关键修改：增加固定纵横比，这对于平面标定非常重要，防止 fx/fy 漂移
                    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
                    self.log(f"使用物理猜测: f_pix={f_pix:.1f} (固定纵横比)")
                except:
                    self.log("物理参数输入错误，回退到自动猜测")
            
            # 3. 执行标定
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, 
                init_camera_matrix if (flags & cv2.CALIB_USE_INTRINSIC_GUESS) else None, 
                None, 
                flags=flags
            )
            
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.reproj_err = ret
            self.rvecs = rvecs
            self.tvecs = tvecs
            
            # 更新位姿索引
            for k, idx in enumerate([i for i, d in enumerate(self.calib_images) if d['found']]):
                self.calib_images[idx]['pose_idx'] = k
            
            self.log(f"标定成功! RMS: {ret:.4f}")
            self.log(f"Fx: {mtx[0,0]:.2f}")
            self.log(f"Fy: {mtx[1,1]:.2f}")
            self.log(f"Cx: {mtx[0,2]:.2f}, Cy: {mtx[1,2]:.2f}")
            
            messagebox.showinfo("成功", f"标定完成！RMS: {ret:.4f}")
            self.draw_image()
            
        except Exception as e:
            self.log(f"Error: {e}")
            import traceback; traceback.print_exc()

    # --- 显示与交互 ---
    def on_select_image(self, event):
        sel = self.lst_images.curselection()
        if not sel: return
        self.current_img_idx = sel[0]
        self.draw_image()

    def draw_image(self):
        if self.current_img_idx < 0: return
        data = self.calib_images[self.current_img_idx]
        img = data['img'].copy()
        
        if data['found']:
            rows, cols = self.var_rows.get(), self.var_cols.get()
            cv2.drawChessboardCorners(img, (cols, rows), data['corners'], True)
            
            # 绘制位姿坐标轴
            if self.camera_matrix is not None and data['pose_idx'] != -1:
                rvec = self.rvecs[data['pose_idx']]
                tvec = self.tvecs[data['pose_idx']]
                axis_len = self.var_spacing.get() * 3
                axis = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
                
                try:
                    ipts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                    ipts = ipts.astype(int)
                    o = tuple(ipts[0].ravel())
                    cv2.line(img, o, tuple(ipts[1].ravel()), (0,0,255), 5) # X
                    cv2.line(img, o, tuple(ipts[2].ravel()), (0,255,0), 5) # Y
                    cv2.line(img, o, tuple(ipts[3].ravel()), (255,0,0), 5) # Z
                    cv2.putText(img, "X", tuple(ipts[1].ravel()), 1, 2, (0,0,255), 2)
                    cv2.putText(img, "Y", tuple(ipts[2].ravel()), 1, 2, (0,255,0), 2)
                except: pass

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

    def save_result(self):
        if self.camera_matrix is None: return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="calibration_result.json")
        if path:
            d = {
                "camera_matrix": self.camera_matrix.tolist(),
                "dist_coeffs": self.dist_coeffs.tolist(),
                "rms": self.reproj_err,
                "image_size": self.calib_images[0]['gray_shape'] if self.calib_images else [0,0]
            }
            with open(path, 'w') as f: json.dump(d, f, indent=4)
            self.log(f"已保存: {path}")

    def on_zoom(self, e):
        self.zoom *= (1.1 if e.delta>0 else 0.9); self.draw_image()
    def on_drag_start(self, e):
        self.drag_start = (e.x, e.y)
    def on_drag_move(self, e):
        if self.drag_start:
            self.pan_x += e.x - self.drag_start[0]; self.pan_y += e.y - self.drag_start[1]
            self.drag_start = (e.x, e.y); self.draw_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationSystemV4(root)
    root.mainloop()
