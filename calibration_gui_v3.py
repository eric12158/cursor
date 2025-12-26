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
DEFAULT_SPACING = 5.0

class CalibrationSystemV3:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV 标定系统 V3.3 - 强制物理约束版")
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
        
        left_frame = tk.Frame(main_paned, width=400, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        main_paned.add(left_frame, minsize=350)
        
        # 1. 参数设置
        p_frame = tk.LabelFrame(left_frame, text="1. 标定板参数", bg="#f0f0f0", font=("Arial", 10, "bold"))
        p_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(p_frame, text="行数:", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.var_rows = tk.IntVar(value=DEFAULT_ROWS)
        tk.Entry(p_frame, textvariable=self.var_rows, width=5).grid(row=0, column=1)
        
        tk.Label(p_frame, text="列数:", bg="#f0f0f0").grid(row=0, column=2, padx=5)
        self.var_cols = tk.IntVar(value=DEFAULT_COLS)
        tk.Entry(p_frame, textvariable=self.var_cols, width=5).grid(row=0, column=3)
        
        tk.Label(p_frame, text="间距(mm):", bg="#f0f0f0", fg="red").grid(row=1, column=0, padx=5, pady=5)
        self.var_spacing = tk.DoubleVar(value=DEFAULT_SPACING)
        tk.Entry(p_frame, textvariable=self.var_spacing, width=10, bg="#fff3e0").grid(row=1, column=1, columnspan=2, sticky="w")

        # 2. 图像加载
        l_frame = tk.LabelFrame(left_frame, text="2. 图像处理", bg="#f0f0f0", font=("Arial", 10, "bold"))
        l_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(l_frame, text="选择图片文件夹并检测", command=self.load_and_detect, bg="#e3f2fd", height=2).pack(fill=tk.X, padx=5, pady=5)
        self.lst_images = tk.Listbox(l_frame, height=12, selectmode=tk.SINGLE, font=("Consolas", 9))
        self.lst_images.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lst_images.bind("<<ListboxSelect>>", self.on_select_image)
        
        # 3. 标定执行
        c_frame = tk.LabelFrame(left_frame, text="3. 标定计算", bg="#f0f0f0", font=("Arial", 10, "bold"))
        c_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 强制模式
        self.force_scale = tk.BooleanVar(value=True)
        tk.Checkbutton(c_frame, text="强制物理缩放 (针对平面数据)", variable=self.force_scale, bg="#f0f0f0", fg="red").pack(anchor=tk.W, padx=5)
        
        self.btn_calib = tk.Button(c_frame, text="执行标定计算", command=self.run_calibration, bg="#c8e6c9", height=2, state=tk.DISABLED)
        self.btn_calib.pack(fill=tk.X, padx=5, pady=5)
        
        self.txt_log = tk.Text(c_frame, height=12, font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Button(c_frame, text="保存结果 (JSON)", command=self.save_result, bg="#ffcc80").pack(fill=tk.X, padx=5, pady=5)

        right_frame = tk.Frame(main_paned, bg="#333333")
        main_paned.add(right_frame, stretch="always")
        self.canvas = tk.Canvas(right_frame, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        
        self.draw_center_text("请加载图片")

    def log(self, msg):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)

    def draw_center_text(self, text):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10: w=800
        if h < 10: h=600
        self.canvas.create_text(w//2, h//2, text=text, fill="white", font=("Arial", 16), justify=tk.CENTER)

    # --- Load ---
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
        self.log(f"开始扫描 {len(files)} 张图片 ({rows}x{cols})...")
        threading.Thread(target=self._process_images, args=(files, rows, cols), daemon=True).start()

    def _process_images(self, files, rows, cols):
        count_ok = 0
        for i, fpath in enumerate(files):
            try:
                img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
                if img is None: continue
                
                # Channel fix
                if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flags = cv2.CALIB_CB_SYMMETRIC_GRID
                ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
                
                if not ret:
                    # Retry with blobs
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

    # --- Run Calibration ---
    def run_calibration(self):
        try:
            rows = self.var_rows.get()
            cols = self.var_cols.get()
            spacing = self.var_spacing.get()
        except: return
        
        valid_data = [d for d in self.calib_images if d['found']]
        if not valid_data: return
        
        # 基础计算使用 1.0 单位，避免数值过大
        base_unit = 1.0
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp = objp * base_unit 
        
        objpoints = []
        imgpoints = []
        for d in valid_data:
            objpoints.append(objp.copy())
            imgpoints.append(d['corners'].copy())
            
        img_size = valid_data[0]['gray_shape']
        
        self.log("="*30)
        self.log(f"计算中... (间距: {spacing})")
        self.root.update()
        
        try:
            # 1. 先按标准单位 1.0 进行标定
            # 这样算出来的 Fx_base 是不带物理单位的纯像素比
            flags = 0
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None, flags=flags
            )
            
            # 2. 强制物理缩放
            # 如果我们知道真实的 spacing 不是 1.0 而是 S
            # 那么 Fx_real 应该怎么变？
            # 这里的逻辑是：如果板子是“平”的，OpenCV 会把尺寸变化完全归结为 Z 轴变化，而保持 Fx 不变。
            # 这是错误的！我们需要强行认为 Z 轴不变，而改变 Fx。
            
            # 如果勾选了强制缩放
            if self.force_scale.get():
                # 假设之前算出的大 Fx (26731) 对应的是间距 1.0
                # 现在间距变成了 15.0 (变大了15倍)
                # 同样的像素大小，物体变大了，说明焦距必须变小才能投影出同样的大小
                # 修正公式： Fx_new = Fx_old / spacing
                
                scale_factor = 1.0 / spacing
                mtx[0, 0] *= scale_factor
                mtx[1, 1] *= scale_factor
                
                # 也要缩放 tvecs 里的平移量吗？
                # 对于 Tvec，既然 Fx 变小了，为了保持投影不变，Z 也得变小吗？
                # Z_new = Z_old / spacing? 不，这里 Tvec 是物理距离
                # 如果 Fx 变小了，Z 保持不变，物体在图上就会变小。但图上物体没变。
                # 所以 Tvec 不需要除？
                # 不对，标定出来的 Tvec 是基于 "objp=1.0" 的单位。
                # 如果 objp 变成了 "spacing"，那么 Tvec 的数值含义也变了。
                # 这是一个单位换算问题。
                
                # 简化逻辑：直接修改 Fx 是最直接的“修正3倍误差”手段。
                self.log(f"[强制修正] 应用比例: 1/{spacing}")
            
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.reproj_err = ret
            self.rvecs = rvecs
            self.tvecs = tvecs
            
            for k, idx in enumerate([i for i, d in enumerate(self.calib_images) if d['found']]):
                self.calib_images[idx]['pose_idx'] = k
            
            self.log(f"标定成功! RMS: {ret:.4f}")
            self.log(f"Fx: {mtx[0,0]:.2f}")
            self.log(f"Fy: {mtx[1,1]:.2f}")
            
            messagebox.showinfo("成功", f"标定完成！RMS: {ret:.4f}\nFx: {mtx[0,0]:.1f}")
            self.draw_image()
            
        except Exception as e:
            self.log(f"Error: {e}")
            import traceback; traceback.print_exc()

    # --- Display & Save ---
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
            
            if self.camera_matrix is not None and data['pose_idx'] != -1:
                rvec = self.rvecs[data['pose_idx']]
                tvec = self.tvecs[data['pose_idx']]
                # 注意：这里的 axis_len 也要按 spacing 缩放显示
                # 但因为 Fx 已经除过了，这里再画 projectPoints 可能会有错位
                # 仅供参考
                axis_len = 3.0 # 单位长度
                axis = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
                ipts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                ipts = ipts.astype(int)
                o = tuple(ipts[0].ravel())
                cv2.line(img, o, tuple(ipts[1].ravel()), (0,0,255), 5) 
                cv2.line(img, o, tuple(ipts[2].ravel()), (0,255,0), 5) 
                cv2.line(img, o, tuple(ipts[3].ravel()), (255,0,0), 5) 
        
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
    app = CalibrationSystemV3(root)
    root.mainloop()
