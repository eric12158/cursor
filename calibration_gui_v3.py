import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import threading
import json
from datetime import datetime

# --- 默认配置 ---
DEFAULT_ROWS = 7
DEFAULT_COLS = 7
DEFAULT_SPACING = 5.0  # mm

class CalibrationSystemV3:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV 标定系统 V3.0 - 终极排查版")
        self.root.geometry("1600x950")
        
        # 核心数据
        self.calib_images = [] # [{'path':..., 'img':..., 'corners':..., 'shape':...}]
        self.camera_matrix = None
        self.dist_coeffs = None
        self.reproj_err = 0.0
        
        # 图像显示状态
        self.current_img_idx = -1
        self.tk_image = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # 左侧控制面板
        left_panel = tk.Frame(self.root, width=350, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # 1. 参数设置
        tk.Label(left_panel, text="1. 标定板参数设置", font=("黑体", 12, "bold"), bg="#f0f0f0").pack(anchor=tk.W, pady=(10,5))
        
        frm_grid = tk.Frame(left_panel, bg="#f0f0f0")
        frm_grid.pack(fill=tk.X, padx=5)
        
        tk.Label(frm_grid, text="行数 (Rows):", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
        self.var_rows = tk.IntVar(value=DEFAULT_ROWS)
        tk.Entry(frm_grid, textvariable=self.var_rows, width=5).grid(row=0, column=1)
        
        tk.Label(frm_grid, text="列数 (Cols):", bg="#f0f0f0").grid(row=0, column=2, sticky="w")
        self.var_cols = tk.IntVar(value=DEFAULT_COLS)
        tk.Entry(frm_grid, textvariable=self.var_cols, width=5).grid(row=0, column=3)
        
        tk.Label(left_panel, text="圆心间距 (mm):", bg="#f0f0f0", fg="red").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.var_spacing = tk.DoubleVar(value=DEFAULT_SPACING)
        tk.Entry(left_panel, textvariable=self.var_spacing, font=("Arial", 12, "bold")).pack(fill=tk.X, padx=5)
        tk.Label(left_panel, text="* 请务必使用卡尺实测，不要信默认值", bg="#f0f0f0", fg="gray", font=("Arial", 8)).pack(anchor=tk.W, padx=5)

        # 2. 图片操作
        tk.Label(left_panel, text="2. 图片加载与检测", font=("黑体", 12, "bold"), bg="#f0f0f0").pack(anchor=tk.W, pady=(20,5))
        tk.Button(left_panel, text="选择文件夹并检测", command=self.load_and_detect, bg="#bbdefb", height=2).pack(fill=tk.X, padx=5)
        
        # 图片列表
        self.lst_images = tk.Listbox(left_panel, height=15, selectmode=tk.SINGLE)
        self.lst_images.pack(fill=tk.X, padx=5, pady=5)
        self.lst_images.bind("<<ListboxSelect>>", self.on_select_image)
        
        # 3. 标定执行
        tk.Label(left_panel, text="3. 执行标定", font=("黑体", 12, "bold"), bg="#f0f0f0").pack(anchor=tk.W, pady=(20,5))
        self.btn_calib = tk.Button(left_panel, text="开始计算参数", command=self.run_calibration, bg="#c8e6c9", height=2, state=tk.DISABLED)
        self.btn_calib.pack(fill=tk.X, padx=5)
        
        # 结果显示
        self.txt_result = tk.Text(left_panel, height=10, font=("Consolas", 9))
        self.txt_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Button(left_panel, text="保存标定结果 (JSON)", command=self.save_result, bg="#ffecb3").pack(fill=tk.X, padx=5, pady=5)

        # 右侧显示区
        right_panel = tk.Frame(self.root, bg="gray")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(right_panel, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 鼠标绑定
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        
        self.draw_overlay_text("请加载图片...")

    def log(self, msg):
        self.txt_result.insert(tk.END, msg + "\n")
        self.txt_result.see(tk.END)

    def draw_overlay_text(self, text):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.create_text(w//2, h//2, text=text, fill="white", font=("Arial", 20))

    # --- 逻辑处理 ---
    def load_and_detect(self):
        folder = filedialog.askdirectory()
        if not folder: return
        
        self.calib_images = []
        self.lst_images.delete(0, tk.END)
        self.txt_result.delete(1.0, tk.END)
        
        files = glob.glob(os.path.join(folder, "*.*"))
        valid_exts = ['.jpg', '.png', '.bmp', '.jpeg', '.tif']
        files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
        files.sort()
        
        if not files:
            messagebox.showwarning("警告", "文件夹为空！")
            return
            
        rows = self.var_rows.get()
        cols = self.var_cols.get()
        
        self.log(f"开始检测 {len(files)} 张图片...")
        self.log(f"目标阵列: {rows} x {cols}")
        
        threading.Thread(target=self._worker_detect, args=(files, rows, cols), daemon=True).start()

    def _worker_detect(self, files, rows, cols):
        success_count = 0
        for i, fpath in enumerate(files):
            try:
                # 兼容中文路径
                img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
                if img is None: continue
                
                # 转灰度
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # 查找圆点
                flags = cv2.CALIB_CB_SYMMETRIC_GRID
                ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
                
                # 如果找不到，尝试 Blob 聚类
                if not ret:
                    params = cv2.SimpleBlobDetector_Params()
                    params.minArea = 10
                    params.minDistBetweenBlobs = 5
                    blob_detector = cv2.SimpleBlobDetector_create(params)
                    ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags | cv2.CALIB_CB_CLUSTERING, blobDetector=blob_detector)
                
                status_mark = "[OK]" if ret else "[XX]"
                if ret: success_count += 1
                
                name = os.path.basename(fpath)
                self.calib_images.append({
                    'path': fpath,
                    'name': name,
                    'img': img, # 缓存图片以便快速显示，内存大可优化
                    'gray_shape': gray.shape[::-1], # w, h
                    'corners': corners,
                    'found': ret
                })
                
                self.root.after(0, self.lst_images.insert, tk.END, f"{status_mark} {name} ({gray.shape[1]}x{gray.shape[0]})")
                self.root.after(0, self.lst_images.see, tk.END)
                
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
        
        self.root.after(0, self.log, f"检测完成: 成功 {success_count}/{len(files)}")
        if success_count >= 3:
            self.root.after(0, self.btn_calib.config, {'state': tk.NORMAL})
        else:
            self.root.after(0, self.log, "有效图片不足3张，无法标定！")

    def on_select_image(self, event):
        sel = self.lst_images.curselection()
        if not sel: return
        idx = sel[0]
        self.current_img_idx = idx
        self.show_image()

    def show_image(self):
        if self.current_img_idx < 0 or self.current_img_idx >= len(self.calib_images): return
        
        data = self.calib_images[self.current_img_idx]
        img_vis = data['img'].copy()
        
        # 绘制角点
        if data['found']:
            cv2.drawChessboardCorners(img_vis, (self.var_cols.get(), self.var_rows.get()), data['corners'], True)
            
            # 画出第一个点(红色)和第二个点(黄色)的距离，用于肉眼验证
            p0 = tuple(data['corners'][0][0].astype(int))
            p1 = tuple(data['corners'][1][0].astype(int))
            cv2.circle(img_vis, p0, 10, (0,0,255), -1) # 0: Red
            cv2.circle(img_vis, p1, 8, (0,255,255), -1) # 1: Yellow
            
            dist_px = np.linalg.norm(data['corners'][0] - data['corners'][1])
            cv2.putText(img_vis, f"P0", (p0[0], p0[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(img_vis, f"Dist: {dist_px:.1f} px", ((p0[0]+p1[0])//2, (p0[1]+p1[1])//2 - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # 转换为PIL并缩放
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # 计算显示尺寸
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # 基础适配
        scale_base = min(cw/w, ch/h)
        scale_final = scale_base * self.zoom
        
        nw, nh = int(w * scale_final), int(h * scale_final)
        if nw<1 or nh<1: return
        
        img_pil = Image.fromarray(cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_NEAREST))
        self.tk_image = ImageTk.PhotoImage(img_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(cw//2 + self.pan_x, ch//2 + self.pan_y, image=self.tk_image, anchor=tk.CENTER)

    # --- 标定核心 ---
    def run_calibration(self):
        rows = self.var_rows.get()
        cols = self.var_cols.get()
        spacing = self.var_spacing.get()
        
        valid_data = [d for d in self.calib_images if d['found']]
        if len(valid_data) < 3: return
        
        # 1. 构建物理坐标 (World Points)
        # 假设 Z=0, 间距 = spacing
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp = objp * spacing 
        
        objpoints = [objp] * len(valid_data)
        imgpoints = [d['corners'] for d in valid_data]
        img_size = valid_data[0]['gray_shape'] # w, h
        
        self.log("正在计算矩阵...")
        self.root.update()
        
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )
            
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.reproj_err = ret
            
            # 报告结果
            self.log("-" * 30)
            self.log(f"标定成功! RMS误差: {ret:.4f}")
            self.log(f"分辨率: {img_size}")
            self.log(f"焦距 fx: {mtx[0,0]:.2f}")
            self.log(f"焦距 fy: {mtx[1,1]:.2f}")
            self.log("-" * 30)
            
            # --- 自动验证：反推间距 ---
            # 我们用算出来的矩阵，去反推第一张图上两个点的物理距离
            # 如果反推出来不是 5mm，那就说明有问题
            self.verify_scale(valid_data[0], mtx, dist, spacing)
            
        except Exception as e:
            self.log(f"标定失败: {e}")
            messagebox.showerror("Error", str(e))

    def verify_scale(self, data, mtx, dist, expected_spacing):
        """自我验证环节"""
        # 取前两个点
        p0 = data['corners'][0]
        p1 = data['corners'][1]
        
        # 这里的验证比较复杂，因为单目无法直接测距（缺少深度）。
        # 但我们知道这俩点在标定板平面上。
        # 我们可以计算它们在图像上的像素距离
        dist_px = np.linalg.norm(p0 - p1)
        
        # 估算： Z = f * real_dist / px_dist
        # 这里只是粗略展示像素密度
        self.log(f"[自我检查] 图上相邻点像素距离: {dist_px:.2f} px")
        self.log(f"[自我检查] 理论物理间距: {expected_spacing:.2f} mm")
        self.log(f"[自我检查] 当前像素密度: {dist_px / expected_spacing:.2f} px/mm")
        
        # 提示用户
        msg = "如果【焦距】看起来特别大(比如>10000)，\n且你的图片分辨率并不是特别高(比如4K)，\n那么请检查：你填的5mm是不是太小了？\n或者图片被裁剪/缩放过？"
        self.log(msg)

    def save_result(self):
        if self.camera_matrix is None: return
        fpath = filedialog.asksaveasfilename(defaultextension=".json", initialfile="calibration_result_v3.json")
        if fpath:
            data = {
                "camera_matrix": self.camera_matrix.tolist(),
                "dist_coeffs": self.dist_coeffs.tolist(),
                "rms": self.reproj_err,
                "calibration_date": str(datetime.now())
            }
            with open(fpath, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("OK", "保存成功")

    # --- 交互 ---
    def on_zoom(self, event):
        if event.delta > 0: self.zoom *= 1.1
        else: self.zoom *= 0.9
        self.show_image()
        
    def on_drag_start(self, event):
        self.drag_start = (event.x, event.y)
        
    def on_drag_move(self, event):
        if not self.drag_start: return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start = (event.x, event.y)
        self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationSystemV3(root)
    root.mainloop()
