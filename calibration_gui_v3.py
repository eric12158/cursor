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

class CalibrationSystemV3:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV 标定系统 V3.0 - 工业级排查版")
        self.root.geometry("1400x900")
        
        # 核心数据
        self.calib_images = [] # 存储图片数据
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
        # 左右分栏
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        left_frame = tk.Frame(main_paned, width=400, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        main_paned.add(left_frame, minsize=350)
        
        # 1. 参数设置区域
        p_frame = tk.LabelFrame(left_frame, text="1. 标定板物理参数", bg="#f0f0f0", font=("Arial", 10, "bold"))
        p_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(p_frame, text="行数 (Rows):", bg="#f0f0f0").grid(row=0, column=0, padx=5, pady=5)
        self.var_rows = tk.IntVar(value=DEFAULT_ROWS)
        tk.Entry(p_frame, textvariable=self.var_rows, width=5).grid(row=0, column=1)
        
        tk.Label(p_frame, text="列数 (Cols):", bg="#f0f0f0").grid(row=0, column=2, padx=5)
        self.var_cols = tk.IntVar(value=DEFAULT_COLS)
        tk.Entry(p_frame, textvariable=self.var_cols, width=5).grid(row=0, column=3)
        
        tk.Label(p_frame, text="圆心间距 (mm):", bg="#f0f0f0", fg="red").grid(row=1, column=0, padx=5, pady=5)
        self.var_spacing = tk.DoubleVar(value=DEFAULT_SPACING)
        tk.Entry(p_frame, textvariable=self.var_spacing, width=10, bg="#fff3e0").grid(row=1, column=1, columnspan=2, sticky="w")

        # 2. 图像加载区域
        l_frame = tk.LabelFrame(left_frame, text="2. 图像处理", bg="#f0f0f0", font=("Arial", 10, "bold"))
        l_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(l_frame, text="选择图片文件夹并检测", command=self.load_and_detect, bg="#e3f2fd", height=2).pack(fill=tk.X, padx=5, pady=5)
        
        # 列表框
        self.lst_images = tk.Listbox(l_frame, height=15, selectmode=tk.SINGLE, font=("Consolas", 9))
        self.lst_images.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lst_images.bind("<<ListboxSelect>>", self.on_select_image)
        
        # 3. 标定执行区域
        c_frame = tk.LabelFrame(left_frame, text="3. 计算与保存", bg="#f0f0f0", font=("Arial", 10, "bold"))
        c_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.btn_calib = tk.Button(c_frame, text="执行标定计算", command=self.run_calibration, bg="#c8e6c9", height=2, state=tk.DISABLED)
        self.btn_calib.pack(fill=tk.X, padx=5, pady=5)
        
        self.txt_log = tk.Text(c_frame, height=10, font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Button(c_frame, text="保存结果 (JSON)", command=self.save_result, bg="#ffcc80").pack(fill=tk.X, padx=5, pady=5)

        # 右侧显示区域
        right_frame = tk.Frame(main_paned, bg="#333333")
        main_paned.add(right_frame, stretch="always")
        
        self.canvas = tk.Canvas(right_frame, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标操作
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        
        # 初始提示
        self.draw_center_text("请加载图片\n(支持滚轮缩放 / 拖拽移动)")

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

    # --- 图像加载逻辑 ---
    def load_and_detect(self):
        folder = filedialog.askdirectory()
        if not folder: return
        
        self.calib_images = []
        self.lst_images.delete(0, tk.END)
        self.txt_log.delete(1.0, tk.END)
        self.btn_calib.config(state=tk.DISABLED)
        
        # 收集文件
        exts = ['*.jpg', '*.png', '*.bmp', '*.jpeg', '*.tif']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        files.sort()
        
        if not files:
            messagebox.showwarning("空文件夹", "未找到图片文件")
            return
            
        rows = self.var_rows.get()
        cols = self.var_cols.get()
        
        self.log(f"开始扫描 {len(files)} 张图片...")
        self.log(f"目标阵列: {rows} 行 x {cols} 列")
        
        # 多线程处理避免卡死
        threading.Thread(target=self._process_images, args=(files, rows, cols), daemon=True).start()

    def _process_images(self, files, rows, cols):
        count_ok = 0
        
        for i, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            
            try:
                # 1. 读取图片 (支持中文路径)
                img_data = np.fromfile(fpath, dtype=np.uint8)
                img = cv2.imdecode(img_data, -1)
                
                if img is None:
                    continue
                
                # 2. 转换为 BGR (如果是灰度或RGBA)
                if len(img.shape) == 2:
                    gray = img.copy()
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                elif len(img.shape) == 3:
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 3. 检测圆点
                flags = cv2.CALIB_CB_SYMMETRIC_GRID
                ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
                
                # 4. 如果失败，尝试 Blob 检测器增强
                if not ret:
                    blob_params = cv2.SimpleBlobDetector_Params()
                    blob_params.minArea = 10
                    blob_params.minDistBetweenBlobs = 5
                    blob_detector = cv2.SimpleBlobDetector_create(blob_params)
                    ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags | cv2.CALIB_CB_CLUSTERING, blobDetector=blob_detector)
                
                # 5. 存储
                item = {
                    "path": fpath,
                    "name": fname,
                    "img": img, # 缓存原图用于显示
                    "gray_shape": gray.shape[::-1], # (w, h)
                    "corners": corners,
                    "found": ret
                }
                self.calib_images.append(item)
                
                # 6. 更新 UI
                tag = "[OK]" if ret else "[--]"
                if ret: count_ok += 1
                
                self.root.after(0, self._update_list, f"{tag} {fname}", i)
                
            except Exception as e:
                print(f"Error processing {fname}: {e}")
        
        self.root.after(0, self._finish_detection, count_ok)

    def _update_list(self, text, idx):
        self.lst_images.insert(tk.END, text)
        if "[OK]" in text:
            self.lst_images.itemconfig(idx, {'bg': '#e8f5e9'})
        else:
            self.lst_images.itemconfig(idx, {'fg': '#999999'})
        self.lst_images.see(tk.END)

    def _finish_detection(self, count):
        self.log("-" * 30)
        self.log(f"检测完成。有效图片: {count}")
        if count >= 3:
            self.btn_calib.config(state=tk.NORMAL)
            self.log("请点击 '执行标定计算'")
        else:
            messagebox.showerror("数量不足", "至少需要 3 张成功检测的图片才能标定！")

    # --- 图像显示与交互 ---
    def on_select_image(self, event):
        sel = self.lst_images.curselection()
        if not sel: return
        idx = sel[0]
        self.current_img_idx = idx
        self.draw_image()

    def draw_image(self):
        if self.current_img_idx < 0 or self.current_img_idx >= len(self.calib_images): return
        
        data = self.calib_images[self.current_img_idx]
        img_vis = data['img'].copy()
        
        # 如果检测成功，绘制角点
        if data['found']:
            rows = self.var_rows.get()
            cols = self.var_cols.get()
            cv2.drawChessboardCorners(img_vis, (cols, rows), data['corners'], True)
            
            # --- 关键调试信息：显示像素距离 ---
            # 画出第0个点和第1个点的连线，并显示像素距离
            p0 = tuple(data['corners'][0][0].astype(int))
            p1 = tuple(data['corners'][1][0].astype(int))
            
            cv2.circle(img_vis, p0, 8, (0, 0, 255), -1) # Red for Start
            cv2.line(img_vis, p0, p1, (0, 255, 255), 2)
            
            px_dist = np.linalg.norm(data['corners'][0] - data['corners'][1])
            mm_dist = self.var_spacing.get()
            
            info = f"Px Dist: {px_dist:.1f}"
            cv2.putText(img_vis, info, (p0[0]+10, p0[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 在图上显示分辨率
            h, w = img_vis.shape[:2]
            cv2.putText(img_vis, f"Res: {w}x{h}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 缩放转 PIL
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10: canvas_w=800
        
        # 计算缩放
        scale_fit = min(canvas_w/w, canvas_h/h)
        final_scale = scale_fit * self.zoom
        
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        if new_w > 0 and new_h > 0:
            img_small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            self.tk_image = ImageTk.PhotoImage(Image.fromarray(img_small))
            
            self.canvas.delete("all")
            # 居中 + 偏移
            cx = canvas_w // 2 + self.pan_x
            cy = canvas_h // 2 + self.pan_y
            self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.tk_image)

    # 鼠标事件
    def on_zoom(self, event):
        if event.delta > 0: self.zoom *= 1.1
        else: self.zoom *= 0.9
        self.draw_image()
    
    def on_drag_start(self, event):
        self.drag_start = (event.x, event.y)
        
    def on_drag_move(self, event):
        if not self.drag_start: return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start = (event.x, event.y)
        self.draw_image()

    # --- 标定计算 ---
    def run_calibration(self):
        rows = self.var_rows.get()
        cols = self.var_cols.get()
        spacing = self.var_spacing.get()
        
        valid_data = [d for d in self.calib_images if d['found']]
        if not valid_data: return
        
        # 1. 准备物理坐标
        # 规则：Z=0，X和Y按间距分布
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp = objp * spacing 
        
        objpoints = [objp] * len(valid_data)
        imgpoints = [d['corners'] for d in valid_data]
        img_size = valid_data[0]['gray_shape'] # w, h
        
        self.log(f"正在计算... (图片数: {len(valid_data)})")
        self.root.update()
        
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )
            
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.reproj_err = ret
            
            # 输出报告
            self.log("=" * 40)
            self.log(f"标定成功！")
            self.log(f"RMS 误差: {ret:.4f} (越小越好)")
            self.log(f"图像分辨率: {img_size}")
            self.log("-" * 20)
            self.log(f"内参矩阵 (Camera Matrix):")
            self.log(f"Fx: {mtx[0,0]:.2f}")
            self.log(f"Fy: {mtx[1,1]:.2f}")
            self.log(f"Cx: {mtx[0,2]:.2f}")
            self.log(f"Cy: {mtx[1,2]:.2f}")
            self.log("-" * 20)
            self.log(f"畸变系数: {np.ravel(dist)}")
            self.log("=" * 40)
            
            messagebox.showinfo("成功", f"标定完成！RMS: {ret:.4f}")
            
        except Exception as e:
            self.log(f"标定崩溃: {e}")
            messagebox.showerror("Error", str(e))

    def save_result(self):
        if self.camera_matrix is None:
            messagebox.showwarning("提示", "请先执行标定！")
            return
            
        fpath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            initialfile="calibration_result_v3.json"
        )
        if fpath:
            data = {
                "camera_matrix": self.camera_matrix.tolist(),
                "dist_coeffs": self.dist_coeffs.tolist(),
                "rms": self.reproj_err,
                "image_size": self.calib_images[0]['gray_shape'] if self.calib_images else [0,0],
                "date": str(datetime.now())
            }
            try:
                with open(fpath, 'w') as f:
                    json.dump(data, f, indent=4)
                self.log(f"结果已保存: {fpath}")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationSystemV3(root)
    root.mainloop()
