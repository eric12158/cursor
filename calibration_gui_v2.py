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

# --- 配置常量 ---
DEFAULT_PATTERN_ROWS = 7
DEFAULT_PATTERN_COLS = 7
DEFAULT_CIRCLE_SPACING = 5.0  # mm

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("相机标定工具 V2.0 - 专业版")
        self.root.geometry("1400x900")
        
        # 状态变量
        self.images_data = []  # 存储图片信息
        self.is_detecting = False
        self.calibration_result = None
        
        # 图像显示相关
        self.current_cv_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # 1. 顶部控制栏
        control_frame = tk.LabelFrame(self.root, text="设置与操作", padx=10, pady=5)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 第一行：路径选择
        path_frame = tk.Frame(control_frame)
        path_frame.pack(fill=tk.X, pady=2)
        tk.Label(path_frame, text="图片文件夹:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        tk.Entry(path_frame, textvariable=self.path_var, width=60).pack(side=tk.LEFT, padx=5)
        tk.Button(path_frame, text="浏览...", command=self.browse_folder).pack(side=tk.LEFT)
        
        # 第二行：标定板参数
        param_frame = tk.Frame(control_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(param_frame, text="标定板行数:").pack(side=tk.LEFT)
        self.rows_var = tk.IntVar(value=DEFAULT_PATTERN_ROWS)
        tk.Spinbox(param_frame, from_=3, to=20, textvariable=self.rows_var, width=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(param_frame, text="标定板列数:").pack(side=tk.LEFT, padx=(10, 0))
        self.cols_var = tk.IntVar(value=DEFAULT_PATTERN_COLS)
        tk.Spinbox(param_frame, from_=3, to=20, textvariable=self.cols_var, width=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(param_frame, text="圆心间距 (mm):").pack(side=tk.LEFT, padx=(10, 0))
        self.spacing_var = tk.DoubleVar(value=DEFAULT_CIRCLE_SPACING)
        tk.Entry(param_frame, textvariable=self.spacing_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 操作按钮
        tk.Frame(param_frame, width=20).pack(side=tk.LEFT) # Spacer
        self.btn_detect = tk.Button(param_frame, text="1. 加载并检测角点", command=self.start_detection, bg="#e1f5fe", font=("Arial", 10, "bold"))
        self.btn_detect.pack(side=tk.LEFT, padx=5)
        
        self.btn_calib = tk.Button(param_frame, text="2. 执行标定计算", command=self.run_calibration, bg="#e8f5e9", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_calib.pack(side=tk.LEFT, padx=5)
        
        self.btn_save = tk.Button(param_frame, text="3. 保存标定结果", command=self.save_results, bg="#fff3e0", state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # 2. 主内容区 (左右分栏)
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧：图片列表
        left_frame = tk.Frame(main_paned)
        tk.Label(left_frame, text="图片列表 (绿色=检测成功):").pack(anchor=tk.W)
        self.listbox = tk.Listbox(left_frame, selectmode=tk.SINGLE, font=("Consolas", 9))
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_image)
        main_paned.add(left_frame, minsize=200, width=250)
        
        # 中间：图像显示
        center_frame = tk.Frame(main_paned, bg="#404040")
        self.canvas = tk.Canvas(center_frame, bg="#404040", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件 (缩放和平移)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        
        # 提示覆盖层
        self.canvas_msg = self.canvas.create_text(400, 300, text="请加载图片\n(支持鼠标滚轮缩放，左键拖拽)", fill="gray", font=("Arial", 14), anchor=tk.CENTER)
        
        main_paned.add(center_frame, minsize=500, stretch="always")
        
        # 右侧：日志与结果
        right_frame = tk.Frame(main_paned)
        tk.Label(right_frame, text="系统日志 & 标定结果:").pack(anchor=tk.W)
        self.log_text = tk.Text(right_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        main_paned.add(right_frame, minsize=250, width=350)
        
        # 3. 底部状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padx=5).pack(fill=tk.X)

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_var.set(folder)
            self.log(f"已选择文件夹: {folder}")

    def start_detection(self):
        folder = self.path_var.get()
        if not os.path.isdir(folder):
            messagebox.showerror("路径错误", "请选择有效的文件夹路径！")
            return
            
        try:
            rows = self.rows_var.get()
            cols = self.cols_var.get()
        except:
            messagebox.showerror("参数错误", "行列数必须为整数！")
            return

        self.is_detecting = True
        self.btn_detect.config(state=tk.DISABLED)
        self.btn_calib.config(state=tk.DISABLED)
        self.listbox.delete(0, tk.END)
        self.images_data = []
        
        self.log(f"开始扫描 (阵列: {rows}x{cols})...")
        
        # 开启线程
        threading.Thread(target=self._detection_process, args=(folder, rows, cols), daemon=True).start()

    def _detection_process(self, folder, rows, cols):
        valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        files = []
        for ext in valid_exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        
        files.sort()
        if not files:
            self.root.after(0, lambda: messagebox.showwarning("提示", "该文件夹下没有找到图片！"))
            self.root.after(0, lambda: self.btn_detect.config(state=tk.NORMAL))
            return
            
        success_count = 0
        
        for i, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            self.root.after(0, lambda s=f"正在处理 ({i+1}/{len(files)}): {fname}": self.status_var.set(s))
            
            # 读取图片 - 处理中文路径
            try:
                img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
                if img is None: raise Exception("Decode failed")
                
                # Check channel count
                if len(img.shape) == 2: # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4: # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
            except:
                self.root.after(0, lambda n=fname: self.log(f"无法读取图片: {n}"))
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 关键：查找圆点阵列
            # 使用对称圆点网格 (Symmetric Circles Grid)
            # 也可以尝试 CALIB_CB_CLUSTERING 以增强鲁棒性
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            ret, centers = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
            
            # 如果没找到，尝试 blob detector 增强 (可选，会变慢)
            if not ret:
                 params = cv2.SimpleBlobDetector_Params()
                 params.maxArea = 100000
                 params.minArea = 10
                 params.minDistBetweenBlobs = 5
                 blobDetector = cv2.SimpleBlobDetector_create(params)
                 ret, centers = cv2.findCirclesGrid(gray, (cols, rows), flags=flags | cv2.CALIB_CB_CLUSTERING, blobDetector=blobDetector)

            # 存储结果
            # 注意：不保存原图 img 以节省内存，只保存路径和角点
            data = {
                'path': fpath,
                'name': fname,
                'shape': gray.shape[::-1], # (width, height)
                'found': ret,
                'corners': centers
            }
            self.images_data.append(data)
            
            if ret:
                success_count += 1
                color_code = "#ccffcc" # 浅绿
                tag = " [OK]"
            else:
                color_code = "#ffcccc" # 浅红
                tag = " [FAIL]"
                
            # 更新UI列表
            self.root.after(0, lambda d=data, c=color_code: self._add_list_item(d, c))
            
        self.root.after(0, lambda: self._detection_finished(success_count, len(files)))

    def _add_list_item(self, data, bg_color):
        idx = self.listbox.size()
        tag_status = "[√]" if data['found'] else "[×]"
        self.listbox.insert(tk.END, f"{tag_status} {data['name']}")
        if data['found']:
             self.listbox.itemconfig(idx, {'bg': '#e8f5e9'})
        else:
             self.listbox.itemconfig(idx, {'bg': '#ffebee', 'fg': 'gray'})

    def _detection_finished(self, success, total):
        self.status_var.set("检测完成")
        self.is_detecting = False
        self.btn_detect.config(state=tk.NORMAL)
        
        self.log("-" * 30)
        self.log(f"检测结束。总数: {total}, 成功: {success}")
        
        if success > 0:
            self.btn_calib.config(state=tk.NORMAL)
            self.log("提示: 请点击列表查看检测效果，确认无误后点击'执行标定计算'")
        else:
            messagebox.showwarning("失败", "没有检测到任何有效的标定板，请检查行列数设置或图片质量！")

    def on_select_image(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        
        idx = selection[0]
        if idx >= len(self.images_data): return
        
        data = self.images_data[idx]
        
        # 重新读取图片用于显示
        try:
            img = cv2.imdecode(np.fromfile(data['path'], dtype=np.uint8), -1)
        except:
            return
            
        if data['found'] and data['corners'] is not None:
            # 绘制角点
            rows = self.rows_var.get()
            cols = self.cols_var.get()
            cv2.drawChessboardCorners(img, (cols, rows), data['corners'], data['found'])
            
        # 重置视图并显示
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.current_cv_img = img
        self.display_image()

    def display_image(self):
        if self.current_cv_img is None: return
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10: canvas_w = 800
        if canvas_h < 10: canvas_h = 600
        
        # 1. 计算适应窗口的基础缩放比
        scale_fit = min(canvas_w / w, canvas_h / h)
        
        # 2. 应用用户缩放
        final_scale = scale_fit * self.scale
        
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        # 避免过大导致内存爆炸或过小看不见
        if new_w < 10 or new_h < 10: return
        if new_w > 10000 or new_h > 10000: return # 限制最大渲染尺寸
        
        # 缩放
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        self.pil_img = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        self.canvas.delete("all")
        
        # 计算居中位置 + 偏移
        cx = canvas_w // 2 + self.offset_x
        cy = canvas_h // 2 + self.offset_y
        
        self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.tk_img)

    # --- 鼠标交互 ---
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

    # --- 标定核心逻辑 ---
    def run_calibration(self):
        # 1. 验证输入
        try:
            spacing = self.spacing_var.get()
            if spacing <= 0: raise ValueError
            rows = self.rows_var.get()
            cols = self.cols_var.get()
        except:
            messagebox.showerror("参数错误", "请检查圆点间距是否正确输入！")
            return
            
        valid_data = [d for d in self.images_data if d['found']]
        if len(valid_data) < 3:
            messagebox.showerror("数量不足", "至少需要 3 张成功检测的图片才能进行标定！")
            return
            
        self.log(f"\n>>> 启动标定计算...")
        self.log(f"使用参数: 行={rows}, 列={cols}, 间距={spacing} mm")
        self.log(f"有效图片数: {len(valid_data)}")
        
        # 2. 准备物体坐标 (Object Points)
        # 圆点阵列坐标系：Z=0
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp = objp * spacing  # 转换到真实物理尺寸
        
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane
        
        img_size = valid_data[0]['shape'] # (w, h)
        
        for d in valid_data:
            objpoints.append(objp)
            imgpoints.append(d['corners'])
            
        # 3. OpenCV 标定
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )
            
            # 4. 输出结果
            self.calibration_result = {
                "rms": ret,
                "camera_matrix": mtx.tolist(),
                "dist_coeffs": dist.tolist(),
                "image_size": img_size,
                "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self._print_result_report(ret, mtx, dist, img_size)
            self.btn_save.config(state=tk.NORMAL)
            messagebox.showinfo("成功", f"标定完成！\n平均误差 (RMS): {ret:.4f} 像素")
            
        except cv2.error as e:
            self.log(f"标定出错: {e}")
            messagebox.showerror("OpenCV错误", str(e))

    def _print_result_report(self, ret, mtx, dist, img_size):
        self.log("\n" + "="*40)
        self.log(f"标定报告")
        self.log("="*40)
        self.log(f"图像尺寸: {img_size}")
        self.log(f"重投影误差 (RMS): {ret:.5f} (越小越好，通常<0.5为佳)")
        self.log("-" * 30)
        self.log(f"内参矩阵 (Camera Matrix):\n{np.array2string(mtx, precision=4, separator=', ')}")
        self.log("-" * 30)
        self.log(f"畸变系数 (k1, k2, p1, p2, k3):\n{np.array2string(dist.ravel(), precision=5, separator=', ')}")
        self.log("="*40)
        
        # 生成可直接复制的代码片段
        code = "\n# === Python 参数代码 (可直接复制) ===\n"
        code += "import numpy as np\n\n"
        code += "CAMERA_MATRIX = np.array([\n"
        code += f"    [{mtx[0,0]:.4f}, {mtx[0,1]:.4f}, {mtx[0,2]:.4f}],\n"
        code += f"    [{mtx[1,0]:.4f}, {mtx[1,1]:.4f}, {mtx[1,2]:.4f}],\n"
        code += f"    [{mtx[2,0]:.4f}, {mtx[2,1]:.4f}, {mtx[2,2]:.4f}]\n"
        code += "], dtype=np.float64)\n\n"
        code += f"DIST_COEFFS = np.array({np.array2string(dist.ravel(), separator=', ')}, dtype=np.float64)\n"
        self.log(code)

    def save_results(self):
        if not self.calibration_result: return
        
        fpath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialfile="calibration_result.json"
        )
        if fpath:
            try:
                with open(fpath, 'w', encoding='utf-8') as f:
                    json.dump(self.calibration_result, f, indent=4)
                self.log(f"结果已保存至: {fpath}")
                messagebox.showinfo("保存成功", "标定结果已保存文件。")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    try:
        # 尝试设置高DPI支持 (Windows)
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    app = CalibrationApp(root)
    root.mainloop()
