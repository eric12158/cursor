import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading

# 默认配置
DEFAULT_DIR = r"D:\Other\Users\ENGINEER\Desktop\cursor-cursor-2d-to-2-5d-camera-2b90\image_folder"
PATTERN_ROWS = 7
PATTERN_COLS = 7
CIRCLE_SPACING = 5.0 # mm

class CalibrationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV 相机标定工具 (圆点阵列 7x7)")
        self.root.geometry("1200x800") # Correct format is "WidthxHeight", not "*"
        
        # Data
        self.images_data = [] # List of dicts: {'path': str, 'found': bool, 'corners': np.array, 'img_shape': tuple}
        self.current_image_idx = -1
        
        # Zoom & Pan State
        self.current_cv_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        
        # UI Layout
        self.setup_ui()
        
        # Set default path if it exists, else use current dir
        if os.path.exists(DEFAULT_DIR):
            self.path_var.set(DEFAULT_DIR)
            
    def setup_ui(self):
        # --- Top Control Panel ---
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(fill=tk.X, padx=10)
        
        tk.Label(control_frame, text="图片文件夹:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        tk.Entry(control_frame, textvariable=self.path_var, width=50).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="浏览...", command=self.browse_folder).pack(side=tk.LEFT)
        
        tk.Label(control_frame, text="   ").pack(side=tk.LEFT) # Spacer
        
        tk.Label(control_frame, text="圆点间距(mm):").pack(side=tk.LEFT)
        self.spacing_var = tk.DoubleVar(value=CIRCLE_SPACING)
        tk.Entry(control_frame, textvariable=self.spacing_var, width=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="圆点间距(mm):").pack(side=tk.LEFT)
        self.spacing_var = tk.DoubleVar(value=CIRCLE_SPACING)
        tk.Entry(control_frame, textvariable=self.spacing_var, width=5).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="1. 加载并检测", command=self.start_detection, bg="#dddddd").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="2. 开始标定", command=self.run_calibration, bg="#aaffaa").pack(side=tk.LEFT, padx=5)

        # --- Main Content Area ---
        content_frame = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=4)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left: Image List
        left_frame = tk.Frame(content_frame, width=200) # Reduce width slightly
        tk.Label(left_frame, text="图片列表 (点击查看):").pack(anchor=tk.W)
        
        self.listbox = tk.Listbox(left_frame, selectmode=tk.SINGLE)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_image)
        
        # Filter checkboxes
        self.show_all_var = tk.BooleanVar(value=True)
        
        content_frame.add(left_frame, minsize=150)
        
        # Center: Image Display (Make it bigger/default)
        center_frame = tk.Frame(content_frame, bg="gray")
        self.canvas = tk.Canvas(center_frame, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind Mouse Events for Zoom/Pan
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        
        content_frame.add(center_frame, minsize=600, stretch="always") # Give more space to center
        
        # Right: Log/Results
        right_frame = tk.Frame(content_frame, width=250)
        tk.Label(right_frame, text="标定结果 / 日志:").pack(anchor=tk.W)
        self.log_text = tk.Text(right_frame, wrap=tk.WORD, width=30)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        content_frame.add(right_frame, minsize=200)

        # Status Bar
        self.status_var = tk.StringVar(value="准备就绪")
        tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_var.set(folder)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def start_detection(self):
        folder = self.path_var.get()
        if not os.path.exists(folder):
            messagebox.showerror("错误", "文件夹路径不存在！")
            return
            
        self.log(f"正在扫描文件夹: {folder}")
        self.listbox.delete(0, tk.END)
        self.images_data = []
        
        # Threading detection to not freeze UI
        threading.Thread(target=self._detect_thread, args=(folder,), daemon=True).start()

    def _detect_thread(self, folder):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
            
        files.sort()
        if not files:
            self.root.after(0, lambda: messagebox.showinfo("提示", "未找到图片！"))
            return
            
        self.root.after(0, lambda: self.log(f"找到 {len(files)} 张图片，开始检测圆点 (7x7)..."))
        
        for idx, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            # Use root.after for thread safety
            self.root.after(0, lambda s=f"正在处理 ({idx+1}/{len(files)}): {fname}": self.status_var.set(s))
            
            img = cv2.imread(fpath)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find Circles Grid
            # pattern_size = (cols, rows) -> (7, 7)
            ret, centers = cv2.findCirclesGrid(gray, (PATTERN_COLS, PATTERN_ROWS), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            
            data = {
                'path': fpath,
                'name': fname,
                'found': ret,
                'corners': centers,
                'shape': gray.shape[::-1], # (w, h)
                'img_original': img # Keep ref? Might be memory heavy. Better re-read for display.
            }
            # We won't store full image in memory to save RAM, just path
            del data['img_original']
            
            self.images_data.append(data)
            
            # Update UI
            tag = "[OK] " if ret else "[FAIL] "
            self.root.after(0, lambda t=tag, n=fname: self.listbox.insert(tk.END, t + n))
            
        self.root.after(0, lambda: self.status_var.set("检测完成。请检查图片，然后点击'开始标定'。"))
        self.root.after(0, lambda: self.log(f"检测完成。成功: {sum(1 for d in self.images_data if d['found'])} / {len(self.images_data)}"))

    def on_select_image(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
            
        idx = selection[0]
        data = self.images_data[idx]
        
        img = cv2.imread(data['path'])
        if img is None:
            return
            
        # Draw corners if found
        if data['found']:
            cv2.drawChessboardCorners(img, (PATTERN_COLS, PATTERN_ROWS), data['corners'], data['found'])
            
        # Reset Zoom/Pan
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.current_cv_img = img
        
        # Initial display
        self.display_image()

    def display_image(self, cv_img=None):
        if cv_img is not None:
            # Should not happen with new logic, but kept for compatibility or direct calls
            self.current_cv_img = cv_img
            
        if self.current_cv_img is None:
            return

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Calculate scale to fit canvas initially, then apply zoom
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10: return
        
        # Base scale to fit
        base_scale = min(canvas_w/w, canvas_h/h)
        
        # Final display size
        final_scale = base_scale * self.scale
        new_w, new_h = int(w * final_scale), int(h * final_scale)
        
        # Resize
        # Use simple interpolation for speed during interaction?
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        self.pil_img = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        self.canvas.delete("all")
        
        # Calculate centered position + offset
        center_x = canvas_w // 2 + self.offset_x
        center_y = canvas_h // 2 + self.offset_y
        
        # Anchor is center to make zooming easier around center? 
        # Or keep NW and calc top-left?
        # Let's use CENTER anchor for the image item.
        self.canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.tk_img)

    def on_mouse_wheel(self, event):
        if self.current_cv_img is None: return
        
        # Determine scroll direction (Windows vs Linux)
        if event.num == 5 or event.delta < 0:
            factor = 0.9
        else:
            factor = 1.1
            
        self.scale *= factor
        # Limit zoom
        if self.scale < 0.1: self.scale = 0.1
        if self.scale > 50.0: self.scale = 50.0
        
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

    def run_calibration(self):
        valid_images = [d for d in self.images_data if d['found']]
        if not valid_images:
            messagebox.showerror("错误", "没有检测到圆点的图片，无法标定！")
            return
            
        try:
            spacing = self.spacing_var.get()
            if spacing <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("错误", "请输入有效的圆点间距！")
            return

        self.log(f"\n>>> 开始计算标定参数 (圆点间距: {spacing} mm)...")
        
        # Prepare Object Points
        # (0,0,0), (1,0,0), (2,0,0) ...., (6,6,0)
        objp = np.zeros((PATTERN_ROWS * PATTERN_COLS, 3), np.float32)
        objp[:, :2] = np.mgrid[0:PATTERN_COLS, 0:PATTERN_ROWS].T.reshape(-1, 2)
        objp = objp * spacing # Scale by user input spacing
        
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        img_size = valid_images[0]['shape'] # (w, h)
        
        for data in valid_images:
            objpoints.append(objp)
            imgpoints.append(data['corners'])
            
        # Calibrate
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )
            
            self.log("-" * 30)
            self.log(f"标定成功！")
            self.log(f"平均重投影误差 (RMS): {ret:.4f} pixels")
            self.log("-" * 30)
            self.log("内参矩阵 (Camera Matrix):\n" + str(mtx))
            self.log("\n畸变系数 (Distortion Coeffs):\n" + str(dist.ravel()))
            self.log("-" * 30)
            
            # Format as Python code for user
            code_str = "\n# --- 复制下面的代码替换之前的参数 ---\n"
            code_str += "import numpy as np\n\n"
            code_str += "CAMERA_MATRIX = np.array([\n"
            code_str += f"    [{mtx[0,0]:.4f}, {mtx[0,1]:.4f}, {mtx[0,2]:.4f}],\n"
            code_str += f"    [{mtx[1,0]:.4f}, {mtx[1,1]:.4f}, {mtx[1,2]:.4f}],\n"
            code_str += f"    [{mtx[2,0]:.4f}, {mtx[2,1]:.4f}, {mtx[2,2]:.4f}]\n"
            code_str += "], dtype=np.float64)\n\n"
            code_str += f"DIST_COEFFS = np.array({np.array2string(dist.ravel(), separator=', ')}, dtype=np.float64)\n"
            
            self.log(code_str)
            messagebox.showinfo("成功", f"标定完成！RMS误差: {ret:.4f}")
            
        except cv2.error as e:
            self.log(f"标定失败: {e}")
            messagebox.showerror("错误", f"标定失败: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationGUI(root)
    root.mainloop()
