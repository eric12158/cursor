import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os

class VerificationSystemV3:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV 通用验证系统 V3.0 - 专业完整版")
        self.root.geometry("1400x900")
        
        # 核心数据
        self.camera_matrix = None
        self.dist_coeffs = None
        self.img_size_calib = None # 标定时的图像尺寸
        self.current_img = None
        self.processed_img = None  # 缓存处理后的图像用于显示
        self.tk_image = None
        
        # 图像交互
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        left_frame = tk.Frame(main_paned, width=350, bg="#f0f0f0")
        left_frame.pack_propagate(False)
        main_paned.add(left_frame, minsize=350)
        
        # 1. 参数加载
        p_frame = tk.LabelFrame(left_frame, text="1. 加载标定参数", bg="#f0f0f0", font=("微软雅黑", 10, "bold"))
        p_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(p_frame, text="读取参数文件 (.json)", command=self.load_params, bg="#ffcc80", height=2).pack(fill=tk.X, padx=5, pady=5)
        self.lbl_param_status = tk.Label(p_frame, text="未加载", fg="red", bg="#f0f0f0")
        self.lbl_param_status.pack(pady=2)
        
        self.txt_params = tk.Text(p_frame, height=8, font=("Consolas", 9), bg="#fafafa")
        self.txt_params.pack(fill=tk.X, padx=5, pady=5)

        # 2. 二维码真实尺寸
        s_frame = tk.LabelFrame(left_frame, text="2. 目标设置", bg="#f0f0f0", font=("微软雅黑", 10, "bold"))
        s_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(s_frame, text="二维码边长 (mm):", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.var_qr_size = tk.DoubleVar(value=100.0)
        tk.Entry(s_frame, textvariable=self.var_qr_size, width=10).grid(row=0, column=1, pady=5)
        
        # 3. 图像检测
        l_frame = tk.LabelFrame(left_frame, text="3. 验证图片", bg="#f0f0f0", font=("微软雅黑", 10, "bold"))
        l_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(l_frame, text="打开测试图片", command=self.load_image, bg="#e3f2fd", height=2).pack(fill=tk.X, padx=5, pady=5)
        
        # 4. 结果显示
        r_frame = tk.LabelFrame(left_frame, text="4. 计算结果", bg="#f0f0f0", font=("微软雅黑", 10, "bold"))
        r_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.txt_result = tk.Text(r_frame, height=15, font=("Consolas", 10), bg="#e8f5e9")
        self.txt_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右侧图像显示
        right_frame = tk.Frame(main_paned, bg="#333333")
        main_paned.add(right_frame, stretch="always")
        
        self.canvas = tk.Canvas(right_frame, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        
        self.draw_center_text("通用验证工具 V3.0\n请先加载参数文件")

    def log(self, msg):
        self.txt_result.insert(tk.END, msg + "\n")
        self.txt_result.see(tk.END)

    def draw_center_text(self, text):
        self.canvas.delete("all")
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        if w < 10: w=800; h=600
        self.canvas.create_text(w//2, h//2, text=text, fill="white", font=("Arial", 16), justify=tk.CENTER)

    # --- 核心逻辑 ---
    def load_params(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not path: return
        try:
            with open(path, 'r') as f: data = json.load(f)
            self.camera_matrix = np.array(data["camera_matrix"])
            self.dist_coeffs = np.array(data["dist_coeffs"])
            self.img_size_calib = tuple(data.get("image_size", [0,0]))
            
            self.lbl_param_status.config(text=f"已加载: {os.path.basename(path)}", fg="green")
            self.txt_params.delete(1.0, tk.END)
            self.txt_params.insert(tk.END, f"Fx: {self.camera_matrix[0,0]:.2f}\n")
            self.txt_params.insert(tk.END, f"Fy: {self.camera_matrix[1,1]:.2f}\n")
            self.txt_params.insert(tk.END, f"Cx: {self.camera_matrix[0,2]:.2f}\n")
            self.txt_params.insert(tk.END, f"Cy: {self.camera_matrix[1,2]:.2f}\n")
            self.txt_params.insert(tk.END, f"RMS: {data.get('rms',0):.4f}\n")
            self.txt_params.insert(tk.END, f"Size: {self.img_size_calib}")
            
            # 如果已有图片，重新计算
            if self.current_img is not None:
                self.process_detection()
                
        except Exception as e:
            messagebox.showerror("错误", f"参数加载失败: {e}")

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        try:
            # 中文路径支持
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            if img is None: raise ValueError("无法读取图片")
            
            # 通道处理
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            self.current_img = img
            self.process_detection()
            
        except Exception as e:
            messagebox.showerror("错误", f"图片加载失败: {e}")

    def process_detection(self):
        if self.current_img is None: return
        
        # 创建一个副本用于绘制结果，避免修改原始数据
        self.processed_img = self.current_img.copy()
        self.txt_result.delete(1.0, tk.END)
        
        if self.camera_matrix is None:
            self.log("错误: 请先加载标定参数")
            self.update_canvas()
            return

        # 自动分辨率匹配
        h, w = self.processed_img.shape[:2]
        mtx = self.camera_matrix.copy()
        
        if self.img_size_calib and self.img_size_calib[0] > 0:
            calib_w, calib_h = self.img_size_calib
            if w != calib_w or h != calib_h:
                scale_x = w / calib_w
                scale_y = h / calib_h
                self.log(f"[提示] 分辨率不匹配，自动缩放内参:")
                self.log(f"  原: {calib_w}x{calib_h} -> 现: {w}x{h}")
                self.log(f"  缩放系数: {scale_x:.2f}")
                mtx[0,0] *= scale_x # fx
                mtx[1,1] *= scale_y # fy
                mtx[0,2] *= scale_x # cx
                mtx[1,2] *= scale_y # cy

        # QR检测
        det = cv2.QRCodeDetector()
        ret, decoded_info, points, _ = det.detectAndDecodeMulti(self.processed_img)
        
        if not ret or points is None:
            self.log("未检测到二维码")
            self.update_canvas()
            return
            
        points = points[0] # 取第一个二维码
        
        # 绘制检测点
        for i, pt in enumerate(points):
            pt = tuple(map(int, pt))
            cv2.circle(self.processed_img, pt, 5, (0, 0, 255), -1) # 红色实心点
            cv2.putText(self.processed_img, str(i), (pt[0]+10, pt[1]-10), 0, 1.0, (0,0,255), 2)
            
        # PnP 解算
        half_sz = self.var_qr_size.get() / 2.0
        # 定义二维码坐标系 (中心为原点，Z轴垂直向外)
        # 顺序: 左上, 右上, 右下, 左下
        obj_pts = np.array([
            [-half_sz, half_sz, 0],
            [ half_sz, half_sz, 0],
            [ half_sz,-half_sz, 0],
            [-half_sz,-half_sz, 0]
        ], dtype=np.float64)
        
        success, rvec, tvec = cv2.solvePnP(obj_pts, points, mtx, self.dist_coeffs)
        
        if success:
            dist = np.linalg.norm(tvec)
            self.log(f"=== 计算结果 ===")
            self.log(f"距离 (Distance): {dist:.2f} mm")
            self.log(f"X平移: {tvec[0][0]:.2f}")
            self.log(f"Y平移: {tvec[1][0]:.2f}")
            self.log(f"Z平移: {tvec[2][0]:.2f}")
            
            # 绘制坐标轴
            cv2.drawFrameAxes(self.processed_img, mtx, self.dist_coeffs, rvec, tvec, half_sz)
            
            # 重投影验证 (黄色点) - 这一步非常重要，用于验证“为什么不对”
            # 如果黄色点和红色点重合，说明数学求解正确，问题出在内参或者物理尺寸输入
            reproj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, self.dist_coeffs)
            for i, pt in enumerate(reproj_pts):
                pt = tuple(map(int, pt.ravel()))
                cv2.circle(self.processed_img, pt, 3, (0, 255, 255), -1) # 黄色小点
                
            self.log("\n[图例说明]")
            self.log("红点: 图像识别点")
            self.log("黄点: 参数反算点 (重投影)")
            self.log("若红黄点重合但距离不对 -> 焦距Fx/Fy错误")
            self.log("若红黄点不重合 -> 畸变或角点检测错误")
        else:
            self.log("PnP解算失败")

        self.update_canvas()

    def update_canvas(self, img=None):
        if img is None:
            # 如果未传入 img，尝试使用已处理的图片，否则使用原始图片
            if self.processed_img is not None:
                img = self.processed_img
            elif self.current_img is not None:
                img = self.current_img
            else:
                return

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

    def on_zoom(self, e):
        self.zoom *= (1.1 if e.delta>0 else 0.9); 
        self.update_canvas()
        
    def on_drag_start(self, e):
        self.drag_start = (e.x, e.y)
    def on_drag_move(self, e):
        if self.drag_start:
            self.pan_x += e.x - self.drag_start[0]; self.pan_y += e.y - self.drag_start[1]
            self.drag_start = (e.x, e.y)
            self.update_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = VerificationSystemV3(root)
    root.mainloop()