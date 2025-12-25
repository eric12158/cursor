import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import json
import itertools
import math

class PermutationDebugger:
    def __init__(self, root):
        self.root = root
        self.root.title("PnP 全排列穷举调试工具")
        self.root.geometry("1500x900")
        
        # 默认参数 (请加载 json)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.results = [] # 存储所有排列的结果
        self.current_cv_img = None
        self.points_original = None # 原始检测到的4个点
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # 左侧控制区
        left_panel = tk.Frame(self.root, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(left_panel, text="1. 加载参数", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=5)
        tk.Button(left_panel, text="加载 JSON 参数文件", command=self.load_params).pack(fill=tk.X)
        self.lbl_params = tk.Label(left_panel, text="未加载", fg="red")
        self.lbl_params.pack(anchor=tk.W)

        tk.Label(left_panel, text="2. 打开图片 & 扫描", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20,5))
        tk.Label(left_panel, text="二维码边长(mm):").pack(anchor=tk.W)
        self.qr_size_var = tk.DoubleVar(value=26.0)
        tk.Entry(left_panel, textvariable=self.qr_size_var).pack(fill=tk.X)
        tk.Button(left_panel, text="打开图片并计算全排列", command=self.run_analysis, bg="#e3f2fd").pack(fill=tk.X, pady=5)

        tk.Label(left_panel, text="3. 排列组合结果 (按误差排序)", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20,5))
        
        # 结果列表
        columns = ("rank", "order", "dist", "error")
        self.tree = ttk.Treeview(left_panel, columns=columns, show="headings", height=20)
        self.tree.heading("rank", text="#")
        self.tree.column("rank", width=30)
        self.tree.heading("order", text="点序索引")
        self.tree.column("order", width=80)
        self.tree.heading("dist", text="距离(mm)")
        self.tree.column("dist", width=80)
        self.tree.heading("error", text="重投影误差")
        self.tree.column("error", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_result)

        # 右侧显示区
        right_panel = tk.Frame(self.root, bg="#404040")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(right_panel, bg="#404040")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)

    def load_params(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not fpath: return
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            self.lbl_params.config(text=f"已加载: {os.path.basename(fpath)}", fg="green")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_analysis(self):
        if self.camera_matrix is None:
            messagebox.showwarning("提示", "请先加载相机参数")
            return
            
        fpath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if not fpath: return
        
        # 读取图片
        self.current_cv_img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), -1)
        if len(self.current_cv_img.shape) == 2:
            self.current_cv_img = cv2.cvtColor(self.current_cv_img, cv2.COLOR_GRAY2BGR)
            
        # 检测二维码
        detector = cv2.QRCodeDetector()
        ret, info, points, _ = detector.detectAndDecodeMulti(self.current_cv_img)
        
        if not ret:
            messagebox.showinfo("提示", "未检测到二维码")
            return
            
        # 取第一个二维码的4个点
        self.points_original = points[0].reshape(4, 2).astype(np.float64)
        
        # 定义真实物理坐标 (Z=0)
        sz = self.qr_size_var.get()
        # 标准顺序：TL, TR, BR, BL
        obj_points = np.array([
            [0, 0, 0],    # TL
            [sz, 0, 0],   # TR
            [sz, sz, 0],  # BR
            [0, sz, 0]    # BL
        ], dtype=np.float64)
        
        # 穷举所有排列
        indices = [0, 1, 2, 3]
        perms = list(itertools.permutations(indices))
        
        self.results = []
        
        for p in perms:
            # 根据当前排列重新组合图像点
            # 这里的 p 比如是 (0, 1, 2, 3) 或者 (1, 0, 3, 2)
            # 我们假设 obj_points 的顺序是固定的 (TL, TR, BR, BL)
            # 我们尝试将图像点的不同顺序匹配给这固定的物理点
            img_points_perm = self.points_original[list(p)]
            
            # PnP
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points_perm, self.camera_matrix, self.dist_coeffs)
            
            if success:
                # 计算重投影误差
                projected_pts, _ = cv2.projectPoints(obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                error = cv2.norm(img_points_perm, projected_pts.reshape(4, 2), cv2.NORM_L2) / 4.0
                
                dist = np.linalg.norm(tvec)
                
                # 计算欧拉角
                rmat, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
                if sy > 1e-6:
                    rx = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
                    ry = math.degrees(math.atan2(-rmat[2,0], sy))
                    rz = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
                else:
                    rx = math.degrees(math.atan2(-rmat[1,2], rmat[1,1]))
                    ry = math.degrees(math.atan2(-rmat[2,0], sy))
                    rz = 0
                
                self.results.append({
                    "order": p,
                    "dist": dist,
                    "error": error,
                    "rvec": rvec,
                    "tvec": tvec,
                    "angles": (rx, ry, rz)
                })
        
        # 按误差排序
        self.results.sort(key=lambda x: x["error"])
        
        # 更新列表
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for i, res in enumerate(self.results):
            self.tree.insert("", "end", values=(i+1, str(res["order"]), f"{res['dist']:.1f}", f"{res['error']:.4f}"))
            
        # 默认选中第一个
        if self.results:
            self.display_result(0)

    def on_select_result(self, event):
        sel = self.tree.selection()
        if not sel: return
        item = self.tree.item(sel[0])
        idx = int(item['values'][0]) - 1
        self.display_result(idx)

    def display_result(self, idx):
        if idx >= len(self.results): return
        res = self.results[idx]
        
        img = self.current_cv_img.copy()
        
        # 1. 绘制当前排列的连线 (绿色框)
        # 注意：这里我们按 res['order'] 的顺序来连线
        # 这代表算法认为的 TL -> TR -> BR -> BL
        ordered_pts = self.points_original[list(res['order'])].astype(np.int32)
        
        for i in range(4):
            cv2.line(img, tuple(ordered_pts[i]), tuple(ordered_pts[(i+1)%4]), (0, 255, 0), 2)
            
        # 2. 标记角点定义
        labels = ["TL(0,0)", "TR(sz,0)", "BR(sz,sz)", "BL(0,sz)"]
        for i, pt in enumerate(ordered_pts):
            cv2.circle(img, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(img, f"{i}:{labels[i]}", (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 3. 绘制坐标轴
        axis_len = 20.0
        axis = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, res['rvec'], res['tvec'], self.camera_matrix, self.dist_coeffs)
        imgpts = imgpts.astype(np.int32)
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3) # X - Red
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3) # Y - Green
        cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3) # Z - Blue (Down)

        # 4. 显示信息
        info = f"Rank #{idx+1} | Error: {res['error']:.2f}\n"
        info += f"Dist: {res['dist']:.1f} mm\n"
        info += f"Rot: {res['angles'][0]:.1f}, {res['angles'][1]:.1f}, {res['angles'][2]:.1f}"
        
        y0, dy = 30, 30
        for i, line in enumerate(info.split('\n')):
            cv2.putText(img, line, (20, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.img_to_show = img
        self.update_canvas()

    def update_canvas(self):
        if self.img_to_show is None: return
        img_rgb = cv2.cvtColor(self.img_to_show, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw<10: cw=800
        
        scale = min(cw/w, ch/h) * self.scale
        nw, nh = int(w*scale), int(h*scale)
        
        img_rs = cv2.resize(img_rgb, (nw, nh))
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(img_rs))
        
        self.canvas.delete("all")
        self.canvas.create_image(cw//2+self.offset_x, ch//2+self.offset_y, anchor=tk.CENTER, image=self.tk_img)

    def on_mouse_wheel(self, event):
        if event.delta > 0: self.scale *= 1.1
        else: self.scale *= 0.9
        self.update_canvas()
    def on_mouse_press(self, event):
        self.last_x, self.last_y = event.x, event.y
    def on_mouse_drag(self, event):
        self.offset_x += event.x - self.last_x
        self.offset_y += event.y - self.last_y
        self.last_x, self.last_y = event.x, event.y
        self.update_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = PermutationDebugger(root)
    root.mainloop()
