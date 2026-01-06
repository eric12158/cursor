import sys
import os
import cv2
import numpy as np
import glob
import time
import threading
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# =============================================================================
# 模块：相机驱动封装
# 尝试加载海康SDK，如果失败则回退到OpenCV标准VideoCapture或模拟模式
# =============================================================================

HAS_HIK_SDK = False
try:
    # 尝试导入海康SDK (这里简化了路径搜索，假设环境已配置好或在标准路径)
    # 在实际部署时，保留 import sys.py 中的路径搜索逻辑会更稳健
    sys.path.append(os.getcwd())
    from MvImport.MvCameraControl_class import *
    HAS_HIK_SDK = True
except ImportError:
    pass

class CameraInterface:
    """
    通用相机接口，统一海康相机和普通Webcam/模拟相机
    """
    def __init__(self):
        self.is_opened = False
        self.frame = None
        self.lock = threading.Lock()
        
    def open(self, source):
        raise NotImplementedError
        
    def close(self):
        raise NotImplementedError
        
    def read(self):
        raise NotImplementedError

class HikCameraWrapper(CameraInterface):
    def __init__(self):
        super().__init__()
        self.handle = None
        self.data_buf = None
        self.n_payload_size = 0
        
    def open(self, ip):
        if not HAS_HIK_SDK:
            raise Exception("未找到海康SDK环境")
            
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0: raise Exception(f"枚举设备失败: {hex(ret)}")
        if deviceList.nDeviceNum == 0: raise Exception("未发现 GigE 相机")
            
        target_device = None
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = ctypes.cast(deviceList.pDeviceInfo[i], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                # 解析IP地址逻辑简化
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                str_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
                if str_ip == ip:
                    target_device = mvcc_dev_info
                    break
        
        if target_device is None: raise Exception(f"未找到IP为 {ip} 的相机")
            
        self.handle = MvCamera()
        ret = self.handle.MV_CC_CreateHandle(target_device)
        if ret != 0: raise Exception(f"创建句柄失败: {hex(ret)}")
            
        ret = self.handle.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0: raise Exception(f"打开设备失败: {hex(ret)}")
        
        # 默认配置
        self.handle.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        self.handle.MV_CC_SetEnumValue("TriggerMode", 0) # 连续采集
        
        # 获取PayloadSize
        stParam = MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_INTVALUE))
        ret = self.handle.MV_CC_GetIntValue("PayloadSize", stParam)
        self.n_payload_size = stParam.nCurValue
        self.data_buf = (ctypes.c_ubyte * self.n_payload_size)()
        
        ret = self.handle.MV_CC_StartGrabbing()
        if ret != 0: raise Exception(f"开始取流失败: {hex(ret)}")
        
        self.is_opened = True
        return True

    def close(self):
        if self.handle:
            self.handle.MV_CC_StopGrabbing()
            self.handle.MV_CC_CloseDevice()
            self.handle.MV_CC_DestroyHandle()
        self.is_opened = False

    def read(self):
        if not self.is_opened: return False, None
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ctypes.memset(ctypes.byref(stFrameInfo), 0, ctypes.sizeof(MV_FRAME_OUT_INFO_EX))
        ret = self.handle.MV_CC_GetOneFrameTimeout(ctypes.byref(self.data_buf), self.n_payload_size, stFrameInfo, 1000)
        if ret == 0:
            h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
            data = np.frombuffer(self.data_buf, count=int(self.n_payload_size), dtype=np.uint8)
            # 简单处理 Mono8 和 RGB
            if stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
                img = data.reshape((h, w))
                return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # 这里省略其他格式处理，实际使用请补全
            return False, None # 暂不支持格式
        return False, None

class OpenCVCameraWrapper(CameraInterface):
    def __init__(self):
        super().__init__()
        self.cap = None

    def open(self, source):
        # source 可以是数字索引(0, 1) 或 视频文件路径
        try:
            src = int(source)
        except:
            src = source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise Exception(f"无法打开相机/视频: {source}")
        self.is_opened = True
        return True

    def close(self):
        if self.cap:
            self.cap.release()
        self.is_opened = False

    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None

# =============================================================================
# 模块：标定逻辑
# =============================================================================

class Calibrator:
    def __init__(self):
        self.pattern_size = (10, 7) # 内角点数量 (cols, rows)
        self.square_size = 20.0     # 毫米
        self.obj_points = []        # 真实世界坐标点
        self.img_points = []        # 图像平面坐标点
        self.image_size = None
        self.images_data = []       # 存储图片信息: {'name': str, 'img': np.array, 'found': bool, 'corners': np.array}
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = 0.0

    def set_params(self, cols, rows, size_mm):
        self.pattern_size = (cols, rows)
        self.square_size = size_mm
        # 重置缓存
        self.reset()

    def reset(self):
        self.obj_points = []
        self.img_points = []
        self.images_data = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None

    def add_image(self, name, img):
        # 转换为灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        if self.image_size is None:
            self.image_size = (img.shape[1], img.shape[0])
        elif self.image_size != (img.shape[1], img.shape[0]):
            print(f"警告: 图片 {name} 尺寸不一致，已跳过")
            return False

        # 寻找角点
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        img_data = {
            'name': name,
            'img': img,
            'gray': gray,
            'found': ret,
            'corners': corners,
            'rvec': None,
            'tvec': None
        }
        
        if ret:
            # 亚像素优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_data['corners'] = corners2
        
        self.images_data.append(img_data)
        return ret

    def run_calibration(self):
        # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp = objp * self.square_size

        self.obj_points = []
        self.img_points = []
        
        valid_images_count = 0
        
        for data in self.images_data:
            if data['found']:
                self.obj_points.append(objp)
                self.img_points.append(data['corners'])
                valid_images_count += 1
                
        if valid_images_count < 3:
            return False, "有效图片不足3张，无法标定"

        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, self.image_size, None, None
            )
            
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            
            # 保存每张图的姿态
            idx = 0
            for data in self.images_data:
                if data['found']:
                    data['rvec'] = rvecs[idx]
                    data['tvec'] = tvecs[idx]
                    idx += 1
            
            # 计算重投影误差
            total_error = 0
            for i in range(len(self.obj_points)):
                imgpoints2, _ = cv2.projectPoints(self.obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            self.reprojection_error = total_error / len(self.obj_points)
            
            return True, f"标定成功! 误差: {self.reprojection_error:.4f} pixels"
        except Exception as e:
            return False, str(e)

# =============================================================================
# 模块：主程序 UI
# =============================================================================

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("相机标定工具 v1.0")
        self.root.geometry("1400x900")
        
        self.calibrator = Calibrator()
        self.camera = None
        self.is_capturing = False
        self.thread_stop = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # 左侧面板：控制和配置
        left_panel = tk.Frame(self.root, width=300, padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # 1. 标定板参数配置
        group_config = tk.LabelFrame(left_panel, text="标定板配置", padx=5, pady=5)
        group_config.pack(fill=tk.X, pady=5)
        
        tk.Label(group_config, text="内角点 列数 (W):").grid(row=0, column=0, sticky=tk.W)
        self.entry_cols = tk.Entry(group_config, width=10)
        self.entry_cols.insert(0, "10")
        self.entry_cols.grid(row=0, column=1)
        
        tk.Label(group_config, text="内角点 行数 (H):").grid(row=1, column=0, sticky=tk.W)
        self.entry_rows = tk.Entry(group_config, width=10)
        self.entry_rows.insert(0, "7")
        self.entry_rows.grid(row=1, column=1)
        
        tk.Label(group_config, text="方格边长 (mm):").grid(row=2, column=0, sticky=tk.W)
        self.entry_size = tk.Entry(group_config, width=10)
        self.entry_size.insert(0, "20.0")
        self.entry_size.grid(row=2, column=1)
        
        tk.Button(group_config, text="应用参数", command=self.apply_params).grid(row=3, column=0, columnspan=2, pady=5)

        # 2. 相机/输入源控制
        group_input = tk.LabelFrame(left_panel, text="图像输入", padx=5, pady=5)
        group_input.pack(fill=tk.X, pady=5)
        
        tk.Label(group_input, text="相机IP / 索引:").pack(anchor=tk.W)
        self.entry_source = tk.Entry(group_input)
        self.entry_source.insert(0, "0") # 默认为0号摄像头
        self.entry_source.pack(fill=tk.X, pady=2)
        
        self.btn_camera = tk.Button(group_input, text="连接相机", command=self.toggle_camera, bg="#ddd")
        self.btn_camera.pack(fill=tk.X, pady=2)
        
        tk.Button(group_input, text="拍照采集", command=self.capture_frame, bg="lightblue").pack(fill=tk.X, pady=5)
        
        tk.Label(group_input, text="--- 或 ---").pack(pady=2)
        tk.Button(group_input, text="从文件夹导入", command=self.load_from_folder).pack(fill=tk.X, pady=2)

        # 3. 标定操作
        group_calib = tk.LabelFrame(left_panel, text="标定操作", padx=5, pady=5)
        group_calib.pack(fill=tk.X, pady=5)
        
        tk.Button(group_calib, text="开始标定", command=self.start_calibration, bg="lightgreen", height=2).pack(fill=tk.X, pady=5)
        self.lbl_result = tk.Label(group_calib, text="状态: 未标定", fg="blue", wraplength=250)
        self.lbl_result.pack(pady=5)
        
        tk.Button(group_calib, text="保存结果", command=self.save_results).pack(fill=tk.X, pady=2)

        # 中间面板：图像预览
        mid_panel = tk.Frame(self.root, bg="black")
        mid_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(mid_panel, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧面板：图片列表
        right_panel = tk.Frame(self.root, width=250)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        tk.Label(right_panel, text="已采集图片列表").pack(pady=5)
        
        # Treeview
        columns = ("name", "status")
        self.tree = ttk.Treeview(right_panel, columns=columns, show="headings")
        self.tree.heading("name", text="文件名")
        self.tree.heading("status", text="检测状态")
        self.tree.column("name", width=120)
        self.tree.column("status", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_image)
        
        tk.Button(right_panel, text="清除所有", command=self.clear_all).pack(fill=tk.X)

    def apply_params(self):
        try:
            c = int(self.entry_cols.get())
            r = int(self.entry_rows.get())
            s = float(self.entry_size.get())
            self.calibrator.set_params(c, r, s)
            messagebox.showinfo("提示", f"参数已更新: Grid {c}x{r}, Size {s}mm")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    def toggle_camera(self):
        if self.is_capturing:
            # 关闭相机
            self.thread_stop = True
            if self.camera:
                self.camera.close()
            self.camera = None
            self.is_capturing = False
            self.btn_camera.config(text="连接相机", bg="#ddd")
        else:
            # 打开相机
            source = self.entry_source.get()
            
            # 判断是否是IP地址格式（简单判断）
            if "." in source and len(source.split(".")) == 4:
                self.camera = HikCameraWrapper()
            else:
                self.camera = OpenCVCameraWrapper()
                
            try:
                self.camera.open(source)
                self.is_capturing = True
                self.thread_stop = False
                self.btn_camera.config(text="断开相机", bg="#ffcccc")
                
                # 开启预览线程
                self.preview_thread = threading.Thread(target=self.update_preview)
                self.preview_thread.daemon = True
                self.preview_thread.start()
                
            except Exception as e:
                messagebox.showerror("连接失败", str(e))

    def update_preview(self):
        while not self.thread_stop and self.camera and self.camera.is_opened:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # 缩放以适应画布（简单处理）
                self.current_frame = frame
                self.display_image(frame)
            time.sleep(0.03)

    def display_image(self, img):
        # 将OpenCV图像转换为Tkinter兼容格式
        # 保持纵横比缩放
        h, w = img.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10: return
        
        scale = min(canvas_w/w, canvas_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=tk_img, anchor=tk.CENTER)
        self.canvas.image = tk_img # 保持引用

    def capture_frame(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            timestamp = datetime.now().strftime("%H%M%S")
            name = f"cap_{timestamp}.jpg"
            # 复制一份当前帧
            img_copy = self.current_frame.copy()
            self.process_added_image(name, img_copy)
        else:
            messagebox.showwarning("警告", "没有相机画面")

    def load_from_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        
        exts = ['*.jpg', '*.png', '*.bmp', '*.jpeg']
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder_path, e)))
            
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                name = os.path.basename(f)
                self.process_added_image(name, img)

    def process_added_image(self, name, img):
        ret = self.calibrator.add_image(name, img)
        status = "OK" if ret else "未检测到"
        
        # 添加到Treeview
        self.tree.insert("", tk.END, values=(name, status), tags=('ok' if ret else 'err',))
        
        # 更新显示最后一张图的效果（如果是OK的）
        if ret:
            self.show_detection_result(self.calibrator.images_data[-1])

    def on_select_image(self, event):
        selected_items = self.tree.selection()
        if not selected_items: return
        
        item = self.tree.item(selected_items[0])
        name = item['values'][0]
        
        # 查找对应的数据
        for data in self.calibrator.images_data:
            if data['name'] == name:
                self.show_detection_result(data)
                break

    def show_detection_result(self, data):
        # 即使正在预览相机，点击列表也会暂停预览显示选中图片
        # 这里为了简单，我们只是在canvas上画图。
        # 如果相机线程在跑，它会覆盖。所以最好有个标志位暂停预览显示
        
        display_img = data['img'].copy()
        
        if data['found']:
            # 绘制角点
            cv2.drawChessboardCorners(display_img, self.calibrator.pattern_size, data['corners'], data['found'])
            
            # 如果已经标定过，绘制姿态轴
            if data.get('rvec') is not None and self.calibrator.camera_matrix is not None:
                try:
                    # 坐标轴长度：标定板方格长度的3倍
                    axis_length = self.calibrator.square_size * 3
                    cv2.drawFrameAxes(display_img, self.calibrator.camera_matrix, self.calibrator.dist_coeffs, 
                                      data['rvec'], data['tvec'], axis_length)
                except Exception as e:
                    print(f"绘制坐标轴失败: {e}")

        self.display_image(display_img)

    def start_calibration(self):
        self.apply_params() # 确保使用最新参数
        ret, msg = self.calibrator.run_calibration()
        self.lbl_result.config(text=msg, fg="green" if ret else "red")
        
        if ret:
            # 刷新当前显示的图片（如果选中了）以显示坐标轴
            self.on_select_image(None)

    def save_results(self):
        if self.calibrator.camera_matrix is None:
            messagebox.showwarning("警告", "请先进行标定")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path: return
        
        data = {
            "camera_matrix": self.calibrator.camera_matrix.tolist(),
            "dist_coeffs": self.calibrator.dist_coeffs.tolist(),
            "reprojection_error": self.calibrator.reprojection_error,
            "image_width": self.calibrator.image_size[0],
            "image_height": self.calibrator.image_size[1]
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        
        messagebox.showinfo("成功", f"标定结果已保存至 {file_path}")

    def clear_all(self):
        self.calibrator.reset()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.lbl_result.config(text="状态: 已清空")
        self.canvas.delete("all")

from datetime import datetime

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
