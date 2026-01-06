import sys
import os
import time
import json
import threading
import socket
import math
import ctypes
from datetime import datetime

# GUI Imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog, filedialog
    from PIL import Image, ImageTk
except ImportError:
    print("Error: Tkinter or Pillow is not installed. Please install them using: pip install tk pillow")
    # We continue so we can show the error if possible, but likely will crash later if run.
    # But since user asked for UI, they likely have it or can install it.
    pass

import cv2
import numpy as np

# =============================================================================
# 第一部分：海康相机 SDK 驱动封装 (HikCameraWrapper)
# =============================================================================
HAS_HIK_SDK = False
try:
    # 尝试常见的 MVS 安装路径 (Windows/Linux)
    possible_paths = [
        r"C:\Program Files (x86)\MVS\Runtime\Win64_x64",
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
        r"/opt/MVS/lib/64",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            if sys.platform == 'win32':
                os.environ['PATH'] = p + ";" + os.environ['PATH']
                if hasattr(os, 'add_dll_directory'): os.add_dll_directory(p)
            break
            
    # 尝试导入 MvImport
    try:
        from MvImport.MvCameraControl_class import *
        HAS_HIK_SDK = True
    except ImportError:
        pass
except Exception:
    pass

class HikCameraWrapper:
    """海康相机 SDK 封装类"""
    def __init__(self):
        self.handle = None
        self.is_opened = False
        self.data_buf = None
        self.n_payload_size = 0
        
    def open_by_ip(self, ip):
        if not HAS_HIK_SDK: raise Exception("未检测到海康 SDK 环境，请安装 MVS 并确保 python 能够调用")
        
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, deviceList)
        if ret != 0: raise Exception(f"枚举设备失败: {hex(ret)}")
        
        target_device = None
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = ctypes.cast(deviceList.pDeviceInfo[i], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                nip = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp
                str_ip = f"{(nip >> 24) & 0xff}.{(nip >> 16) & 0xff}.{(nip >> 8) & 0xff}.{nip & 0xff}"
                if str_ip == ip:
                    target_device = mvcc_dev_info
                    break
        
        if target_device is None: raise Exception(f"未找到 IP 为 {ip} 的相机")
            
        self.handle = MvCamera()
        if self.handle.MV_CC_CreateHandle(target_device) != 0: raise Exception("创建句柄失败")
        if self.handle.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0: raise Exception("打开设备失败")
        
        # 配置
        self.handle.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        self.handle.MV_CC_SetEnumValue("TriggerMode", 0) # 连续采集
        
        stParam = MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(stParam), 0, ctypes.sizeof(MVCC_INTVALUE))
        self.handle.MV_CC_GetIntValue("PayloadSize", stParam)
        self.n_payload_size = stParam.nCurValue
        self.data_buf = (ctypes.c_ubyte * self.n_payload_size)()
        
        if self.handle.MV_CC_StartGrabbing() != 0: raise Exception("开始取流失败")
        self.is_opened = True

    def read(self):
        if not self.is_opened: return False, None
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ctypes.memset(ctypes.byref(stFrameInfo), 0, ctypes.sizeof(MV_FRAME_OUT_INFO_EX))
        ret = self.handle.MV_CC_GetOneFrameTimeout(ctypes.byref(self.data_buf), self.n_payload_size, stFrameInfo, 1000)
        if ret == 0:
            h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
            data = np.frombuffer(self.data_buf, count=int(self.n_payload_size), dtype=np.uint8)
            
            # 处理像素格式
            if stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
                img = data.reshape((h, w))
                return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
                img = data.reshape((h, w, 3))
                return True, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 默认尝试 Bayer 转 RGB
            try:
                img = data.reshape((h, w)) 
                return True, cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR) # 假设 BayerRG
            except:
                return False, None
        return False, None

    def set_exposure(self, exp_us):
        if self.is_opened:
            self.handle.MV_CC_SetEnumValue("ExposureAuto", 0)
            self.handle.MV_CC_SetFloatValue("ExposureTime", float(exp_us))

    def set_gain(self, gain_db):
        if self.is_opened:
            self.handle.MV_CC_SetEnumValue("GainAuto", 0)
            self.handle.MV_CC_SetFloatValue("Gain", float(gain_db))
            
    def release(self):
        if self.handle:
            self.handle.MV_CC_StopGrabbing()
            self.handle.MV_CC_CloseDevice()
            self.handle.MV_CC_DestroyHandle()
        self.is_opened = False

# =============================================================================
# 第二部分：核心算法库 (Algorithm)
# =============================================================================
class AlgorithmUtils:
    @staticmethod
    def euler_to_matrix(rx, ry, rz, unit='deg'):
        """欧拉角转旋转矩阵 (XYZ顺序)"""
        if unit == 'deg':
            rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
            
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        
        # R = Rz * Ry * Rx (对应固定轴XYZ旋转)
        return Rz @ Ry @ Rx

    @staticmethod
    def pose_to_homogeneous(x, y, z, rx, ry, rz, unit='deg'):
        """六自由度位姿转 4x4 齐次矩阵"""
        R = AlgorithmUtils.euler_to_matrix(rx, ry, rz, unit)
        t = np.array([[x], [y], [z]])
        RT = np.column_stack((R, t))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
        return RT

    @staticmethod
    def homogeneous_to_pose(RT):
        """4x4 矩阵转位姿 [x,y,z,rx,ry,rz] (deg)"""
        R_mat = RT[:3, :3]
        t_vec = RT[:3, 3]
        
        sy = math.sqrt(R_mat[0,0] * R_mat[0,0] +  R_mat[1,0] * R_mat[1,0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R_mat[2,1] , R_mat[2,2])
            y = math.atan2(-R_mat[2,0], sy)
            z = math.atan2(R_mat[1,0], R_mat[0,0])
        else:
            x = math.atan2(-R_mat[1,2], R_mat[1,1])
            y = math.atan2(-R_mat[2,0], sy)
            z = 0

        rx, ry, rz = math.degrees(x), math.degrees(y), math.degrees(z)
        return [t_vec[0], t_vec[1], t_vec[2], rx, ry, rz]

# =============================================================================
# 第三部分：GUI 主程序
# =============================================================================
class HandEyeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手眼标定综合工具箱 (All-in-One)")
        self.root.geometry("1300x850")
        
        # --- 全局变量与状态 ---
        self.config_file = "hand_eye_config.json"
        self.data_file = "hand_eye_data.json" # 修改文件名避免冲突
        
        # 自动加载配置和数据，如果不存在则使用默认值，不再报错
        self.cfg = self.load_config()
        self.calib_data = self.load_data()
        
        self.cam_running = False
        self.camera = None
        self.curr_frame = None
        self.lock = threading.Lock()
        
        self.tcp_running = False
        self.tcp_server = None
        self.latest_robot_pose = None # 缓存最新的机械臂坐标
        
        # --- UI 初始化 ---
        self.setup_ui()
        
        # --- 启动状态恢复 ---
        self.update_ui_from_config()

        # 确保保存目录存在
        if not os.path.exists("captured_images"):
            os.makedirs("captured_images")

    # ------------------ 配置管理 ------------------
    def load_config(self):
        default = {
            "camera": {
                "type": "opencv", # opencv 或 sdk
                "ip": "192.168.1.64",
                "index": 0,
                "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0,
                "dist": [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "board": {
                "type": "circles", # circles 或 chessboard
                "rows": 7, "cols": 7, "spacing": 15.0
            },
            "robot": {
                "port": 8000, "angle_unit": "deg"
            }
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    # 简单合并，防止key缺失
                    for k, v in default.items():
                        if k not in data: data[k] = v
                        else:
                            for sub_k, sub_v in v.items():
                                if sub_k not in data[k]: data[k][sub_k] = sub_v
                    return data
            except: 
                print("配置文件读取失败，使用默认配置")
                pass
        return default

    def save_config(self):
        self.update_config_from_ui()
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.cfg, f, indent=4)
            self.log("系统配置已保存")
        except Exception as e:
            self.log(f"配置保存失败: {e}")

    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    # 将列表还原为numpy数组以便后续处理
                    valid_data = []
                    for d in data:
                        if 'corners' in d and d['corners']:
                            d['corners'] = np.array(d['corners'], dtype=np.float32)
                        valid_data.append(d)
                    return valid_data
            except: 
                print("数据文件读取失败，初始化为空")
                pass
        return []

    def save_data(self):
        # 序列化保存（numpy转list）
        save_list = []
        for d in self.calib_data:
            item = d.copy()
            if 'corners' in item and isinstance(item['corners'], np.ndarray):
                item['corners'] = item['corners'].tolist()
            save_list.append(item)
        try:
            with open(self.data_file, 'w') as f:
                json.dump(save_list, f, indent=4)
            self.log(f"标定数据已自动保存 ({len(save_list)} 条)")
        except Exception as e:
            self.log(f"数据保存失败: {e}")

    # ------------------ UI 布局 ------------------
    def setup_ui(self):
        # 顶部：Tab 页签
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: 图像采集 & 实时监控
        self.tab_capture = tk.Frame(self.tabs)
        self.tabs.add(self.tab_capture, text="1. 图像采集 & 监控")
        self.setup_tab_capture()
        
        # Tab 2: 参数配置
        self.tab_config = tk.Frame(self.tabs)
        self.tabs.add(self.tab_config, text="2. 系统参数配置")
        self.setup_tab_config()
        
        # Tab 3: 标定计算
        self.tab_calib = tk.Frame(self.tabs)
        self.tabs.add(self.tab_calib, text="3. 标定计算 & 验证")
        self.setup_tab_calib()
        
        # 底部：日志输出
        log_frame = tk.LabelFrame(self.root, text="系统日志")
        log_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.BOTTOM)
        self.log_text = tk.Text(log_frame, height=6, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, padx=2, pady=2)

    def setup_tab_capture(self):
        # 左侧控制栏
        left_panel = tk.Frame(self.tab_capture, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 相机控制
        cam_group = tk.LabelFrame(left_panel, text="相机控制")
        cam_group.pack(fill=tk.X, pady=5)
        
        tk.Button(cam_group, text="打开相机", command=self.start_camera, bg="#e1f5fe").grid(row=0, column=0, padx=5, pady=5)
        tk.Button(cam_group, text="关闭相机", command=self.stop_camera, bg="#ffebee").grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(cam_group, text="曝光:").grid(row=1, column=0)
        self.scale_exp = tk.Scale(cam_group, from_=100, to=50000, orient=tk.HORIZONTAL, command=self.on_exp_change)
        self.scale_exp.set(5000)
        self.scale_exp.grid(row=1, column=1, sticky="ew")
        
        tk.Label(cam_group, text="增益:").grid(row=2, column=0)
        self.scale_gain = tk.Scale(cam_group, from_=0, to=20, orient=tk.HORIZONTAL, command=self.on_gain_change)
        self.scale_gain.grid(row=2, column=1, sticky="ew")

        # 机械臂通信
        robot_group = tk.LabelFrame(left_panel, text="机械臂通讯 (TCP Server)")
        robot_group.pack(fill=tk.X, pady=5)
        
        self.btn_tcp = tk.Button(robot_group, text="启动监听 (Port:8000)", command=self.toggle_tcp)
        self.btn_tcp.pack(fill=tk.X, padx=5, pady=5)
        self.lbl_tcp_status = tk.Label(robot_group, text="状态: 未启动", fg="gray")
        self.lbl_tcp_status.pack()
        tk.Label(robot_group, text="提示: 机械臂应连接此端口并发送\n x,y,z,rx,ry,rz 格式字符串", fg="blue", font=("Arial", 8)).pack()

        # 采集操作
        cap_group = tk.LabelFrame(left_panel, text="采集数据", fg="blue", font=("Arial", 10, "bold"))
        cap_group.pack(fill=tk.X, pady=10)
        
        tk.Button(cap_group, text="📸 拍照并记录坐标", command=self.capture_and_record, height=2, bg="#c8e6c9").pack(fill=tk.X, padx=5, pady=10)
        
        self.lbl_last_pose = tk.Label(cap_group, text="最新接收坐标:\n暂无", justify=tk.LEFT, bg="#f0f0f0", relief=tk.SUNKEN)
        self.lbl_last_pose.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(cap_group, text="手动输入坐标...", command=self.manual_input_pose).pack(fill=tk.X, padx=5)

        # 右侧：图像显示
        self.canvas_frame = tk.Frame(self.tab_capture, bg="black")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def setup_tab_config(self):
        f = tk.Frame(self.tab_config)
        f.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 1. 相机连接配置
        lf_cam = tk.LabelFrame(f, text="相机连接参数")
        lf_cam.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        tk.Label(lf_cam, text="驱动类型:").grid(row=0, column=0, sticky="e")
        self.var_cam_type = tk.StringVar(value="opencv")
        ttk.Combobox(lf_cam, textvariable=self.var_cam_type, values=["opencv", "sdk"]).grid(row=0, column=1)
        
        tk.Label(lf_cam, text="相机IP (SDK用):").grid(row=1, column=0, sticky="e")
        self.var_cam_ip = tk.StringVar()
        tk.Entry(lf_cam, textvariable=self.var_cam_ip).grid(row=1, column=1)
        
        tk.Label(lf_cam, text="索引/ID (OpenCV用):").grid(row=2, column=0, sticky="e")
        self.var_cam_idx = tk.IntVar()
        tk.Entry(lf_cam, textvariable=self.var_cam_idx).grid(row=2, column=1)

        # 2. 标定板参数
        lf_board = tk.LabelFrame(f, text="标定板规格")
        lf_board.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        tk.Label(lf_board, text="类型:").grid(row=0, column=0, sticky="e")
        self.var_board_type = tk.StringVar(value="circles")
        ttk.Combobox(lf_board, textvariable=self.var_board_type, values=["circles", "chessboard"]).grid(row=0, column=1)
        
        tk.Label(lf_board, text="行数 (Rows):").grid(row=1, column=0, sticky="e")
        self.var_board_rows = tk.IntVar(value=7)
        tk.Entry(lf_board, textvariable=self.var_board_rows).grid(row=1, column=1)
        
        tk.Label(lf_board, text="列数 (Cols):").grid(row=2, column=0, sticky="e")
        self.var_board_cols = tk.IntVar(value=7)
        tk.Entry(lf_board, textvariable=self.var_board_cols).grid(row=2, column=1)
        
        tk.Label(lf_board, text="间距 (mm):").grid(row=3, column=0, sticky="e")
        self.var_board_space = tk.DoubleVar(value=15.0)
        tk.Entry(lf_board, textvariable=self.var_board_space).grid(row=3, column=1)

        # 3. 相机内参
        lf_intr = tk.LabelFrame(f, text="相机内参 (Intrinsics)")
        lf_intr.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        frame_intr = tk.Frame(lf_intr)
        frame_intr.pack(padding=10)
        
        # fx, fy, cx, cy
        for i, key in enumerate(["fx", "fy", "cx", "cy"]):
            tk.Label(frame_intr, text=f"{key}:").grid(row=0, column=i*2)
            setattr(self, f"var_{key}", tk.DoubleVar())
            tk.Entry(frame_intr, textvariable=getattr(self, f"var_{key}"), width=10).grid(row=0, column=i*2+1, padx=5)
            
        tk.Label(frame_intr, text="畸变系数 (k1,k2,p1,p2,k3):").grid(row=1, column=0, columnspan=2, pady=10)
        self.var_dist = tk.StringVar()
        tk.Entry(frame_intr, textvariable=self.var_dist, width=50).grid(row=1, column=2, columnspan=6)

        # 4. 机械臂设置
        lf_robot = tk.LabelFrame(f, text="机械臂设置")
        lf_robot.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)
        
        tk.Label(lf_robot, text="TCP监听端口:").pack(side=tk.LEFT, padx=5)
        self.var_tcp_port = tk.IntVar(value=8000)
        tk.Entry(lf_robot, textvariable=self.var_tcp_port, width=8).pack(side=tk.LEFT)
        
        tk.Label(lf_robot, text="角度单位:").pack(side=tk.LEFT, padx=15)
        self.var_angle_unit = tk.StringVar(value="deg")
        ttk.Radiobutton(lf_robot, text="度 (deg)", variable=self.var_angle_unit, value="deg").pack(side=tk.LEFT)
        ttk.Radiobutton(lf_robot, text="弧度 (rad)", variable=self.var_angle_unit, value="rad").pack(side=tk.LEFT)
        
        # 底部保存按钮
        tk.Button(f, text="💾 保存所有配置", command=self.save_config, height=2, bg="#2196f3", fg="white").grid(row=3, column=0, columnspan=2, sticky="ew", pady=20)

    def setup_tab_calib(self):
        paned = tk.PanedWindow(self.tab_calib, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：数据列表
        left_f = tk.Frame(paned, width=400)
        paned.add(left_f)
        
        tk.Label(left_f, text="已采集数据列表:").pack(anchor="w")
        
        cols = ("id", "img", "robot_status")
        self.tree = ttk.Treeview(left_f, columns=cols, show="headings", height=20)
        self.tree.heading("id", text="#")
        self.tree.column("id", width=40)
        self.tree.heading("img", text="图片")
        self.tree.column("img", width=150)
        self.tree.heading("robot_status", text="坐标")
        self.tree.column("robot_status", width=100)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        
        btn_bar = tk.Frame(left_f)
        btn_bar.pack(fill=tk.X, pady=5)
        tk.Button(btn_bar, text="删除选中", command=self.delete_data).pack(side=tk.LEFT)
        tk.Button(btn_bar, text="清空所有", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_bar, text="重新识别角点", command=self.reprocess_corners).pack(side=tk.RIGHT)

        # 右侧：计算控制
        right_f = tk.Frame(paned)
        paned.add(right_f)
        
        tk.Label(right_f, text="计算流程", font=("bold", 12)).pack(pady=10)
        
        step1 = tk.LabelFrame(right_f, text="Step 1: 相机内参")
        step1.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(step1, text="若无内参，可先点击此按钮计算\n(需要采集3张以上标定板图片)", fg="gray").pack()
        tk.Button(step1, text="计算并更新相机内参", command=self.calc_intrinsics, bg="#e0f2f1").pack(fill=tk.X, padx=5, pady=5)
        
        step2 = tk.LabelFrame(right_f, text="Step 2: 手眼矩阵")
        step2.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(step2, text="计算手眼矩阵 (眼在手上)", command=self.calc_hand_eye, bg="#ffecb3", height=2).pack(fill=tk.X, padx=5, pady=5)
        
        self.result_text = tk.Text(right_f, height=15, font=("Consolas", 9))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Button(right_f, text="验证选中行数据的误差", command=self.verify_selected).pack(pady=5)

    # ------------------ UI 数据同步 ------------------
    def update_ui_from_config(self):
        c = self.cfg
        self.var_cam_type.set(c["camera"]["type"])
        self.var_cam_ip.set(c["camera"]["ip"])
        self.var_cam_idx.set(c["camera"]["index"])
        self.var_fx.set(c["camera"]["fx"])
        self.var_fy.set(c["camera"]["fy"])
        self.var_cx.set(c["camera"]["cx"])
        self.var_cy.set(c["camera"]["cy"])
        self.var_dist.set(",".join(map(str, c["camera"]["dist"])))
        
        self.var_board_type.set(c["board"]["type"])
        self.var_board_rows.set(c["board"]["rows"])
        self.var_board_cols.set(c["board"]["cols"])
        self.var_board_space.set(c["board"]["spacing"])
        
        self.var_tcp_port.set(c["robot"]["port"])
        self.var_angle_unit.set(c["robot"]["angle_unit"])
        
        self.refresh_data_list()

    def update_config_from_ui(self):
        try:
            dist_list = [float(x.strip()) for x in self.var_dist.get().split(",")]
        except:
            dist_list = [0.0]*5
            
        self.cfg = {
            "camera": {
                "type": self.var_cam_type.get(),
                "ip": self.var_cam_ip.get(),
                "index": self.var_cam_idx.get(),
                "fx": self.var_fx.get(), "fy": self.var_fy.get(),
                "cx": self.var_cx.get(), "cy": self.var_cy.get(),
                "dist": dist_list
            },
            "board": {
                "type": self.var_board_type.get(),
                "rows": self.var_board_rows.get(),
                "cols": self.var_board_cols.get(),
                "spacing": self.var_board_space.get()
            },
            "robot": {
                "port": self.var_tcp_port.get(),
                "angle_unit": self.var_angle_unit.get()
            }
        }

    def refresh_data_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, d in enumerate(self.calib_data):
            status = "√ OK" if d.get("robot_pose") else "× 缺坐标"
            self.tree.insert("", "end", values=(i+1, os.path.basename(d["img_path"]), status))

    def log(self, msg):
        t = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"[{t}] {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        print(f"[{t}] {msg}")

    # ------------------ 相机与采集逻辑 ------------------
    def start_camera(self):
        if self.cam_running: return
        self.update_config_from_ui() # 确保参数最新
        
        mode = self.var_cam_type.get()
        try:
            if mode == "sdk":
                if not HAS_HIK_SDK: raise Exception("未检测到海康SDK，请检查驱动安装")
                self.camera = HikCameraWrapper()
                self.camera.open_by_ip(self.var_cam_ip.get())
            else:
                try:
                    self.camera = cv2.VideoCapture(self.var_cam_idx.get(), cv2.CAP_DSHOW) # Win下尝试DSHOW优先
                    if not self.camera.isOpened():
                        self.camera = cv2.VideoCapture(self.var_cam_idx.get())
                except:
                    self.camera = cv2.VideoCapture(self.var_cam_idx.get())
                    
                if not self.camera.isOpened():
                    raise Exception("无法打开摄像头，请检查索引号或占用情况")
                    
            self.cam_running = True
            threading.Thread(target=self.camera_loop, daemon=True).start()
            self.log("相机已启动")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def stop_camera(self):
        self.cam_running = False
        if self.camera:
            if hasattr(self.camera, "release"): self.camera.release()
            self.camera = None
        self.log("相机已关闭")

    def camera_loop(self):
        while self.cam_running:
            ret, frame = False, None
            try:
                if hasattr(self.camera, "read"):
                    if isinstance(self.camera, cv2.VideoCapture):
                        ret, frame = self.camera.read()
                    else:
                        ret, frame = self.camera.read()
            except: pass
            
            if ret and frame is not None:
                with self.lock:
                    self.curr_frame = frame.copy()
                
                # 缩放显示
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]
                    cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
                    if cw > 10 and ch > 10:
                        scale = min(cw/w, ch/h)
                        nw, nh = int(w*scale), int(h*scale)
                        img_rgb = cv2.resize(img_rgb, (nw, nh))
                        
                    self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
                    self.root.after(0, lambda: self.canvas.create_image(cw//2, ch//2, image=self.tk_img, anchor=tk.CENTER))
                except: pass
            
            time.sleep(0.03)

    def on_exp_change(self, val):
        if self.camera and isinstance(self.camera, HikCameraWrapper):
            self.camera.set_exposure(val)
        elif self.camera and isinstance(self.camera, cv2.VideoCapture):
            self.camera.set(cv2.CAP_PROP_EXPOSURE, float(val))

    def on_gain_change(self, val):
        if self.camera and isinstance(self.camera, HikCameraWrapper):
            self.camera.set_gain(val)

    # ------------------ 机械臂通信 ------------------
    def toggle_tcp(self):
        if self.tcp_running:
            self.tcp_running = False
            if self.tcp_server: self.tcp_server.close()
            self.btn_tcp.config(text="启动监听", bg="#f0f0f0")
            self.lbl_tcp_status.config(text="状态: 已停止", fg="red")
        else:
            try:
                port = self.var_tcp_port.get()
                self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_server.bind(("0.0.0.0", port))
                self.tcp_server.listen(1)
                self.tcp_running = True
                threading.Thread(target=self.tcp_loop, daemon=True).start()
                self.btn_tcp.config(text="停止监听", bg="#ffccbc")
                self.lbl_tcp_status.config(text=f"状态: 监听中 Port {port}", fg="green")
            except Exception as e:
                messagebox.showerror("端口错误", str(e))

    def tcp_loop(self):
        while self.tcp_running:
            try:
                self.tcp_server.settimeout(1.0)
                try:
                    client, addr = self.tcp_server.accept()
                except socket.timeout:
                    continue
                
                self.log(f"机械臂连接: {addr}")
                with client:
                    while self.tcp_running:
                        data = client.recv(1024)
                        if not data: break
                        msg = data.decode('utf-8').strip()
                        # 解析: x,y,z,rx,ry,rz
                        try:
                            # 简单过滤，取出数字
                            parts = msg.replace(" ", "").split(",")
                            if len(parts) >= 6:
                                pose = [float(p) for p in parts[:6]]
                                self.latest_robot_pose = pose
                                # UI 更新
                                self.root.after(0, lambda p=pose: self.lbl_last_pose.config(
                                    text=f"最新: X={p[0]:.1f}, Y={p[1]:.1f}, Z={p[2]:.1f}\nRx={p[3]:.1f}, Ry={p[4]:.1f}, Rz={p[5]:.1f}", fg="green"
                                ))
                        except:
                            pass
            except Exception as e:
                # self.log(f"TCP Error: {e}")
                pass

    def manual_input_pose(self):
        res = tk.simpledialog.askstring("手动输入", "请输入机械臂坐标 (x,y,z,rx,ry,rz)，逗号分隔:")
        if res:
            try:
                pose = [float(x) for x in res.replace(" ", "").split(",")]
                if len(pose) == 6:
                    self.latest_robot_pose = pose
                    self.lbl_last_pose.config(text=f"手动: {pose}", fg="blue")
                else:
                    messagebox.showwarning("格式错误", "必须包含6个数字")
            except:
                messagebox.showwarning("格式错误", "解析失败，请输入数字")

    # ------------------ 采集逻辑 ------------------
    def capture_and_record(self):
        # 1. 获取图片
        with self.lock:
            frame = self.curr_frame.copy() if self.curr_frame is not None else None
        
        if frame is None:
            messagebox.showwarning("警告", "未获取到相机图像")
            return

        # 2. 获取坐标
        pose = self.latest_robot_pose
        if pose is None:
            if not messagebox.askyesno("无坐标", "未接收到机械臂坐标，是否继续采集（稍后手动录入）？"):
                return
        
        # 3. 保存
        save_dir = "captured_images"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        filename = f"{save_dir}/img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        
        # 4. 识别角点（预处理）
        corners = self.detect_corners(frame)
        
        # 5. 记录数据
        self.calib_data.append({
            "img_path": filename,
            "robot_pose": pose,
            "corners": corners
        })
        
        self.save_data()
        self.refresh_data_list()
        self.log(f"已采集: {filename}, 坐标: {pose}")
        
        # 视觉反馈
        if corners is not None:
            vis = frame.copy()
            type_ = self.var_board_type.get()
            cols = self.var_board_cols.get()
            rows = self.var_board_rows.get()
            
            if type_ == "chessboard":
                cv2.drawChessboardCorners(vis, (cols-1, rows-1), corners, True)
            else:
                cv2.drawChessboardCorners(vis, (cols, rows), corners, True)
                
            # 弹窗显示一下
            try:
                win_name = "Detected Corners"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name, vis)
                cv2.waitKey(1000)
                cv2.destroyWindow(win_name)
            except: pass

    def detect_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows = self.var_board_rows.get()
        cols = self.var_board_cols.get()
        type_ = self.var_board_type.get()
        
        corners = None
        ret = False
        
        if type_ == "chessboard":
            # 棋盘格内角点是 rows-1 * cols-1
            pattern = (cols-1, rows-1)
            ret, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        else:
            # 圆点是 rows * cols
            pattern = (cols, rows)
            ret, corners = cv2.findCirclesGrid(gray, pattern, cv2.CALIB_CB_SYMMETRIC_GRID)
            
        return corners if ret else None

    # ------------------ 标定计算逻辑 ------------------
    def reprocess_corners(self):
        """重新对所有图片进行角点检测（防止配置修改后之前的无效）"""
        count = 0
        for d in self.calib_data:
            if os.path.exists(d["img_path"]):
                img = cv2.imread(d["img_path"])
                d["corners"] = self.detect_corners(img)
                if d["corners"] is not None: count += 1
        self.save_data()
        self.log(f"重新处理完成，有效角点数据: {count}/{len(self.calib_data)}")

    def get_obj_points(self):
        """生成标定板3D点"""
        rows = self.var_board_rows.get()
        cols = self.var_board_cols.get()
        spacing = self.var_board_space.get()
        type_ = self.var_board_type.get()
        
        if type_ == "chessboard":
            # (rows-1) * (cols-1)
            objp = np.zeros(((rows-1)*(cols-1), 3), np.float32)
            objp[:,:2] = np.mgrid[0:cols-1, 0:rows-1].T.reshape(-1,2) * spacing
        else:
            objp = np.zeros((rows*cols, 3), np.float32)
            objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2) * spacing
        return objp

    def calc_intrinsics(self):
        valid_data = [d for d in self.calib_data if d["corners"] is not None]
        if len(valid_data) < 3:
            messagebox.showwarning("数据不足", "至少需要3组包含角点的数据才能进行标定")
            return
            
        objp = self.get_obj_points()
        objpoints = [objp] * len(valid_data)
        imgpoints = [d["corners"] for d in valid_data]
        
        # 获取图像尺寸
        img = cv2.imread(valid_data[0]["img_path"])
        h, w = img.shape[:2]
        
        self.log("开始计算相机内参...")
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)
            
            if ret:
                msg = f"内参标定成功! Error: {ret:.4f}\n"
                msg += f"fx={mtx[0,0]:.2f}, fy={mtx[1,1]:.2f}\n"
                msg += f"cx={mtx[0,2]:.2f}, cy={mtx[1,2]:.2f}\n"
                msg += f"Dist={dist.ravel()}"
                self.result_text.insert(tk.END, "\n=== 相机内参结果 ===\n" + msg + "\n")
                
                # 更新配置
                self.var_fx.set(mtx[0,0])
                self.var_fy.set(mtx[1,1])
                self.var_cx.set(mtx[0,2])
                self.var_cy.set(mtx[1,2])
                self.var_dist.set(",".join([f"{x:.4f}" for x in dist.ravel()]))
                self.save_config()
                messagebox.showinfo("成功", "内参已更新并保存")
            else:
                messagebox.showerror("失败", "标定计算失败")
        except Exception as e:
            messagebox.showerror("错误", f"标定过程出错: {e}")

    def calc_hand_eye(self):
        valid_data = [d for d in self.calib_data if d["corners"] is not None and d["robot_pose"] is not None]
        if len(valid_data) < 3:
            messagebox.showwarning("数据不足", "至少需要3组完整数据（含角点+机械臂坐标）")
            return
        
        # 准备数据
        objp = self.get_obj_points()
        camera_matrix = np.array([
            [self.var_fx.get(), 0, self.var_cx.get()],
            [0, self.var_fy.get(), self.var_cy.get()],
            [0, 0, 1]
        ], dtype=float)
        dist_coeffs = np.array([float(x) for x in self.var_dist.get().split(",")], dtype=float)
        
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        angle_unit = self.var_angle_unit.get() # deg or rad
        
        for d in valid_data:
            # 1. 计算标定板在相机坐标系位姿 (Target to Camera)
            ret, rvec, tvec = cv2.solvePnP(objp, d["corners"], camera_matrix, dist_coeffs)
            R_t2c, _ = cv2.Rodrigues(rvec)
            R_target2cam.append(R_t2c)
            t_target2cam.append(tvec)
            
            # 2. 计算末端在基座坐标系位姿 (Gripper to Base)
            rx, ry, rz = d["robot_pose"][3:]
            x, y, z = d["robot_pose"][:3]
            
            # 生成 T_base_to_gripper
            RT_b2g = AlgorithmUtils.pose_to_homogeneous(x, y, z, rx, ry, rz, angle_unit)
            R_b2g = RT_b2g[:3, :3]
            t_b2g = RT_b2g[:3, 3].reshape(3,1)
            
            R_gripper2base.append(R_b2g)
            t_gripper2base.append(t_b2g)
            
        self.log("开始手眼标定计算...")
        try:
            # method=cv2.CALIB_HAND_EYE_TSAI
            # OpenCV: R_cam2g, t_cam2g. 这里的g是gripper.
            # 但我们需要的是 T_cam_to_end (Camera to EndEffector)
            # 实际上，CALIB_HAND_EYE_TSAI 求解的是 AX=XB 问题
            # 输入 R_gripper2base (Base -> Gripper), R_target2cam (Cam -> Target)
            # 输出的是 T_cam_to_gripper (Eye-in-Hand)
            
            R_cam2g, t_cam2g = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base, 
                R_target2cam, t_target2cam, 
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # 结果组合
            RT_cam2g = np.column_stack((R_cam2g, t_cam2g))
            RT_cam2g = np.row_stack((RT_cam2g, np.array([0, 0, 0, 1])))
            
            # 保存结果到内存供验证
            self.last_RT_cam2g = RT_cam2g
            
            # 输出显示
            res_str = "\n=== 手眼标定结果 (Camera -> EndEffector) ===\n"
            res_str += str(RT_cam2g) + "\n"
            
            # 转换为欧拉角方便阅读
            pose_res = AlgorithmUtils.homogeneous_to_pose(RT_cam2g)
            res_str += f"\n平移(mm): X={pose_res[0]:.2f}, Y={pose_res[1]:.2f}, Z={pose_res[2]:.2f}\n"
            res_str += f"旋转(deg): Rx={pose_res[3]:.2f}, Ry={pose_res[4]:.2f}, Rz={pose_res[5]:.2f}\n"
            
            self.result_text.insert(tk.END, res_str)
            self.log("手眼标定计算完成")
            
            # 自动保存到文件
            with open("hand_eye_result.txt", "w") as f:
                f.write(res_str)
            
        except Exception as e:
            self.log(f"计算出错: {e}")
            messagebox.showerror("错误", str(e))

    def verify_selected(self):
        sel = self.tree.selection()
        if not sel: return
        idx = int(self.tree.item(sel[0], "values")[0]) - 1
        d = self.calib_data[idx]
        
        if not hasattr(self, 'last_RT_cam2g'):
            messagebox.showwarning("提示", "请先执行手眼标定计算")
            return
            
        # 1. T_cam_to_board (solvePnP)
        objp = self.get_obj_points()
        camera_matrix = np.array([[self.var_fx.get(), 0, self.var_cx.get()], [0, self.var_fy.get(), self.var_cy.get()], [0, 0, 1]], dtype=float)
        dist = np.array([float(x) for x in self.var_dist.get().split(",")], dtype=float)
        ret, rvec, tvec = cv2.solvePnP(objp, d["corners"], camera_matrix, dist)
        R_t2c, _ = cv2.Rodrigues(rvec)
        RT_t2c = np.column_stack((R_t2c, tvec))
        RT_t2c = np.row_stack((RT_t2c, [0,0,0,1]))
        
        # 2. T_end_to_base (Robot)
        p = d["robot_pose"]
        RT_e2b = AlgorithmUtils.pose_to_homogeneous(p[0], p[1], p[2], p[3], p[4], p[5], self.var_angle_unit.get())
        
        # 3. T_cam_to_end (Calibration Result)
        RT_c2e = self.last_RT_cam2g
        
        # 链式变换: Board -> Cam -> End -> Base
        # T_base_to_board = T_end_to_base * T_cam_to_end * T_target_to_cam
        
        RT_final = RT_e2b @ RT_c2e @ RT_t2c
        
        pos = RT_final[:3, 3]
        msg = f"该图验证结果 (标定板在基座系下坐标):\nX={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}"
        self.result_text.insert(tk.END, f"\n[验证 ID {idx+1}] {msg}\n")
        self.log(f"验证 ID {idx+1}: {pos}")

    # ------------------ 其他 ------------------
    def delete_data(self):
        sel = self.tree.selection()
        if not sel: return
        # 倒序删除避免索引错位
        indices = sorted([int(self.tree.item(s, "values")[0]) - 1 for s in sel], reverse=True)
        for i in indices:
            del self.calib_data[i]
        self.save_data()
        self.refresh_data_list()

    def clear_data(self):
        if messagebox.askyesno("确认", "确定清空所有数据吗？"):
            self.calib_data = []
            self.save_data()
            self.refresh_data_list()

    def on_tree_double_click(self, event):
        sel = self.tree.selection()
        if not sel: return
        idx = int(self.tree.item(sel[0], "values")[0]) - 1
        path = self.calib_data[idx]["img_path"]
        if os.path.exists(path):
            try:
                if sys.platform == 'win32': os.startfile(path)
                else: 
                    # simple viewer using cv2
                    img = cv2.imread(path)
                    cv2.imshow("Preview", cv2.resize(img, (800, 600)))
            except: pass

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = HandEyeApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        input("Press Enter to exit...")
