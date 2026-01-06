import sys
import os
import time
import json
import threading
import socket
import math
import ctypes
from datetime import datetime
import glob

# GUI Imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog, filedialog
    from PIL import Image, ImageTk
except ImportError:
    print("Error: Tkinter or Pillow is not installed. Please install them using: pip install tk pillow")
    pass

import cv2
import numpy as np

# =============================================================================
# 模块一：海康相机 SDK 驱动封装
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
            
            if stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
                img = data.reshape((h, w))
                return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
                img = data.reshape((h, w, 3))
                return True, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            try:
                img = data.reshape((h, w)) 
                return True, cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR) 
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
# 模块二：核心算法库
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
        
        # R = Rz * Ry * Rx
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
# 模块三：机械臂客户端 (请求指令)
# =============================================================================
class RobotClient:
    def __init__(self, ip, port, logger_func=print):
        self.ip = ip
        self.port = int(port)
        self.log = logger_func

    def get_pose(self, command="tcp_pose", timeout=2.0):
        """向机械臂发送指令并获取返回"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((self.ip, self.port))
            
            # 发送命令 (以\r结尾，根据import sys.py的逻辑)
            cmd_str = command + '\r' if command == 'tcp_pose' else command + '\n'
            sock.sendall(cmd_str.encode('utf-8'))
            
            # 接收响应
            response = b''
            sock.settimeout(1.0)
            try:
                while True:
                    chunk = sock.recv(1024)
                    if not chunk: break
                    response += chunk
                    if b'\n' in chunk or b'\r' in chunk: break
            except socket.timeout:
                pass
            
            sock.close()
            
            resp_str = response.decode('utf-8').strip()
            if not resp_str:
                return False, "未收到响应"
            
            return True, resp_str
            
        except Exception as e:
            return False, str(e)

    def parse_pose_string(self, resp_str):
        """解析如 'x,y,z,rx,ry,rz' 格式的字符串"""
        try:
            # 清理
            s = resp_str.strip().replace('\r', '').replace('\n', '')
            parts = s.split(',')
            # 过滤空值
            parts = [p.strip() for p in parts if p.strip()]
            
            # 如果有 OK 前缀则去掉
            if len(parts) > 0 and parts[0].upper() in ["OK", "SUCCESS"]:
                parts = parts[1:]
                
            if len(parts) >= 6:
                return [float(p) for p in parts[:6]]
            return None
        except:
            return None

# =============================================================================
# 模块四：GUI 主程序
# =============================================================================
class HandEyeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手眼标定专家版 v3.0 (集成TCP指令)")
        self.root.geometry("1300x900")
        
        # --- 全局变量 ---
        self.config_file = "hand_eye_config.json"
        self.data_file = "hand_eye_data.json"
        
        self.cfg = self.load_config()
        self.calib_data = self.load_data()
        
        self.cam_running = False
        self.camera = None
        self.curr_frame = None
        self.lock = threading.Lock()
        
        # 控制采集状态：False=实时预览(采集), True=暂停(拍照后停止)
        self.is_preview_frozen = False
        
        self.robot_client = None
        
        # --- UI 初始化 ---
        self.setup_ui()
        self.update_ui_from_config()
        
        # 确保保存目录
        if not os.path.exists("captured_images"):
            os.makedirs("captured_images")

    def load_config(self):
        default = {
            "camera": {
                "type": "opencv", "ip": "192.168.1.64", "index": 0,
                "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0,
                "dist": [0.0]*5
            },
            "board": {
                "type": "circles", "rows": 7, "cols": 7, "spacing": 15.0
            },
            "robot": {
                "ip": "192.168.1.100", "port": 8080, "angle_unit": "deg", "cmd": "tcp_pose"
            }
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    # Merge default
                    for k, v in default.items():
                        if k not in data: data[k] = v
                        else:
                            for sk, sv in v.items():
                                if sk not in data[k]: data[k][sk] = sv
                    return data
            except: pass
        return default

    def save_config(self):
        self.update_config_from_ui()
        with open(self.config_file, 'w') as f:
            json.dump(self.cfg, f, indent=4)
        self.log("配置已保存")

    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    valid = []
                    for d in data:
                        if 'corners' in d and d['corners']:
                            d['corners'] = np.array(d['corners'], dtype=np.float32)
                        valid.append(d)
                    return valid
            except: pass
        return []

    def save_data(self):
        save_list = []
        for d in self.calib_data:
            item = d.copy()
            if 'corners' in item and isinstance(item['corners'], np.ndarray):
                item['corners'] = item['corners'].tolist()
            save_list.append(item)
        with open(self.data_file, 'w') as f:
            json.dump(save_list, f, indent=4)
        self.log(f"已保存 {len(save_list)} 条数据")

    # ---------------- UI 构建 ----------------
    def setup_ui(self):
        # 顶部：连接状态栏
        top_bar = tk.Frame(self.root, bg="#eee", height=40)
        top_bar.pack(fill=tk.X)
        self.lbl_status_cam = tk.Label(top_bar, text="相机: 未连接", fg="red", bg="#eee", width=20)
        self.lbl_status_cam.pack(side=tk.LEFT, padx=5)
        self.lbl_status_robot = tk.Label(top_bar, text="机械臂: 待命", fg="orange", bg="#eee", width=20)
        self.lbl_status_robot.pack(side=tk.LEFT, padx=5)
        
        # 主体：Notebook
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: 硬件设置与测试
        self.tab_hw = tk.Frame(self.nb)
        self.nb.add(self.tab_hw, text="1. 硬件连接配置")
        self.setup_tab_hw()
        
        # Tab 2: 数据采集
        self.tab_cap = tk.Frame(self.nb)
        self.nb.add(self.tab_cap, text="2. 数据采集")
        self.setup_tab_cap()
        
        # Tab 3: 标定计算
        self.tab_cal = tk.Frame(self.nb)
        self.nb.add(self.tab_cal, text="3. 标定计算")
        self.setup_tab_cal()
        
        # 底部日志
        log_frame = tk.LabelFrame(self.root, text="操作日志")
        log_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        self.log_txt = tk.Text(log_frame, height=6, font=("Consolas", 9), state='disabled')
        self.log_txt.pack(fill=tk.BOTH)

    def log(self, msg):
        t = datetime.now().strftime("%H:%M:%S")
        self.log_txt.config(state='normal')
        self.log_txt.insert(tk.END, f"[{t}] {msg}\n")
        self.log_txt.see(tk.END)
        self.log_txt.config(state='disabled')
        print(f"[{t}] {msg}")

    # --- Tab 1: 硬件设置 ---
    def setup_tab_hw(self):
        f = tk.Frame(self.tab_hw)
        f.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 左：相机设置
        lf_cam = tk.LabelFrame(f, text="相机设置", width=400)
        lf_cam.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(lf_cam, text="驱动类型:").grid(row=0, column=0, sticky="e", pady=5)
        self.var_cam_type = tk.StringVar(value="opencv")
        ttk.Combobox(lf_cam, textvariable=self.var_cam_type, values=["opencv", "sdk"]).grid(row=0, column=1)
        
        tk.Label(lf_cam, text="海康IP:").grid(row=1, column=0, sticky="e", pady=5)
        self.var_cam_ip = tk.StringVar()
        tk.Entry(lf_cam, textvariable=self.var_cam_ip).grid(row=1, column=1)
        
        tk.Label(lf_cam, text="USB索引:").grid(row=2, column=0, sticky="e", pady=5)
        self.var_cam_idx = tk.IntVar(value=0)
        tk.Entry(lf_cam, textvariable=self.var_cam_idx).grid(row=2, column=1)
        
        tk.Button(lf_cam, text="测试打开相机", command=self.test_camera_open, bg="#e1f5fe").grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

        # 右：机械臂设置
        lf_rob = tk.LabelFrame(f, text="机械臂通讯设置 (TCP Client)", width=400)
        lf_rob.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(lf_rob, text="机械臂IP:").grid(row=0, column=0, sticky="e", pady=5)
        self.var_rob_ip = tk.StringVar(value="192.168.1.100")
        tk.Entry(lf_rob, textvariable=self.var_rob_ip).grid(row=0, column=1)
        
        tk.Label(lf_rob, text="端口:").grid(row=1, column=0, sticky="e", pady=5)
        self.var_rob_port = tk.IntVar(value=8080)
        tk.Entry(lf_rob, textvariable=self.var_rob_port).grid(row=1, column=1)
        
        tk.Label(lf_rob, text="查询指令:").grid(row=2, column=0, sticky="e", pady=5)
        self.var_rob_cmd = tk.StringVar(value="tcp_pose")
        tk.Entry(lf_rob, textvariable=self.var_rob_cmd).grid(row=2, column=1)
        
        tk.Label(lf_rob, text="角度单位:").grid(row=3, column=0, sticky="e", pady=5)
        self.var_angle_unit = tk.StringVar(value="deg")
        ttk.Combobox(lf_rob, textvariable=self.var_angle_unit, values=["deg", "rad"]).grid(row=3, column=1)
        
        tk.Button(lf_rob, text="发送指令测试连接", command=self.test_robot_conn, bg="#e8f5e9").grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")
        
        # 底部：标定板 & 内参
        lf_board = tk.LabelFrame(f, text="标定板与内参配置")
        lf_board.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        # 标定板
        f_b = tk.Frame(lf_board)
        f_b.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(f_b, text="类型:").pack(side=tk.LEFT)
        self.var_b_type = tk.StringVar(value="circles")
        ttk.Combobox(f_b, textvariable=self.var_b_type, values=["circles", "chessboard"], width=10).pack(side=tk.LEFT)
        tk.Label(f_b, text="行:").pack(side=tk.LEFT)
        self.var_b_row = tk.IntVar(value=7)
        tk.Entry(f_b, textvariable=self.var_b_row, width=5).pack(side=tk.LEFT)
        tk.Label(f_b, text="列:").pack(side=tk.LEFT)
        self.var_b_col = tk.IntVar(value=7)
        tk.Entry(f_b, textvariable=self.var_b_col, width=5).pack(side=tk.LEFT)
        tk.Label(f_b, text="间距(mm):").pack(side=tk.LEFT)
        self.var_b_space = tk.DoubleVar(value=15.0)
        tk.Entry(f_b, textvariable=self.var_b_space, width=5).pack(side=tk.LEFT)
        
        # 内参
        f_i = tk.Frame(lf_board)
        f_i.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(f_i, text="内参 fx:").pack(side=tk.LEFT)
        self.var_fx = tk.DoubleVar(value=1000)
        tk.Entry(f_i, textvariable=self.var_fx, width=8).pack(side=tk.LEFT)
        tk.Label(f_i, text="fy:").pack(side=tk.LEFT)
        self.var_fy = tk.DoubleVar(value=1000)
        tk.Entry(f_i, textvariable=self.var_fy, width=8).pack(side=tk.LEFT)
        tk.Label(f_i, text="cx:").pack(side=tk.LEFT)
        self.var_cx = tk.DoubleVar(value=640)
        tk.Entry(f_i, textvariable=self.var_cx, width=8).pack(side=tk.LEFT)
        tk.Label(f_i, text="cy:").pack(side=tk.LEFT)
        self.var_cy = tk.DoubleVar(value=360)
        tk.Entry(f_i, textvariable=self.var_cy, width=8).pack(side=tk.LEFT)
        tk.Label(f_i, text="畸变(k1,k2,p1,p2,k3):").pack(side=tk.LEFT)
        self.var_dist = tk.StringVar(value="0,0,0,0,0")
        tk.Entry(f_i, textvariable=self.var_dist, width=20).pack(side=tk.LEFT)
        
        tk.Button(f, text="保存所有配置", command=self.save_config, bg="blue", fg="white").pack(side=tk.BOTTOM, pady=10)

    # --- Tab 2: 采集 ---
    def setup_tab_cap(self):
        paned = tk.PanedWindow(self.tab_cap, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制
        f_left = tk.Frame(paned, width=300, bg="#f5f5f5")
        paned.add(f_left)
        
        # 按钮组
        btn_frame = tk.LabelFrame(f_left, text="采集控制", bg="#f5f5f5", font=("bold", 10))
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # 采集按钮 (Resume)
        self.btn_collect = tk.Button(btn_frame, text="▶ 开始采集/预览", command=self.click_collect, height=2, bg="#e1f5fe")
        self.btn_collect.pack(fill=tk.X, padx=5, pady=5)
        
        # 拍照按钮 (Freeze & Save)
        self.btn_snap = tk.Button(btn_frame, text="📸 拍照 (停止采集)", command=self.click_snap, height=2, bg="#c8e6c9")
        self.btn_snap.pack(fill=tk.X, padx=5, pady=5)
        
        # 辅助功能
        tk.Button(f_left, text="仅测试获取坐标", command=self.req_robot_pose).pack(fill=tk.X, padx=5, pady=5)
        tk.Button(f_left, text="从文件夹导入图片", command=self.import_images).pack(fill=tk.X, padx=5, pady=5)

        tk.Label(f_left, text="当前机械臂坐标:", bg="#f5f5f5").pack(anchor="w", padx=5, pady=(20,0))
        self.lbl_curr_pose = tk.Label(f_left, text="未知", fg="blue", bg="white", relief="sunken", height=2)
        self.lbl_curr_pose.pack(fill=tk.X, padx=5)

        # 右侧预览
        self.canvas_frame = tk.Frame(paned, bg="black")
        paned.add(self.canvas_frame)
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # --- Tab 3: 计算 ---
    def setup_tab_cal(self):
        paned = tk.PanedWindow(self.tab_cal, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧列表
        f_list = tk.Frame(paned, width=400)
        paned.add(f_list)
        
        cols = ("id", "img", "pose_status")
        self.tree = ttk.Treeview(f_list, columns=cols, show="headings")
        self.tree.heading("id", text="ID")
        self.tree.column("id", width=40)
        self.tree.heading("img", text="图片名")
        self.tree.column("img", width=150)
        self.tree.heading("pose_status", text="坐标状态")
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        btn_f = tk.Frame(f_list)
        btn_f.pack(fill=tk.X)
        tk.Button(btn_f, text="删除选中", command=self.del_data).pack(side=tk.LEFT)
        tk.Button(btn_f, text="清空", command=self.clear_data).pack(side=tk.LEFT)
        tk.Button(btn_f, text="补录坐标", command=self.manual_pose_entry).pack(side=tk.RIGHT)
        
        # 右侧结果
        f_res = tk.Frame(paned)
        paned.add(f_res)
        
        tk.Button(f_res, text="1. 计算相机内参 (更新配置)", command=self.run_calib_intrinsics).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(f_res, text="2. 计算手眼矩阵 (眼在手上)", command=self.run_hand_eye, bg="#ffe0b2").pack(fill=tk.X, padx=10, pady=5)
        
        self.txt_res = tk.Text(f_res, font=("Consolas", 10))
        self.txt_res.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # ---------------- 功能实现 ----------------
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
        
        self.var_b_type.set(c["board"]["type"])
        self.var_b_row.set(c["board"]["rows"])
        self.var_b_col.set(c["board"]["cols"])
        self.var_b_space.set(c["board"]["spacing"])
        
        self.var_rob_ip.set(c["robot"]["ip"])
        self.var_rob_port.set(c["robot"]["port"])
        self.var_angle_unit.set(c["robot"]["angle_unit"])
        self.var_rob_cmd.set(c["robot"]["cmd"])
        
        self.refresh_tree()

    def update_config_from_ui(self):
        try:
            d_list = [float(x) for x in self.var_dist.get().split(",")]
        except: d_list = [0.0]*5
        
        self.cfg = {
            "camera": {
                "type": self.var_cam_type.get(),
                "ip": self.var_cam_ip.get(),
                "index": self.var_cam_idx.get(),
                "fx": self.var_fx.get(), "fy": self.var_fy.get(),
                "cx": self.var_cx.get(), "cy": self.var_cy.get(),
                "dist": d_list
            },
            "board": {
                "type": self.var_b_type.get(),
                "rows": self.var_b_row.get(),
                "cols": self.var_b_col.get(),
                "spacing": self.var_b_space.get()
            },
            "robot": {
                "ip": self.var_rob_ip.get(),
                "port": self.var_rob_port.get(),
                "angle_unit": self.var_angle_unit.get(),
                "cmd": self.var_rob_cmd.get()
            }
        }

    # --- 硬件控制 ---
    def test_camera_open(self):
        self.update_config_from_ui()
        try:
            if self.cfg["camera"]["type"] == "sdk":
                cam = HikCameraWrapper()
                cam.open_by_ip(self.cfg["camera"]["ip"])
                cam.release()
            else:
                cap = cv2.VideoCapture(self.cfg["camera"]["index"])
                if not cap.isOpened(): raise Exception("无法打开OpenCV相机")
                cap.release()
            messagebox.showinfo("成功", "相机连接测试成功")
        except Exception as e:
            messagebox.showerror("失败", str(e))

    def test_robot_conn(self):
        self.update_config_from_ui()
        client = RobotClient(self.cfg["robot"]["ip"], self.cfg["robot"]["port"], self.log)
        ok, res = client.get_pose(self.cfg["robot"]["cmd"])
        if ok:
            messagebox.showinfo("成功", f"收到响应:\n{res}")
        else:
            messagebox.showerror("失败", f"通信失败: {res}")

    def init_camera(self):
        if self.cam_running: return True
        self.update_config_from_ui()
        try:
            if self.cfg["camera"]["type"] == "sdk":
                self.camera = HikCameraWrapper()
                self.camera.open_by_ip(self.cfg["camera"]["ip"])
            else:
                self.camera = cv2.VideoCapture(self.cfg["camera"]["index"])
            
            self.cam_running = True
            threading.Thread(target=self.loop_cam, daemon=True).start()
            self.lbl_status_cam.config(text="相机: 运行中", fg="green")
            return True
        except Exception as e:
            messagebox.showerror("错误", f"无法启动相机: {e}")
            return False

    def loop_cam(self):
        while self.cam_running:
            try:
                if hasattr(self.camera, "read"):
                    if isinstance(self.camera, cv2.VideoCapture):
                        ret, frame = self.camera.read()
                    else:
                        ret, frame = self.camera.read()
                    
                    if ret and frame is not None:
                        with self.lock:
                            self.curr_frame = frame.copy()
                        
                        # 仅在未冻结时刷新界面
                        if not self.is_preview_frozen:
                            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w = img_rgb.shape[:2]
                            cw = self.canvas.winfo_width()
                            ch = self.canvas.winfo_height()
                            if cw>10 and ch>10:
                                s = min(cw/w, ch/h)
                                nw, nh = int(w*s), int(h*s)
                                img_rs = cv2.resize(img_rgb, (nw, nh))
                                self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_rs))
                                self.root.after(0, lambda: self.canvas.create_image(cw//2, ch//2, image=self.tk_img, anchor=tk.CENTER))
            except: pass
            time.sleep(0.03)

    def req_robot_pose(self):
        self.update_config_from_ui()
        client = RobotClient(self.cfg["robot"]["ip"], self.cfg["robot"]["port"], self.log)
        ok, res = client.get_pose(self.cfg["robot"]["cmd"])
        if ok:
            pose = client.parse_pose_string(res)
            if pose:
                self.lbl_curr_pose.config(text=f"X:{pose[0]:.1f}, Y:{pose[1]:.1f}, Z:{pose[2]:.1f}\nRx:{pose[3]:.1f}, Ry:{pose[4]:.1f}, Rz:{pose[5]:.1f}")
                return pose
            else:
                self.log(f"坐标解析失败: {res}")
        else:
            self.log(f"获取坐标失败: {res}")
        return None

    # --- 新增：按钮逻辑 ---
    def click_collect(self):
        """点击开始/继续采集"""
        if not self.cam_running:
            if not self.init_camera(): return

        self.is_preview_frozen = False
        self.btn_collect.config(text="✔ 采集/预览进行中", bg="#81c784", state="disabled")
        self.btn_snap.config(text="📸 拍照 (停止采集)", bg="#e1f5fe", state="normal")
        self.log("恢复实时预览")

    def click_snap(self):
        """点击拍照：停止采集 -> 保存"""
        if not self.cam_running:
            messagebox.showwarning("提示", "相机未运行")
            return

        # 1. 冻结画面
        self.is_preview_frozen = True
        
        # 2. 更新按钮状态
        self.btn_collect.config(text="▶ 点击继续采集", bg="#ffcc80", state="normal")
        self.btn_snap.config(text="已拍照", bg="#eeeeee", state="disabled")
        
        # 3. 执行拍照保存逻辑
        self.capture_and_save()
        self.log("画面已定格，等待继续采集")

    def capture_and_save(self):
        with self.lock:
            frame = self.curr_frame.copy() if self.curr_frame is not None else None
        
        if frame is None:
            messagebox.showwarning("无图像", "未获取到图像")
            return
            
        # 自动尝试获取坐标
        pose = self.req_robot_pose()
        if pose is None:
            if not messagebox.askyesno("无坐标", "未获取到机械臂坐标，是否继续保存图片（后续手动录入）？"):
                return
                
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"captured_images/{ts}.jpg"
        cv2.imwrite(path, frame)
        
        # 识别角点
        corners = self.detect_corners(frame)
        
        self.calib_data.append({
            "img_path": path,
            "robot_pose": pose,
            "corners": corners
        })
        self.save_data()
        self.refresh_tree()
        
        # 视觉反馈 (在Canvas上绘制一下角点，让用户知道拍到了)
        if corners is not None:
            vis = frame.copy()
            c = self.cfg["board"]
            pattern = (c["cols"]-1, c["rows"]-1) if c["type"]=="chessboard" else (c["cols"], c["rows"])
            cv2.drawChessboardCorners(vis, pattern, corners, True)
            
            # 临时显示带角点的图到Canvas，虽然现在frozen了，但我们可以手动更新一次Canvas显示结果图
            img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw>10 and ch>10:
                s = min(cw/w, ch/h)
                nw, nh = int(w*s), int(h*s)
                img_rs = cv2.resize(img_rgb, (nw, nh))
                self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_rs))
                self.canvas.create_image(cw//2, ch//2, image=self.tk_img, anchor=tk.CENTER)

    def detect_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = self.cfg["board"]
        rows, cols = c["rows"], c["cols"]
        
        corners = None
        ret = False
        
        if c["type"] == "chessboard":
            pattern = (cols-1, rows-1)
            ret, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        else:
            pattern = (cols, rows)
            ret, corners = cv2.findCirclesGrid(gray, pattern, cv2.CALIB_CB_SYMMETRIC_GRID)
            
        return corners if ret else None

    # --- 数据管理 ---
    def refresh_tree(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for i, d in enumerate(self.calib_data):
            st = "√" if d["robot_pose"] else "×"
            self.tree.insert("", "end", values=(i+1, os.path.basename(d["img_path"]), st))

    def del_data(self):
        sel = self.tree.selection()
        if not sel: return
        indices = sorted([int(self.tree.item(s, "values")[0])-1 for s in sel], reverse=True)
        for i in indices: del self.calib_data[i]
        self.save_data()
        self.refresh_tree()

    def clear_data(self):
        self.calib_data = []
        self.save_data()
        self.refresh_tree()

    def manual_pose_entry(self):
        sel = self.tree.selection()
        if not sel: return
        idx = int(self.tree.item(sel[0], "values")[0])-1
        
        res = simpledialog.askstring("补录坐标", "请输入 x,y,z,rx,ry,rz (逗号分隔):")
        if res:
            try:
                parts = [float(x) for x in res.replace(" ", "").split(",")]
                if len(parts) == 6:
                    self.calib_data[idx]["robot_pose"] = parts
                    self.save_data()
                    self.refresh_tree()
            except: messagebox.showerror("错误", "格式不正确")

    def import_images(self):
        d = filedialog.askdirectory()
        if not d: return
        files = glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))
        cnt = 0
        for f in files:
            img = cv2.imread(f)
            if img is None: continue
            corners = self.detect_corners(img)
            self.calib_data.append({
                "img_path": f, "robot_pose": None, "corners": corners
            })
            cnt += 1
        self.save_data()
        self.refresh_tree()
        messagebox.showinfo("完成", f"已导入 {cnt} 张图片")

    # --- 标定计算 ---
    def get_obj_points(self):
        c = self.cfg["board"]
        r, cl, s = c["rows"], c["cols"], c["spacing"]
        if c["type"] == "chessboard":
            objp = np.zeros(((r-1)*(cl-1), 3), np.float32)
            objp[:,:2] = np.mgrid[0:cl-1, 0:r-1].T.reshape(-1,2) * s
        else:
            objp = np.zeros((r*cl, 3), np.float32)
            objp[:,:2] = np.mgrid[0:cl, 0:r].T.reshape(-1,2) * s
        return objp

    def run_calib_intrinsics(self):
        valid = [d for d in self.calib_data if d["corners"] is not None]
        if len(valid) < 3:
            messagebox.showwarning("数据不足", "至少需要3组有效角点数据")
            return
            
        objp = self.get_obj_points()
        objpoints = [objp]*len(valid)
        imgpoints = [d["corners"] for d in valid]
        
        img = cv2.imread(valid[0]["img_path"])
        h, w = img.shape[:2]
        
        try:
            ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)
            self.txt_res.insert(tk.END, f"\n=== 内参标定完成 (Error={ret:.4f}) ===\n")
            self.txt_res.insert(tk.END, f"fx={mtx[0,0]:.2f}, fy={mtx[1,1]:.2f}\n")
            self.txt_res.insert(tk.END, f"cx={mtx[0,2]:.2f}, cy={mtx[1,2]:.2f}\n")
            self.txt_res.insert(tk.END, f"dist={dist.ravel()}\n")
            
            # 更新UI
            self.var_fx.set(mtx[0,0])
            self.var_fy.set(mtx[1,1])
            self.var_cx.set(mtx[0,2])
            self.var_cy.set(mtx[1,2])
            self.var_dist.set(",".join([f"{x:.4f}" for x in dist.ravel()]))
            self.save_config()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run_hand_eye(self):
        valid = [d for d in self.calib_data if d["corners"] is not None and d["robot_pose"] is not None]
        if len(valid) < 3:
            messagebox.showwarning("数据不足", "至少需要3组完整数据(含坐标)")
            return
        
        objp = self.get_obj_points()
        K = np.array([[self.var_fx.get(), 0, self.var_cx.get()], [0, self.var_fy.get(), self.var_cy.get()], [0, 0, 1]])
        D = np.array([float(x) for x in self.var_dist.get().split(",")])
        
        R_g2b, t_g2b = [], []
        R_t2c, t_t2c = [], []
        
        for d in valid:
            # 1. Target to Cam
            _, rvec, tvec = cv2.solvePnP(objp, d["corners"], K, D)
            rmat, _ = cv2.Rodrigues(rvec)
            R_t2c.append(rmat)
            t_t2c.append(tvec)
            
            # 2. Gripper to Base
            p = d["robot_pose"]
            RT_b2g = AlgorithmUtils.pose_to_homogeneous(p[0], p[1], p[2], p[3], p[4], p[5], self.cfg["robot"]["angle_unit"])
            R_g2b.append(RT_b2g[:3, :3])
            t_g2b.append(RT_b2g[:3, 3].reshape(3,1))
            
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
            
            RT = np.eye(4)
            RT[:3, :3] = R_c2g
            RT[:3, 3] = t_c2g.flatten()
            
            self.txt_res.insert(tk.END, f"\n=== 手眼标定结果 (Cam -> End) ===\n")
            self.txt_res.insert(tk.END, str(RT) + "\n")
            
            # 存文件
            with open("hand_eye_result.txt", "w") as f: f.write(str(RT))
            
            # 验证
            self.txt_res.insert(tk.END, "--- 验证结果 (标定板在基座下坐标一致性) ---\n")
            for i, d in enumerate(valid):
                RT_t2c_mat = np.eye(4)
                RT_t2c_mat[:3,:3] = R_t2c[i]
                RT_t2c_mat[:3,3] = t_t2c[i].flatten()
                
                RT_g2b_mat = np.eye(4)
                RT_g2b_mat[:3,:3] = R_g2b[i]
                RT_g2b_mat[:3,3] = t_g2b[i].flatten()
                
                RT_final = RT_g2b_mat @ RT @ RT_t2c_mat
                pos = RT_final[:3, 3]
                self.txt_res.insert(tk.END, f"Img {i+1}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}\n")
                
        except Exception as e:
            self.log(f"计算失败: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandEyeApp(root)
    root.mainloop()
