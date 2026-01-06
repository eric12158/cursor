#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
手眼标定程序 (Hand-Eye Calibration) - GUI版

功能说明:
    1. 支持JSON配置文件输入相机内参和畸变系数
    2. 支持棋盘格(chessboard)和圆点(circles)两种标定板类型
    3. 从指定目录加载图片，自动检测标定板
    4. 支持通过TCP发送tcp_pose命令获取机械臂位姿
    5. 使用OpenCV calibrateHandEye进行手眼标定
    6. 输出完整的手眼标定矩阵和验证结果

使用方法:
    1. 运行程序：python hand_eye_calibrate.py
    2. 在配置页面设置相机内参、机械臂IP/端口、标定板参数
    3. 点击"从文件夹加载图片"加载标定图片
    4. 选中每行数据，点击"获取机械臂位姿"发送tcp_pose命令
    5. 所有数据就绪后，点击"计算手眼标定"

作者: 基于原始MATLAB标定版本重构
版本: 3.0 (GUI版 + TCP位姿获取)
日期: 2024
'''

import os
import sys
import json
import glob
import socket
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from math import sin, cos, pi

# =============================================================================
# 配置文件管理
# =============================================================================

DEFAULT_CONFIG = {
    "camera_intrinsics": {
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 640.0,
        "cy": 480.0
    },
    "distortion_coeffs": {
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0
    },
    "calibration_board": {
        "type": "chessboard",
        "rows": 9,
        "cols": 6,
        "spacing": 25.0
    },
    "robot_net": {
        "robot_ip": "192.168.1.100",
        "robot_port": 8080,
        "timeout": 2.0
    },
    "paths": {
        "image_dir": "./calibration_images",
        "output_dir": "./calibration_results"
    },
    "robot_config": {
        "euler_order": "xyz",
        "angle_unit": "degrees",
        "pose_format": "x,y,z,rx,ry,rz"
    },
    "hand_eye_method": "TSAI"
}

CONFIG_FILE = "calibration_config.json"
CALIB_DATA_FILE = "calibration_data.json"


def load_config(config_path=CONFIG_FILE):
    """加载配置文件"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            config = merge_config(DEFAULT_CONFIG, user_config)
            return config
        except Exception as e:
            print(f"[警告] 加载配置失败: {e}")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def merge_config(default, user):
    """递归合并配置"""
    result = default.copy()
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config, config_path=CONFIG_FILE):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


# =============================================================================
# 标定板检测
# =============================================================================

def detect_calibration_board(image, config):
    """检测标定板角点/圆心"""
    board_config = config["calibration_board"]
    board_type = board_config["type"]
    rows = board_config["rows"]
    cols = board_config["cols"]
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if board_type == "chessboard":
        pattern_size = (cols, rows)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        success, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        
        if success:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
    elif board_type == "circles":
        pattern_size = (cols, rows)
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 50000
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.87
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        detector = cv2.SimpleBlobDetector_create(params)
        
        success, corners = cv2.findCirclesGrid(
            gray, pattern_size, None,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=detector
        )
        
        if not success:
            success, corners = cv2.findCirclesGrid(
                gray, pattern_size, None,
                flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                blobDetector=detector
            )
    else:
        return False, None
    
    return success, corners


def get_object_points(config):
    """生成标定板3D坐标点"""
    board_config = config["calibration_board"]
    rows = board_config["rows"]
    cols = board_config["cols"]
    spacing = board_config["spacing"]
    
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * spacing
    
    return objp


# =============================================================================
# 欧拉角转换
# =============================================================================

def euler_to_rotation_matrix(rx, ry, rz, order='xyz'):
    """欧拉角转旋转矩阵"""
    Rx = np.array([
        [1, 0, 0],
        [0, cos(rx), -sin(rx)],
        [0, sin(rx), cos(rx)]
    ])
    
    Ry = np.array([
        [cos(ry), 0, sin(ry)],
        [0, 1, 0],
        [-sin(ry), 0, cos(ry)]
    ])
    
    Rz = np.array([
        [cos(rz), -sin(rz), 0],
        [sin(rz), cos(rz), 0],
        [0, 0, 1]
    ])
    
    rotations = {'x': Rx, 'y': Ry, 'z': Rz}
    order = order.lower()
    
    if len(order) == 3:
        R = rotations[order[2]] @ rotations[order[1]] @ rotations[order[0]]
    else:
        R = Rz @ Ry @ Rx
    
    return R


def pose_to_transform_matrix(x, y, z, rx, ry, rz, config):
    """将机械臂位姿转换为4x4齐次变换矩阵"""
    robot_config = config["robot_config"]
    euler_order = robot_config.get("euler_order", "xyz")
    angle_unit = robot_config.get("angle_unit", "degrees")
    
    if angle_unit == "degrees":
        rx = rx * pi / 180
        ry = ry * pi / 180
        rz = rz * pi / 180
    
    R = euler_to_rotation_matrix(rx, ry, rz, euler_order)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


# =============================================================================
# 手眼标定方法
# =============================================================================

def get_hand_eye_method(method_name):
    """获取OpenCV手眼标定方法常量"""
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS
    }
    return methods.get(method_name.upper(), cv2.CALIB_HAND_EYE_TSAI)


# =============================================================================
# GUI主程序
# =============================================================================

class HandEyeCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手眼标定程序 v3.0 - GUI版")
        self.root.geometry("1200x800")
        
        self.config = load_config()
        self.calib_data_list = []
        
        self.load_calib_data()
        self.setup_ui()
        
    def load_calib_data(self):
        """加载标定数据"""
        if os.path.exists(CALIB_DATA_FILE):
            try:
                with open(CALIB_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for d in data:
                        if 'corners' in d and d['corners'] is not None:
                            d['corners'] = np.array(d['corners'], dtype=np.float32)
                    self.calib_data_list = data
                    print(f"[系统] 已恢复 {len(data)} 条标定数据")
            except Exception as e:
                print(f"[警告] 加载标定数据失败: {e}")
    
    def save_calib_data(self, show_message=True):
        """保存标定数据"""
        serializable_list = []
        for d in self.calib_data_list:
            item = d.copy()
            if 'corners' in item and item['corners'] is not None:
                item['corners'] = item['corners'].tolist()
            serializable_list.append(item)
        
        try:
            with open(CALIB_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(serializable_list, f, indent=4, ensure_ascii=False)
            if show_message:
                self.log(f"已保存 {len(self.calib_data_list)} 条标定数据")
        except Exception as e:
            self.log(f"[错误] 保存标定数据失败: {e}")
    
    def setup_ui(self):
        """设置UI界面"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 标定数据页面
        frame_calib = tk.Frame(notebook)
        notebook.add(frame_calib, text="1. 标定数据采集")
        self.setup_calib_ui(frame_calib)
        
        # 配置页面
        frame_config = tk.Frame(notebook)
        notebook.add(frame_config, text="2. 系统配置")
        self.setup_config_ui(frame_config)
        
        # 结果页面
        frame_result = tk.Frame(notebook)
        notebook.add(frame_result, text="3. 标定结果")
        self.setup_result_ui(frame_result)
    
    def setup_calib_ui(self, parent):
        """设置标定数据采集界面"""
        # 顶部按钮区
        top_frame = tk.Frame(parent, pady=10)
        top_frame.pack(fill=tk.X, padx=10)
        
        tk.Button(top_frame, text="📁 从文件夹加载图片", command=self.load_images_from_folder,
                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="📡 获取选中行的机械臂位姿", command=self.get_robot_pose_for_selected,
                  bg="#2196F3", fg="white", font=("Arial", 10, "bold"), width=25).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="✎ 手动输入位姿", command=self.manual_input_pose,
                  bg="#FF9800", fg="white", font=("Arial", 10), width=15).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="🗑️ 删除选中行", command=self.delete_selected_row,
                  bg="#f44336", fg="white", font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="💾 保存数据", command=lambda: self.save_calib_data(True),
                  bg="#9C27B0", fg="white", font=("Arial", 10), width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="▶ 计算手眼标定", command=self.run_hand_eye_calibration,
                  bg="#FF5722", fg="white", font=("Arial", 11, "bold"), width=15).pack(side=tk.RIGHT, padx=5)
        
        # 状态显示
        status_frame = tk.Frame(parent)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.lbl_pose_status = tk.Label(status_frame, text="", fg="blue", font=("Arial", 10))
        self.lbl_pose_status.pack(side=tk.LEFT)
        
        # 数据表格
        columns = ("id", "img", "x", "y", "z", "rx", "ry", "rz", "status")
        self.tree_calib = ttk.Treeview(parent, columns=columns, show="headings", height=15)
        
        self.tree_calib.heading("id", text="ID")
        self.tree_calib.column("id", width=40, anchor="center")
        self.tree_calib.heading("img", text="图片文件名")
        self.tree_calib.column("img", width=200)
        
        for c in ["x", "y", "z", "rx", "ry", "rz"]:
            self.tree_calib.heading(c, text=c.upper())
            self.tree_calib.column(c, width=80, anchor="center")
        
        self.tree_calib.heading("status", text="状态")
        self.tree_calib.column("status", width=80, anchor="center")
        
        # 滚动条
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.tree_calib.yview)
        self.tree_calib.configure(yscrollcommand=scrollbar.set)
        
        self.tree_calib.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, side=tk.LEFT)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT, pady=5)
        
        self.tree_calib.bind("<Double-1>", self.view_calib_image)
        
        # 日志区域
        log_frame = tk.LabelFrame(parent, text="日志", font=("Arial", 10, "bold"))
        log_frame.pack(fill=tk.X, padx=10, pady=5, side=tk.BOTTOM)
        
        self.txt_log = tk.Text(log_frame, height=6, font=("Consolas", 9))
        self.txt_log.pack(fill=tk.X, padx=5, pady=5)
        
        # 刷新列表
        self.refresh_calib_list()
    
    def setup_config_ui(self, parent):
        """设置配置界面"""
        # 相机内参
        cam_frame = tk.LabelFrame(parent, text="相机内参", font=("Arial", 10, "bold"))
        cam_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = tk.Frame(cam_frame)
        row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(row, text="fx:", width=5).pack(side=tk.LEFT)
        self.var_fx = tk.StringVar(value=str(self.config["camera_intrinsics"]["fx"]))
        tk.Entry(row, textvariable=self.var_fx, width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="fy:", width=5).pack(side=tk.LEFT)
        self.var_fy = tk.StringVar(value=str(self.config["camera_intrinsics"]["fy"]))
        tk.Entry(row, textvariable=self.var_fy, width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="cx:", width=5).pack(side=tk.LEFT)
        self.var_cx = tk.StringVar(value=str(self.config["camera_intrinsics"]["cx"]))
        tk.Entry(row, textvariable=self.var_cx, width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="cy:", width=5).pack(side=tk.LEFT)
        self.var_cy = tk.StringVar(value=str(self.config["camera_intrinsics"]["cy"]))
        tk.Entry(row, textvariable=self.var_cy, width=12).pack(side=tk.LEFT, padx=5)
        
        # 畸变系数
        dist_frame = tk.LabelFrame(parent, text="畸变系数", font=("Arial", 10, "bold"))
        dist_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = tk.Frame(dist_frame)
        row.pack(fill=tk.X, padx=10, pady=5)
        
        dist_labels = ["k1", "k2", "p1", "p2", "k3"]
        self.var_dist = {}
        for lbl in dist_labels:
            tk.Label(row, text=f"{lbl}:", width=4).pack(side=tk.LEFT)
            self.var_dist[lbl] = tk.StringVar(value=str(self.config["distortion_coeffs"][lbl]))
            tk.Entry(row, textvariable=self.var_dist[lbl], width=12).pack(side=tk.LEFT, padx=3)
        
        # 机械臂通信
        robot_frame = tk.LabelFrame(parent, text="机械臂TCP通信", font=("Arial", 10, "bold"))
        robot_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = tk.Frame(robot_frame)
        row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(row, text="机械臂IP:", width=10).pack(side=tk.LEFT)
        self.var_robot_ip = tk.StringVar(value=self.config["robot_net"]["robot_ip"])
        tk.Entry(row, textvariable=self.var_robot_ip, width=15).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="端口:", width=5).pack(side=tk.LEFT)
        self.var_robot_port = tk.StringVar(value=str(self.config["robot_net"]["robot_port"]))
        tk.Entry(row, textvariable=self.var_robot_port, width=8).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="超时(秒):", width=8).pack(side=tk.LEFT)
        self.var_robot_timeout = tk.StringVar(value=str(self.config["robot_net"]["timeout"]))
        tk.Entry(row, textvariable=self.var_robot_timeout, width=6).pack(side=tk.LEFT, padx=5)
        
        tk.Button(row, text="测试连接", command=self.test_robot_connection,
                  bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=20)
        
        # 标定板参数
        board_frame = tk.LabelFrame(parent, text="标定板参数", font=("Arial", 10, "bold"))
        board_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = tk.Frame(board_frame)
        row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(row, text="类型:", width=5).pack(side=tk.LEFT)
        self.var_board_type = tk.StringVar(value=self.config["calibration_board"]["type"])
        ttk.Combobox(row, textvariable=self.var_board_type, values=["chessboard", "circles"], 
                     width=12, state="readonly").pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="行数:", width=5).pack(side=tk.LEFT)
        self.var_board_rows = tk.StringVar(value=str(self.config["calibration_board"]["rows"]))
        tk.Entry(row, textvariable=self.var_board_rows, width=6).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="列数:", width=5).pack(side=tk.LEFT)
        self.var_board_cols = tk.StringVar(value=str(self.config["calibration_board"]["cols"]))
        tk.Entry(row, textvariable=self.var_board_cols, width=6).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="间距(mm):", width=9).pack(side=tk.LEFT)
        self.var_board_spacing = tk.StringVar(value=str(self.config["calibration_board"]["spacing"]))
        tk.Entry(row, textvariable=self.var_board_spacing, width=8).pack(side=tk.LEFT, padx=5)
        
        # 机械臂配置
        robot_cfg_frame = tk.LabelFrame(parent, text="机械臂位姿配置", font=("Arial", 10, "bold"))
        robot_cfg_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = tk.Frame(robot_cfg_frame)
        row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(row, text="欧拉角顺序:", width=10).pack(side=tk.LEFT)
        self.var_euler_order = tk.StringVar(value=self.config["robot_config"]["euler_order"])
        ttk.Combobox(row, textvariable=self.var_euler_order, 
                     values=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"], 
                     width=8, state="readonly").pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="角度单位:", width=8).pack(side=tk.LEFT)
        self.var_angle_unit = tk.StringVar(value=self.config["robot_config"]["angle_unit"])
        ttk.Combobox(row, textvariable=self.var_angle_unit, 
                     values=["degrees", "radians"], 
                     width=10, state="readonly").pack(side=tk.LEFT, padx=5)
        
        tk.Label(row, text="标定方法:", width=8).pack(side=tk.LEFT)
        self.var_method = tk.StringVar(value=self.config["hand_eye_method"])
        ttk.Combobox(row, textvariable=self.var_method, 
                     values=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"], 
                     width=12, state="readonly").pack(side=tk.LEFT, padx=5)
        
        # 保存按钮
        tk.Button(parent, text="💾 保存配置", command=self.save_config_from_ui,
                  bg="#2196F3", fg="white", font=("Arial", 11, "bold"), 
                  width=20, height=2).pack(pady=20)
    
    def setup_result_ui(self, parent):
        """设置结果显示界面"""
        self.txt_result = tk.Text(parent, font=("Consolas", 10))
        self.txt_result.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.txt_result.insert(tk.END, "手眼标定结果将在此显示...\n\n")
        self.txt_result.insert(tk.END, "使用步骤:\n")
        self.txt_result.insert(tk.END, "1. 在配置页面设置相机内参和机械臂通信参数\n")
        self.txt_result.insert(tk.END, "2. 点击'从文件夹加载图片'加载标定图片\n")
        self.txt_result.insert(tk.END, "3. 选中每行数据，点击'获取机械臂位姿'发送tcp_pose命令\n")
        self.txt_result.insert(tk.END, "4. 确保所有数据的位姿已获取（状态显示'已就绪'）\n")
        self.txt_result.insert(tk.END, "5. 点击'计算手眼标定'执行标定\n")
    
    def log(self, msg):
        """输出日志"""
        self.txt_log.insert(tk.END, f"{msg}\n")
        self.txt_log.see(tk.END)
        print(msg)
    
    def refresh_calib_list(self):
        """刷新标定数据列表"""
        for item in self.tree_calib.get_children():
            self.tree_calib.delete(item)
        
        for d in self.calib_data_list:
            pose = d.get("robot_pose")
            status = "已就绪" if pose else "待获取"
            
            vals = [d["id"], os.path.basename(d["img_path"])]
            if pose:
                vals.extend([f"{x:.2f}" for x in pose])
            else:
                vals.extend(["-"] * 6)
            vals.append(status)
            
            self.tree_calib.insert("", "end", values=vals)
    
    def update_config_from_ui(self):
        """从UI更新配置"""
        try:
            self.config["camera_intrinsics"]["fx"] = float(self.var_fx.get())
            self.config["camera_intrinsics"]["fy"] = float(self.var_fy.get())
            self.config["camera_intrinsics"]["cx"] = float(self.var_cx.get())
            self.config["camera_intrinsics"]["cy"] = float(self.var_cy.get())
            
            for key in self.var_dist:
                self.config["distortion_coeffs"][key] = float(self.var_dist[key].get())
            
            self.config["robot_net"]["robot_ip"] = self.var_robot_ip.get()
            self.config["robot_net"]["robot_port"] = int(self.var_robot_port.get())
            self.config["robot_net"]["timeout"] = float(self.var_robot_timeout.get())
            
            self.config["calibration_board"]["type"] = self.var_board_type.get()
            self.config["calibration_board"]["rows"] = int(self.var_board_rows.get())
            self.config["calibration_board"]["cols"] = int(self.var_board_cols.get())
            self.config["calibration_board"]["spacing"] = float(self.var_board_spacing.get())
            
            self.config["robot_config"]["euler_order"] = self.var_euler_order.get()
            self.config["robot_config"]["angle_unit"] = self.var_angle_unit.get()
            self.config["hand_eye_method"] = self.var_method.get()
            
        except ValueError as e:
            messagebox.showerror("配置错误", f"请输入有效的数值: {e}")
            return False
        return True
    
    def save_config_from_ui(self):
        """保存配置"""
        if self.update_config_from_ui():
            save_config(self.config)
            messagebox.showinfo("成功", "配置已保存")
            self.log("配置已保存")
    
    def load_images_from_folder(self):
        """从文件夹加载标定图片"""
        folder_path = filedialog.askdirectory(title="选择包含标定图片的文件夹")
        if not folder_path:
            return
        
        self.update_config_from_ui()
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))
        
        if not image_files:
            messagebox.showwarning("警告", "所选文件夹中没有找到图片文件")
            return
        
        image_files.sort()
        self.log(f"从文件夹加载: {len(image_files)} 张图片")
        
        success_count = 0
        fail_count = 0
        
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.log(f"[跳过] 无法读取: {os.path.basename(img_path)}")
                    fail_count += 1
                    continue
                
                success, corners = detect_calibration_board(img, self.config)
                
                if success:
                    self.calib_data_list.append({
                        "id": len(self.calib_data_list) + 1,
                        "img_path": img_path,
                        "corners": corners,
                        "robot_pose": None
                    })
                    success_count += 1
                    self.log(f"[成功] {os.path.basename(img_path)}: 检测到标定板")
                else:
                    self.log(f"[失败] {os.path.basename(img_path)}: 未检测到标定板")
                    fail_count += 1
                    
            except Exception as e:
                self.log(f"[错误] {os.path.basename(img_path)}: {e}")
                fail_count += 1
        
        self.refresh_calib_list()
        self.save_calib_data(show_message=False)
        
        messagebox.showinfo("加载完成", 
            f"成功: {success_count} 张\n失败: {fail_count} 张\n总计: {len(image_files)} 张")
    
    def send_command_to_robot(self, command, timeout=2.0):
        """向机械臂发送TCP命令并接收响应"""
        self.update_config_from_ui()
        robot_ip = self.config["robot_net"]["robot_ip"]
        robot_port = self.config["robot_net"]["robot_port"]
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            sock.connect((robot_ip, robot_port))
            self.log(f"已连接到机械臂 {robot_ip}:{robot_port}")
            
            # 发送命令（使用回车符）
            cmd = command + '\r'
            sock.sendall(cmd.encode('utf-8'))
            self.log(f"已发送命令: {command}")
            
            # 接收响应
            response = b''
            sock.settimeout(1.0)
            try:
                while True:
                    chunk = sock.recv(1024)
                    if not chunk:
                        break
                    response += chunk
                    if b'\n' in chunk or b'\r' in chunk:
                        break
            except socket.timeout:
                pass
            
            response_str = response.decode('utf-8').strip()
            if response_str:
                self.log(f"收到响应: {response_str}")
            
            sock.close()
            return True, response_str if response_str else "未收到响应"
            
        except socket.timeout:
            self.log(f"[错误] 连接超时")
            return False, "连接超时"
        except ConnectionRefusedError:
            self.log(f"[错误] 连接被拒绝")
            return False, "连接被拒绝"
        except Exception as e:
            self.log(f"[错误] 连接失败: {e}")
            return False, str(e)
    
    def parse_robot_pose(self, response_str):
        """解析机械臂返回的坐标数据"""
        try:
            response_str = response_str.strip().replace('\r', '').replace('\n', '')
            
            if "," in response_str:
                parts = response_str.split(",")
                parts = [p.strip() for p in parts if p.strip()]
                
                # 跳过前缀
                if len(parts) > 0 and parts[0].upper() in ["OK", "SUCCESS"]:
                    parts = parts[1:]
                
                if len(parts) >= 6:
                    pose = [float(p) for p in parts[:6]]
                    self.log(f"解析成功: X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}, "
                            f"Rx={pose[3]:.2f}, Ry={pose[4]:.2f}, Rz={pose[5]:.2f}")
                    return True, pose
                else:
                    self.log(f"[错误] 数据不足6个: {len(parts)}个")
                    return False, None
            else:
                self.log(f"[错误] 响应格式错误（无逗号分隔）")
                return False, None
                
        except ValueError as e:
            self.log(f"[错误] 解析失败: {e}")
            return False, None
    
    def get_robot_pose_for_selected(self):
        """获取选中行的机械臂位姿"""
        sel = self.tree_calib.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中一行数据")
            return
        
        idx = self.tree_calib.index(sel[0])
        if idx >= len(self.calib_data_list):
            messagebox.showerror("错误", "数据索引错误")
            return
        
        self.lbl_pose_status.config(text="正在获取坐标...", fg="blue")
        self.root.update()
        
        # 发送tcp_pose命令
        timeout = self.config["robot_net"].get("timeout", 2.0)
        success, response = self.send_command_to_robot("tcp_pose", timeout)
        
        if success:
            parse_ok, pose = self.parse_robot_pose(response)
            if parse_ok:
                self.calib_data_list[idx]["robot_pose"] = pose
                self.refresh_calib_list()
                self.save_calib_data(show_message=False)
                self.lbl_pose_status.config(text=f"✓ 已获取位姿", fg="green")
                messagebox.showinfo("成功", 
                    f"已成功获取位姿:\n"
                    f"X: {pose[0]:.2f} mm\n"
                    f"Y: {pose[1]:.2f} mm\n"
                    f"Z: {pose[2]:.2f} mm\n"
                    f"Rx: {pose[3]:.2f}°\n"
                    f"Ry: {pose[4]:.2f}°\n"
                    f"Rz: {pose[5]:.2f}°")
            else:
                self.lbl_pose_status.config(text="✗ 解析失败", fg="red")
                messagebox.showerror("错误", f"无法解析返回数据:\n{response}")
        else:
            self.lbl_pose_status.config(text="✗ 连接失败", fg="red")
            messagebox.showerror("错误", f"无法连接机械臂:\n{response}")
    
    def manual_input_pose(self):
        """手动输入机械臂位姿"""
        sel = self.tree_calib.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中一行数据")
            return
        
        idx = self.tree_calib.index(sel[0])
        if idx >= len(self.calib_data_list):
            messagebox.showerror("错误", "数据索引错误")
            return
        
        # 创建输入对话框
        win = tk.Toplevel(self.root)
        win.title("手动输入机械臂位姿")
        win.geometry("600x200")
        win.transient(self.root)
        win.grab_set()
        
        tk.Label(win, text="请输入机械臂当前位姿", font=("Arial", 11, "bold")).pack(pady=10)
        
        f_main = tk.Frame(win)
        f_main.pack(pady=10)
        
        entries = []
        labels = ["X (mm)", "Y (mm)", "Z (mm)", "Rx (°)", "Ry (°)", "Rz (°)"]
        
        # 获取现有位姿作为默认值
        existing_pose = self.calib_data_list[idx].get("robot_pose")
        
        for i, lbl in enumerate(labels):
            f = tk.Frame(f_main)
            f.grid(row=0, column=i, padx=8)
            tk.Label(f, text=lbl, font=("Arial", 9)).pack()
            e = tk.Entry(f, width=10, font=("Arial", 10))
            e.pack()
            if existing_pose:
                e.insert(0, str(existing_pose[i]))
            entries.append(e)
        
        def confirm():
            try:
                vals = [float(e.get()) for e in entries]
                self.calib_data_list[idx]["robot_pose"] = vals
                self.refresh_calib_list()
                self.save_calib_data(show_message=False)
                self.log(f"已手动输入位姿: X={vals[0]:.2f}, Y={vals[1]:.2f}, Z={vals[2]:.2f}")
                win.destroy()
                messagebox.showinfo("成功", "位姿已保存")
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字")
        
        f_btn = tk.Frame(win)
        f_btn.pack(pady=15)
        tk.Button(f_btn, text="确认保存", command=confirm, bg="#4CAF50", fg="white", 
                  width=12, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(f_btn, text="取消", command=win.destroy, bg="#f44336", fg="white", 
                  width=12, height=2).pack(side=tk.LEFT, padx=10)
    
    def delete_selected_row(self):
        """删除选中行"""
        sel = self.tree_calib.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的行")
            return
        
        idx = self.tree_calib.index(sel[0])
        if idx >= len(self.calib_data_list):
            return
        
        data = self.calib_data_list[idx]
        img_name = os.path.basename(data.get("img_path", ""))
        
        if not messagebox.askyesno("确认删除", f"确定要删除第 {idx+1} 行数据吗？\n\n图片: {img_name}"):
            return
        
        del self.calib_data_list[idx]
        
        # 重新编号
        for i, d in enumerate(self.calib_data_list):
            d["id"] = i + 1
        
        self.refresh_calib_list()
        self.save_calib_data(show_message=False)
        self.log(f"已删除第 {idx+1} 行数据")
    
    def view_calib_image(self, event):
        """双击查看标定图片"""
        sel = self.tree_calib.selection()
        if not sel:
            return
        
        idx = self.tree_calib.index(sel[0])
        if idx >= len(self.calib_data_list):
            return
        
        data = self.calib_data_list[idx]
        img_path = data["img_path"]
        
        if not os.path.exists(img_path):
            messagebox.showerror("错误", f"图片不存在: {img_path}")
            return
        
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("错误", f"无法读取图片: {img_path}")
            return
        
        # 绘制检测结果
        corners = data.get("corners")
        if corners is not None:
            board_type = self.config["calibration_board"]["type"]
            rows = self.config["calibration_board"]["rows"]
            cols = self.config["calibration_board"]["cols"]
            
            if board_type == "chessboard":
                cv2.drawChessboardCorners(img, (cols, rows), corners, True)
            else:
                cv2.drawChessboardCorners(img, (cols, rows), corners, True)
        
        # 缩放显示
        h, w = img.shape[:2]
        max_size = 800
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        cv2.imshow(f"标定图片 - {os.path.basename(img_path)}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def test_robot_connection(self):
        """测试机械臂连接"""
        self.update_config_from_ui()
        robot_ip = self.config["robot_net"]["robot_ip"]
        robot_port = self.config["robot_net"]["robot_port"]
        timeout = self.config["robot_net"]["timeout"]
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((robot_ip, robot_port))
            sock.close()
            messagebox.showinfo("成功", f"成功连接到机械臂\n{robot_ip}:{robot_port}")
            self.log(f"测试连接成功: {robot_ip}:{robot_port}")
        except Exception as e:
            messagebox.showerror("连接失败", f"无法连接到机械臂\n{robot_ip}:{robot_port}\n\n错误: {e}")
            self.log(f"测试连接失败: {e}")
    
    def run_hand_eye_calibration(self):
        """执行手眼标定"""
        self.update_config_from_ui()
        
        # 获取有效数据
        valid_data = [d for d in self.calib_data_list if d.get("robot_pose") is not None]
        
        if len(valid_data) < 3:
            messagebox.showwarning("警告", 
                f"有效数据不足！\n\n"
                f"当前有效: {len(valid_data)} 组\n"
                f"至少需要: 3 组\n\n"
                f"请先获取机械臂位姿（选中数据行后点击'获取机械臂位姿'）")
            return
        
        try:
            # 准备相机内参
            fx = self.config["camera_intrinsics"]["fx"]
            fy = self.config["camera_intrinsics"]["fy"]
            cx = self.config["camera_intrinsics"]["cx"]
            cy = self.config["camera_intrinsics"]["cy"]
            
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([
                self.config["distortion_coeffs"]["k1"],
                self.config["distortion_coeffs"]["k2"],
                self.config["distortion_coeffs"]["p1"],
                self.config["distortion_coeffs"]["p2"],
                self.config["distortion_coeffs"]["k3"]
            ], dtype=np.float64)
            
            objp = get_object_points(self.config)
            
            R_all_board_to_cam = []
            T_all_board_to_cam = []
            R_all_end_to_base = []
            T_all_end_to_base = []
            
            for d in valid_data:
                corners = d["corners"]
                pose = d["robot_pose"]
                
                # solvePnP计算标定板相对于相机的位姿
                success, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
                if not success:
                    continue
                
                R_board_to_cam, _ = cv2.Rodrigues(rvec)
                R_all_board_to_cam.append(R_board_to_cam)
                T_all_board_to_cam.append(tvec)
                
                # 计算末端到基座的变换矩阵
                RT = pose_to_transform_matrix(
                    pose[0], pose[1], pose[2],
                    pose[3], pose[4], pose[5],
                    self.config
                )
                R_all_end_to_base.append(RT[:3, :3])
                T_all_end_to_base.append(RT[:3, 3].reshape((3, 1)))
            
            if len(R_all_board_to_cam) < 3:
                messagebox.showerror("错误", "有效数据不足")
                return
            
            # 执行手眼标定
            method = get_hand_eye_method(self.config["hand_eye_method"])
            R_cam2end, T_cam2end = cv2.calibrateHandEye(
                R_all_end_to_base, T_all_end_to_base,
                R_all_board_to_cam, T_all_board_to_cam,
                method=method
            )
            
            # 构建4x4变换矩阵
            RT_cam2end = np.column_stack((R_cam2end, T_cam2end))
            RT_cam2end = np.vstack((RT_cam2end, [[0, 0, 0, 1]]))
            
            # 计算欧拉角
            try:
                from scipy.spatial.transform import Rotation as Rot
                euler = Rot.from_matrix(R_cam2end).as_euler('xyz', degrees=True)
            except:
                euler = [0, 0, 0]
            
            # 显示结果
            self.txt_result.delete(1.0, tk.END)
            self.txt_result.insert(tk.END, "=" * 60 + "\n")
            self.txt_result.insert(tk.END, "手眼标定结果 (Camera to End-Effector)\n")
            self.txt_result.insert(tk.END, "=" * 60 + "\n\n")
            
            self.txt_result.insert(tk.END, "【旋转矩阵 (3x3)】\n")
            for i in range(3):
                row_str = "  ["
                for j in range(3):
                    row_str += f"{R_cam2end[i, j]:12.8f}"
                    if j < 2:
                        row_str += ", "
                row_str += "]\n"
                self.txt_result.insert(tk.END, row_str)
            
            self.txt_result.insert(tk.END, "\n【平移向量 (mm)】\n")
            self.txt_result.insert(tk.END, f"  X = {T_cam2end[0][0]:12.4f} mm\n")
            self.txt_result.insert(tk.END, f"  Y = {T_cam2end[1][0]:12.4f} mm\n")
            self.txt_result.insert(tk.END, f"  Z = {T_cam2end[2][0]:12.4f} mm\n")
            
            self.txt_result.insert(tk.END, "\n【欧拉角 (xyz顺序, 度)】\n")
            self.txt_result.insert(tk.END, f"  Rx = {euler[0]:12.4f}°\n")
            self.txt_result.insert(tk.END, f"  Ry = {euler[1]:12.4f}°\n")
            self.txt_result.insert(tk.END, f"  Rz = {euler[2]:12.4f}°\n")
            
            self.txt_result.insert(tk.END, "\n【变换矩阵 (4x4)】\n")
            for i in range(4):
                row_str = "  ["
                for j in range(4):
                    row_str += f"{RT_cam2end[i, j]:12.8f}"
                    if j < 3:
                        row_str += ", "
                row_str += "]\n"
                self.txt_result.insert(tk.END, row_str)
            
            self.txt_result.insert(tk.END, "\n" + "=" * 60 + "\n")
            self.txt_result.insert(tk.END, f"使用数据: {len(R_all_board_to_cam)} 组\n")
            self.txt_result.insert(tk.END, f"标定方法: {self.config['hand_eye_method']}\n")
            
            # 验证结果
            self.txt_result.insert(tk.END, "\n" + "=" * 60 + "\n")
            self.txt_result.insert(tk.END, "标定结果验证\n")
            self.txt_result.insert(tk.END, "=" * 60 + "\n")
            
            positions = []
            for i in range(len(R_all_board_to_cam)):
                RT_end_to_base = np.column_stack((R_all_end_to_base[i], T_all_end_to_base[i]))
                RT_end_to_base = np.vstack((RT_end_to_base, [[0, 0, 0, 1]]))
                
                RT_board_to_cam = np.column_stack((R_all_board_to_cam[i], T_all_board_to_cam[i]))
                RT_board_to_cam = np.vstack((RT_board_to_cam, [[0, 0, 0, 1]]))
                
                RT_board_to_base = RT_end_to_base @ RT_cam2end @ RT_board_to_cam
                RT_base_to_board = np.linalg.inv(RT_board_to_base)
                
                positions.append(RT_base_to_board[:3, 3])
                self.txt_result.insert(tk.END, 
                    f"第{i+1}组: X={RT_base_to_board[0,3]:.2f}, Y={RT_base_to_board[1,3]:.2f}, Z={RT_base_to_board[2,3]:.2f}\n")
            
            positions = np.array(positions)
            std_pos = np.std(positions, axis=0)
            total_std = np.sqrt(np.sum(std_pos**2))
            
            self.txt_result.insert(tk.END, f"\n标准差: X={std_pos[0]:.2f}, Y={std_pos[1]:.2f}, Z={std_pos[2]:.2f} mm\n")
            self.txt_result.insert(tk.END, f"综合标准差: {total_std:.2f} mm\n")
            
            if total_std < 5:
                self.txt_result.insert(tk.END, "评估: ★★★ 标定精度优秀！\n")
            elif total_std < 10:
                self.txt_result.insert(tk.END, "评估: ★★☆ 标定精度良好\n")
            elif total_std < 20:
                self.txt_result.insert(tk.END, "评估: ★☆☆ 标定精度一般\n")
            else:
                self.txt_result.insert(tk.END, "评估: ☆☆☆ 标定精度较差\n")
            
            # 保存结果
            output_dir = self.config["paths"]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存变换矩阵
            matrix_file = os.path.join(output_dir, "cam2end.txt")
            with open(matrix_file, 'w') as f:
                f.write("# 相机到末端的变换矩阵 (4x4)\n")
                for row in RT_cam2end:
                    f.write(' '.join(map(str, row)) + '\n')
            
            # 保存JSON结果
            result = {
                "rotation_matrix": R_cam2end.tolist(),
                "translation_vector": T_cam2end.flatten().tolist(),
                "transformation_matrix_4x4": RT_cam2end.tolist(),
                "euler_angles_xyz_deg": list(euler),
                "camera_intrinsics": self.config["camera_intrinsics"],
                "calibration_board": self.config["calibration_board"]
            }
            
            json_file = os.path.join(output_dir, "calibration_result.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            self.txt_result.insert(tk.END, f"\n结果已保存到: {output_dir}\n")
            
            self.log("手眼标定完成！")
            messagebox.showinfo("成功", "手眼标定完成！\n\n结果已显示在'标定结果'页面")
            
        except Exception as e:
            self.log(f"[错误] 标定失败: {e}")
            messagebox.showerror("错误", f"标定失败: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    np.set_printoptions(suppress=True, precision=8)
    
    root = tk.Tk()
    app = HandEyeCalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
