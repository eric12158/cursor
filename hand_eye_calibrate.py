#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
手眼标定程序 (Hand-Eye Calibration)

功能说明:
    1. 支持JSON配置文件输入相机内参和畸变系数
    2. 支持棋盘格(chessboard)和圆点(circles)两种标定板类型
    3. 从指定目录自动加载图片进行标定板检测
    4. 从文本文件读取对应的机械臂末端位姿
    5. 使用OpenCV calibrateHandEye进行手眼标定
    6. 输出完整的手眼标定矩阵和验证结果

使用方法:
    1. 准备配置文件 calibration_config.json（首次运行会自动生成默认配置）
    2. 准备标定图片，放入配置中指定的目录（默认：./calibration_images/）
    3. 准备机械臂位姿文件（默认：./calibration_images/robot_poses.txt）
       格式：每行 x,y,z,rx,ry,rz（单位：mm和弧度）
    4. 运行程序：python hand_eye_calibrate.py

配置文件说明:
    - camera_intrinsics: 相机内参 (fx, fy, cx, cy)
    - distortion_coeffs: 畸变系数 (k1, k2, p1, p2, k3)
    - calibration_board: 标定板参数 (类型、行列数、间距)
    - paths: 图片目录和位姿文件路径
    - euler_order: 机械臂欧拉角顺序 (如 "xyz", "zyx" 等)
    - angle_unit: 机械臂角度单位 ("radians" 或 "degrees")

作者: 基于原始MATLAB标定版本重构
版本: 2.0 (配置文件驱动版)
日期: 2024
'''

import os
import sys
import json
import glob
import cv2
import numpy as np
from math import sin, cos, pi

# =============================================================================
# 配置文件管理
# =============================================================================

# 默认配置模板
DEFAULT_CONFIG = {
    "camera_intrinsics": {
        "fx": 1000.0,           # 焦距x (像素)
        "fy": 1000.0,           # 焦距y (像素)
        "cx": 640.0,            # 主点x (像素)
        "cy": 480.0             # 主点y (像素)
    },
    "distortion_coeffs": {
        "k1": 0.0,              # 径向畸变系数1
        "k2": 0.0,              # 径向畸变系数2
        "p1": 0.0,              # 切向畸变系数1
        "p2": 0.0,              # 切向畸变系数2
        "k3": 0.0               # 径向畸变系数3
    },
    "calibration_board": {
        "type": "chessboard",   # 标定板类型: "chessboard" (棋盘格) 或 "circles" (圆点)
        "rows": 9,              # 行数（棋盘格：行方向角点数，圆点：行方向圆点数）
        "cols": 6,              # 列数（棋盘格：列方向角点数，圆点：列方向圆点数）
        "spacing": 25.0         # 间距 (mm)：棋盘格为方格边长，圆点为圆心间距
    },
    "paths": {
        "image_dir": "./calibration_images",      # 标定图片目录
        "robot_pose_file": "./calibration_images/robot_poses.txt",  # 机械臂位姿文件
        "output_dir": "./calibration_results"     # 输出结果目录
    },
    "robot_config": {
        "euler_order": "xyz",   # 机械臂欧拉角顺序: "xyz", "zyx", "zyz" 等
        "angle_unit": "radians", # 角度单位: "radians" (弧度) 或 "degrees" (度)
        "pose_format": "x,y,z,rx,ry,rz"  # 位姿格式说明
    },
    "hand_eye_method": "TSAI"   # 手眼标定方法: "TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"
}

CONFIG_FILE = "calibration_config.json"


def load_config(config_path=CONFIG_FILE):
    """加载配置文件，如果不存在则创建默认配置"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            # 递归合并配置，补全缺失字段
            config = merge_config(DEFAULT_CONFIG, user_config)
            print(f"[配置] 成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            print(f"[警告] 加载配置文件失败: {e}，使用默认配置")
            return DEFAULT_CONFIG.copy()
    else:
        # 创建默认配置文件
        save_config(DEFAULT_CONFIG, config_path)
        print(f"[配置] 已创建默认配置文件: {config_path}")
        print(f"[提示] 请根据实际情况修改配置文件后重新运行程序")
        return DEFAULT_CONFIG.copy()


def merge_config(default, user):
    """递归合并配置，用户配置优先，缺失字段使用默认值"""
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
    print(f"[配置] 配置已保存到: {config_path}")


# =============================================================================
# 相机内参和畸变系数处理
# =============================================================================

def get_camera_matrix(config):
    """从配置中构建相机内参矩阵"""
    intrinsics = config["camera_intrinsics"]
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def get_distortion_coeffs(config):
    """从配置中获取畸变系数"""
    dist = config["distortion_coeffs"]
    D = np.array([dist["k1"], dist["k2"], dist["p1"], dist["p2"], dist["k3"]], dtype=np.float64)
    return D


# =============================================================================
# 标定板检测
# =============================================================================

def detect_calibration_board(image, config):
    """
    检测标定板角点/圆心
    
    参数:
        image: BGR图像
        config: 配置字典
    
    返回:
        success: 是否成功检测
        corners: 角点/圆心坐标 (N, 1, 2) 或 None
    """
    board_config = config["calibration_board"]
    board_type = board_config["type"]
    rows = board_config["rows"]
    cols = board_config["cols"]
    
    # 转为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if board_type == "chessboard":
        # 棋盘格检测
        # 注意：OpenCV的findChessboardCorners参数是(cols, rows)
        pattern_size = (cols, rows)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        success, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        
        if success:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
    elif board_type == "circles":
        # 圆点标定板检测
        pattern_size = (cols, rows)
        # 使用SimpleBlobDetector
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
            # 尝试非对称网格
            success, corners = cv2.findCirclesGrid(
                gray, pattern_size, None,
                flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                blobDetector=detector
            )
    else:
        print(f"[错误] 不支持的标定板类型: {board_type}")
        return False, None
    
    return success, corners


def get_object_points(config):
    """
    生成标定板3D坐标点
    
    返回:
        objp: 3D点坐标 (N, 3)，单位mm
    """
    board_config = config["calibration_board"]
    board_type = board_config["type"]
    rows = board_config["rows"]
    cols = board_config["cols"]
    spacing = board_config["spacing"]
    
    # 生成标定板角点的3D坐标
    # 假设标定板在Z=0平面上
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * spacing
    
    return objp


# =============================================================================
# 机械臂位姿处理
# =============================================================================

def read_robot_poses(file_path, config):
    """
    读取机械臂末端位姿文件
    
    文件格式: 每行 x,y,z,rx,ry,rz (逗号分隔)
    
    参数:
        file_path: 位姿文件路径
        config: 配置字典
    
    返回:
        poses: 位姿列表 [[x,y,z,rx,ry,rz], ...]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"机械臂位姿文件不存在: {file_path}")
    
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):  # 跳过空行和注释
            continue
        
        # 去除方括号
        line = line.replace('[', '').replace(']', '')
        
        try:
            # 尝试逗号分隔
            if ',' in line:
                values = [float(v.strip()) for v in line.split(',')]
            else:
                # 尝试空格分隔
                values = [float(v) for v in line.split()]
            
            if len(values) != 6:
                print(f"[警告] 第{line_num}行数据格式错误，跳过: {line}")
                continue
            
            poses.append(values)
        except ValueError as e:
            print(f"[警告] 第{line_num}行解析失败，跳过: {line} ({e})")
            continue
    
    print(f"[位姿] 成功读取 {len(poses)} 组机械臂位姿数据")
    return poses


def euler_to_rotation_matrix(rx, ry, rz, order='xyz'):
    """
    欧拉角转旋转矩阵
    
    参数:
        rx, ry, rz: 欧拉角 (弧度)
        order: 旋转顺序，如 'xyz', 'zyx' 等
    
    返回:
        R: 3x3 旋转矩阵
    """
    # 基本旋转矩阵
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
    
    # 根据顺序组合
    rotations = {'x': Rx, 'y': Ry, 'z': Rz}
    order = order.lower()
    
    if len(order) == 3:
        R = rotations[order[2]] @ rotations[order[1]] @ rotations[order[0]]
    else:
        # 默认xyz顺序: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx
    
    return R


def pose_to_transform_matrix(x, y, z, rx, ry, rz, config):
    """
    将机械臂位姿转换为4x4齐次变换矩阵
    
    参数:
        x, y, z: 平移 (mm)
        rx, ry, rz: 欧拉角
        config: 配置字典
    
    返回:
        T: 4x4 齐次变换矩阵
    """
    robot_config = config["robot_config"]
    euler_order = robot_config.get("euler_order", "xyz")
    angle_unit = robot_config.get("angle_unit", "radians")
    
    # 单位转换
    if angle_unit == "degrees":
        rx = rx * pi / 180
        ry = ry * pi / 180
        rz = rz * pi / 180
    
    # 计算旋转矩阵
    R = euler_to_rotation_matrix(rx, ry, rz, euler_order)
    
    # 构建4x4齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


# =============================================================================
# 图像加载
# =============================================================================

def load_calibration_images(image_dir):
    """
    从目录加载所有标定图片
    
    参数:
        image_dir: 图片目录路径
    
    返回:
        image_files: 图片文件路径列表（按文件名排序）
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    
    # 支持的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    # 按文件名排序
    image_files.sort()
    
    if not image_files:
        raise FileNotFoundError(f"未在目录 {image_dir} 中找到图片文件")
    
    print(f"[图像] 找到 {len(image_files)} 张标定图片")
    return image_files


# =============================================================================
# 手眼标定核心算法 (保留原有算法)
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


def run_hand_eye_calibration(config):
    """
    执行手眼标定主流程
    
    参数:
        config: 配置字典
    
    返回:
        R_cam2end: 相机到末端的旋转矩阵 (3x3)
        T_cam2end: 相机到末端的平移向量 (3x1)
        RT_cam2end: 相机到末端的变换矩阵 (4x4)
    """
    print("\n" + "=" * 60)
    print("手眼标定程序 (Hand-Eye Calibration)")
    print("=" * 60)
    
    # 获取配置参数
    K = get_camera_matrix(config)
    D = get_distortion_coeffs(config)
    objp = get_object_points(config)
    
    print(f"\n[相机内参]")
    print(f"  fx = {K[0,0]:.2f}")
    print(f"  fy = {K[1,1]:.2f}")
    print(f"  cx = {K[0,2]:.2f}")
    print(f"  cy = {K[1,2]:.2f}")
    print(f"\n[畸变系数]")
    print(f"  k1 = {D[0]:.6f}")
    print(f"  k2 = {D[1]:.6f}")
    print(f"  p1 = {D[2]:.6f}")
    print(f"  p2 = {D[3]:.6f}")
    print(f"  k3 = {D[4]:.6f}")
    
    board_config = config["calibration_board"]
    print(f"\n[标定板配置]")
    print(f"  类型: {board_config['type']}")
    print(f"  尺寸: {board_config['rows']} x {board_config['cols']}")
    print(f"  间距: {board_config['spacing']} mm")
    
    # 加载图片
    image_dir = config["paths"]["image_dir"]
    image_files = load_calibration_images(image_dir)
    
    # 读取机械臂位姿
    pose_file = config["paths"]["robot_pose_file"]
    robot_poses = read_robot_poses(pose_file, config)
    
    # 检查数据数量匹配
    if len(image_files) != len(robot_poses):
        raise ValueError(
            f"图片数量 ({len(image_files)}) 与机械臂位姿数量 ({len(robot_poses)}) 不匹配！\n"
            f"请确保每张图片都有对应的机械臂位姿数据。"
        )
    
    # 检测标定板并计算位姿
    R_all_board_to_cam = []  # 标定板到相机的旋转矩阵
    T_all_board_to_cam = []  # 标定板到相机的平移向量
    R_all_end_to_base = []   # 末端到基座的旋转矩阵
    T_all_end_to_base = []   # 末端到基座的平移向量
    
    valid_indices = []
    
    print(f"\n[标定板检测]")
    for i, (img_file, pose) in enumerate(zip(image_files, robot_poses)):
        img = cv2.imread(img_file)
        if img is None:
            print(f"  [{i+1}] 无法读取图片: {os.path.basename(img_file)}")
            continue
        
        success, corners = detect_calibration_board(img, config)
        
        if success:
            # 使用solvePnP计算标定板相对于相机的位姿
            success_pnp, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
            
            if success_pnp:
                # 将旋转向量转换为旋转矩阵
                R_board_to_cam, _ = cv2.Rodrigues(rvec)
                R_all_board_to_cam.append(R_board_to_cam)
                T_all_board_to_cam.append(tvec)
                
                # 计算末端到基座的变换矩阵
                RT_end_to_base = pose_to_transform_matrix(
                    pose[0], pose[1], pose[2],
                    pose[3], pose[4], pose[5],
                    config
                )
                R_all_end_to_base.append(RT_end_to_base[:3, :3])
                T_all_end_to_base.append(RT_end_to_base[:3, 3].reshape((3, 1)))
                
                valid_indices.append(i)
                print(f"  [{i+1}] ✓ 检测成功: {os.path.basename(img_file)}")
            else:
                print(f"  [{i+1}] ✗ PnP求解失败: {os.path.basename(img_file)}")
        else:
            print(f"  [{i+1}] ✗ 标定板检测失败: {os.path.basename(img_file)}")
    
    # 检查有效数据数量
    if len(valid_indices) < 3:
        raise ValueError(
            f"有效标定数据不足！成功检测 {len(valid_indices)} 组，至少需要 3 组。\n"
            f"请检查标定板配置是否正确，或增加更多标定图片。"
        )
    
    print(f"\n[数据统计]")
    print(f"  总图片数: {len(image_files)}")
    print(f"  有效数据: {len(valid_indices)} 组")
    
    # 执行手眼标定 (保留原有算法)
    method = get_hand_eye_method(config.get("hand_eye_method", "TSAI"))
    print(f"\n[执行手眼标定]")
    print(f"  方法: {config.get('hand_eye_method', 'TSAI')}")
    
    R_cam2end, T_cam2end = cv2.calibrateHandEye(
        R_all_end_to_base, T_all_end_to_base,
        R_all_board_to_cam, T_all_board_to_cam,
        method=method
    )
    
    # 构建4x4变换矩阵
    RT_cam2end = np.column_stack((R_cam2end, T_cam2end))
    RT_cam2end = np.vstack((RT_cam2end, [[0, 0, 0, 1]]))
    
    return R_cam2end, T_cam2end, RT_cam2end, \
           R_all_board_to_cam, T_all_board_to_cam, \
           R_all_end_to_base, T_all_end_to_base, valid_indices


def verify_calibration_result(R_cam2end, T_cam2end,
                               R_all_board_to_cam, T_all_board_to_cam,
                               R_all_end_to_base, T_all_end_to_base):
    """
    验证手眼标定结果
    
    原理: 标定板相对于基座的位姿应该是固定的
    RT_board_to_base = RT_end_to_base @ RT_cam2end @ RT_board_to_cam
    """
    print("\n" + "=" * 60)
    print("标定结果验证")
    print("=" * 60)
    print("(原理: 标定板相对于基座的位姿应保持一致)")
    print("-" * 60)
    
    # 构建cam2end变换矩阵
    RT_cam2end = np.column_stack((R_cam2end, T_cam2end))
    RT_cam2end = np.vstack((RT_cam2end, [[0, 0, 0, 1]]))
    
    positions = []
    
    for i in range(len(R_all_board_to_cam)):
        # 构建末端到基座的变换矩阵
        RT_end_to_base = np.column_stack((R_all_end_to_base[i], T_all_end_to_base[i]))
        RT_end_to_base = np.vstack((RT_end_to_base, [[0, 0, 0, 1]]))
        
        # 构建标定板到相机的变换矩阵
        RT_board_to_cam = np.column_stack((R_all_board_to_cam[i], T_all_board_to_cam[i]))
        RT_board_to_cam = np.vstack((RT_board_to_cam, [[0, 0, 0, 1]]))
        
        # 计算标定板相对于基座的位姿
        RT_board_to_base = RT_end_to_base @ RT_cam2end @ RT_board_to_cam
        RT_base_to_board = np.linalg.inv(RT_board_to_base)
        
        print(f"\n第 {i+1} 组数据:")
        print(f"  标定板在基座坐标系下的位姿矩阵:")
        for row in RT_base_to_board[:3, :]:
            print(f"    [{row[0]:12.4f} {row[1]:12.4f} {row[2]:12.4f} {row[3]:12.4f}]")
        
        positions.append(RT_base_to_board[:3, 3])
    
    # 计算位置偏差
    positions = np.array(positions)
    mean_pos = np.mean(positions, axis=0)
    std_pos = np.std(positions, axis=0)
    max_deviation = np.max(np.abs(positions - mean_pos), axis=0)
    
    print("\n" + "-" * 60)
    print("位置一致性分析:")
    print(f"  平均位置: X={mean_pos[0]:.2f}, Y={mean_pos[1]:.2f}, Z={mean_pos[2]:.2f} mm")
    print(f"  标准差:   X={std_pos[0]:.2f}, Y={std_pos[1]:.2f}, Z={std_pos[2]:.2f} mm")
    print(f"  最大偏差: X={max_deviation[0]:.2f}, Y={max_deviation[1]:.2f}, Z={max_deviation[2]:.2f} mm")
    
    total_std = np.sqrt(np.sum(std_pos**2))
    print(f"\n  综合标准差: {total_std:.2f} mm")
    
    if total_std < 5:
        print("  [评估] ★★★ 标定精度优秀！")
    elif total_std < 10:
        print("  [评估] ★★☆ 标定精度良好")
    elif total_std < 20:
        print("  [评估] ★☆☆ 标定精度一般，建议重新采集数据")
    else:
        print("  [评估] ☆☆☆ 标定精度较差，请检查数据质量")
    
    return mean_pos, std_pos


def save_results(R_cam2end, T_cam2end, RT_cam2end, config):
    """保存标定结果"""
    output_dir = config["paths"]["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存变换矩阵到文本文件
    matrix_file = os.path.join(output_dir, "cam2end.txt")
    with open(matrix_file, 'w') as f:
        f.write("# 相机到末端的变换矩阵 (4x4)\n")
        f.write("# Camera to End-Effector Transformation Matrix\n")
        for row in RT_cam2end:
            f.write(' '.join(map(str, row)) + '\n')
    
    # 保存详细结果到JSON
    result = {
        "rotation_matrix": R_cam2end.tolist(),
        "translation_vector": T_cam2end.flatten().tolist(),
        "transformation_matrix_4x4": RT_cam2end.tolist(),
        "camera_intrinsics": config["camera_intrinsics"],
        "distortion_coeffs": config["distortion_coeffs"],
        "calibration_board": config["calibration_board"]
    }
    
    json_file = os.path.join(output_dir, "calibration_result.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"\n[保存] 结果已保存到:")
    print(f"  变换矩阵: {matrix_file}")
    print(f"  完整结果: {json_file}")


def print_final_result(R_cam2end, T_cam2end, RT_cam2end):
    """打印最终标定结果"""
    print("\n" + "=" * 60)
    print("手眼标定最终结果")
    print("=" * 60)
    
    print("\n【手眼矩阵分解得到的旋转矩阵】")
    print(R_cam2end)
    
    print("\n【手眼矩阵分解得到的平移矩阵】")
    print(T_cam2end)
    
    print("\n【相机相对于末端的变换矩阵 (4x4)】")
    print(RT_cam2end)
    
    # 提取欧拉角 (使用scipy或手动计算)
    try:
        from scipy.spatial.transform import Rotation as Rot
        euler = Rot.from_matrix(R_cam2end).as_euler('xyz', degrees=True)
        print(f"\n【欧拉角 (xyz顺序, 度)】")
        print(f"  Rx = {euler[0]:.4f}°")
        print(f"  Ry = {euler[1]:.4f}°")
        print(f"  Rz = {euler[2]:.4f}°")
    except ImportError:
        pass
    
    print(f"\n【平移分量 (mm)】")
    print(f"  X = {T_cam2end[0][0]:.4f} mm")
    print(f"  Y = {T_cam2end[1][0]:.4f} mm")
    print(f"  Z = {T_cam2end[2][0]:.4f} mm")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序入口"""
    # 设置NumPy打印选项
    np.set_printoptions(suppress=True, precision=8)
    
    print("\n" + "=" * 60)
    print("手眼标定程序 v2.0")
    print("配置文件驱动版 - 支持棋盘格和圆点标定板")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    
    # 检查关键路径是否存在
    image_dir = config["paths"]["image_dir"]
    pose_file = config["paths"]["robot_pose_file"]
    
    if not os.path.exists(image_dir):
        print(f"\n[错误] 图片目录不存在: {image_dir}")
        print(f"[提示] 请创建目录并放入标定图片，或修改配置文件中的 paths.image_dir")
        os.makedirs(image_dir, exist_ok=True)
        print(f"[提示] 已为您创建空目录: {image_dir}")
        return
    
    if not os.path.exists(pose_file):
        print(f"\n[错误] 机械臂位姿文件不存在: {pose_file}")
        print(f"[提示] 请创建位姿文件，格式为每行: x,y,z,rx,ry,rz")
        # 创建示例文件
        example_content = """# 机械臂末端位姿文件
# 格式: x,y,z,rx,ry,rz (单位: mm, 弧度或度，根据配置)
# 每行对应一张标定图片
# 示例:
# 100.0, 200.0, 300.0, 0.1, 0.2, 0.3
"""
        with open(pose_file, 'w') as f:
            f.write(example_content)
        print(f"[提示] 已为您创建示例位姿文件: {pose_file}")
        return
    
    try:
        # 执行手眼标定
        R_cam2end, T_cam2end, RT_cam2end, \
        R_all_board_to_cam, T_all_board_to_cam, \
        R_all_end_to_base, T_all_end_to_base, valid_indices = run_hand_eye_calibration(config)
        
        # 打印最终结果
        print_final_result(R_cam2end, T_cam2end, RT_cam2end)
        
        # 验证标定结果
        verify_calibration_result(
            R_cam2end, T_cam2end,
            R_all_board_to_cam, T_all_board_to_cam,
            R_all_end_to_base, T_all_end_to_base
        )
        
        # 保存结果
        save_results(R_cam2end, T_cam2end, RT_cam2end, config)
        
        print("\n" + "=" * 60)
        print("手眼标定完成！")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
    except ValueError as e:
        print(f"\n[错误] {e}")
    except Exception as e:
        print(f"\n[错误] 标定过程中发生异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
