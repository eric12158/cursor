import cv2
import numpy as np
import sys
from math import sin, cos, atan2, asin

# ==========================================
# 1. 基础配置与辅助函数
# ==========================================

# 模拟的相机内参 (实际使用时需替换为 camera.ini 或标定结果)
CAMERA_MATRIX = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=float)
DIST_COEFFS = np.zeros((5, 1))

# 定义 Aruco 字典 (推荐使用 4X4 或 5X5)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# 二维码/Aruco 真实物理尺寸 (米)
MARKER_SIZE = 0.05  # 例如 5cm

def euler_to_matrix(roll, pitch, yaw):
    """欧拉角转旋转矩阵 (XYZ顺序)"""
    Rx = np.array([[1, 0, 0],
                   [0, cos(roll), -sin(roll)],
                   [0, sin(roll), cos(roll)]])
    Ry = np.array([[cos(pitch), 0, sin(pitch)],
                   [0, 1, 0],
                   [-sin(pitch), 0, cos(pitch)]])
    Rz = np.array([[cos(yaw), -sin(yaw), 0],
                   [sin(yaw), cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def get_transform_matrix(rvec, tvec):
    """将旋转向量和平移向量转换为 4x4 齐次变换矩阵"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

# ==========================================
# 2. 核心流程：视觉定位
# ==========================================

def detect_and_estimate_pose(image):
    """
    检测 Aruco 并估计位姿 T_cam_marker
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    
    if ids is not None and len(ids) > 0:
        # 假设我们只关心 ID 为 0 的码
        target_index = np.where(ids == 0)[0][0]
        
        # 姿态估计
        # objPoints, imgPoints -> solvePnP
        # Aruco 库提供了直接的 estimatePoseSingleMarkers
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS)
        
        rvec = rvecs[target_index]
        tvec = tvecs[target_index]
        
        # 绘制检测结果 (调试用)
        cv2.drawFrameAxes(image, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.1)
        cv2.aruco.drawDetectedMarkers(image, corners)
        
        # 返回 T_cam_marker
        return get_transform_matrix(rvec, tvec), image
    
    return None, image

# ==========================================
# 3. 核心流程：坐标链计算
# ==========================================

def calculate_grasp_pose(T_base_end, T_end_cam, T_cam_marker):
    """
    计算最终抓取位姿 T_base_grasp
    
    链式法则: T_base_marker = T_base_end * T_end_cam * T_cam_marker
    最终目标: T_base_grasp  = T_base_marker * T_marker_grasp
    """
    
    # 1. 计算 Marker 在基座标系下的位姿
    # T_base_marker = T_base_end @ T_end_cam @ T_cam_marker
    # 注意：矩阵乘法顺序需根据你的坐标系定义确认，通常是左乘父坐标系
    T_base_marker = T_base_end @ T_end_cam @ T_cam_marker
    
    # 2. 定义抓取点相对于 Marker 的固定偏移 (由物体几何决定)
    # 例如：物体在 Marker 中心左侧 10cm，且需要侧面抓取 (绕 Y 轴旋转 90 度)
    # 这里的数值需要根据实际物体测量得到
    offset_x = -0.10 
    offset_y = 0.0
    offset_z = 0.0
    
    # 定义旋转偏移 (例如侧面抓取可能需要绕某个轴转动)
    # 假设 Marker 平贴在侧面，Z轴朝外。我们需要沿 Z 轴反方向抓取。
    # 这部分高度依赖具体的抓取策略
    R_grasp = np.eye(3) 
    t_grasp = np.array([offset_x, offset_y, offset_z])
    
    T_marker_grasp = np.eye(4)
    T_marker_grasp[:3, :3] = R_grasp
    T_marker_grasp[:3, 3] = t_grasp
    
    # 3. 计算最终机械臂需要到达的 TCP 位姿
    T_base_grasp = T_base_marker @ T_marker_grasp
    
    return T_base_grasp

# ==========================================
# 4. 主程序模拟
# ==========================================

if __name__ == "__main__":
    print("启动定位抓取程序...")
    
    # 1. 模拟获取当前机械臂位姿 (从机器人控制器读取)
    # 假设当前末端在基座系的 [0.4, 0, 0.5] 位置，无旋转
    curr_rvec = np.array([0, 3.14, 0], dtype=float) # 指向下
    curr_tvec = np.array([0.4, 0.0, 0.5], dtype=float)
    T_base_end = get_transform_matrix(curr_rvec, curr_tvec)
    
    # 2. 读取手眼标定结果 (T_end_cam)
    # 假设相机装在末端，且有一定偏移
    # 从你的 hand_eye_calibrate.py 生成的文件读取
    # 这里手动模拟一个
    T_end_cam = np.eye(4)
    T_end_cam[:3, 3] = np.array([0.05, 0.0, 0.05]) # 假设相机在法兰中心前方5cm，上方5cm
    
    # 3. 获取图像并检测
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # 这里创建一个虚拟图像用来测试代码逻辑是否跑通
    frame = np.zeros((720, 1280, 3), dtype=np.uint8) 
    
    T_cam_marker, debug_img = detect_and_estimate_pose(frame)
    
    if T_cam_marker is not None:
        print("检测到目标！")
        print("T_cam_marker:\n", T_cam_marker)
        
        # 计算抓取位姿
        target_pose = calculate_grasp_pose(T_base_end, T_end_cam, T_cam_marker)
        
        print("-" * 30)
        print("计算出的抓取目标位姿 (T_base_grasp):")
        print(target_pose)
        print("-" * 30)
        
        # TODO: 将 target_pose 发送给机械臂控制器进行规划和运动
        
    else:
        print("未检测到 Marker，请移动机械臂寻找目标...")
