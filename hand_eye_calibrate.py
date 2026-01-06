'''
Description: 
使用OpenCV进行手眼标定 (Hand-Eye Calibration)
- 替代原有的MATLAB工具箱依赖，支持直接配置相机内参
- 模仿 import sys.py 的数据处理方式
- 包含标定板角点检测、位姿读取、手眼标定及结果验证

Author: Assistant
Date: 2026-01-06
'''

import cv2
import numpy as np
import glob
import os
import configparser
from scipy.spatial.transform import Rotation as R

# ================= 配置区域 (Configuration) =================

# 1. 相机内参 (Camera Intrinsics) - 请根据实际相机参数修改
# 格式: [fx, fy, cx, cy]
CAMERA_INTRINSICS = [2463.8, 2461.9, 1225.6, 1013.2] 

# 2. 畸变系数 (Distortion Coefficients)
# 格式: [k1, k2, p1, p2, k3]
DISTORTION_COEFFS = [-0.1065, 0.1345, 0.0, 0.0, 0.0]

# 3. 标定板设置 (Calibration Board Settings)
# checkerboard_size: (columns, rows) - 内角点数量 (例如 11x8 的方格，角点是 10x7)
# square_size: 格子边长 (单位: mm 或 m，需保持一致)
BOARD_CONFIG = {
    'checkerboard_size': (11, 8),  
    'square_size': 15.0            
}

# 4. 数据路径 (Data Paths)
# 图片文件夹路径
IMAGE_DIR = 'images'
# 机械臂位姿文件路径 (格式: x, y, z, rx, ry, rz)
POSE_FILE = 'images/pos.txt'
# 图片格式
IMAGE_EXT = '*.jpg' # 或 *.bmp, *.png

# ==========================================================

class HandEyeCalibration:
    def __init__(self, camera_intrinsics, distortion_coeffs, board_config):
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        self.dist_coeffs = np.array(distortion_coeffs, dtype=np.float64)
        self.board_size = board_config['checkerboard_size']
        self.square_size = board_config['square_size']
        
        # 构造相机矩阵 K
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 准备标定板的三维坐标 (Object Points)
        # 坐标系建立在标定板上，Z=0
        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        self.objp = self.objp * self.square_size

    def load_images(self, image_dir, extension='*.bmp'):
        """读取指定文件夹下的所有图片路径，并排序"""
        images = glob.glob(os.path.join(image_dir, extension))
        images.sort() # 确保顺序与pose文件一致
        if not images:
            print(f"[警告] 在 {image_dir} 未找到 {extension} 图片")
        return images

    def load_robot_poses(self, file_path):
        """
        读取机械臂位姿文件
        格式每一行: x, y, z, rx, ry, rz (逗号分隔)
        """
        poses = []
        if not os.path.exists(file_path):
            print(f"[错误] 未找到位姿文件: {file_path}")
            return []
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace('[', '').replace(']', '')
                if not line: continue
                parts = line.split(',')
                try:
                    vals = [float(v) for v in parts if v.strip()]
                    if len(vals) >= 6:
                        poses.append(vals[:6])
                except ValueError:
                    print(f"[跳过] 无法解析行: {line}")
        return poses

    def process_data(self, image_paths, robot_poses):
        """
        处理每一对图片和位姿:
        1. 检测图片中的标定板 -> 得到 T_chess_to_cam (标定板相对于相机)
        2. 解析机械臂位姿 -> 得到 T_end_to_base (末端相对于基座)
        """
        R_chess_to_cam = []
        T_chess_to_cam = []
        R_end_to_base = []
        T_end_to_base = []
        
        valid_indices = [] # 记录成功的索引

        # 确保图片数量和位姿数量一致 (或取最小值)
        num_samples = min(len(image_paths), len(robot_poses))
        print(f"正在处理 {num_samples} 组数据...")

        for i in range(num_samples):
            img_path = image_paths[i]
            pose = robot_poses[i] # [x, y, z, rx, ry, rz]
            
            # --- 1. 图片处理 (求标定板位姿) ---
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 寻找角点
            found, corners = cv2.findChessboardCorners(gray, self.board_size, None)
            
            if found:
                # 亚像素精确化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # SolvePnP 计算标定板相对于相机的位姿
                # rvec, tvec 是标定板在相机坐标系下的旋转和平移
                ret, rvec, tvec = cv2.solvePnP(self.objp, corners2, self.K, self.dist_coeffs)
                
                # 将旋转向量转换为旋转矩阵
                rot_mat, _ = cv2.Rodrigues(rvec)
                
                R_chess_to_cam.append(rot_mat)
                T_chess_to_cam.append(tvec)
                
                # --- 2. 机械臂位姿处理 (求末端位姿) ---
                # pose: [x, y, z, rx, ry, rz]
                # 模仿 import sys.py 的处理方式: 使用 scipy R.from_euler('xyz', ...)
                # 注意: 这里的单位如果是 mm，需要统一。手眼标定通常对单位不敏感，只要一致即可。
                # 但通常相机 calibration 出来的 tvec 是以 square_size 为单位 (这里是 15mm)，所以 input pose 也要是 mm
                
                x, y, z, rx, ry, rz = pose
                
                # 平移向量
                t_robot = np.array([[x], [y], [z]])
                
                # 旋转矩阵 (假设输入是欧拉角，顺序 xyz，单位度)
                # 对应 import sys.py: R.from_euler('xyz', robot_pose[3:], degrees=True).as_matrix()
                r_robot = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
                
                R_end_to_base.append(r_robot)
                T_end_to_base.append(t_robot)
                
                valid_indices.append(i)
                print(f"[成功] 样本 {i}: 图片 {os.path.basename(img_path)} 检测到角点")
            else:
                print(f"[失败] 样本 {i}: 图片 {os.path.basename(img_path)} 未检测到角点")

        return R_end_to_base, T_end_to_base, R_chess_to_cam, T_chess_to_cam, valid_indices

    def calibrate(self, R_base, T_base, R_target, T_target):
        """执行手眼标定"""
        print("\n开始计算手眼标定矩阵...")
        # cv2.calibrateHandEye 输入要求:
        # R_gripper2base, t_gripper2base: 机械臂末端到基座
        # R_target2cam, t_target2cam: 标定板(Target)到相机
        # 默认方法: cv2.CALIB_HAND_EYE_TSAI
        
        R_cam2end, T_cam2end = cv2.calibrateHandEye(
            R_base, T_base, 
            R_target, T_target, 
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        return R_cam2end, T_cam2end

    def save_camera_ini(self):
        """保存相机参数到 camera.ini (保留原有功能)"""
        config = configparser.ConfigParser()
        config['camera_parameters'] = {
            'fx': str(self.fx),
            'fy': str(self.fy),
            'cx': str(self.cx),
            'cy': str(self.cy)
        }
        config['distortion_parameters'] = {
            'k1': str(self.dist_coeffs[0]),
            'k2': str(self.dist_coeffs[1]),
            'p1': str(self.dist_coeffs[2]),
            'p2': str(self.dist_coeffs[3]),
            'k3': str(self.dist_coeffs[4])
        }
        with open('camera.ini', 'w') as configfile:
            config.write(configfile)
        print("已保存相机参数到 camera.ini")

def main():
    # 设置打印选项
    np.set_printoptions(suppress=True, precision=4)

    # 1. 初始化标定类
    calibrator = HandEyeCalibration(CAMERA_INTRINSICS, DISTORTION_COEFFS, BOARD_CONFIG)
    
    # 2. 保存相机配置 (兼容旧程序)
    calibrator.save_camera_ini()
    
    # 3. 加载数据
    print(f"正在加载图片: {os.path.join(IMAGE_DIR, IMAGE_EXT)}")
    image_paths = calibrator.load_images(IMAGE_DIR, IMAGE_EXT)
    
    print(f"正在加载位姿: {POSE_FILE}")
    robot_poses = calibrator.load_robot_poses(POSE_FILE)
    
    if not image_paths or not robot_poses:
        print("错误: 缺少图片或位姿数据，无法继续。")
        return

    # 4. 处理数据 (检测角点 + 转换位姿)
    R_base, T_base, R_target, T_target, valid_indices = calibrator.process_data(image_paths, robot_poses)
    
    if len(R_base) < 3:
        print("有效数据不足 3 组，无法进行标定。")
        return

    # 5. 执行标定
    # 得到相机相对于末端的变换矩阵 (Eye-in-Hand 模式下通常求 Cam to End)
    # OpenCV calibrateHandEye 返回的是 R_cam2gripper, t_cam2gripper (即 cam 在 gripper 坐标系下的位姿)
    R_cam_to_end, T_cam_to_end = calibrator.calibrate(R_base, T_base, R_target, T_target)

    # 6. 输出结果
    print("\n=== 标定结果: 相机相对于末端 (Cam to End) ===")
    print("旋转矩阵 R:")
    print(R_cam_to_end)
    print("平移向量 T:")
    print(T_cam_to_end)
    
    # 组合成 4x4 矩阵
    RT_cam_to_end = np.column_stack((R_cam_to_end, T_cam_to_end))
    RT_cam_to_end = np.row_stack((RT_cam_to_end, np.array([0, 0, 0, 1])))
    
    print("\n齐次变换矩阵 RT:")
    print(RT_cam_to_end)
    
    # 7. 保存结果到文件
    with open("cam2end.txt", 'w') as f:
        for row in RT_cam_to_end:
            row_str = ' '.join(map(str, row))
            f.write(row_str + '\n')
    print("\n结果已保存至 cam2end.txt")

    # 8. 结果验证 (Result Validation)
    print("\n=== 结果验证 ===")
    print("计算每一幅图像中，标定板相对于机器人基座的固定位姿 (理论上应保持一致)")
    
    for i in range(len(valid_indices)):
        # 构造齐次矩阵
        
        # End to Base (末端到基座)
        RT_end_to_base = np.column_stack((R_base[i], T_base[i]))
        RT_end_to_base = np.row_stack((RT_end_to_base, np.array([0, 0, 0, 1])))
        
        # Board to Cam (标定板到相机)
        RT_chess_to_cam = np.column_stack((R_target[i], T_target[i]))
        RT_chess_to_cam = np.row_stack((RT_chess_to_cam, np.array([0, 0, 0, 1])))
        
        # Cam to End (相机到末端 - 标定结果)
        # 验证公式: T_board_to_base = T_end_to_base * T_cam_to_end * T_board_to_cam
        # 注意: 这里的乘法顺序取决于坐标系定义。
        # OpenCV calibrateHandEye 文档: R_gripper2base * R_cam2gripper * R_target2cam = R_target2base
        
        RT_chess_to_base = RT_end_to_base @ RT_cam_to_end @ RT_chess_to_cam
        
        # 原程序中似乎还求了逆，可能是为了看 Base to Board? 
        # 原文: RT_chess_to_base = np.linalg.inv(RT_chess_to_base)
        # 我们这里直接输出 Board to Base 的位置，理论上 XYZ 应该很稳定
        
        print(f"第 {valid_indices[i]} 组 - 标定板在基座坐标系下的位置 (X, Y, Z):")
        print(RT_chess_to_base[:3, 3].T)

if __name__ == '__main__':
    main()
