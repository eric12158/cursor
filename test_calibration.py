#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手眼标定测试程序
生成模拟的标定图片和机械臂位姿数据进行测试
"""

import os
import cv2
import numpy as np
import json

def generate_chessboard_image(rows, cols, square_size, image_size=(1280, 960)):
    """生成棋盘格图案"""
    img = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255
    
    # 计算棋盘格的起始位置（居中）
    board_width = cols * square_size
    board_height = rows * square_size
    start_x = (image_size[0] - board_width) // 2
    start_y = (image_size[1] - board_height) // 2
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                x1 = start_x + j * square_size
                y1 = start_y + i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), 0, -1)
    
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def create_test_data():
    """创建测试数据"""
    
    # 创建目录
    os.makedirs("calibration_images", exist_ok=True)
    os.makedirs("calibration_results", exist_ok=True)
    
    # 相机内参（模拟值）
    fx, fy = 1000.0, 1000.0
    cx, cy = 640.0, 480.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    # 标定板参数
    rows, cols = 9, 6
    square_size = 25.0  # mm
    
    # 生成3D点
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    
    # 模拟机械臂位姿（单位：mm和弧度）
    robot_poses = [
        [300, 0, 400, 0.0, 0.0, 0.0],
        [320, 20, 380, 0.1, 0.0, 0.0],
        [280, -10, 420, 0.0, 0.1, 0.0],
        [310, 15, 390, 0.05, 0.05, 0.0],
        [290, -5, 410, -0.05, 0.0, 0.05],
    ]
    
    print("生成测试图片...")
    
    for i, pose in enumerate(robot_poses):
        # 生成棋盘格图像
        img = generate_chessboard_image(rows + 1, cols + 1, 50, (1280, 960))
        
        # 保存图片
        img_path = f"calibration_images/image_{i+1:03d}.jpg"
        cv2.imwrite(img_path, img)
        print(f"  保存: {img_path}")
    
    # 保存机械臂位姿
    pose_file = "calibration_images/robot_poses.txt"
    with open(pose_file, 'w') as f:
        f.write("# 机械臂末端位姿\n")
        f.write("# 格式: x,y,z,rx,ry,rz (mm, radians)\n")
        for pose in robot_poses:
            f.write(f"{pose[0]}, {pose[1]}, {pose[2]}, {pose[3]}, {pose[4]}, {pose[5]}\n")
    print(f"  保存位姿文件: {pose_file}")
    
    # 更新配置文件
    config = {
        "camera_intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
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
            "rows": rows,
            "cols": cols,
            "spacing": square_size
        },
        "paths": {
            "image_dir": "./calibration_images",
            "robot_pose_file": "./calibration_images/robot_poses.txt",
            "output_dir": "./calibration_results"
        },
        "robot_config": {
            "euler_order": "xyz",
            "angle_unit": "radians",
            "pose_format": "x,y,z,rx,ry,rz"
        },
        "hand_eye_method": "TSAI"
    }
    
    with open("calibration_config.json", 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  更新配置文件: calibration_config.json")
    
    print("\n测试数据生成完成！")
    print("现在可以运行 python hand_eye_calibrate.py 进行测试")


if __name__ == "__main__":
    create_test_data()
