import cv2
import numpy as np
import glob
import os
import argparse

def calibrate_and_measure(image_folder, pattern_size=(7, 7), circle_spacing=1.5):
    # ---------------------------------------------------------
    # 1. 配置更强的斑点检测器 (关键步骤)
    # ---------------------------------------------------------
    blobParams = cv2.SimpleBlobDetector_Params()
    
    # 过滤面积：圆点不能太小也不能太大
    blobParams.filterByArea = True
    blobParams.minArea = 10      # 最小像素面积 (根据图片分辨率调整，太小会被噪点干扰)
    blobParams.maxArea = 500000  # 最大像素面积 (允许大圆点)
    
    # 形状限制：放宽，防止侧拍时圆变椭圆导致识别失败
    blobParams.filterByCircularity = False 
    blobParams.filterByConvexity = False
    blobParams.filterByInertia = False 
    
    # 颜色：寻找黑色斑点(0)还是白色斑点(255)
    blobParams.filterByColor = True
    blobParams.blobColor = 0 # 寻找黑色圆点
    
    # 创建检测器
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)
    # ---------------------------------------------------------

    # 准备 3D 坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= circle_spacing 

    objpoints = [] 
    imgpoints = [] 

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not images:
        print(f"在 '{image_folder}' 未找到图片！")
        return

    print(f"--> 开始处理 {len(images)} 张图片...")
    print(f"--> 使用增强型检测参数 (忽略黑框/椭圆)")
    
    valid_images = []

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用自定义 detector
        ret, centers = cv2.findCirclesGrid(
            gray, 
            pattern_size, 
            flags=cv2.CALIB_CB_SYMMETRIC_GRID, 
            blobDetector=blobDetector
        )

        filename = os.path.basename(fname)
        if ret:
            objpoints.append(objp)
            imgpoints.append(centers)
            valid_images.append(img.shape[::-1]) # 记录分辨率
            print(f"  [成功] {filename}")
            
            # (可选) 如果你想看它识别了哪里，可以把这几行注释取消
            # debug_img = cv2.drawChessboardCorners(img.copy(), pattern_size, centers, ret)
            # cv2.imwrite(f"debug_{filename}", debug_img)
            
        else:
            print(f"  [失败] {filename} - 未找到圆点阵列")

    if len(objpoints) > 0:
        print(f"\n--> 正在进行标定计算 (基于 {len(objpoints)} 张有效图片)...")
        
        # 标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, valid_images[0], None, None)
        
        print("\n" + "="*50)
        print("        标定结果 & 距离测试")
        print("="*50)
        print(f"内参矩阵 mtx:\n{np.array2string(mtx, separator=', ')}")
        print(f"畸变系数 dist:\n{np.array2string(dist, separator=', ')}")
        print(f"重投影误差: {ret:.4f}")
        print("-" * 50)
        
        # --- 计算每张图的距离 (验证环节) ---
        print("\n[验证] 标定板在每张图中的计算距离:")
        for i, tvec in enumerate(tvecs):
            # tvec[2] 是 Z轴距离
            z_distance = tvec[2][0]
            print(f"  图片 {i+1}: Z = {z_distance:.2f} mm")
            
        return mtx, dist
    else:
        print("\n所有图片均识别失败。可能原因：")
        print("1. minArea 太大或太小 (当前设置: 10~500000)")
        print("2. 图像太暗，圆点和背景对比度不够")

if __name__ == "__main__":
    # 你的默认配置
    TARGET_DIR = r"D:\Other\Users\ENGINEER\Desktop\cursor-cursor-2d-to-2-5d-camera-2b90\image_folder"
    SPACING = 1.5 # mm
    ROWS = 7
    COLS = 7
    
    calibrate_and_measure(TARGET_DIR, (COLS, ROWS), SPACING)
