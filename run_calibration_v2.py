import cv2
import numpy as np
import glob
import os
import argparse

def calibrate_and_measure(image_folder, pattern_size=(7, 7), circle_spacing=5.0):
    # ---------------------------------------------------------
    # 1. 优化后的极速参数 (针对清晰标定板)
    # ---------------------------------------------------------
    blobParams = cv2.SimpleBlobDetector_Params()
    
    # 面积过滤：根据你的图片，圆点应该挺明显的
    blobParams.filterByArea = True
    blobParams.minArea = 50      # 调大最小面积，忽略噪点
    blobParams.maxArea = 100000  # 稍微限制最大面积
    
    # 形状过滤：开启这些能极大提高速度！
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.7  # 必须比较圆
    
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.8    # 必须是凸形状
    
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.4 # 允许一定的椭圆度(侧拍)
    
    blobParams.filterByColor = True
    blobParams.blobColor = 0 # 寻找黑色圆点
    
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
    print(f"--> 标定板: {pattern_size}, 间距: {circle_spacing}mm")
    
    img_size = None 

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            h, w = gray.shape[:2]
            img_size = (w, h)

        # 这里的 findCirclesGrid 如果参数不对会很慢，现在加了过滤应该很快
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
            print(f"  [成功] {filename}")
        else:
            print(f"  [失败] {filename}")

    if len(objpoints) > 0:
        print(f"\n--> 正在进行标定计算 (基于 {len(objpoints)} 张有效图片)...")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        
        print("\n" + "="*50)
        print("        标定结果 & 距离测试")
        print("="*50)
        print(f"内参矩阵 mtx:\n{np.array2string(mtx, separator=', ')}")
        print(f"畸变系数 dist:\n{np.array2string(dist, separator=', ')}")
        print(f"重投影误差: {ret:.4f}")
        print("-" * 50)
        
        print("\n[验证] 标定板在每张图中的计算距离 (Z轴):")
        for i, tvec in enumerate(tvecs):
            z_distance = tvec[2][0]
            print(f"  图片 {i+1}: Z = {z_distance:.2f} mm")
            
        return mtx, dist
    else:
        print("\n错误：所有图片均识别失败。")

if __name__ == "__main__":
    TARGET_DIR = r"D:\Other\Users\ENGINEER\Desktop\cursor-cursor-2d-to-2-5d-camera-2b90\image_folder"
    
    # 你的参数
    SPACING = 5.0 # mm
    ROWS = 7
    COLS = 7
    
    calibrate_and_measure(TARGET_DIR, (COLS, ROWS), SPACING)
