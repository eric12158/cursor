import cv2
import numpy as np
import glob
import os
import argparse

def calibrate_and_measure(image_folder, pattern_size=(7, 7), circle_spacing=1.5):
    # ---------------------------------------------------------
    # 1. 配置更强的斑点检测器
    # ---------------------------------------------------------
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.filterByArea = True
    blobParams.minArea = 10      
    blobParams.maxArea = 500000  
    blobParams.filterByCircularity = False 
    blobParams.filterByConvexity = False
    blobParams.filterByInertia = False 
    blobParams.filterByColor = True
    blobParams.blobColor = 0 # 黑色圆点
    
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
    
    # 记录图像尺寸 (width, height)
    img_size = None 

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 记录尺寸 (宽, 高)
        if img_size is None:
            h, w = gray.shape[:2]
            img_size = (w, h)

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
    # 你的默认配置
    TARGET_DIR = r"D:\Other\Users\ENGINEER\Desktop\cursor-cursor-2d-to-2-5d-camera-2b90\image_folder"
    
    # --- 修改这里：尝试将间距从 1.5 改为 1.6 ---
    # 如果你是 10mm 总长 7个点，那间距可能是 10/6 = 1.666
    # 这里我们根据 385 -> 410 的比例反推，填 1.6 试试
    SPACING = 1.6 
    
    ROWS = 7
    COLS = 7
    
    calibrate_and_measure(TARGET_DIR, (COLS, ROWS), SPACING)
