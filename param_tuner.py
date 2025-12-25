import cv2
import numpy as np
import glob
import os

# 你的图片路径
IMAGE_FOLDER = r"D:\Other\Users\ENGINEER\Desktop\cursor-cursor-2d-to-2-5d-camera-2b90\image_folder"
PATTERN_SIZE = (7, 7)

def nothing(x):
    pass

def tune_params():
    # 找一张图片
    images = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    if not images:
        print("未找到图片")
        return
    
    # 读取第一张图
    img_orig = cv2.imread(images[0])
    
    # 如果图片太大，缩小一点方便显示
    h, w = img_orig.shape[:2]
    scale = 1.0
    if w > 1200:
        scale = 1200 / w
        img_orig = cv2.resize(img_orig, None, fx=scale, fy=scale)
    
    cv2.namedWindow('Tuner', cv2.WINDOW_NORMAL)
    
    # 创建滑动条
    cv2.createTrackbar('Min Area', 'Tuner', 50, 2000, nothing) # 默认50
    cv2.createTrackbar('Max Area', 'Tuner', 5000, 20000, nothing)
    cv2.createTrackbar('Min Circ', 'Tuner', 10, 100, nothing) # 实际值 / 100
    cv2.createTrackbar('Min Conv', 'Tuner', 10, 100, nothing) # 实际值 / 100
    
    print("按 'q' 退出，按 's' 保存参数")

    while True:
        # 获取滑动条的值
        min_area = cv2.getTrackbarPos('Min Area', 'Tuner')
        max_area = cv2.getTrackbarPos('Max Area', 'Tuner')
        min_circ = cv2.getTrackbarPos('Min Circ', 'Tuner') / 100.0
        min_conv = cv2.getTrackbarPos('Min Conv', 'Tuner') / 100.0
        
        # 确保 max > min
        if max_area <= min_area: max_area = min_area + 1
        
        # 设置参数
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = float(min_area)
        params.maxArea = float(max_area)
        
        params.filterByCircularity = True
        params.minCircularity = min_circ
        
        params.filterByConvexity = True
        params.minConvexity = min_conv
        
        params.filterByInertia = False
        params.filterByColor = True
        params.blobColor = 0 # 找黑色
        
        detector = cv2.SimpleBlobDetector_create(params)
        
        # 检测
        keypoints = detector.detect(img_orig)
        
        # 绘制结果
        img_disp = cv2.drawKeypoints(img_orig, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # 尝试找 Grid
        gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        ret, centers = cv2.findCirclesGrid(gray, PATTERN_SIZE, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector)
        
        status_text = f"Blobs: {len(keypoints)}"
        color = (0, 0, 255)
        
        if ret:
            status_text += " | GRID FOUND! (Success)"
            color = (0, 255, 0)
            cv2.drawChessboardCorners(img_disp, PATTERN_SIZE, centers, ret)
        
        cv2.putText(img_disp, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Tuner', img_disp)
        
        k = cv2.waitKey(100)
        if k == ord('q'):
            break
        if k == ord('s'):
            print("\n" + "="*30)
            print("最佳参数如下 (请填入 run_calibration_v2.py):")
            print(f"blobParams.minArea = {min_area}")
            print(f"blobParams.maxArea = {max_area}")
            print(f"blobParams.minCircularity = {min_circ}")
            print(f"blobParams.minConvexity = {min_conv}")
            print("="*30)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    tune_params()
