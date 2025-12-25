import cv2
import numpy as np
import glob
import os

# --- 配置 ---
IMAGE_FOLDER = r"D:\Other\Users\ENGINEER\Desktop\cursor-cursor-2d-to-2-5d-camera-2b90\image_folder"
PATTERN_SIZE = (7, 7) # 列, 行

def debug_corners():
    # 配置斑点检测器
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.filterByArea = True
    blobParams.minArea = 10      
    blobParams.maxArea = 500000  
    blobParams.filterByColor = True
    blobParams.blobColor = 0 # 黑色圆点
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    images = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob.glob(os.path.join(IMAGE_FOLDER, "*.bmp"))
    
    if not images:
        print("没找到图")
        return

    print(f"正在检查 {len(images)} 张图片...")

    # 只检查第一张成功的图
    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, centers = cv2.findCirclesGrid(
            gray, 
            PATTERN_SIZE, 
            flags=cv2.CALIB_CB_SYMMETRIC_GRID, 
            blobDetector=blobDetector
        )

        if ret:
            print(f"✅ 在 {os.path.basename(fname)} 中找到了 {PATTERN_SIZE} 个点")
            print("正在保存可视化结果到 debug_view.jpg ...")
            
            # 画出来
            cv2.drawChessboardCorners(img, PATTERN_SIZE, centers, ret)
            
            # 保存到当前目录
            cv2.imwrite("debug_view.jpg", img)
            print("请打开 'debug_view.jpg' 查看它到底连了哪些点！")
            break
        else:
            print(f"❌ {os.path.basename(fname)} 识别失败")

if __name__ == "__main__":
    debug_corners()
