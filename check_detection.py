import cv2
import numpy as np
import glob
import os

# 你的路径
IMAGE_FOLDER = r"D:\Other\Users\ENGINEER\Desktop\cursor-cursor-2d-to-2-5d-camera-2b90\image_folder"
PATTERN_SIZE = (7, 7)

def check_detection():
    # 使用你刚才的参数
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.filterByArea = True
    blobParams.minArea = 50     
    blobParams.maxArea = 20000  # 稍微放宽一点上限
    blobParams.filterByCircularity = True 
    blobParams.minCircularity = 0.1 # 放宽一点圆度，防止侧拍识别不到
    blobParams.filterByConvexity = False
    blobParams.filterByInertia = False 
    blobParams.filterByColor = True
    blobParams.blobColor = 0
    
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    images = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    if not images:
        print("没找到图片")
        return

    print(f"正在检查 {len(images)} 张图片...")
    
    # 随便找一张处理
    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 找点
        ret, centers = cv2.findCirclesGrid(
            gray, 
            PATTERN_SIZE, 
            flags=cv2.CALIB_CB_SYMMETRIC_GRID, 
            blobDetector=blobDetector
        )

        filename = os.path.basename(fname)
        if ret:
            print(f"✅ {filename}: 成功检测到 {len(centers)} 个点")
            
            # --- 关键：把点画出来 ---
            # drawChessboardCorners 会画出连线，非常直观
            cv2.drawChessboardCorners(img, PATTERN_SIZE, centers, ret)
            
            # 另外画一个醒目的黄色矩形框住所有点，方便看范围
            pts = centers.reshape(-1, 2).astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            save_path = f"DEBUG_{filename}"
            cv2.imwrite(save_path, img)
            print(f"👉 已保存可视化结果: {save_path} (请务必打开查看！)")
            break # 只看一张就够了
        else:
            print(f"❌ {filename}: 未检测到")

if __name__ == "__main__":
    check_detection()
