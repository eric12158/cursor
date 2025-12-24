import cv2
import numpy as np
import glob
import os

def calibrate_camera_circle(image_folder, pattern_size=(7, 7), circle_spacing=10.0):
    """
    Args:
        pattern_size: (列数, 行数) -> 对应这张图是 (7, 7)
        circle_spacing: 两个圆心之间的实际距离 (单位: mm)
    """
    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ...
    # 对于对称圆网格，逻辑坐标和棋盘格一样
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= circle_spacing # 乘以圆心距

    objpoints = [] # 3D点
    imgpoints = [] # 2D点

    # 支持 jpg 和 png
    images = glob.glob(f'{image_folder}/*.jpg') + glob.glob(f'{image_folder}/*.png')
    
    if not images:
        print(f"在 {image_folder} 文件夹下未找到图片！")
        print("请把拍摄的标定板照片直接拖入左侧文件列表的 'calibration_images' 文件夹中。")
        return

    print(f"找到 {len(images)} 张图片，开始检测圆点...")
    found_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找圆点
        # cv2.CALIB_CB_SYMMETRIC_GRID 表示这是对称圆网格
        ret, centers = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

        if ret == True:
            found_count += 1
            objpoints.append(objp)
            imgpoints.append(centers)
            print(f"  [OK] {os.path.basename(fname)} - 检测成功")
        else:
            print(f"  [FAIL] {os.path.basename(fname)} - 未能检测到完整圆点阵列")

    if found_count > 0:
        print(f"\n成功在 {found_count} 张图片中检测到圆点阵列。正在计算内参...")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("\n======== 标定结果 (直接复制下面的值) ========")
        print(f"内参矩阵 (camera_matrix):\n{np.array2string(mtx, separator=', ')}")
        print(f"\n畸变系数 (dist_coeffs):\n{np.array2string(dist, separator=', ')}")
        print(f"\n平均重投影误差: {ret:.4f}")
        return mtx, dist
    else:
        print("\n所有图片都未能检测到圆点，请检查：")
        print("1. pattern_size 是否正确 (你的板子是 7x7)")
        print("2. 照片是否清晰，光照是否均匀")
        print("3. 圆点是否全部都在画面内")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 你的板子参数：7行7列
    PATTERN_SIZE = (7, 7) 
    
    # 这一步很关键：用尺子量两个圆心的距离(mm)，比如10mm或15mm，填在这里
    CIRCLE_SPACING = 10.0 
    
    FOLDER_PATH = "/workspace/calibration_images"
    
    calibrate_camera_circle(FOLDER_PATH, PATTERN_SIZE, CIRCLE_SPACING)
