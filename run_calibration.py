import cv2
import numpy as np
import glob
import os
import argparse
import sys

def calibrate_camera_circle(image_folder, pattern_size=(7, 7), circle_spacing=1.5):
    """
    Args:
        image_folder: 图片文件夹路径
        pattern_size: (列数, 行数) -> 对应这张图是 (7, 7)
        circle_spacing: 两个圆心之间的实际距离 (单位: mm)
    """
    # 检查路径是否存在
    if not os.path.isdir(image_folder):
        print(f"错误: 路径 '{image_folder}' 不存在或不是一个目录。")
        print("请检查路径拼写，或确保文件夹已创建。")
        return

    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= circle_spacing # 乘以圆心距

    objpoints = [] # 3D点
    imgpoints = [] # 2D点

    # 支持 jpg, png, jpeg, bmp
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        search_path = os.path.join(image_folder, ext)
        images.extend(glob.glob(search_path))
    
    if not images:
        print(f"在 '{image_folder}' 下未找到任何图片 (jpg, png, bmp)。")
        return

    print(f"--> 找到 {len(images)} 张图片，开始检测圆点...")
    print(f"--> 标定板规格: {pattern_size[0]}x{pattern_size[1]}, 间距: {circle_spacing}mm")
    print(f"--> 图片文件夹: {image_folder}")
    
    found_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找圆点
        ret, centers = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

        filename = os.path.basename(fname)
        if ret == True:
            found_count += 1
            objpoints.append(objp)
            imgpoints.append(centers)
            print(f"  [OK] {filename}")
        else:
            print(f"  [FAIL] {filename} - 未能识别")

    if found_count > 0:
        print(f"\n成功在 {found_count} 张图片中检测到圆点。正在计算内参...")
        
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
            print("\n" + "="*50)
            print("        标定结果 (请复制保存)")
            print("="*50)
            print(f"平均重投影误差: {ret:.4f} (越小越好，通常<0.5)")
            print("\n1. 内参矩阵 (camera_matrix):")
            print("np.array(" + np.array2string(mtx, separator=', ') + ")")
            print("\n2. 畸变系数 (dist_coeffs):")
            print("np.array(" + np.array2string(dist, separator=', ') + ")")
            print("="*50)
            return mtx, dist
        except cv2.error as e:
            print(f"\n标定计算出错: {e}")
    else:
        print("\n错误：所有图片都未能检测到圆点！")
        print("请检查：1. 行列数是否填对 2. 图片是否清晰 3. 是否有反光干扰")

if __name__ == "__main__":
    # --- 用户配置区域 ---
    
    # 你的图片文件夹绝对路径 (注意前面加 r 防止转义)
    TARGET_DIR = r"D:\Other\Users\ENGINEER\Desktop\cursor-cursor-2d-to-2-5d-camera-2b90\image_folder"
    
    # 你的标定板参数 (请确认间距是否为1.5mm，若不是请修改)
    SPACING = 1.5   # mm
    ROWS = 7
    COLS = 7
    
    # ------------------
    
    # 优先使用命令行参数，如果没有则使用上面的默认配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=TARGET_DIR)
    parser.add_argument('--spacing', type=float, default=SPACING)
    parser.add_argument('--rows', type=int, default=ROWS)
    parser.add_argument('--cols', type=int, default=COLS)
    
    args = parser.parse_args()
    
    calibrate_camera_circle(args.dir, (args.cols, args.rows), args.spacing)
