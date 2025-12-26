import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import json

# 原始标定参数 (OpenCV结果)
ORIGINAL_CAMERA_MATRIX = np.array([
    [12847.6168, 0.0000, 2735.4775],
    [0.0000, 12898.0137, 1823.5581],
    [0.0000, 0.0000, 1.0000]
], dtype=np.float64)

ORIGINAL_DIST_COEFFS = np.array([-1.73024818e+00,  3.58438784e+01,  3.28795144e-03,  1.02873492e-02,
 -3.27885199e+02], dtype=np.float64)

def solve_pnp_distance(img_points, obj_points, camera_matrix, dist_coeffs):
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    if not success:
        return None, None, None
    return np.linalg.norm(tvec), rvec, tvec

def run_verification():
    root = tk.Tk()
    root.withdraw()

    # 1. 选择图片
    print("请选择包含二维码的图片...")
    img_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Images", "*.jpg *.png *.bmp")])
    if not img_path:
        return

    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片")
        return

    # 2. 检测二维码
    detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(img)
    
    if not retval:
        messagebox.showerror("错误", "未检测到二维码！请使用清晰的图片。")
        return

    # 假设使用第一个检测到的二维码
    qr_idx = 0
    img_points = points[qr_idx].reshape(4, 2).astype(np.float64)
    
    # 3. 输入二维码真实边长
    qr_size = simpledialog.askfloat("输入参数", "请输入二维码的实际边长 (mm):", initialvalue=26.0)
    if qr_size is None: return

    obj_points = np.array([
        [0, 0, 0],
        [qr_size, 0, 0],
        [qr_size, qr_size, 0],
        [0, qr_size, 0]
    ], dtype=np.float64)

    # 4. 使用当前参数计算距离
    calc_dist, rvec, tvec = solve_pnp_distance(img_points, obj_points, ORIGINAL_CAMERA_MATRIX, ORIGINAL_DIST_COEFFS)
    
    print("-" * 50)
    print(f"当前标定参数计算出的距离: {calc_dist:.2f} mm")
    print("-" * 50)

    # 5. 输入真实距离以进行校正
    real_dist = simpledialog.askfloat("校正", f"当前计算距离为 {calc_dist:.2f} mm。\n请输入相机到二维码中心的 **真实测量距离** (mm):", initialvalue=calc_dist)
    
    if real_dist is None: return

    # 6. 计算修正系数
    # 原理: Z approx f * (RealSize / ImageSize)
    # 所以 f_new / f_old = Z_real / Z_calc
    correction_factor = real_dist / calc_dist
    
    new_camera_matrix = ORIGINAL_CAMERA_MATRIX.copy()
    new_camera_matrix[0, 0] *= correction_factor # fx
    new_camera_matrix[1, 1] *= correction_factor # fy
    
    print("\n" + "=" * 50)
    print("校正分析报告")
    print("=" * 50)
    print(f"目标真实距离: {real_dist:.2f} mm")
    print(f"原始计算距离: {calc_dist:.2f} mm")
    print(f"误差比例: {calc_dist/real_dist:.2f}倍 (OpenCV结果偏大说明标定时输入的格子尺寸偏小)")
    print(f"建议修正系数: {correction_factor:.6f}")
    
    print("\n修正后的内参矩阵 (复制并在代码中使用):")
    print("-" * 20)
    code = "NEW_CAMERA_MATRIX = np.array([\n"
    code += f"    [{new_camera_matrix[0,0]:.4f}, 0.0000, {new_camera_matrix[0,2]:.4f}],\n"
    code += f"    [0.0000, {new_camera_matrix[1,1]:.4f}, {new_camera_matrix[1,2]:.4f}],\n"
    code += f"    [0.0000, 0.0000, 1.0000]\n"
    code += "], dtype=np.float64)"
    print(code)
    print("-" * 20)
    
    # 7. 验证修正后的结果
    new_calc_dist, _, _ = solve_pnp_distance(img_points, obj_points, new_camera_matrix, ORIGINAL_DIST_COEFFS)
    print(f"验证 - 使用新矩阵计算的距离: {new_calc_dist:.2f} mm (应接近 {real_dist:.2f})")
    
    # 8. 可视化
    h, w = img.shape[:2]
    scale = 1000.0 / max(h, w)
    disp_img = cv2.resize(img, (int(w*scale), int(h*scale)))
    
    cv2.putText(disp_img, f"Old Dist: {calc_dist:.1f}mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(disp_img, f"Real Dist: {real_dist:.1f}mm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(disp_img, f"Factor: {correction_factor:.4f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv2.imshow("Calibration Correction", disp_img)
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_verification()
