import numpy as np

def halcon_to_opencv(halcon_params):
    """
    将 Halcon 标定结果转换为 OpenCV 格式
    
    参数 halcon_params 字典包含:
    - f: 焦距 (mm)
    - sx: 单个像元宽 (um)
    - sy: 单个像元高 (um)
    - cx: 中心点列坐标 (px)
    - cy: 中心点行坐标 (px)
    - kappa: 畸变系数 (1/m^2)
    - image_width: 图像宽 (px)
    - image_height: 图像高 (px)
    """
    
    # 1. 提取参数
    f_mm = halcon_params['f']
    sx_mm = halcon_params['sx'] / 1000.0  # um 转 mm
    sy_mm = halcon_params['sy'] / 1000.0  # um 转 mm
    cx = halcon_params['cx']
    cy = halcon_params['cy']
    kappa = halcon_params['kappa']
    
    # 2. 计算内参矩阵 (Camera Matrix)
    # fx = f / sx
    # fy = f / sy
    fx = f_mm / sx_mm
    fy = f_mm / sy_mm
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 3. 计算畸变系数 (Distortion Coefficients)
    # Halcon Division Model Kappa (1/m^2) -> OpenCV k1
    # 转换公式: k1 = kappa * f^2 (注意 f 要换算成米)
    f_m = f_mm / 1000.0
    k1 = kappa * (f_m ** 2)
    
    # Halcon 的 Division 模型只对应 k1，其他通常为 0
    # OpenCV 顺序: [k1, k2, p1, p2, k3]
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    
    return camera_matrix, dist_coeffs

# --- 用户提供的 Halcon 数据 ---
params = {
    'f': 6.35831,       # 焦距 mm
    'sx': 8.30095,      # 像元宽 um
    'sy': 8.3,          # 像元高 um
    'kappa': -102.251,  # Kappa 1/m^2
    'cx': 2725.5,       # Cx 像素
    'cy': 1821.36,      # Cy 像素
    'image_width': 5472,
    'image_height': 3648
}

mtx, dist = halcon_to_opencv(params)

print("="*40)
print("Halcon -> OpenCV 转换结果")
print("="*40)
print("\n1. 内参矩阵 (camera_matrix):")
print("np.array(" + np.array2string(mtx, separator=', ') + ")")
print(f"\n   fx = {mtx[0,0]:.5f}")
print(f"   fy = {mtx[1,1]:.5f}")
print(f"   cx = {mtx[0,2]:.5f}")
print(f"   cy = {mtx[1,2]:.5f}")

print("\n2. 畸变系数 (dist_coeffs):")
print("np.array(" + np.array2string(dist, separator=', ') + ")")
print(f"\n   k1 = {dist[0]:.8f} (由 Kappa 转换)")
