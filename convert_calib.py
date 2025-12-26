import numpy as np
import json
import math

# Input parameters from the image
f_mm = 26.9271
sx_um = 8.29135
sy_um = 8.3
cx = 2684.71
cy = 1979.3
kappa = 36.6694 # m^-2
width = 5472
height = 3648

# Pose parameters
tx_mm = -44.4442
ty_mm = 52.0128
tz_mm = 343.65
rot_x_deg = 3.69983
rot_y_deg = 1.75751
rot_z_deg = 358.724

# 1. Intrinsic Matrix (K)
sx_mm = sx_um / 1000.0
sy_mm = sy_um / 1000.0

fx = f_mm / sx_mm
fy = f_mm / sy_mm

K = [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
]

# 2. Distortion Coefficients (D)
f_m = f_mm / 1000.0
k_factor = kappa * (f_m**2)

# Taylor expansion approximation
# k1 corresponds to -kappa * f^2
k1 = -k_factor
k2 = k_factor**2
k3 = -(k_factor**3)
p1 = 0.0
p2 = 0.0

dist_coeffs = [k1, k2, p1, p2, k3]

# 3. Extrinsics (R, T)
tvec = [tx_mm, ty_mm, tz_mm]

rx_rad = np.deg2rad(rot_x_deg)
ry_rad = np.deg2rad(rot_y_deg)
rz_rad = np.deg2rad(rot_z_deg)

Rx = np.array([
    [1, 0, 0],
    [0, np.cos(rx_rad), -np.sin(rx_rad)],
    [0, np.sin(rx_rad), np.cos(rx_rad)]
])

Ry = np.array([
    [np.cos(ry_rad), 0, np.sin(ry_rad)],
    [0, 1, 0],
    [-np.sin(ry_rad), 0, np.cos(ry_rad)]
])

Rz = np.array([
    [np.cos(rz_rad), -np.sin(rz_rad), 0],
    [np.sin(rz_rad), np.cos(rz_rad), 0],
    [0, 0, 1]
])

# Halcon 'gba' corresponds to Z-Y-X Euler angles (extrinsic)
# R = Rz * Ry * Rx
R = Rz @ Ry @ Rx

def matrix_to_rodrigues(R):
    theta = math.acos((np.trace(R) - 1) / 2)
    if abs(theta) < 1e-6:
        return [0.0, 0.0, 0.0]
    
    factor = theta / (2 * math.sin(theta))
    rx = (R[2, 1] - R[1, 2]) * factor
    ry = (R[0, 2] - R[2, 0]) * factor
    rz = (R[1, 0] - R[0, 1]) * factor
    return [rx, ry, rz]

rvec = matrix_to_rodrigues(R)

output = {
    "camera_matrix": K,
    "dist_coeffs": dist_coeffs,
    "rvec": rvec,
    "tvec": tvec,
    "image_width": width,
    "image_height": height,
    "halcon_params": {
        "f_mm": f_mm,
        "kappa": kappa,
        "sx_um": sx_um,
        "sy_um": sy_um
    }
}

print(json.dumps(output, indent=4))
