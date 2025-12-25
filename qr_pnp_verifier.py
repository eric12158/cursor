import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math
import sys

# --- 1. 标定参数配置 (用户提供) ---

# [修正] 手动修正焦距，基于之前的验证结果
# 原始计算: f_pixel = 712.7 (基于 Halcon 输出的 f=5.9mm, Sx=8.3um)
# 实际验证: 距离 420mm 时，二维码占 243.1 像素，反推焦距应为 3403.4
# 原因分析: Halcon 的 Sx 极有可能填错了，真实 Sx 约为 1.74um (5.9mm / 3403 * 1000)
# 结论: 我们必须使用修正后的焦距才能得到正确的物理距离。

CORRECTED_F = 3403.4

# 内参矩阵 (Camera Matrix)
CAMERA_MATRIX = np.array([
    [CORRECTED_F,   0.0,          2725.88], # fx
    [0.0,           CORRECTED_F,  1817.03], # fy
    [0.0,           0.0,          1.0    ]
], dtype=np.float64)

# 畸变系数 (Distortion Coefficients)
# Halcon Kappa (division model) -> OpenCV k1 (approx: k1 = -Kappa * f^2)
DIST_COEFFS = np.array([0.003538, 0., 0., 0., 0.], dtype=np.float64)

# 旋转向量 (Rotation Vector) - 从Halcon Euler角转换而来
# 注意：这里我们只用内参和畸变，因为我们要计算的是二维码相对于相机的位姿，
# 而不是使用之前标定板的固定外参。PnP会算出新的rvec和tvec。

class CalibrationVerifier:
    def __init__(self):
        self.points = [] 
        self.img_original = None
        self.img_display = None
        self.scale_factor = 1.0
        
        # QR Code parameters
        self.qr_size = 30.0 # mm
        
        # Prepare 3D object points for PnP
        # Assuming QR code is flat on Z=0, centered or top-left corner
        # Standard QR detector usually returns corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.obj_points = np.array([
            [0, 0, 0],              # Top-Left
            [self.qr_size, 0, 0],   # Top-Right
            [self.qr_size, self.qr_size, 0], # Bottom-Right
            [0, self.qr_size, 0]    # Bottom-Left
        ], dtype=np.float64)

    def detect_and_calculate_pnp(self):
        if self.img_original is None:
            return

        # Initialize QR detector
        detector = cv2.QRCodeDetector()
        
        # Detect and decode
        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(self.img_original)
        
        # Prepare display image
        self.img_display = cv2.resize(self.img_original, None, fx=self.scale_factor, fy=self.scale_factor)
        
        if retval:
            print(f"检测到 {len(points)} 个二维码。")
            
            for i in range(len(points)):
                # Get the 4 corners of the i-th QR code
                img_points = points[i].reshape(4, 2)
                
                # Verify points are valid floats
                img_points = img_points.astype(np.float64)
                
                # Solve PnP
                # We want to find the rotation (rvec) and translation (tvec) 
                # that minimizes reprojection error from obj_points to img_points
                success, rvec, tvec = cv2.solvePnP(self.obj_points, img_points, CAMERA_MATRIX, DIST_COEFFS)
                
                if success:
                    # Calculate distance
                    dist_mm = np.linalg.norm(tvec)
                    
                    # Calculate Pixel Width for debugging
                    pixel_width = np.linalg.norm(img_points[0] - img_points[1])
                    
                    # Draw on display image
                    self.draw_result(img_points, rvec, tvec, dist_mm)
                    
                    # Print info
                    print(f"\n--- 二维码 #{i+1} ---")
                    print(f"内容: {decoded_info[i] if i < len(decoded_info) else 'Unknown'}")
                    print(f"检测到的像素宽度 (边长): {pixel_width:.1f} pixels")
                    print(f"当前计算距离: {dist_mm:.4f} mm")
                    print(f"相机坐标 (X, Y, Z): ({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})")

        else:
            print("未检测到二维码。请确保图片清晰且二维码完整。")
            # Fallback text
            cv2.putText(self.img_display, "No QR Code Detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
        cv2.imshow("QR PnP Verification", self.img_display)

    def draw_result(self, img_points, rvec, tvec, dist):
        # Draw corners
        # Need to scale points for display
        scaled_pts = (img_points * self.scale_factor).astype(np.int32)
        
        # Draw bounding box
        for j in range(4):
            p1 = tuple(scaled_pts[j])
            p2 = tuple(scaled_pts[(j+1)%4])
            cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
            
        # Draw axes
        axis_length = 20.0 # mm
        axis_points = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, -axis_length] # Z axis usually points away from camera, but here we define local obj coord.
                                 # Standard OpenCV camera: Z forward. 
                                 # If we want to draw Z pointing OUT of the QR code (normal), 
                                 # and we defined points as 0,0,0 ... 
                                 # Let's just draw standard XYZ local axes
        ], dtype=np.float64)
        
        # Project axis points
        img_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
        scaled_axis = (img_axis_points.reshape(-1, 2) * self.scale_factor).astype(np.int32)
        
        origin = tuple(scaled_axis[0])
        pt_x = tuple(scaled_axis[1])
        pt_y = tuple(scaled_axis[2])
        pt_z = tuple(scaled_axis[3]) # Note: this might point "into" the paper depending on coord sys
        
        # X: Red, Y: Green, Z: Blue
        cv2.line(self.img_display, origin, pt_x, (0, 0, 255), 2) 
        cv2.line(self.img_display, origin, pt_y, (0, 255, 0), 2)
        cv2.line(self.img_display, origin, pt_z, (255, 0, 0), 2)
        
        # Draw Distance Text
        center_x = int(np.mean(scaled_pts[:, 0]))
        center_y = int(np.mean(scaled_pts[:, 1]))
        
        text = f"Dist: {dist:.1f}mm"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Background
        cv2.rectangle(self.img_display, (center_x - 10, center_y - th - 10), (center_x + tw + 10, center_y + 10), (0,0,0), -1)
        cv2.putText(self.img_display, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def load_image(self):
        root = tk.Tk()
        root.withdraw() 
        root.attributes('-topmost', True)
        
        print("请选择一张包含二维码的图片...")
        file_path = filedialog.askopenfilename(
            title="选择二维码图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        
        root.destroy()
        
        if not file_path:
            return False
            
        print(f"正在加载: {file_path}")
        try:
            self.img_original = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        except:
            self.img_original = cv2.imread(file_path)
            
        if self.img_original is None:
            print("无法读取图片")
            return False
            
        h, w = self.img_original.shape[:2]
        max_dim = 1200
        self.scale_factor = max_dim / float(max(h, w)) if max(h, w) > max_dim else 1.0
        
        return True

    def run(self):
        if not self.load_image():
            return

        cv2.namedWindow("QR PnP Verification", cv2.WINDOW_NORMAL)
        
        # Run detection automatically
        self.detect_and_calculate_pnp()
        
        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verifier = CalibrationVerifier()
    verifier.run()
