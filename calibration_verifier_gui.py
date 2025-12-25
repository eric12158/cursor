import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math
import sys

# --- 1. 标定参数配置 (用户提供) ---
# 内参矩阵 (Camera Matrix)
CAMERA_MATRIX = np.array([
    [712.688,   0.0,      2725.88],
    [0.0,       712.671,  1817.03],
    [0.0,       0.0,      1.0    ]
], dtype=np.float64)

# 畸变系数 (Distortion Coefficients)
DIST_COEFFS = np.array([0.003538, 0., 0., 0., 0.], dtype=np.float64)

# 旋转向量 (Rotation Vector)
RVEC = np.array([0.004413, -0.000994, -1.308492], dtype=np.float64)

# 平移向量 (Translation Vector)
TVEC = np.array([-3.53697, -3.84724, 81.1027], dtype=np.float64)


class CalibrationVerifier:
    def __init__(self):
        self.points = [] # Store clicked points (original coordinates)
        self.img_original = None
        self.img_display = None
        self.scale_factor = 1.0
        
        # Pre-calculate rotation matrix and inverse homography for Z=0 plane
        self.setup_matrices()

    def setup_matrices(self):
        # Calculate Rotation Matrix
        R, _ = cv2.Rodrigues(RVEC)
        
        # Calculate Homography for Z=0 plane
        # P_cam = R * P_world + T
        # We assume Z_world = 0
        # Col 1 of R corresponds to X, Col 2 to Y. 
        # T is the translation.
        
        # H = K * [r1 r2 t]
        # Where r1 is first column of R, r2 is second column.
        
        # Construct the projection matrix for the plane Z=0
        # P_plane = [r1, r2, T]
        P_plane = np.hstack((R[:, 0:1], R[:, 1:2], TVEC.reshape(3, 1)))
        
        # H = K * P_plane
        self.H = CAMERA_MATRIX @ P_plane
        
        # Inverse H to map pixel -> world (X, Y)
        self.H_inv = np.linalg.inv(self.H)
        
        print("初始化完成：矩阵计算完毕")

    def pixel_to_world(self, u, v):
        """
        Convert pixel coordinates (u, v) to world coordinates (x, y) on Z=0 plane.
        """
        # Undistort point first? 
        # Ideally, we should undistort the point using cv2.undistortPoints
        # But given the distortion is small, we can try direct mapping or proper undistortion.
        # Let's use cv2.undistortPoints for accuracy.
        
        src = np.array([[[u, v]]], dtype=np.float64)
        
        # undistortPoints returns normalized coordinates if P is empty (default), 
        # or new pixel coordinates if P is provided.
        # Here we want to go to world. 
        # Let's stick to the Homography approach which assumes linear model (lens distortion removed or ignored).
        # Since dist_coeffs are provided, let's undistort the pixel first.
        
        undistorted_pt = cv2.undistortPoints(src, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX)
        u_prime, v_prime = undistorted_pt[0][0]
        
        # Now use Homography on undistorted pixel coordinate
        uv_point = np.array([u_prime, v_prime, 1.0])
        world_point_hom = self.H_inv @ uv_point
        
        world_x = world_point_hom[0] / world_point_hom[2]
        world_y = world_point_hom[1] / world_point_hom[2]
        
        return world_x, world_y

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinate to original image coordinate
            orig_x = x / self.scale_factor
            orig_y = y / self.scale_factor
            
            self.points.append((orig_x, orig_y))
            
            # Limit to last 2 points
            if len(self.points) > 2:
                self.points.pop(0)
            
            self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clear points on right click
            self.points = []
            self.update_display()

    def calculate_camera_distance(self, u, v):
        """
        Calculate the distance from the camera center to the point (u, v) on the Z=0 plane.
        """
        # 1. Get world coordinates on Z=0 plane
        wx, wy = self.pixel_to_world(u, v)
        wz = 0.0
        
        # 2. Convert world point to camera coordinates
        # P_cam = R * P_world + T
        P_world = np.array([[wx], [wy], [wz]], dtype=np.float64)
        R, _ = cv2.Rodrigues(RVEC)
        P_cam = R @ P_world + TVEC.reshape(3, 1)
        
        # 3. Calculate Euclidean distance from camera origin (0,0,0) to P_cam
        dist_cam = np.linalg.norm(P_cam)
        
        return dist_cam, P_cam.flatten()

    def update_display(self):
        if self.img_original is None:
            return

        # Start with a clean copy of the resized image
        self.img_display = cv2.resize(self.img_original, None, fx=self.scale_factor, fy=self.scale_factor)
        
        # Draw text info
        info_color = (0, 255, 0)
        cv2.putText(self.img_display, "Left Click: Select QR Code Center | Right Click: Clear", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)

        # Draw points and lines
        for i, pt in enumerate(self.points):
            # Convert back to display coords
            disp_x = int(pt[0] * self.scale_factor)
            disp_y = int(pt[1] * self.scale_factor)
            
            cv2.circle(self.img_display, (disp_x, disp_y), 5, (0, 0, 255), -1)
            
            # Calculate distance to camera
            dist_cam, p_cam = self.calculate_camera_distance(pt[0], pt[1])
            
            # Display info
            text_lines = [
                f"Dist to Cam: {dist_cam:.2f} mm",
                f"Cam Coords: ({p_cam[0]:.1f}, {p_cam[1]:.1f}, {p_cam[2]:.1f})"
            ]
            
            for j, line in enumerate(text_lines):
                y_offset = disp_y + 20 + (j * 20)
                # Background
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(self.img_display, (disp_x + 10, y_offset - th), (disp_x + 10 + tw, y_offset + 5), (0,0,0), -1)
                cv2.putText(self.img_display, line, (disp_x + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            print(f"点{i+1} - 像素:({pt[0]:.1f}, {pt[1]:.1f}) -> 相机距离: {dist_cam:.4f} mm | 相机坐标(XYZ): {p_cam}")

        cv2.imshow("Verification Tool", self.img_display)

    def load_image(self):
        root = tk.Tk()
        root.withdraw() # Hide the main window
        
        print("请选择一张图片...")
        file_path = filedialog.askopenfilename(
            title="选择用于验证的图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        root.destroy()
        
        if not file_path:
            print("未选择图片，程序退出。")
            return False
            
        print(f"正在加载图片: {file_path}")
        self.img_original = cv2.imread(file_path)
        
        if self.img_original is None:
            print("无法读取图片！")
            return False
            
        # Calculate scale factor to fit screen (e.g., max dim 1200)
        h, w = self.img_original.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            self.scale_factor = max_dim / float(max(h, w))
        else:
            self.scale_factor = 1.0
            
        print(f"图片尺寸: {w}x{h}, 显示缩放比例: {self.scale_factor:.3f}")
        return True

    def run(self):
        if not self.load_image():
            return

        cv2.namedWindow("Verification Tool", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Verification Tool", self.mouse_callback)
        
        self.update_display()
        
        print("\n=== 操作说明 ===")
        print("1. 左键点击: 选择测量点 (自动保留最后两个点)")
        print("2. 右键点击: 清除所有点")
        print("3. 按 'ESC' 或 'q': 退出程序")
        print("================")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC or q
                break
            
            # Check if window was closed manually
            if cv2.getWindowProperty("Verification Tool", cv2.WND_PROP_VISIBLE) < 1:
                break
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verifier = CalibrationVerifier()
    verifier.run()
