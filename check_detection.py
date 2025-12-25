import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
import math

# 注意：这里请填入你通过 calibration_gui.py 重新标定后得到的 NEW_CAMERA_MATRIX
# 不要用之前那个 12847 的，也不要用 Halcon 的，用你重新标定的结果
# 暂时用一个占位符，请你替换成正确的值
CAMERA_MATRIX = np.array([
    [4282.0, 0.0000, 2735.4775],
    [0.0000, 4299.0, 1823.5581],
    [0.0000, 0.0000, 1.0000]
], dtype=np.float64)

DIST_COEFFS = np.array([-1.73024818e+00,  3.58438784e+01,  3.28795144e-03,  1.02873492e-02,
 -3.27885199e+02], dtype=np.float64)

class CalibrationVerifier:
    def __init__(self):
        self.img_original = None
        self.img_display = None
        self.scale_factor = 1.0
        self.qr_size = 26.0 # 默认值，运行时会询问

    def detect_and_show(self):
        if self.img_original is None:
            return

        # 1. 检测二维码
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(self.img_original)
        
        # 准备显示图像
        h, w = self.img_original.shape[:2]
        max_dim = 1200
        self.scale_factor = max_dim / float(max(h, w)) if max(h, w) > max_dim else 1.0
        new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
        self.img_display = cv2.resize(self.img_original, (new_w, new_h))

        if retval:
            print(f"检测到 {len(points)} 个二维码。")
            
            for i in range(len(points)):
                # 获取角点 (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
                # 形状: (4, 2)
                img_points = points[i].reshape(4, 2).astype(np.float64)
                
                # --- 可视化：只画出这4个点 ---
                # 将坐标转换到显示缩放后的坐标系
                pts_display = (img_points * self.scale_factor).astype(np.int32)
                
                # 画框 (绿色)
                for j in range(4):
                    cv2.line(self.img_display, tuple(pts_display[j]), tuple(pts_display[(j+1)%4]), (0, 255, 0), 2)
                
                # 画点并标记顺序 (红色圆点 + 数字)
                # 这很重要，用于确认检测到的角点顺序是否正确 (TL->TR->BR->BL)
                labels = ["TL", "TR", "BR", "BL"]
                for j, pt in enumerate(pts_display):
                    cv2.circle(self.img_display, tuple(pt), 5, (0, 0, 255), -1)
                    cv2.putText(self.img_display, f"{j}:{labels[j]}", (pt[0]+10, pt[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # --- 计算 PnP ---
                # 定义真实世界的 3D 坐标 (Z=0 平面)
                obj_points = np.array([
                    [0, 0, 0],              # Top-Left
                    [self.qr_size, 0, 0],   # Top-Right
                    [self.qr_size, self.qr_size, 0], # Bottom-Right
                    [0, self.qr_size, 0]    # Bottom-Left
                ], dtype=np.float64)
                
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, CAMERA_MATRIX, DIST_COEFFS)
                
                if success:
                    dist_mm = np.linalg.norm(tvec)
                    
                    # 简单显示结果
                    center_x = int(np.mean(pts_display[:, 0]))
                    center_y = int(np.mean(pts_display[:, 1]))
                    
                    text = f"Dist: {dist_mm:.1f}mm"
                    cv2.putText(self.img_display, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    print(f"二维码 #{i+1}: 距离 = {dist_mm:.2f} mm")

        else:
            print("未检测到二维码。")
            cv2.putText(self.img_display, "No QR Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("QR Verification", self.img_display)

    def run(self):
        root = tk.Tk()
        root.withdraw()
        
        # 输入二维码尺寸
        self.qr_size = simpledialog.askfloat("设置", "请输入二维码真实边长 (mm):", initialvalue=26.0)
        if self.qr_size is None: return

        print("请选择图片...")
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if not file_path: return
        
        # 读取图片
        self.img_original = cv2.imread(file_path)
        if self.img_original is None:
            print("无法读取图片")
            return

        cv2.namedWindow("QR Verification", cv2.WINDOW_NORMAL)
        self.detect_and_show()
        
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verifier = CalibrationVerifier()
    verifier.run()
