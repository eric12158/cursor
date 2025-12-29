import cv2
import numpy as np

# 1. 二维码真实物理尺寸 (mm)
QR_SIZE = 50.0

# 2. 二维码的物理坐标（世界坐标系）
# 用户提供的当前使用的坐标系
obj_points = np.array([[QR_SIZE,0,0], [0,0,0], [0,QR_SIZE,0], [QR_SIZE,QR_SIZE,0]], dtype=np.float32)

# 3. 相机内参矩阵
# 相机焦距
f = 8
wMachine = 12.8
hMachine = 9.6
width = 5472
height = 3648
# 传感器尺寸
fx = width / wMachine * 8
fy = height / hMachine * 8

K = np.array([[fx, 0, width/2], [0, fy, height/2], [0, 0, 1]], dtype=np.float32)

# 4. 相机畸变系数
dist_coeffs = np.array([0.1, -0.05, 0, 0, 0], dtype=np.float32)

def solve_pose_for_points(obj_pts, img_pts, camera_matrix, dist_coeffs):
    # 2. 去畸变矫正像素点
    img_points_undist = cv2.undistortPoints(img_pts, camera_matrix, dist_coeffs, P=camera_matrix)
    img_points_undist = img_points_undist.reshape(-1, 2)

    # 3. 解算单应性矩阵H
    H, _ = cv2.findHomography(obj_pts[:, 0:2], img_points_undist, cv2.RANSAC, 5.0)

    if H is None:
        return None, None, None

    # 4. 分解H得到旋转矩阵R和平移矩阵T
    _, Rs, Ts, _ = cv2.decomposeHomographyMat(H, camera_matrix)

    # 5. 选择合理的旋转矩阵（取Z轴正方向的）
    R = None
    T = None
    best_idx = -1
    for i in range(len(Rs)):
        if Ts[i][2] > 0:  # 二维码在相机前方
            R = Rs[i]
            T = Ts[i]
            best_idx = i
            break
    
    if R is None:
        # Fallback if none found positive (unlikely if detection is good)
        R = Rs[0]
        T = Ts[0]

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    # 6. 从旋转矩阵解算欧拉角
    pitch = np.arctan2(-R[2, 0], sy) * 180 / np.pi
    yaw = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
    roll = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi

    return pitch, yaw, roll

def calc_qr_angle(img):
    # 1. 识别二维码，提取4个顶点像素坐标
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(img)
    
    if not retval or points is None:
        return None

    img_points = points[0].astype(np.float32)
    
    # --- VISUALIZATION ---
    draw_points = img_points.astype(int)
    
    # Draw connections and indices
    for i, point in enumerate(draw_points):
        cv2.circle(img, tuple(point), 15, (0, 0, 255), -1)
        cv2.putText(img, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        next_point = draw_points[(i + 1) % 4]
        cv2.line(img, tuple(point), tuple(next_point), (0, 255, 0), 5)

    # Calculate for all 4 possible cyclic shifts (rotations)
    results = []
    
    # The detector usually returns points in a loop (e.g. TL, TR, BR, BL)
    # obj_points are fixed. We shift img_points to align with obj_points in different ways.
    
    for shift in range(4):
        # Roll the image points: shift=0 (0,1,2,3), shift=1 (3,0,1,2), etc.
        # Actually numpy roll shift=1 moves last to first. 
        # But logically we want to map:
        # Shift 0: img[0]->obj[0]
        # Shift 1: img[1]->obj[0] (effectively rotating the correspondence)
        
        shifted_img_points = np.roll(img_points, -shift, axis=0) # Negative shift to rotate "left"
        
        pitch, yaw, roll = solve_pose_for_points(obj_points, shifted_img_points, K, dist_coeffs)
        
        if pitch is not None:
            results.append({
                "shift": shift,
                "mapping": f"Img[{shift}]->Obj[0]",
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            })

    # Draw results on the left side
    y_start = 100
    x_start = 50
    line_height = 80
    
    # Draw a semi-transparent background for text
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (1400, 500), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    cv2.putText(img, "Pose Estimation Results (All Shifts):", (x_start, y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    for i, res in enumerate(results):
        text = f"Shift {res['shift']} ({res['mapping']}): P={res['pitch']:.1f}, Y={res['yaw']:.1f}, R={res['roll']:.1f}"
        color = (0, 255, 255) if res['shift'] == 0 else (200, 200, 200)
        cv2.putText(img, text, (x_start, y_start + (i + 1) * line_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)

    return results

if __name__ == '__main__':
    image_path = "./img/Image_20251229090820982.bmp"
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        # Dummy image
        img = np.zeros((3648, 5472, 3), dtype=np.uint8)
    else:
        results = calc_qr_angle(img)
        if results:
            print("Calculated poses for all shifts:")
            for res in results:
                print(f"Shift {res['shift']}: Pitch={res['pitch']:.2f}, Yaw={res['yaw']:.2f}, Roll={res['roll']:.2f}")
            
            cv2.imwrite("qr_marked_result.jpg", img)
            print("Saved marked image to qr_marked_result.jpg")
            
            # Display scaled down
            display_scale = 0.2
            disp_img = cv2.resize(img, None, fx=display_scale, fy=display_scale)
            # cv2.imshow not available in headless, rely on saved image
        else:
            print("Failed to detect QR code")
