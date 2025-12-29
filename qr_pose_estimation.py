import cv2
import numpy as np

# 1. 二维码真实物理尺寸 (mm)
QR_SIZE = 50.0
# 2. 二维码的物理坐标（世界坐标系，左上、右上、右下、左下）
# obj_points = np.array([[0,0,0], [QR_SIZE,0,0], [QR_SIZE,QR_SIZE,0], [0,QR_SIZE,0]], dtype=np.float32)
obj_points = np.array([[QR_SIZE,0,0], [0,0,0], [0,QR_SIZE,0], [QR_SIZE,QR_SIZE,0]], dtype=np.float32)
# 2. 二维码的物理坐标（世界坐标系，左下、左上、右上、右下）
# obj_points = np.array([[0,QR_SIZE,0], [0,0,0], [QR_SIZE,0,0], [QR_SIZE,QR_SIZE,0]], dtype=np.float32)


# 3. 相机内参矩阵（替换成你的标定结果）
# K = np.array([[712.70, 0,      2725.88], [0,      712.67, 1817.03], [0, 0, 1]], dtype=np.float32)
# K = np.array([[3333.33, 0,      2725.88], [0,      3333.33, 1817.03], [0, 0, 1]], dtype=np.float32)
# K = np.array([[6095.23, 0,      2725.88], [0,      5495.01, 1817.03], [0, 0, 1]], dtype=np.float32)
# K = np.array([[2310.23, 0,      2725.88], [0,      2310.97, 1817.03], [0, 0, 1]], dtype=np.float32)

# 相机焦距
f = 8
wMachine = 12.8
hMachine = 9.6
width = 5472
height = 3648
# 传感器尺寸  fx = 宽度 / 对应传感器物理宽度 * 焦距   fy = 高度 / 对应传感器物理高度 * 焦距
fx = width / wMachine * 8
fy = height / hMachine * 8

K = np.array([[fx, 0,      width/2], [0,      fy, height/2], [0, 0, 1]], dtype=np.float32)
# K = np.array([[3420, 0,      2736], [0,      3040, 1824], [0, 0, 1]], dtype=np.float32)
# 4. 相机畸变系数（替换成你的标定结果，一般5个参数）
dist_coeffs = np.array([0.1, -0.05, 0, 0, 0], dtype=np.float32)


def calc_qr_angle(img):
    # 1. 识别二维码，提取4个顶点像素坐标
    qr_detector = cv2.QRCodeDetector()
    # retval, points, _ = qr_detector.detectAndDecode(img)
    retval,decoded_info, points, _ = qr_detector.detectAndDecodeMulti(img)
    if not retval or points is None:
        return None, None, None

    img_points = points[0].astype(np.float32)
    
    # 绘制4个顶点
    # Convert to integer for drawing
    draw_points = img_points.astype(int)
    for i, point in enumerate(draw_points):
        # Draw circle at each corner (Red)
        cv2.circle(img, tuple(point), 15, (0, 0, 255), -1)
        # Draw index number
        cv2.putText(img, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        # Draw lines connecting the points (Green)
        next_point = draw_points[(i + 1) % 4]
        cv2.line(img, tuple(point), tuple(next_point), (0, 255, 0), 5)

    # 2. 去畸变矫正像素点
    img_points_undist = cv2.undistortPoints(img_points, K, dist_coeffs, P=K)
    img_points_undist = img_points_undist.reshape(-1, 2)

    # 3. 解算单应性矩阵H
    H, _ = cv2.findHomography(obj_points[:, 0:2], img_points_undist, cv2.RANSAC, 5.0)

    # 4. 分解H得到旋转矩阵R和平移矩阵T
    _, Rs, Ts, _ = cv2.decomposeHomographyMat(H, K)

    # 5. 选择合理的旋转矩阵（取Z轴正方向的）
    R = None
    T = None
    for i in range(len(Rs)):
        if Ts[i][2] > 0:  # 二维码在相机前方
            R = Rs[i]
            T = Ts[i]
            break
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    # 6. 从旋转矩阵解算欧拉角（俯仰角、偏航角、滚转角）
    pitch = np.arctan2(-R[2, 0],sy) * 180 / np.pi
    yaw = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
    roll = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi

    return pitch, yaw, roll

if __name__ == '__main__':
    # Make sure the image path is correct or handle error
    image_path = "./img/Image_20251229090820982.bmp"
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        # Create a dummy image for testing logic if file doesn't exist
        print("Creating dummy image with QR code for testing...")
        img = np.zeros((3648, 5472, 3), dtype=np.uint8)
        # This dummy won't have a real QR so detection will fail, but script will run.
        # Ideally we stop here.
    else:
        pitch, yaw, roll = calc_qr_angle(img)
        if pitch is not None:
            print(f"俯仰角: {pitch:.2f}°  偏航角: {yaw:.2f}°  滚转角: {roll:.2f}°")
            
            # Resize for display if image is too large
            display_scale = 0.2
            disp_img = cv2.resize(img, None, fx=display_scale, fy=display_scale)
            
            cv2.imshow("QR Detection", disp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save the marked image
            cv2.imwrite("qr_marked_result.jpg", img)
            print("Saved marked image to qr_marked_result.jpg")
        else:
            print("Failed to detect QR code")
