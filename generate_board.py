import cv2
import numpy as np

# ================= 配置区域 (Configuration) =================

# 棋盘格的【方块】数量
# 注意：OpenCV标定通常使用的是"内角点"数量。
# 如果你想要生成一个标准的 9x6 标定板（指的是内角点），那么方块数量应该是 10x7。
# 这里的配置指的是具体的黑白方块数量。
BOARD_WIDTH_SQUARES = 11   # 水平方向方块数量 (列数)
BOARD_HEIGHT_SQUARES = 8   # 垂直方向方块数量 (行数)

# 每个方块的像素大小
SQUARE_SIZE_PIXELS = 100

# 边缘留白的像素大小 (白色边框)
MARGIN_PIXELS = 50

# 输出文件名
OUTPUT_FILENAME = "calibration_board.png"

# ==========================================================

def generate_calibration_board():
    """
    生成OpenCV棋盘格标定板图片
    """
    # 计算图像的总宽度和高度
    image_width = (BOARD_WIDTH_SQUARES * SQUARE_SIZE_PIXELS) + (2 * MARGIN_PIXELS)
    image_height = (BOARD_HEIGHT_SQUARES * SQUARE_SIZE_PIXELS) + (2 * MARGIN_PIXELS)

    # 创建一个白色背景的图像 (255 表示白色)
    image = np.ones((image_height, image_width), dtype=np.uint8) * 255

    print(f"正在生成标定板...")
    print(f"方块布局: {BOARD_WIDTH_SQUARES} x {BOARD_HEIGHT_SQUARES}")
    print(f"图像尺寸: {image_width} x {image_height} 像素")

    # 开始绘制黑色方块
    # 也就是当行号和列号之和为奇数（或偶数，取决于起始颜色）时绘制黑色
    for r in range(BOARD_HEIGHT_SQUARES):
        for c in range(BOARD_WIDTH_SQUARES):
            # 这里的逻辑决定了左上角第一个方块是黑还是白
            # (r + c) % 2 == 1 通常会产生左上角为白色的棋盘格（如果0,0是偶数）
            if (r + c) % 2 == 1:
                # 计算方块的左上角坐标 (x, y)
                # 注意：x对应列(c), y对应行(r)
                start_x = MARGIN_PIXELS + c * SQUARE_SIZE_PIXELS
                start_y = MARGIN_PIXELS + r * SQUARE_SIZE_PIXELS
                
                # 计算方块的右下角坐标
                end_x = start_x + SQUARE_SIZE_PIXELS
                end_y = start_y + SQUARE_SIZE_PIXELS

                # 绘制黑色矩形 (颜色值 0)
                # -1 表示填充
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), 0, -1)

    # 保存图像
    cv2.imwrite(OUTPUT_FILENAME, image)
    print(f"标定板已保存至: {OUTPUT_FILENAME}")
    
    # 提示内角点数量 (OpenCV标定函数 findChessboardCorners 需要这个参数)
    inner_corners_w = BOARD_WIDTH_SQUARES - 1
    inner_corners_h = BOARD_HEIGHT_SQUARES - 1
    print(f"\n[提示] 使用此标定板进行OpenCV标定时，patternSize 参数应设为: ({inner_corners_w}, {inner_corners_h})")

if __name__ == "__main__":
    generate_calibration_board()
