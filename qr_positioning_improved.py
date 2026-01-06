"""
二维码定位系统
功能：根据二维码位置和角度，计算固定相对位置的物体坐标
假设：二维码与物体相对位置固定，当二维码移动或旋转时，重新计算物体位置
"""

import math

def getDrAndDl(pA, pB):
    """
    计算从点A（二维码）到点B（物体）的相对距离和角度
    
    参数:
        pA: [x, y, angle_deg] - 二维码坐标和角度（度）
        pB: [x, y] 或 [x, y, angle] - 物体坐标
    
    返回:
        dr: 相对角度（弧度）- 从二维码指向物体的角度，相对于二维码的方向
        dl: 距离
    """
    dx = pB[0] - pA[0]
    dy = pB[1] - pA[1]
    print(f"dx: {dx}, dy: {dy}")
    
    # 注意：原始代码使用 atan2(dx, dy)，这可能是因为：
    # 1. 使用图像坐标系（y轴向下）
    # 2. 角度定义方式不同（从y轴开始测量）
    # 标准用法是 atan2(dy, dx)，但这里保持原逻辑
    ddr1 = math.atan2(dx, dy)  # 从A到B的角度
    
    pointR = math.radians(pA[2])  # 二维码的旋转角度（弧度）
    ddr2 = ddr1 - pointR  # 相对于二维码方向的角度（修正：应该是减法）
    
    dl = math.sqrt(dx*dx + dy*dy)  # 距离
    
    print(f"从A到B的角度: {ddr1:.6f} 弧度 ({math.degrees(ddr1):.2f} 度)")
    print(f"二维码角度: {pointR:.6f} 弧度 ({pA[2]:.2f} 度)")
    print(f"相对角度: {ddr2:.6f} 弧度 ({math.degrees(ddr2):.2f} 度)")
    print(f"距离: {dl:.6f}")
    
    return ddr2, dl  # 返回相对角度和距离
 
def getPointB(pointB, dr, dl, c):
    """
    根据二维码的新位置，计算物体的新坐标
    
    参数:
        pointB: [x, y, angle_deg] - 新的二维码坐标和角度（度）
        dr: 相对角度（弧度）- 从二维码指向物体的角度
        dl: 距离 - 二维码到物体的距离
        c: 角度差（度）- 二维码旋转的角度差，用于补偿旋转
    
    返回:
        [x, y, angle] - 计算出的物体新坐标
    """
    pointR = math.radians(c)  # 角度差转换为弧度
    d = dr + pointR  # 最终角度 = 相对角度 + 旋转补偿
    
    print(f"角度差: {pointR:.6f} 弧度 ({c:.2f} 度)")
    print(f"最终角度: {d:.6f} 弧度 ({math.degrees(d):.2f} 度)")

    # 计算偏移量
    # 注意：这里 x 使用 sin，y 使用 cos，可能是因为：
    # 1. 角度 d 是从 y 轴正方向开始测量的
    # 2. 或者使用图像坐标系（y轴向下）
    l1 = dl * math.cos(d)  # y方向偏移（或图像坐标系中的行偏移）
    l2 = dl * math.sin(d)  # x方向偏移（或图像坐标系中的列偏移）

    print(f"偏移量: x={l2:.6f}, y={l1:.6f}")

    x1 = pointB[0] + l2
    y1 = pointB[1] + l1
    
    return [x1, y1, pointB[2]]


if __name__ == "__main__":
    # 测试用例
    print("=" * 60)
    print("二维码定位系统测试")
    print("=" * 60)
    
    # 原始位置：二维码和物体的坐标
    a = [546.727, 291.281, -2.130]   # 二维码坐标 [x, y, angle(度)]
    b = [403.19048982679544, 168.1912881500751]  # 物体坐标 [x, y]
    
    print("\n步骤1: 计算从二维码到物体的相对位置")
    print("-" * 60)
    dr, dl = getDrAndDl(a, b)
    
    print("\n步骤2: 根据新的二维码位置计算物体位置")
    print("-" * 60)
    # 新的二维码位置（用于重新定位）
    point = [546.727, 291.281, -2.130]   # 这里使用相同值作为测试
    c = a[2] - point[2]  # 角度差（如果二维码旋转了）
    
    result = getPointB(point, dr, dl, c)
    
    print("\n结果对比")
    print("-" * 60)
    print(f"计算出的物体坐标: [{result[0]:.6f}, {result[1]:.6f}, {result[2]:.2f}]")
    print(f"原始物体坐标:     [{b[0]:.6f}, {b[1]:.6f}]")
    print(f"误差: x={abs(result[0]-b[0]):.6f}, y={abs(result[1]-b[1]):.6f}")
    
    # 如果误差很小，说明计算正确
    error = math.sqrt((result[0]-b[0])**2 + (result[1]-b[1])**2)
    print(f"总误差: {error:.6f}")
    
    if error < 0.01:
        print("✓ 计算正确！误差在可接受范围内")
    else:
        print("⚠ 误差较大，可能需要检查坐标系定义或角度计算方式")
