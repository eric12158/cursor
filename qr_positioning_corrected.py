"""
二维码定位系统 - 修正版
功能：根据二维码位置和角度，计算固定相对位置的物体坐标
"""

import math

def getDrAndDl(pA, pB):
    """
    计算从点A（二维码）到点B（物体）的相对距离和角度
    
    参数:
        pA: [x, y, angle_deg] - 二维码坐标和角度（度）
        pB: [x, y] 或 [x, y, angle] - 物体坐标
    
    返回:
        dr: 相对角度（弧度）- 从二维码指向物体的角度
        dl: 距离
    """
    dx = pB[0] - pA[0]
    dy = pB[1] - pA[1]
    print(f"dx: {dx}, dy: {dy}")
    
    # 注意：使用 atan2(dx, dy) 可能是为了适应特定的坐标系定义
    # 标准用法是 atan2(dy, dx)，但这里保持原逻辑
    ddr1 = math.atan2(dx, dy)
    pointR = math.radians(pA[2])
    ddr2 = ddr1 + pointR
    
    dl = math.sqrt(dx*dx + dy*dy)
    print(f"弧度: {ddr2}, 两者之间的角度: {ddr1}, 长度: {dl}")
    
    return ddr2, dl  # 修复：添加返回值
 
def getPointB(pointB, dr, dl, c):
    """
    根据新的二维码位置，计算物体的新坐标
    
    参数:
        pointB: [x, y, angle_deg] - 新的二维码坐标和角度（度）
        dr: 相对角度（弧度）- 从二维码指向物体的角度
        dl: 距离 - 二维码到物体的距离
        c: 角度差（度）- 二维码旋转的角度差
    
    返回:
        [x, y, angle] - 计算出的物体新坐标
    """
    pointR = math.radians(c)
    d = dr + pointR
    print(f"需要旋转的弧度: {pointR}, 最终弧度: {d}")

    # 注意：这里 x 使用 sin，y 使用 cos，可能是为了适应特定的坐标系定义
    l1 = dl * math.cos(d)
    l2 = dl * math.sin(d)

    print(f"偏移量: {l1}, {l2}")
    x1 = pointB[0] + l2
    y1 = pointB[1] + l1
    return [x1, y1, pointB[2]]


if __name__ == "__main__":
    # 原始位置：二维码和物体的坐标
    a = [546.727, 291.281, -2.130]   # 二维码坐标
    b = [403.19048982679544, 168.1912881500751]  # 物体坐标
    
    # 修复：使用函数返回值，而不是手动输入
    dr, dl = getDrAndDl(a, b)
    
    # 新的二维码位置（用于重新定位）
    # 注意：实际使用时，point 应该是新的二维码位置
    point = [546.727, 291.281, -2.130]   # 校验（二维码坐标）
    c = a[2] - point[2]  # 角度差
    
    result = getPointB(point, dr, dl, c)
    print(f"坐标点: {result[0]}, {result[1]}, {result[2]}")
    
    # 验证：计算误差
    error_x = abs(result[0] - b[0])
    error_y = abs(result[1] - b[1])
    error_total = math.sqrt(error_x**2 + error_y**2)
    print(f"\n验证结果:")
    print(f"原始物体坐标: [{b[0]}, {b[1]}]")
    print(f"计算物体坐标: [{result[0]}, {result[1]}]")
    print(f"误差: x={error_x:.6f}, y={error_y:.6f}, 总误差={error_total:.6f}")
