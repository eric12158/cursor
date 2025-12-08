"""
二维码定位系统 - 角度规范化版本
功能：根据二维码位置和角度，计算固定相对位置的物体坐标
修复：添加角度规范化，处理角度跨越±180°的情况
"""

import math

def normalize_angle_deg(angle_deg):
    """
    将角度规范化到 [-180, 180] 度范围
    
    参数:
        angle_deg: 角度（度）
    
    返回:
        规范化后的角度（度），范围 [-180, 180]
    """
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg

def normalize_angle_rad(angle_rad):
    """
    将角度规范化到 [-π, π] 弧度范围
    
    参数:
        angle_rad: 角度（弧度）
    
    返回:
        规范化后的角度（弧度），范围 [-π, π]
    """
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad

def angle_diff_deg(angle1_deg, angle2_deg):
    """
    计算两个角度之间的最短角度差（度）
    例如：179° 和 -179° 之间的差是 2°，而不是 358°
    返回 angle1 - angle2 的最短角度差
    
    参数:
        angle1_deg: 第一个角度（度）
        angle2_deg: 第二个角度（度）
    
    返回:
        角度差（度），范围 [-180, 180]，表示从 angle2 到 angle1 的最短旋转角度
    """
    # 先规范化两个角度到 [-180, 180] 范围
    angle1 = normalize_angle_deg(angle1_deg)
    angle2 = normalize_angle_deg(angle2_deg)
    
    # 计算差值
    diff = angle1 - angle2
    
    # 规范化差值到 [-180, 180] 范围（取最短路径）
    # 如果差值绝对值大于180°，取相反方向（更短的路径）
    if diff > 180:
        diff = diff - 360
    elif diff < -180:
        diff = diff + 360
    
    # 特殊情况：如果差值是 -180 或 180，通常返回 180（而不是 -180）
    # 但这里保持原样，因为 -180 和 180 在数学上等价
    
    return diff

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
    ddr1 = math.atan2(dx, dy)
    
    # 规范化二维码角度
    normalized_angle = normalize_angle_deg(pA[2])
    pointR = math.radians(normalized_angle)
    
    ddr2 = ddr1 + pointR
    # 规范化相对角度
    ddr2 = normalize_angle_rad(ddr2)
    
    dl = math.sqrt(dx*dx + dy*dy)
    print(f"弧度: {ddr2}, 两者之间的角度: {ddr1}, 长度: {dl}")
    print(f"二维码角度: {normalized_angle}° (原始: {pA[2]}°)")
    
    return ddr2, dl
 
def getPointB(pointB, dr, dl, c):
    """
    根据新的二维码位置，计算物体的新坐标
    
    参数:
        pointB: [x, y, angle_deg] - 新的二维码坐标和角度（度）
        dr: 相对角度（弧度）- 从二维码指向物体的角度
        dl: 距离 - 二维码到物体的距离
        c: 角度差（度）- 二维码旋转的角度差（已规范化）
    
    返回:
        [x, y, angle] - 计算出的物体新坐标
    """
    # 规范化角度差
    normalized_c = normalize_angle_deg(c)
    pointR = math.radians(normalized_c)
    
    d = dr + pointR
    # 规范化最终角度
    d = normalize_angle_rad(d)
    
    print(f"角度差: {normalized_c}° (原始: {c}°)")
    print(f"需要旋转的弧度: {pointR}, 最终弧度: {d}")

    # 注意：这里 x 使用 sin，y 使用 cos，可能是为了适应特定的坐标系定义
    l1 = dl * math.cos(d)
    l2 = dl * math.sin(d)

    print(f"偏移量: {l1}, {l2}")
    x1 = pointB[0] + l2
    y1 = pointB[1] + l1
    return [x1, y1, pointB[2]]


if __name__ == "__main__":
    print("=" * 60)
    print("二维码定位系统 - 角度规范化版本")
    print("=" * 60)
    
    # 原始位置：二维码和物体的坐标
    a = [546.727, 291.281, -2.130]   # 二维码坐标
    b = [403.19048982679544, 168.1912881500751]  # 物体坐标
    
    print("\n步骤1: 计算从二维码到物体的相对位置")
    print("-" * 60)
    dr, dl = getDrAndDl(a, b)
    
    print("\n步骤2: 根据新的二维码位置计算物体位置")
    print("-" * 60)
    # 新的二维码位置（用于重新定位）
    point = [546.727, 291.281, -2.130]   # 校验（二维码坐标）
    
    # 修复：使用角度规范化计算角度差
    c = angle_diff_deg(a[2], point[2])
    print(f"角度差计算: {a[2]}° - {point[2]}° = {c}°")
    
    result = getPointB(point, dr, dl, c)
    print(f"\n坐标点: {result[0]}, {result[1]}, {result[2]}")
    
    # 验证
    error_x = abs(result[0] - b[0])
    error_y = abs(result[1] - b[1])
    error_total = math.sqrt(error_x**2 + error_y**2)
    print(f"\n验证结果:")
    print(f"原始物体坐标: [{b[0]}, {b[1]}]")
    print(f"计算物体坐标: [{result[0]}, {result[1]}]")
    print(f"误差: x={error_x:.6f}, y={error_y:.6f}, 总误差={error_total:.6f}")
    
    # 测试角度规范化功能
    print("\n" + "=" * 60)
    print("角度规范化测试")
    print("=" * 60)
    test_cases = [
        (179, -179, 2),      # 应该得到 2°，而不是 358°
        (-179, 179, -2),     # 应该得到 -2°，而不是 -358°
        (10, 350, 20),       # 应该得到 20°，而不是 -340°
        (350, 10, -20),      # 应该得到 -20°，而不是 340°
        (0, 360, 0),         # 应该得到 0°
        (180, -180, 0),      # 应该得到 0°
    ]
    
    for angle1, angle2, expected in test_cases:
        diff = angle_diff_deg(angle1, angle2)
        status = "✓" if abs(diff - expected) < 0.01 else "✗"
        print(f"{status} {angle1}° - {angle2}° = {diff}° (期望: {expected}°)")
