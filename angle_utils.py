"""
角度工具函数
提供角度规范化功能，将角度规范化到标准范围
"""

import math


def normalize_angle_deg(angle_deg):
    """
    将角度规范化到 [-180, 180] 度范围
    
    参数:
        angle_deg: 角度（度），可以是任意值
    
    返回:
        规范化后的角度（度），范围 [-180, 180]
    
    示例:
        >>> normalize_angle_deg(181)
        -179
        >>> normalize_angle_deg(-181)
        179
        >>> normalize_angle_deg(360)
        0
        >>> normalize_angle_deg(179)
        179
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
        angle_rad: 角度（弧度），可以是任意值
    
    返回:
        规范化后的角度（弧度），范围 [-π, π]
    
    示例:
        >>> normalize_angle_rad(math.pi + 0.1)
        -3.041592653589793
        >>> normalize_angle_rad(-math.pi - 0.1)
        3.041592653589793
        >>> normalize_angle_rad(2 * math.pi)
        0.0
    """
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad


def angle_diff_deg(angle1_deg, angle2_deg):
    """
    计算两个角度之间的最短角度差（度）
    处理角度跨越±180°的情况，例如：179° 和 -179° 之间的差是 2°，而不是 358°
    
    参数:
        angle1_deg: 第一个角度（度）
        angle2_deg: 第二个角度（度）
    
    返回:
        角度差（度），范围 [-180, 180]，表示从 angle2 到 angle1 的最短旋转角度
    
    示例:
        >>> angle_diff_deg(179, -179)
        -2
        >>> angle_diff_deg(-179, 179)
        2
        >>> angle_diff_deg(10, 350)
        20
    """
    # 先规范化两个角度
    angle1 = normalize_angle_deg(angle1_deg)
    angle2 = normalize_angle_deg(angle2_deg)
    
    # 计算差值
    diff = angle1 - angle2
    
    # 规范化差值到 [-180, 180] 范围（取最短路径）
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    
    return diff


if __name__ == "__main__":
    # 测试代码
    print("测试 normalize_angle_deg:")
    print(f"181° -> {normalize_angle_deg(181)}°")
    print(f"-181° -> {normalize_angle_deg(-181)}°")
    print(f"360° -> {normalize_angle_deg(360)}°")
    print(f"540° -> {normalize_angle_deg(540)}°")
    
    print("\n测试 normalize_angle_rad:")
    print(f"π + 0.1 -> {normalize_angle_rad(math.pi + 0.1):.4f}")
    print(f"-π - 0.1 -> {normalize_angle_rad(-math.pi - 0.1):.4f}")
    print(f"2π -> {normalize_angle_rad(2 * math.pi):.4f}")
    
    print("\n测试 angle_diff_deg:")
    print(f"179° - (-179°) = {angle_diff_deg(179, -179)}°")
    print(f"-179° - 179° = {angle_diff_deg(-179, 179)}°")
    print(f"10° - 350° = {angle_diff_deg(10, 350)}°")
