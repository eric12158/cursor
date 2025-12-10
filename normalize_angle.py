def normalize_angle(angle):
    """
    归一化角度值：
    - 如果角度是负数，将其转换为对应的正数（加90度）
    - 如果角度是正数，保持不变
    
    参数:
        angle: 输入的角度值（浮点数）
    
    返回:
        归一化后的角度值
    
    示例:
        normalize_angle(-86.41) -> 3.59
        normalize_angle(-43.4) -> 46.6
        normalize_angle(4.453) -> 4.453
        normalize_angle(43.543) -> 43.543
    """
    if angle < 0:
        return angle + 90
    else:
        return angle


# 测试函数
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        (-86.41, 3.59),
        (-43.4, 46.6),  # 注意：-43.4 + 90 = 46.6，不是46.4
        (4.453, 4.453),
        (43.543, 43.543),
    ]
    
    print("角度归一化测试：")
    print("-" * 40)
    for input_angle, expected in test_cases:
        result = normalize_angle(input_angle)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"{status} 输入: {input_angle:8.3f} -> 输出: {result:8.3f} (期望: {expected:8.3f})")
