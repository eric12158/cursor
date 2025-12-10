"""
角度归一化算法的多种实现方式
"""

def normalize_angle_v1(angle):
    """
    版本1：负数加90度，正数保持不变
    这是最符合您示例的版本
    """
    if angle < 0:
        return angle + 90
    else:
        return angle


def normalize_angle_v2(angle):
    """
    版本2：使用模运算，将角度归一化到0-90度范围
    适用于需要处理超出范围的角度
    """
    angle = angle % 90
    if angle < 0:
        angle += 90
    return angle


def normalize_angle_v3(angle):
    """
    版本3：如果角度在-90到0之间，转换为0-90之间
    如果角度已经是0-90，保持不变
    """
    if -90 <= angle < 0:
        return angle + 90
    elif angle < -90:
        # 处理小于-90的情况
        return normalize_angle_v3(angle + 90)
    else:
        return angle


def normalize_angle_v4(angle):
    """
    版本4：使用绝对值，然后从90减去
    这种方式：-86.41 -> 90 - 86.41 = 3.59
    """
    if angle < 0:
        return 90 - abs(angle)
    else:
        return angle


# 测试所有版本
if __name__ == "__main__":
    test_cases = [-86.41, -43.4, 4.453, 43.543, -90, 90, -180, 180]
    
    print("角度归一化算法对比测试：")
    print("=" * 70)
    print(f"{'输入角度':<12} {'版本1':<12} {'版本2':<12} {'版本3':<12} {'版本4':<12}")
    print("-" * 70)
    
    for angle in test_cases:
        v1 = normalize_angle_v1(angle)
        v2 = normalize_angle_v2(angle)
        v3 = normalize_angle_v3(angle)
        v4 = normalize_angle_v4(angle)
        print(f"{angle:>11.3f}  {v1:>11.3f}  {v2:>11.3f}  {v3:>11.3f}  {v4:>11.3f}")
