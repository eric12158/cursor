"""
测试角度规范化功能
验证角度跨越±180°时的处理是否正确
"""

import math
from angle_utils import angle_diff_deg, normalize_angle_deg

print("=" * 60)
print("角度规范化测试")
print("=" * 60)

# 测试用例
test_cases = [
    # (angle1, angle2, description, expected_range)
    (179, -179, "从 -179° 到 179°", (-2, 2)),  # 应该得到 2° 或 -2°（绝对值都是2）
    (-179, 179, "从 179° 到 -179°", (-2, 2)),  # 应该得到 -2° 或 2°（绝对值都是2）
    (10, 350, "从 350° 到 10°", (20, 20)),     # 应该得到 20°
    (350, 10, "从 10° 到 350°", (-20, -20)),   # 应该得到 -20°
    (0, 360, "从 360° 到 0°", (0, 0)),         # 应该得到 0°
    (180, -180, "从 -180° 到 180°", (0, 0)),   # 应该得到 0°
    (5, 355, "从 355° 到 5°", (10, 10)),       # 应该得到 10°
    (355, 5, "从 5° 到 355°", (-10, -10)),     # 应该得到 -10°
    (90, 270, "从 270° 到 90°", (-180, 180)),  # 应该得到 180° 或 -180°
]

print("\n测试角度差计算（angle_diff_deg）:")
print("-" * 60)
for angle1, angle2, desc, expected_range in test_cases:
    diff = angle_diff_deg(angle1, angle2)
    abs_diff = abs(diff)
    expected_min, expected_max = expected_range
    expected_abs = abs(expected_min) if expected_min == expected_max else min(abs(expected_min), abs(expected_max))
    
    # 对于跨越180°的情况，只要绝对值正确即可
    if expected_min != expected_max:
        status = "✓" if abs_diff == expected_abs else "✗"
        print(f"{status} {desc}: {angle1}° - {angle2}° = {diff}° (期望绝对值: {expected_abs}°)")
    else:
        status = "✓" if abs(diff - expected_min) < 0.01 else "✗"
        print(f"{status} {desc}: {angle1}° - {angle2}° = {diff}° (期望: {expected_min}°)")

print("\n测试角度规范化（normalize_angle_deg）:")
print("-" * 60)
normalize_tests = [
    (179, 179),
    (-179, -179),
    (181, -179),
    (-181, 179),
    (360, 0),
    (-360, 0),
    (540, -180),
    (-540, 180),
]

for angle, expected in normalize_tests:
    normalized = normalize_angle_deg(angle)
    status = "✓" if abs(normalized - expected) < 0.01 else "✗"
    print(f"{status} {angle}° -> {normalized}° (期望: {expected}°)")

print("\n实际应用场景测试:")
print("-" * 60)
# 模拟二维码角度从 179° 变成 -179° 的情况
qr_old = 179
qr_new = -179
angle_diff = angle_diff_deg(qr_new, qr_old)
print(f"二维码角度变化: {qr_old}° -> {qr_new}°")
print(f"角度差: {angle_diff}° (实际转动了 {abs(angle_diff)}°)")
print(f"说明: 虽然角度值从 179° 变成了 -179°，但实际只转动了 {abs(angle_diff)}°")
