import math

def getDrAndDl(pA, pB):
    """
    计算从点A到点B的距离和角度
    
    参数:
        pA: [x, y, angle] - 起点（二维码坐标）
        pB: [x, y] 或 [x, y, angle] - 终点（物体坐标）
    
    返回:
        dr: 从A到B的角度（弧度）
        dl: 从A到B的距离
    """
    dx = pB[0] - pA[0]
    dy = pB[1] - pA[1]
    print(f"dx: {dx}, dy: {dy}")
    
    # 修正：math.atan2 的标准用法是 atan2(y, x)，即 atan2(dy, dx)
    ddr1 = math.atan2(dy, dx)  # 从A到B的角度（相对于x轴正方向）
    
    pointR = math.radians(pA[2])  # 二维码的旋转角度（弧度）
    ddr2 = ddr1 - pointR  # 相对于二维码方向的角度
    
    dl = math.sqrt(dx*dx + dy*dy)  # 距离
    
    print(f"从A到B的角度（相对于x轴）: {ddr1} 弧度 ({math.degrees(ddr1)} 度)")
    print(f"二维码旋转角度: {pointR} 弧度 ({pA[2]} 度)")
    print(f"相对角度: {ddr2} 弧度 ({math.degrees(ddr2)} 度)")
    print(f"距离: {dl}")
    
    return ddr2, dl  # 返回相对角度和距离
 
def getPointB(pointB, dr, dl, c):
    """
    根据给定的点、角度、距离和旋转角度，计算新的点位置
    
    参数:
        pointB: [x, y, angle] - 基准点（二维码坐标）
        dr: 相对角度（弧度）
        dl: 距离
        c: 角度差（度）- 用于补偿二维码旋转
    
    返回:
        [x, y, angle] - 计算出的新点坐标
    """
    pointR = math.radians(c)  # 角度差转换为弧度
    d = dr + pointR  # 最终角度
    print(f"需要旋转的弧度: {pointR}, 最终弧度: {d}")
    
    # 计算偏移量
    # 注意：如果坐标系是图像坐标系（y轴向下），可能需要调整
    # 这里假设标准数学坐标系（y轴向上）
    l1 = dl * math.cos(d)  # x方向偏移
    l2 = dl * math.sin(d)  # y方向偏移
    
    print(f"偏移量: x={l1}, y={l2}")
    
    x1 = pointB[0] + l1
    y1 = pointB[1] + l2
    
    return [x1, y1, pointB[2]]


if __name__ == "__main__":
    # 测试用例
    a = [546.727, 291.281, -2.130]   # 二维码坐标 [x, y, angle(度)]
    b = [403.19048982679544, 168.1912881500751]  # 物体坐标 [x, y]
    
    print("=== 计算从二维码到物体的相对位置 ===")
    dr, dl = getDrAndDl(a, b)
    
    print("\n=== 根据相对位置计算新坐标 ===")
    point = [546.727, 291.281, -2.130]   # 新的二维码坐标（用于重新定位）
    c = a[2] - point[2]  # 角度差（如果二维码旋转了）
    
    result = getPointB(point, dr, dl, c)    
    print(f"\n计算出的坐标点: [{result[0]}, {result[1]}, {result[2]}]")
    print(f"原始物体坐标: [{b[0]}, {b[1]}]")
    print(f"误差: x={abs(result[0]-b[0]):.2f}, y={abs(result[1]-b[1]):.2f}")
