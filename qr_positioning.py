import math

def normalize_angle_deg(angle_deg):
    """
    将角度规范化到 [-180, 180] 度范围
    """
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg

def normalize_angle_rad(angle_rad):
    """
    将角度规范化到 [-π, π] 弧度范围
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
        角度差（度），范围 [-180, 180]
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

def getDrAndDl(pA, pB):
    dx = pB[0] - pA[0]
    dy = pB[1] - pA[1]
    print(dx,dy)
    ddr1 =  math.atan2(dx,dy)  
    
    # 规范化二维码角度
    normalized_angle = normalize_angle_deg(pA[2])
    pointR = math.radians(normalized_angle)
    
    ddr2 = ddr1 + pointR
    # 规范化相对角度
    ddr2 = normalize_angle_rad(ddr2)
    
    print(f"弧度: {ddr2},两者之间的角度:{ddr1}, 长度: {math.sqrt(dx*dx + dy*dy)} ")
    return ddr2, math.sqrt(dx*dx + dy*dy)  # 修复：添加返回值
 
def getPointB(pointB, dr, dl, c):
    # c 已经是规范化后的角度差，直接使用
    pointR = math.radians(c)
    
    d = dr + pointR
    # 规范化最终角度
    d = normalize_angle_rad(d)
    
    print(f"需要旋转的弧度 {pointR}, 最终弧度: {d}")

    l1= dl* math.cos(d)
    l2= dl* math.sin(d) 

    print(f"偏移量:{l1},{l2}")
    x1 = pointB[0] + l2
    y1 = pointB[1] + l1
    return [x1,y1,pointB[2]]

if __name__ == "__main__":

    a=[546.727,291.281,-2.130]   #二维码坐标
    b=[403.19048982679544,168.1912881500751]         #调试坐标
    dr, dl = getDrAndDl(a,b)  # 修复：使用返回值

    point =[546.727,291.281,-2.130]   #校验（二维码坐标）
    
    # 修复：使用角度规范化计算角度差，处理角度跨越±180°的情况
    c = angle_diff_deg(a[2], point[2])
    print(f"角度差: {a[2]}° - {point[2]}° = {c}°")
    
    result = getPointB(point, dr, dl, c)    
    print(f"坐标点: {result[0]},{result[1]}, {result[2]}")
