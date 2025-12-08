import math

def getDrAndDl(pA, pB):
    dx = pB[0] - pA[0]
    dy = pB[1] - pA[1]
    print(dx,dy)
    ddr1 =  math.atan2(dx,dy)  
    pointR = math.radians (pA[2])
    ddr2 = ddr1 + pointR
    print(f"弧度: {ddr2},两者之间的角度:{ddr1}, 长度: {math.sqrt(dx*dx + dy*dy)} ")
 
def getPointB(pointB, dr, dl,c):

    pointR =math.radians (c)
    d = dr + (pointR)
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
    getDrAndDl(a,b)

    point =[546.727,291.281,-2.130]   #校验（二维码坐标）
    dl =  189.0867708645954 # A(二维码点)与B(放货点的) 距离

    dr = -2.3168326805365653 # A(二维码点)与B(放货点的) 弧度
    c=a[2]-point[2]
    result = getPointB(point, dr,dl,c)    
    print(f"坐标点: {result[0]},{result[1]}, {result[2]}")
