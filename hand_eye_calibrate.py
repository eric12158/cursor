'''
Description: 
继matlab标定后，使用OpenCV进行手眼标定；
首先使用matlab camera calibrate 工具箱进行标定，保存标定文件cameraParams.mat；
version: 
Author: lxl
Date: 2024-05-10 16:14:13
LastEditors: Luoxl
LastEditTime: 2024-05-16 10:10:08
'''
from scipy.io import loadmat
import cv2
import numpy as np
from math import *

# 将MATLAB结构体转换为Python字典
def struct_to_dict(matlab_struct):
    fields = matlab_struct.dtype.names
    items = [(field, matlab_struct[field]) for field in fields]
    return {key: value[0] if len(value) == 1 else value for key, value in items}
 
#用于根据位姿计算变换矩阵
def pose_robot(x, y, z, Rx, Ry, Rz):

    thetaX = Rx #/ 180 * pi
    thetaY = Ry #/ 180 * pi
    thetaZ = Rz #/ 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[x], [y], [z]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    # RT1=np.linalg.inv(RT1)
    return RT1

#用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    # R = Rx@Ry@Rz
    return R

#读取末端pose文件
def read_end_pose(file_path):
    # 打开 .txt 文件
    with open(file_path, 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()
    # 创建一个空列表来存储数据
    data_list = []
    # 遍历每一行数据
    for line in lines:
        # 去除每行末尾的换行符，并去除首尾的空格
        line = line.strip()
        # 去除方括号 "[" 和 "]"
        line = line.replace('[', '').replace(']', '')
        # 使用逗号分割数据
        values = line.split(',')
        # 将字符串转换为浮点数，并添加到数据列表中
        data_list.append([float(value) for value in values])
    # print(data_list)
    return data_list
    # 打印读取的数据列表



# 设置 NumPy 的打印选项，禁用科学计数法
np.set_printoptions(suppress=True)
# 读取matlab标定文件
mat_file_path = 'cameraParams.mat'
mat_data = loadmat(mat_file_path)
# MATLAB结构体被加载为Python字典
matlab_struct = mat_data['data']
calibrate_dict = struct_to_dict(matlab_struct)

K = calibrate_dict['K'][0]
camera_parameters = [K[0][0], K[1][1], K[0][2], K[1][2]]
print(f"camera_parameters:{camera_parameters}")
distortion_parameters = [calibrate_dict['RadialDistortion'][0][0][0],calibrate_dict['RadialDistortion'][0][0][1],calibrate_dict['TangentialDistortion'][0][0][0],calibrate_dict['TangentialDistortion'][0][0][1], calibrate_dict['RadialDistortion'][0][0][2]] #calibrate_dict['RadialDistortion'][0][0][2]]
print(f"distortion_parameters:{distortion_parameters}")

# 写出相机内参到配置文件camera.ini
import configparser
# 创建一个 ConfigParser 对象
config = configparser.ConfigParser()
# 设置相机参数部分
config['camera_parameters'] = {
    'fx': str(camera_parameters[0]),
    'fy': str(camera_parameters[1]),
    'cx': str(camera_parameters[2]),
    'cy': str(camera_parameters[3])
}
# 设置畸变参数部分
config['distortion_parameters'] = {
    'k1': str(distortion_parameters[0]),
    'k2': str(distortion_parameters[1]),
    'p1': str(distortion_parameters[2]),
    'p2': str(distortion_parameters[3]),
    'k3': str(distortion_parameters[4])
}
# 将数据写入到 INI 文件
with open('camera.ini', 'w') as configfile:
    config.write(configfile)

RotationVectors = calibrate_dict['RotationVectors'][0]
# print(f"{RotationVectors}")
TranslationVectors = calibrate_dict['TranslationVectors'][0]
# print(f"{TranslationVectors}")

#计算board to cam 变换矩阵
R_all_chess_to_cam_1=[]
T_all_chess_to_cam_1=[]
for i in range(len(RotationVectors)):
    R_chess_to_cam, _ = cv2.Rodrigues(np.array(RotationVectors[i]))
    R_all_chess_to_cam_1.append(R_chess_to_cam)
    T_all_chess_to_cam_1.append(np.array(TranslationVectors[i]))

#计算end to base变换矩阵
R_all_end_to_base_1=[]
T_all_end_to_base_1=[]    
end_pose = read_end_pose('images/pos.txt')

for pose in end_pose:
    RT = pose_robot(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
    R_all_end_to_base_1.append(RT[:3, :3])
    T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))

R,T=cv2.calibrateHandEye(R_all_end_to_base_1, T_all_end_to_base_1, R_all_chess_to_cam_1, T_all_chess_to_cam_1)#手眼标定

print("手眼矩阵分解得到的旋转矩阵")
print(R)
print("\n")


print("手眼矩阵分解得到的平移矩阵")
print(T)

RT=np.column_stack((R,T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))#即为cam to end变换矩阵
print("\n")
print('相机相对于末端的变换矩阵为：')
print(RT)
# 指定保存的文件路径
file_path = "cam2end.txt"

# 将矩阵写入文本文件
with open(file_path, 'w') as file:
    for row in RT:
        row_str = ' '.join(map(str, row))  # 将矩阵的每一行转换为字符串
        file.write(row_str + '\n')  # 写入文件并添加换行符


#结果验证，原则上来说，每次结果相差较小
for i in range(len(RotationVectors)):

    # 得到机械手末端到基座的变换矩阵，通过机械手末端到基座的旋转矩阵与平移向量先按列合并，然后按行合并形成变换矩阵格式
    RT_end_to_base=np.column_stack((R_all_end_to_base_1[i],T_all_end_to_base_1[i]))
    RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    # print(RT_end_to_base)

    # 标定版相对于相机的齐次矩阵
    RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1[i],T_all_chess_to_cam_1[i]))
    RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    # print(RT_chess_to_cam)

    # 手眼标定变换矩阵
    RT_cam_to_end=np.column_stack((R,T))
    RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
    # print(RT_cam_to_end)

    # 即为固定的棋盘格相对于机器人基坐标系位姿
    RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam
    RT_chess_to_base=np.linalg.inv(RT_chess_to_base)
    print('第',i,'次')

    print(f"{RT_chess_to_base[:3,:]}")
    print('')
