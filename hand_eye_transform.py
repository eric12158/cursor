import numpy as np
import math

class HandEyeTransform:
    """手眼标定坐标变换工具"""
    
    def __init__(self):
        # 手眼标定结果 (相机相对于法兰)
        self.T_cam_to_flange = None
        
    def euler_to_rotation_matrix(self, rx, ry, rz, degrees=True):
        """欧拉角转旋转矩阵 (XYZ顺规)"""
        if degrees:
            rx = math.radians(rx)
            ry = math.radians(ry)
            rz = math.radians(rz)
        
        # 绕X轴
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        
        # 绕Y轴
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        
        # 绕Z轴
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转 R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        return R
    
    def rotation_matrix_to_euler(self, R):
        """旋转矩阵转欧拉角 (XYZ顺规)"""
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return math.degrees(x), math.degrees(y), math.degrees(z)
    
    def make_transform_matrix(self, tx, ty, tz, rx, ry, rz, degrees=True):
        """构造4x4变换矩阵"""
        R = self.euler_to_rotation_matrix(rx, ry, rz, degrees)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T
    
    def load_hand_eye_calibration(self, tx, ty, tz, rx, ry, rz, degrees=True):
        """加载手眼标定结果 (相机->法兰)"""
        self.T_cam_to_flange = self.make_transform_matrix(tx, ty, tz, rx, ry, rz, degrees)
        print("=" * 60)
        print("手眼标定矩阵已加载 (T_Camera_to_Flange)")
        print(self.T_cam_to_flange)
        return self.T_cam_to_flange
    
    def qr_in_camera_to_flange(self, qr_tx, qr_ty, qr_tz, qr_rx, qr_ry, qr_rz, degrees=True):
        """
        将二维码在相机坐标系的位姿，转换到法兰坐标系
        
        参数:
            qr_tx, qr_ty, qr_tz: 二维码相对于相机的平移 (mm)
            qr_rx, qr_ry, qr_rz: 二维码相对于相机的旋转角度
        
        返回:
            二维码在法兰坐标系的位姿
        """
        if self.T_cam_to_flange is None:
            raise ValueError("请先加载手眼标定结果！")
        
        # 二维码在相机系的变换矩阵
        T_qr_to_cam = self.make_transform_matrix(qr_tx, qr_ty, qr_tz, qr_rx, qr_ry, qr_rz, degrees)
        
        # 坐标变换链: Flange <- Camera <- QR
        # T_qr_to_flange = T_cam_to_flange @ T_qr_to_cam
        T_qr_to_flange = self.T_cam_to_flange @ T_qr_to_cam
        
        # 提取位姿
        tx_flange = T_qr_to_flange[0, 3]
        ty_flange = T_qr_to_flange[1, 3]
        tz_flange = T_qr_to_flange[2, 3]
        
        rx_flange, ry_flange, rz_flange = self.rotation_matrix_to_euler(T_qr_to_flange[:3, :3])
        
        print("\n" + "=" * 60)
        print("二维码在法兰坐标系的位姿")
        print("=" * 60)
        print(f"平移 X: {tx_flange:.4f} mm")
        print(f"平移 Y: {ty_flange:.4f} mm")
        print(f"平移 Z: {tz_flange:.4f} mm")
        print(f"旋转 Rx: {rx_flange:.4f} deg")
        print(f"旋转 Ry: {ry_flange:.4f} deg")
        print(f"旋转 Rz: {rz_flange:.4f} deg")
        print("\n[变换矩阵]")
        print(T_qr_to_flange)
        
        return tx_flange, ty_flange, tz_flange, rx_flange, ry_flange, rz_flange
    
    def qr_in_camera_to_base(self, qr_tx, qr_ty, qr_tz, qr_rx, qr_ry, qr_rz, 
                              flange_tx, flange_ty, flange_tz, flange_rx, flange_ry, flange_rz, 
                              degrees=True):
        """
        将二维码在相机坐标系的位姿，转换到机械臂基座坐标系
        
        参数:
            qr_*: 二维码相对于相机的位姿
            flange_*: 当前法兰相对于基座的位姿 (从机械臂读取)
        
        返回:
            二维码在基座坐标系的位姿
        """
        if self.T_cam_to_flange is None:
            raise ValueError("请先加载手眼标定结果！")
        
        # 1. 二维码在相机系
        T_qr_to_cam = self.make_transform_matrix(qr_tx, qr_ty, qr_tz, qr_rx, qr_ry, qr_rz, degrees)
        
        # 2. 法兰在基座系
        T_flange_to_base = self.make_transform_matrix(flange_tx, flange_ty, flange_tz, 
                                                       flange_rx, flange_ry, flange_rz, degrees)
        
        # 3. 坐标变换链: Base <- Flange <- Camera <- QR
        # T_qr_to_base = T_flange_to_base @ T_cam_to_flange @ T_qr_to_cam
        T_qr_to_base = T_flange_to_base @ self.T_cam_to_flange @ T_qr_to_cam
        
        # 提取位姿
        tx_base = T_qr_to_base[0, 3]
        ty_base = T_qr_to_base[1, 3]
        tz_base = T_qr_to_base[2, 3]
        
        rx_base, ry_base, rz_base = self.rotation_matrix_to_euler(T_qr_to_base[:3, :3])
        
        print("\n" + "=" * 60)
        print("⭐ 二维码在机械臂基座坐标系的位姿 (最终结果)")
        print("=" * 60)
        print(f"平移 X: {tx_base:.4f} mm")
        print(f"平移 Y: {ty_base:.4f} mm")
        print(f"平移 Z: {tz_base:.4f} mm")
        print(f"旋转 Rx: {rx_base:.4f} deg")
        print(f"旋转 Ry: {ry_base:.4f} deg")
        print(f"旋转 Rz: {rz_base:.4f} deg")
        print("\n[变换矩阵]")
        print(T_qr_to_base)
        print("\n💡 用途：将此位姿发送给机械臂，即可让机械臂移动到二维码位置！")
        
        return tx_base, ty_base, tz_base, rx_base, ry_base, rz_base
    
    def generate_grasp_pose(self, qr_tx, qr_ty, qr_tz, qr_rx, qr_ry, qr_rz,
                           flange_tx, flange_ty, flange_tz, flange_rx, flange_ry, flange_rz,
                           offset_z=50, approach_angle=0, degrees=True):
        """
        生成抓取位姿 (在二维码上方offset_z mm处)
        
        参数:
            offset_z: 抓取高度偏移 (mm)，正值表示在二维码上方
            approach_angle: 接近角度偏移 (deg)
        """
        # 先计算二维码在基座系的位姿
        tx, ty, tz, rx, ry, rz = self.qr_in_camera_to_base(
            qr_tx, qr_ty, qr_tz, qr_rx, qr_ry, qr_rz,
            flange_tx, flange_ty, flange_tz, flange_rx, flange_ry, flange_rz, degrees
        )
        
        # 生成抓取位姿（在二维码上方）
        grasp_tx = tx
        grasp_ty = ty
        grasp_tz = tz + offset_z  # 上方offset_z mm
        grasp_rx = rx + approach_angle
        grasp_ry = ry
        grasp_rz = rz
        
        print("\n" + "=" * 60)
        print(f"🤖 推荐抓取位姿 (在二维码上方 {offset_z} mm)")
        print("=" * 60)
        print(f"平移 X: {grasp_tx:.4f} mm")
        print(f"平移 Y: {grasp_ty:.4f} mm")
        print(f"平移 Z: {grasp_tz:.4f} mm")
        print(f"旋转 Rx: {grasp_rx:.4f} deg")
        print(f"旋转 Ry: {grasp_ry:.4f} deg")
        print(f"旋转 Rz: {grasp_rz:.4f} deg")
        print("\n💡 抓取流程:")
        print("   1. 机械臂移动到此位姿 (二维码上方)")
        print("   2. 下降 Z 轴接近物体")
        print("   3. 闭合夹爪")
        print("   4. 提升并移动到目标位置")
        
        return grasp_tx, grasp_ty, grasp_tz, grasp_rx, grasp_ry, grasp_rz


# ===================== 使用示例 =====================

if __name__ == "__main__":
    
    print("\n" + "🚀" * 30)
    print("手眼标定坐标变换计算工具")
    print("🚀" * 30)
    
    # 创建工具实例
    transformer = HandEyeTransform()
    
    # ========== 第1步：加载手眼标定结果 ==========
    print("\n📋 步骤1: 加载手眼标定结果")
    transformer.load_hand_eye_calibration(
        tx=-21.0093,    # mm
        ty=-7.6736,     # mm
        tz=576.7124,    # mm
        rx=20.8621,     # deg
        ry=11.3074,     # deg
        rz=117.7986,    # deg
        degrees=True
    )
    
    # ========== 第2步：输入二维码检测结果 ==========
    print("\n📋 步骤2: 输入二维码在相机坐标系的位姿")
    qr_in_camera = {
        'tx': -15.35,   # mm (相机坐标系)
        'ty': 17.04,    # mm
        'tz': 304.30,   # mm
        'rx': 5.24,     # deg
        'ry': 1.95,     # deg
        'rz': -3.24     # deg
    }
    print(f"二维码位姿 (相机系): X={qr_in_camera['tx']:.2f}, Y={qr_in_camera['ty']:.2f}, Z={qr_in_camera['tz']:.2f}")
    print(f"                     Rx={qr_in_camera['rx']:.2f}°, Ry={qr_in_camera['ry']:.2f}°, Rz={qr_in_camera['rz']:.2f}°")
    
    # ========== 第3步：计算二维码在法兰系的位姿 ==========
    print("\n📋 步骤3: 计算二维码在法兰坐标系的位姿")
    qr_in_flange = transformer.qr_in_camera_to_flange(
        qr_tx=qr_in_camera['tx'],
        qr_ty=qr_in_camera['ty'],
        qr_tz=qr_in_camera['tz'],
        qr_rx=qr_in_camera['rx'],
        qr_ry=qr_in_camera['ry'],
        qr_rz=qr_in_camera['rz']
    )
    
    # ========== 第4步：计算二维码在基座系的位姿 ==========
    print("\n📋 步骤4: 计算二维码在机械臂基座坐标系的位姿")
    print("⚠️  需要从机械臂读取当前法兰位姿！")
    print("    示例：假设当前法兰位姿为...")
    
    # 示例：当前法兰在基座的位姿（需要从机械臂实时读取）
    current_flange_pose = {
        'tx': 300.0,    # mm (示例值，需替换为实际值)
        'ty': 100.0,    # mm
        'tz': 400.0,    # mm
        'rx': 180.0,    # deg
        'ry': 0.0,      # deg
        'rz': 90.0      # deg
    }
    
    qr_in_base = transformer.qr_in_camera_to_base(
        qr_tx=qr_in_camera['tx'],
        qr_ty=qr_in_camera['ty'],
        qr_tz=qr_in_camera['tz'],
        qr_rx=qr_in_camera['rx'],
        qr_ry=qr_in_camera['ry'],
        qr_rz=qr_in_camera['rz'],
        flange_tx=current_flange_pose['tx'],
        flange_ty=current_flange_pose['ty'],
        flange_tz=current_flange_pose['tz'],
        flange_rx=current_flange_pose['rx'],
        flange_ry=current_flange_pose['ry'],
        flange_rz=current_flange_pose['rz']
    )
    
    # ========== 第5步：生成抓取位姿 ==========
    print("\n📋 步骤5: 生成推荐抓取位姿")
    grasp_pose = transformer.generate_grasp_pose(
        qr_tx=qr_in_camera['tx'],
        qr_ty=qr_in_camera['ty'],
        qr_tz=qr_in_camera['tz'],
        qr_rx=qr_in_camera['rx'],
        qr_ry=qr_in_camera['ry'],
        qr_rz=qr_in_camera['rz'],
        flange_tx=current_flange_pose['tx'],
        flange_ty=current_flange_pose['ty'],
        flange_tz=current_flange_pose['tz'],
        flange_rx=current_flange_pose['rx'],
        flange_ry=current_flange_pose['ry'],
        flange_rz=current_flange_pose['rz'],
        offset_z=50  # 在二维码上方50mm
    )
    
    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("📊 数据流总结")
    print("=" * 60)
    print("数据流向:")
    print("  1️⃣  相机检测二维码 → 得到二维码在相机系的位姿")
    print("  2️⃣  手眼标定矩阵 → 相机到法兰的变换关系")
    print("  3️⃣  机械臂实时位姿 → 法兰到基座的变换关系")
    print("  4️⃣  坐标变换链计算 → 二维码在基座系的位姿")
    print("  5️⃣  生成抓取位姿 → 机械臂可执行的目标位置")
    
    print("\n💡 实际应用价值:")
    print("  ✅ 视觉引导抓取：机械臂知道目标物体的准确位置")
    print("  ✅ 自动化装配：精确定位零件位置")
    print("  ✅ 物料分拣：识别并抓取指定物体")
    print("  ✅ 质量检测：测量物体位姿偏差")
    
    print("\n🎯 下一步:")
    print("  1. 将二维码贴在需要抓取的物体上")
    print("  2. 从机械臂读取实时法兰位姿")
    print("  3. 运行此脚本计算目标位姿")
    print("  4. 发送目标位姿给机械臂执行抓取")
    
    print("\n" + "🚀" * 30 + "\n")
