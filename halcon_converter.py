import cv2
import numpy as np
import math

class HalconConverter:
    def __init__(self, sx_um, sy_um, f_mm, kappa_m2, cx, cy, img_w, img_h, pose_params):
        """
        Initialize with Halcon calibration results.
        
        Args:
            sx_um (float): Pixel width in microns
            sy_um (float): Pixel height in microns
            f_mm (float): Focal length in mm
            kappa_m2 (float): Radial distortion (kappa) in 1/m^2
            cx (float): Principal point X
            cy (float): Principal point Y
            img_w (int): Image width
            img_h (int): Image height
            pose_params (list): [Tx, Ty, Tz, RotX, RotY, RotZ] 
                                Translation in mm, Rotation in degrees.
                                Halcon standard order usually implies RotZ * RotY * RotX (gba) 
                                or similar depending on function. 
                                We assume standard Halcon 3D pose: 'gba' (Z-Y-X).
        """
        self.sx = sx_um / 1000.0  # convert to mm
        self.sy = sy_um / 1000.0  # convert to mm
        self.f = f_mm
        self.kappa_phys = kappa_m2 / 1e6  # convert 1/m^2 to 1/mm^2 (1 m^2 = 10^6 mm^2, so 1/m^2 = 10^-6 / mm^2)
        self.cx = cx
        self.cy = cy
        self.width = img_w
        self.height = img_h
        
        self.pose_t = np.array(pose_params[:3], dtype=np.float64) # Tx, Ty, Tz
        self.pose_r_deg = np.array(pose_params[3:], dtype=np.float64) # Rx, Ry, Rz (deg)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        
        self._convert()

    def _euler_to_rotation_matrix(self, rx, ry, rz, order='gba'):
        """
        Convert Euler angles (degrees) to Rotation Matrix.
        Halcon 'gba' order means: Rotate Z(gamma), then Y(beta), then X(alpha).
        R = Rz * Ry * Rx
        """
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        # Standard Halcon sequence 'gba' -> R = Rz * Ry * Rx
        # This applies Rx first (to vector), then Ry, then Rz.
        if order == 'gba':
            R = Rz @ Ry @ Rx
        elif order == 'abg': # x-y-z
            R = Rx @ Ry @ Rz
        else:
            raise ValueError(f"Unknown order: {order}")
            
        return R

    def _convert(self):
        # 1. Intrinsics (Camera Matrix)
        fx = self.f / self.sx
        fy = self.f / self.sy
        
        self.camera_matrix = np.array([
            [fx, 0, self.cx],
            [0, fy, self.cy],
            [0,  0,       1]
        ], dtype=np.float64)
        
        # 2. Distortion
        # Halcon Division Model: r_dist = r_undist / (1 + kappa * r_undist^2)
        # OpenCV Polynomial: r_dist = r_undist * (1 + k1 * r_undist^2 + ...)
        # Taylor: 1 / (1 + K*r^2) approx 1 - K*r^2
        # So k1_norm = - Kappa_norm
        # Kappa_norm = Kappa_phys * f^2
        
        kappa_norm = self.kappa_phys * (self.f ** 2)
        k1 = -kappa_norm
        
        # We assume k2=0, p1=0, p2=0, k3=0 for this approximation
        self.dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
        
        # 3. Extrinsics
        # Pose T is translation (mm)
        self.tvec = self.pose_t.reshape(3, 1)
        
        # Pose R (Euler) -> Matrix -> Rvec
        # Using 'gba' (Z-Y-X) which is standard for many industrial params including Halcon
        # Note: Check if your specific Halcon script used 'abg' or 'gba'. 
        # 'gba' (yaw-pitch-roll order) is most common.
        R = self._euler_to_rotation_matrix(
            self.pose_r_deg[0], 
            self.pose_r_deg[1], 
            self.pose_r_deg[2], 
            order='gba'
        )
        
        self.rvec, _ = cv2.Rodrigues(R)

    def print_params(self):
        print("=== OpenCV Parameters Converted from Halcon ===")
        print("# Copy these into your Python/OpenCV script:")
        print("import numpy as np")
        print(f"camera_matrix = np.array({np.array2string(self.camera_matrix, separator=', ')}, dtype=np.float64)")
        print(f"dist_coeffs = np.array({np.array2string(self.dist_coeffs, separator=', ')}, dtype=np.float64)")
        print(f"rvec = np.array({np.array2string(self.rvec.flatten(), separator=', ')}, dtype=np.float64)")
        print(f"tvec = np.array({np.array2string(self.tvec.flatten(), separator=', ')}, dtype=np.float64)")
        print("===============================================")

    def pixel_to_world(self, u, v, z_world=0):
        """
        Back-project a pixel (u, v) to the world plane at Z = z_world.
        
        Equation:
        s * [u, v, 1]^T = K * (R * [X, Y, Z]^T + T)
        
        We want X, Y given Z.
        Let P_cam = R * P_world + T
        [x_c, y_c, z_c]^T = R * [X, Y, Z]^T + T
        
        Also:
        z_c * [u, v, 1]^T = K * [x_c, y_c, z_c]^T
        inv(K) * [u, v, 1]^T * z_c = [x_c, y_c, z_c]^T
        
        Let UV_norm = inv(K) * [u, v, 1]^T  (Normalized coords)
        
        [x_c, y_c, z_c]^T = z_c * UV_norm
        
        So:
        z_c * UV_norm = R * [X, Y, Z]^T + T
        z_c * UV_norm - T = R * [X, Y, Z]^T
        inv(R) * (z_c * UV_norm - T) = [X, Y, Z]^T
        
        This is a system of equations with unknowns X, Y and z_c.
        But we assume Z (world) is known (e.g. 0).
        
        Let M = inv(R) * inv(K) * z_c  <-- No, z_c is unknown scaling factor.
        
        Alternative approach using homography for a plane:
        If Z=0, we can compute homography H.
        [u, v, 1]^T ~ K * [r1 r2 t] * [X, Y, 1]^T
        H = K * [r1 r2 t]
        [X, Y, 1]^T ~ inv(H) * [u, v, 1]^T
        
        If Z != 0, we can adjust T' = T + r3 * Z.
        H_z = K * [r1 r2 (t + r3*Z)]
        """
        
        # Rotation Matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        # Construct Homography for the plane Z = z_world
        # P_cam = R * P_w + T
        # P_cam = r1*X + r2*Y + r3*Z + T
        # P_cam = r1*X + r2*Y + (r3*Z + T)
        
        # Effective translation for this plane
        T_eff = self.tvec + R[:, 2:3] * z_world
        
        # Projection matrix for the plane columns: [r1, r2, T_eff]
        P_plane = np.hstack((R[:, 0:1], R[:, 1:2], T_eff))
        
        # H = K * P_plane
        H = self.camera_matrix @ P_plane
        
        # Invert H
        H_inv = np.linalg.inv(H)
        
        # Project point
        uv_point = np.array([u, v, 1.0])
        world_point_hom = H_inv @ uv_point
        
        # Normalize
        world_x = world_point_hom[0] / world_point_hom[2]
        world_y = world_point_hom[1] / world_point_hom[2]
        
        return world_x, world_y

    def verify_distance(self, p1_uv, p2_uv, z_world=0):
        x1, y1 = self.pixel_to_world(p1_uv[0], p1_uv[1], z_world)
        x2, y2 = self.pixel_to_world(p2_uv[0], p2_uv[1], z_world)
        
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        print(f"Point 1 (px): {p1_uv} -> World (mm): ({x1:.4f}, {y1:.4f})")
        print(f"Point 2 (px): {p2_uv} -> World (mm): ({x2:.4f}, {y2:.4f})")
        print(f"Calculated Distance: {dist:.4f} mm")
        return dist

if __name__ == "__main__":
    # --- Input Parameters from Halcon ---
    # Sx: 8.2998 um
    # Sy: 8.3 um
    # f: 5.91517 mm
    # Kappa: -101.106 1/m^2
    # Cx: 2725.88
    # Cy: 1817.03
    # Img: 5472 x 3648
    # Pose: X=-3.53697, Y=-3.84724, Z=81.1027 (mm)
    # Pose Rot: X=0.21886, Y=0.101122, Z=285.029 (deg)
    
    converter = HalconConverter(
        sx_um=8.2998,
        sy_um=8.3,
        f_mm=5.91517,
        kappa_m2=-101.106,
        cx=2725.88,
        cy=1817.03,
        img_w=5472,
        img_h=3648,
        pose_params=[-3.53697, -3.84724, 81.1027, 0.21886, 0.101122, 285.029]
    )
    
    converter.print_params()
    
    # --- Verification ---
    print("\n=== Verification Example ===")
    print("Assuming points are on the calibration plane (Z=0).")
    
    # Example: Center of image and a point 100 pixels away
    # You can change these to actual pixel coordinates you want to test
    p1 = (converter.cx, converter.cy)
    p2 = (converter.cx + 100, converter.cy) 
    
    converter.verify_distance(p1, p2, z_world=0)
    
    print("\nNote: To verify height, you would need two points with known Z difference")
    print("or one point and reconstruct its Z if you have stereo/laser data.")
    print("If you just want to measure distance on the Z=0 plane (e.g. the calibration plate),")
    print("use the verify_distance function with pixel coordinates from your image.")
