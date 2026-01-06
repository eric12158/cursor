import cv2
import numpy as np
import json
import os
import sys
from math import *
from scipy.spatial.transform import Rotation as R

# =============================================================================
# Configuration Loading
# =============================================================================
def load_config(config_path="hand_eye_config.json"):
    if not os.path.exists(config_path):
        print(f"[Error] Configuration file '{config_path}' not found.")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# =============================================================================
# Helper Functions
# =============================================================================
def get_object_points(rows, cols, spacing, board_type):
    """
    Generate 3D coordinates of the calibration board.
    mimics import sys.py logic (implied).
    """
    objp = []
    
    if board_type == "chessboard":
        # Usually for chessboard, we look for inner corners
        # If config rows/cols are number of squares, we use rows-1, cols-1
        pattern_size = (cols - 1, rows - 1)
        for i in range(pattern_size[1]):
            for j in range(pattern_size[0]):
                objp.append([j * spacing, i * spacing, 0])
    elif board_type == "circles":
        # For circles grid, we look for centers
        pattern_size = (cols, rows)
        for i in range(pattern_size[1]):
            for j in range(pattern_size[0]):
                objp.append([j * spacing, i * spacing, 0])
    else:
        raise ValueError(f"Unknown board type: {board_type}")
        
    return np.array(objp, dtype=np.float32), pattern_size

def process_robot_pose(pose_list):
    """
    Convert robot pose [x, y, z, rx, ry, rz] to 4x4 homogenous matrix.
    Assumes Euler angles are in degrees and order is XYZ (extrinsic).
    Matches import sys.py: R.from_euler('xyz', pose[3:], degrees=True)
    """
    x, y, z = pose_list[:3]
    rx, ry, rz = pose_list[3:]
    
    # Translation
    t = np.array([x, y, z]).reshape(3, 1)
    
    # Rotation (scipy implementation is robust)
    # import sys.py uses: R.from_euler('xyz', ..., degrees=True)
    r_mat = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    
    # Compose 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = r_mat
    T[:3, 3] = t.flatten()
    return T

# =============================================================================
# Main Calibration Logic
# =============================================================================
def run_calibration():
    # 1. Load Config
    cfg = load_config()
    
    # Parse Intrinsics
    intrinsics = cfg["camera_intrinsics"]
    camera_matrix = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.array([
        intrinsics["k1"], intrinsics["k2"], 
        intrinsics["p1"], intrinsics["p2"], 
        intrinsics["k3"]
    ], dtype=np.float64)
    
    print("Loaded Camera Intrinsics:")
    print(camera_matrix)
    print(f"Distortion: {dist_coeffs}\n")
    
    # 2. Load Data
    data_path = cfg["paths"]["data_file"]
    if not os.path.exists(data_path):
        print(f"[Error] Data file '{data_path}' not found.")
        print("Please create a JSON file with a list of entries containing 'img_path' and 'robot_pose'.")
        sys.exit(1)
        
    with open(data_path, 'r', encoding='utf-8') as f:
        data_entries = json.load(f)
        
    print(f"Loaded {len(data_entries)} data entries.")
    
    # 3. Prepare Object Points
    rows = cfg["calibration"]["rows"]
    cols = cfg["calibration"]["cols"]
    spacing = cfg["calibration"]["spacing"]
    board_type = cfg["calibration"].get("type", "chessboard")
    
    objp, pattern_size = get_object_points(rows, cols, spacing, board_type)
    
    # Lists to store transforms
    R_base_to_end_list = []
    T_base_to_end_list = []
    R_target_to_cam_list = []
    T_target_to_cam_list = []
    
    valid_frames = 0
    
    for entry in data_entries:
        img_path = entry.get("img_path")
        robot_pose = entry.get("robot_pose")
        
        if not img_path or not robot_pose:
            print(f"[Skip] Invalid entry: {entry}")
            continue
            
        if not os.path.exists(img_path):
            print(f"[Skip] Image not found: {img_path}")
            continue
            
        # -- Image Processing --
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        found = False
        corners = None
        
        if board_type == "chessboard":
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if found:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        elif board_type == "circles":
            found, corners = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            
        if not found:
            print(f"[Skip] Board not found in {img_path}")
            continue
            
        # -- Pose Estimation (Board to Camera) --
        # solvePnP returns rotation vector and translation vector from Object Frame to Camera Frame
        success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        if not success:
            print(f"[Skip] solvePnP failed for {img_path}")
            continue
            
        R_target_to_cam, _ = cv2.Rodrigues(rvec)
        
        # -- Robot Pose (Base to End) --
        # robot_pose is [x, y, z, rx, ry, rz]
        T_base_to_end_mat = process_robot_pose(robot_pose)
        R_base_to_end = T_base_to_end_mat[:3, :3]
        t_base_to_end = T_base_to_end_mat[:3, 3].reshape(3, 1)
        
        # Store
        R_base_to_end_list.append(R_base_to_end)
        T_base_to_end_list.append(t_base_to_end)
        R_target_to_cam_list.append(R_target_to_cam)
        T_target_to_cam_list.append(tvec)
        
        valid_frames += 1
        print(f"[OK] Processed {img_path}")
        
    if valid_frames < 3:
        print(f"\n[Error] Not enough valid data points. Need at least 3, got {valid_frames}.")
        sys.exit(1)
        
    # 4. Hand-Eye Calibration
    print(f"\nRunning Hand-Eye Calibration with {valid_frames} frames...")
    
    # method: CV_CALIB_ROBOT_WORLD_HAND_EYE_SHAH (or others)
    # The default is cv2.CALIB_HAND_EYE_TSAI
    # We want R_cam_to_end, T_cam_to_end
    # Inputs:
    # R_gripper2base (R_base_to_end_list ? No, usually the API expects R_base2gripper or R_gripper2base depending on implementation)
    # OpenCV documentation says: 
    # R_gripper2base: Rotation part of the transformation from the gripper frame to the robot base frame.
    # t_gripper2base: Translation part...
    # R_target2cam: ... from target frame to camera frame.
    # t_target2cam: ...
    #
    # We calculated R_base_to_end (which is R_base_to_gripper).
    # Wait, OpenCV docs say: "R_gripper2base".
    # This usually means T_base_from_gripper (Coordinate in Base = T * Coordinate in Gripper).
    # So it matches our T_base_to_end matrix.
    
    try:
        R_cam_to_end, T_cam_to_end = cv2.calibrateHandEye(
            R_base_to_end_list, 
            T_base_to_end_list, 
            R_target_to_cam_list, 
            T_target_to_cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        print("\n====== Calibration Result (Camera -> End Effector) ======")
        print("Rotation Matrix:\n", R_cam_to_end)
        print("Translation Vector (mm):\n", T_cam_to_end)
        
        # Combine to 4x4
        T_cam_to_end_mat = np.eye(4)
        T_cam_to_end_mat[:3, :3] = R_cam_to_end
        T_cam_to_end_mat[:3, 3] = T_cam_to_end.flatten()
        
        print("\nHomogeneous Matrix (4x4):\n", T_cam_to_end_mat)
        
        # Save to file
        output_file = "cam2end.txt"
        np.savetxt(output_file, T_cam_to_end_mat, fmt='%.8f')
        print(f"\nResult saved to {output_file}")
        
        # Validation
        print("\n====== Validation (Reprojection Error Check) ======")
        for i in range(len(R_base_to_end_list)):
             # T_base_to_target = T_base_to_end * T_end_to_cam * T_cam_to_target
             # Actually we check consistency:
             # T_base_to_target should be constant (the board didn't move relative to base)
             
             T_base_to_end = np.eye(4)
             T_base_to_end[:3, :3] = R_base_to_end_list[i]
             T_base_to_end[:3, 3] = T_base_to_end_list[i].flatten()
             
             T_target_to_cam = np.eye(4)
             T_target_to_cam[:3, :3] = R_target_to_cam_list[i]
             T_target_to_cam[:3, 3] = T_target_to_cam_list[i].flatten()
             
             # T_cam_to_target = inv(T_target_to_cam)
             T_cam_to_target = np.linalg.inv(T_target_to_cam)
             
             # T_end_to_cam = inv(T_cam_to_end)
             T_end_to_cam = np.linalg.inv(T_cam_to_end_mat)
             
             # Chain: Base <- End <- Cam <- Target
             T_base_to_target = T_base_to_end @ T_end_to_cam @ T_cam_to_target
             
             # Just print the translation part to see if it's stable
             print(f"Frame {i}: Board Pos in Base = {T_base_to_target[:3, 3].flatten()}")

    except Exception as e:
        print(f"[Error] Calibration failed: {e}")

if __name__ == "__main__":
    run_calibration()
