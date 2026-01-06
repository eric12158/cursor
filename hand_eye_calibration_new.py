import cv2
import numpy as np
import json
import os
import glob
import sys
from math import cos, sin, pi

# Default configuration to be written if config.json doesn't exist
DEFAULT_CONFIG = {
    "camera_intrinsics": {
        "fx": 2500.0,
        "fy": 2500.0,
        "cx": 1280.0,
        "cy": 960.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0
    },
    "calibration_board": {
        "type": "circles",  # "circles" or "chessboard"
        "rows": 7,          # Number of rows (circles) or inner corners (chessboard)
        "cols": 7,          # Number of columns
        "spacing": 15.0     # Spacing in mm
    },
    "data_source": {
        "images_dir": "./images",
        "poses_file": "./images/pos.txt",
        "image_extension": "jpg"
    },
    "output": {
        "result_file": "cam2end.txt"
    }
}

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Creating default...")
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    
    with open(config_path, 'r') as f:
        return json.load(f)

def pose_robot(x, y, z, Rx, Ry, Rz):
    """
    Convert robot pose (x, y, z, Rx, Ry, Rz) to homogeneous transformation matrix.
    Note: The rotation convention depends on the robot controller. 
    Here we use the one from the original script (RPY or similar).
    """
    # Note: original script commented out conversion to radians, implying input is already radians?
    # Or maybe input is degrees and the user commented it out? 
    # Let's check the original script's context. 
    # Original script: thetaX = Rx #/ 180 * pi
    # If the input file is pos.txt, usually robot controllers export in mm and degrees or radians.
    # We will assume the input matches what the original script expected.
    # If necessary, we can add a config option "pose_angle_unit": "deg" or "rad".
    
    thetaX = Rx 
    thetaY = Ry 
    thetaZ = Rz 
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[x], [y], [z]])
    RT1 = np.column_stack([R, t])
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    return RT1

def myRPY2R_robot(x, y, z):
    # Rz * Ry * Rx
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R

def read_end_pose(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pose file not found: {file_path}")
        
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data_list = []
    for line in lines:
        line = line.strip()
        if not line: continue
        line = line.replace('[', '').replace(']', '')
        values = line.split(',') # Assuming comma separated
        # If not comma, try space
        if len(values) < 6:
            values = line.split()
            
        data_list.append([float(value) for value in values])
    return data_list

def get_board_object_points(rows, cols, spacing, type_):
    """
    Generate 3D object points for the calibration board.
    Z = 0
    """
    objp = []
    if type_ == "chessboard":
        # Chessboard inner corners: (rows-1) x (cols-1) usually, but input config usually specifies detected points.
        # Let's assume 'rows' and 'cols' in config ARE the number of feature points.
        # Original script used standard logic.
        for i in range(rows):
            for j in range(cols):
                objp.append([j * spacing, i * spacing, 0])
    elif type_ == "circles":
        # Symmetric circles grid
        for i in range(rows):
            for j in range(cols):
                objp.append([j * spacing, i * spacing, 0])
    
    return np.array(objp, dtype=np.float32)

def detect_corners(img, rows, cols, type_):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = False
    corners = None
    
    if type_ == "chessboard":
        # Note: flags can be tuned
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            # Subpixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
    elif type_ == "circles":
        # Circles grid
        flags = cv2.CALIB_CB_SYMMETRIC_GRID
        found, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=flags)
        # Note: findCirclesGrid usually doesn't need cornerSubPix as it uses blob centers
        
    return found, corners

def main():
    # 1. Load Config
    config = load_config()
    print("Configuration loaded.")
    
    # 2. Setup Camera Matrix
    intrinsics = config["camera_intrinsics"]
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist = np.array([
        intrinsics["k1"], intrinsics["k2"], 
        intrinsics["p1"], intrinsics["p2"], 
        intrinsics["k3"]
    ], dtype=np.float64)
    
    print(f"Camera Matrix:\n{K}")
    print(f"Distortion Coeffs: {dist}")
    
    # 3. Load Images and Poses
    img_dir = config["data_source"]["images_dir"]
    pose_file = config["data_source"]["poses_file"]
    ext = config["data_source"]["image_extension"]
    
    # Load robot poses
    try:
        robot_poses = read_end_pose(pose_file)
        print(f"Loaded {len(robot_poses)} robot poses.")
    except Exception as e:
        print(f"Error loading poses: {e}")
        return

    # Load images
    # We assume images are named or sorted in a way that matches the poses order.
    # Usually 0.jpg, 1.jpg or similar.
    # We'll use glob and sort.
    image_files = sorted(glob.glob(os.path.join(img_dir, f"*.{ext}")))
    if len(image_files) == 0:
        # Try finding uppercase/lowercase extension
        image_files = sorted(glob.glob(os.path.join(img_dir, f"*.{ext.upper()}")))
        
    print(f"Found {len(image_files)} images.")
    
    if len(image_files) != len(robot_poses):
        print("WARNING: Number of images and poses do not match!")
        # We will iterate up to the minimum count
        count = min(len(image_files), len(robot_poses))
    else:
        count = len(image_files)
        
    # 4. Calibration Loop
    board_cfg = config["calibration_board"]
    rows = board_cfg["rows"]
    cols = board_cfg["cols"]
    spacing = board_cfg["spacing"]
    b_type = board_cfg["type"]
    
    objp = get_board_object_points(rows, cols, spacing, b_type)
    
    R_base2end = []
    t_base2end = []
    R_target2cam = []
    t_target2cam = []
    
    valid_count = 0
    
    for i in range(count):
        img_path = image_files[i]
        pose = robot_poses[i]
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        found, corners = detect_corners(img, rows, cols, b_type)
        
        if found:
            # Calculate Board to Camera pose using PnP
            ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
            if ret:
                R_board2cam, _ = cv2.Rodrigues(rvec)
                
                # Store Board -> Cam
                R_target2cam.append(R_board2cam)
                t_target2cam.append(tvec)
                
                # Store Base -> End
                # Pose from file: x, y, z, rx, ry, rz
                # Convert to Matrix
                RT_base2end = pose_robot(*pose)
                R_base2end.append(RT_base2end[:3, :3])
                t_base2end.append(RT_base2end[:3, 3].reshape(3, 1))
                
                valid_count += 1
                print(f"[{i+1}/{count}] Success: {os.path.basename(img_path)}")
            else:
                 print(f"[{i+1}/{count}] PnP Failed: {os.path.basename(img_path)}")
        else:
            print(f"[{i+1}/{count}] Board not detected: {os.path.basename(img_path)}")
            
    print(f"\nValid pairs for calibration: {valid_count}")
    
    if valid_count < 3:
        print("Not enough valid data for calibration (need at least 3).")
        return

    # 5. Run Hand-Eye Calibration
    # cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
    # Note on coordinate systems:
    # input 1: R_base2gripper (or gripper2base?) 
    # OpenCV docs: "rotation part of the transformation from gripper frame to robot base frame" -> Base to End? 
    # Wait, usually it says `R_gripper2base`. That means vector in gripper -> vector in base.
    # So it IS Base_T_End (transformation matrix describing End frame relative to Base).
    #
    # Input 2: `R_target2cam`.
    # Output: `R_cam2gripper`, `t_cam2gripper`. (Eye to Hand)
    #
    # Let's check existing script logic.
    # It passes `R_all_end_to_base_1` (which is Base_T_End) and `R_all_chess_to_cam_1`.
    # And expects output R, T to be cam to end.
    
    try:
        # methods: CALIB_HAND_EYE_TSAI, CALIB_HAND_EYE_PARK, ...
        # Default is TSAI
        R_cam2end, t_cam2end = cv2.calibrateHandEye(
            R_base2end, t_base2end, 
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        print("\n=== Calibration Result (Camera to End-Effector) ===")
        print("Rotation Matrix:\n", R_cam2end)
        print("Translation Vector:\n", t_cam2end)
        
        RT_cam2end = np.column_stack((R_cam2end, t_cam2end))
        RT_cam2end = np.row_stack((RT_cam2end, np.array([0, 0, 0, 1])))
        
        print("\nHomogeneous Matrix:\n", RT_cam2end)
        
        # Save to file
        out_file = config["output"]["result_file"]
        np.savetxt(out_file, RT_cam2end, fmt='%.8f')
        print(f"Saved result to {out_file}")
        
        # 6. Verification
        print("\n=== Verification ===")
        print("Calculating Board to Base pose (should be constant)...")
        
        for i in range(len(R_base2end)):
            # Base_T_End
            RT_b2e = np.eye(4)
            RT_b2e[:3, :3] = R_base2end[i]
            RT_b2e[:3, 3] = t_base2end[i].flatten()
            
            # Cam_T_End (Result) -> End_T_Cam?
            # Wait, calibrateHandEye returns Cam -> End (Camera frame in End frame, or point in Camera to point in End?)
            # OpenCV docs: "returns rotation ... from camera frame to gripper frame". 
            # So P_end = R * P_cam + T. This is Cam_T_End? No, usually T_A_B means B relative to A. 
            # If P_A = T_A_B * P_B, then it is transformation from B to A.
            # "From camera to gripper" usually means `Gripper_T_Camera` (transforms point in Cam to point in Gripper).
            # Let's assume the variable name `RT_cam2end` means "Transform FROM Camera TO End", i.e. P_end = T * P_cam.
            
            RT_c2e = RT_cam2end
            
            # Target_T_Cam (Board to Camera)
            RT_t2c = np.eye(4)
            RT_t2c[:3, :3] = R_target2cam[i]
            RT_t2c[:3, 3] = t_target2cam[i].flatten()
            
            # Chain: Base <- End <- Camera <- Target
            # Base_T_Target = Base_T_End * End_T_Camera * Camera_T_Target
            # We have:
            # Base_T_End (RT_b2e)
            # Camera_T_End (RT_c2e) -> This is P_end = T * P_cam. So it is End_T_Camera.
            # Target_T_Camera (RT_t2c) -> This is P_cam = T * P_target. So it is Camera_T_Target.
            
            RT_b2t = RT_b2e @ RT_c2e @ RT_t2c
            
            # In original script:
            # RT_chess_to_base = RT_end_to_base @ RT_cam_to_end @ RT_chess_to_cam
            # And it prints inverse of that? 
            # RT_chess_to_base = inv(RT_chess_to_base) -> Base_T_Chess?
            
            print(f"Image {i}: Translation [x,y,z] = {RT_b2t[:3, 3].flatten()}")
            
    except Exception as e:
        print(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
