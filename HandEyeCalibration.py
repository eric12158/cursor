import cv2
import numpy as np
import json
import os
import glob
from math import *

class HandEyeCalibration:
    def __init__(self, config_file="config.json"):
        self.load_config(config_file)
        self.R_all_end_to_base = []
        self.T_all_end_to_base = []
        self.R_all_chess_to_cam = []
        self.T_all_chess_to_cam = []
        self.image_points = []
        self.object_points = []
        
    def load_config(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found.")
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        print("Loaded configuration:")
        print(json.dumps(self.config, indent=4))
        
        # Camera Intrinsics
        self.fx = self.config['camera']['fx']
        self.fy = self.config['camera']['fy']
        self.cx = self.config['camera']['cx']
        self.cy = self.config['camera']['cy']
        self.dist = np.array(self.config['camera']['dist'], dtype=float)
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)
        
        # Board Settings
        self.board_type = self.config['calibration']['type'] # "circles" or "chessboard"
        self.rows = self.config['calibration']['rows']
        self.cols = self.config['calibration']['cols']
        self.spacing = self.config['calibration']['spacing']

    def myRPY2R_robot(self, x, y, z):
        # Adapted from hand_eye_calibrate.py
        Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
        Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R

    def pose_robot(self, x, y, z, Rx, Ry, Rz):
        # Adapted from hand_eye_calibrate.py
        # Note: Input angles are assumed to be in radians as per original code usage of math.cos/sin
        # If input is degrees, conversion happens before calling this or inside if modified.
        # Checking original code: "thetaX = Rx #/ 180 * pi" -> It seems it expects radians or the user commented out conversion.
        # But import sys.py uses degrees. I will add a check/flag for degree conversion.
        
        # Original code logic:
        thetaX = Rx 
        thetaY = Ry 
        thetaZ = Rz 
        R = self.myRPY2R_robot(thetaX, thetaY, thetaZ)
        t = np.array([[x], [y], [z]])
        RT1 = np.column_stack([R, t])
        RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
        return RT1

    def process_data(self, data_file):
        """
        Reads data from a JSON file (mimicking import sys.py's backup format)
        or searches for images in a directory if configured.
        """
        data_list = []
        if data_file.endswith('.json'):
            with open(data_file, 'r') as f:
                data_list = json.load(f)
        else:
            raise ValueError("Unsupported data file format. Please use JSON.")

        print(f"Processing {len(data_list)} images...")
        
        # Prepare object points (3D points of board)
        if self.board_type == "chessboard":
            # (rows-1) * (cols-1) inner corners
            # Mimic import sys.py logic
            objp = np.zeros(((self.rows-1) * (self.cols-1), 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.cols-1, 0:self.rows-1].T.reshape(-1, 2) * self.spacing
        else:
            # circles: rows * cols
            objp = np.zeros((self.rows * self.cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * self.spacing

        for item in data_list:
            img_path = item.get('img_path')
            pose = item.get('robot_pose') # [x, y, z, rx, ry, rz]
            
            if not img_path or not pose:
                print(f"Skipping invalid item: {item}")
                continue
                
            if not os.path.exists(img_path):
                # Try relative path
                if os.path.exists(os.path.basename(img_path)):
                    img_path = os.path.basename(img_path)
                else:
                    print(f"Image not found: {img_path}")
                    continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect Corners
            found = False
            corners = None
            
            if self.board_type == "chessboard":
                ret, corners = cv2.findChessboardCorners(gray, (self.cols-1, self.rows-1), 
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    found = True
            else:
                ret, corners = cv2.findCirclesGrid(gray, (self.cols, self.rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
                if ret:
                    found = True
            
            if found:
                # Calculate Board to Camera Pose (solvePnP)
                ret, rvec, tvec = cv2.solvePnP(objp, corners, self.K, self.dist)
                
                # Convert rvec to Rotation Matrix
                R_chess_to_cam, _ = cv2.Rodrigues(rvec)
                
                self.R_all_chess_to_cam.append(R_chess_to_cam)
                self.T_all_chess_to_cam.append(tvec)
                
                # Robot Pose Handling
                # Check if angles are degrees or radians. 
                # import sys.py seems to use degrees for display/storage (Rx, Ry, Rz from robot).
                # hand_eye_calibrate.py used radians (commented out #/180*pi).
                # I will assume degrees in input and convert to radians for calculation if config says so, 
                # or just assume the pose_robot function handles it.
                # Let's assume input is degrees and convert to radians for pose_robot if needed.
                # Standard robots usually output degrees or radians.
                # For safety, I will add a config option "angle_unit": "deg" or "rad". Default "deg".
                
                x, y, z, rx, ry, rz = pose
                if self.config.get('robot_pose_angle_unit', 'deg') == 'deg':
                    rx = rx * pi / 180.0
                    ry = ry * pi / 180.0
                    rz = rz * pi / 180.0
                
                RT_robot = self.pose_robot(x, y, z, rx, ry, rz)
                
                self.R_all_end_to_base.append(RT_robot[:3, :3])
                self.T_all_end_to_base.append(RT_robot[:3, 3].reshape((3, 1)))
                
                print(f"Processed {os.path.basename(img_path)}: Found {len(corners)} points.")
            else:
                print(f"Corners not found in {os.path.basename(img_path)}")

    def run_calibration(self):
        if len(self.R_all_end_to_base) < 3:
            print("Not enough valid data points for calibration (need at least 3).")
            return

        print("\nRunning Hand-Eye Calibration...")
        # cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
        R, T = cv2.calibrateHandEye(
            self.R_all_end_to_base, 
            self.T_all_end_to_base, 
            self.R_all_chess_to_cam, 
            self.T_all_chess_to_cam
        )

        print("Hand-Eye Calibration Result (Rotation Matrix):")
        print(R)
        print("\nHand-Eye Calibration Result (Translation Vector):")
        print(T)

        RT = np.column_stack((R, T))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
        
        print("\nCamera to End-Effector Transformation Matrix:")
        print(RT)
        
        # Save results
        output_file = "cam2end.txt"
        with open(output_file, 'w') as f:
            for row in RT:
                row_str = ' '.join(map(str, row))
                f.write(row_str + '\n')
        print(f"\nResult saved to {output_file}")
        
        self.R_cam_to_end = R
        self.T_cam_to_end = T
        self.verify_results()

    def verify_results(self):
        print("\nVerifying Results...")
        for i in range(len(self.R_all_end_to_base)):
            # T_end_to_base
            RT_end_to_base = np.column_stack((self.R_all_end_to_base[i], self.T_all_end_to_base[i]))
            RT_end_to_base = np.row_stack((RT_end_to_base, np.array([0, 0, 0, 1])))
            
            # T_chess_to_cam
            RT_chess_to_cam = np.column_stack((self.R_all_chess_to_cam[i], self.T_all_chess_to_cam[i]))
            RT_chess_to_cam = np.row_stack((RT_chess_to_cam, np.array([0, 0, 0, 1])))
            
            # T_cam_to_end
            RT_cam_to_end = np.column_stack((self.R_cam_to_end, self.T_cam_to_end))
            RT_cam_to_end = np.row_stack((RT_cam_to_end, np.array([0, 0, 0, 1])))
            
            # T_chess_to_base = T_end_to_base * T_cam_to_end * T_chess_to_cam
            RT_chess_to_base = RT_end_to_base @ RT_cam_to_end @ RT_chess_to_cam
            
            # The board should be fixed relative to base, so RT_chess_to_base should be constant-ish.
            # In the original code, they printed: inv(RT_chess_to_base)
            # "即为固定的棋盘格相对于机器人基坐标系位姿" -> That usually means T_base_to_chess
            # If RT_chess_to_base is Chess -> Base, then inv is Base -> Chess.
            
            RT_chess_to_base_inv = np.linalg.inv(RT_chess_to_base)
            
            print(f"Sample {i}:")
            # print(RT_chess_to_base_inv[:3, :])
            # Only printing translation to check consistency easily
            print(f"Translation (Base->Chess): {RT_chess_to_base[:3, 3].flatten()}")

if __name__ == "__main__":
    import sys
    
    config_file = "config.json"
    data_file = "calibration_data.json"
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    if len(sys.argv) > 2:
        data_file = sys.argv[2]
        
    if not os.path.exists(config_file):
        # Create default config if not exists
        default_config = {
            "camera": {
                "fx": 1000.0, "fy": 1000.0,
                "cx": 640.0, "cy": 360.0,
                "dist": [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "calibration": {
                "type": "circles",
                "rows": 7,
                "cols": 7,
                "spacing": 15.0
            },
            "robot_pose_angle_unit": "deg"
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default config file: {config_file}")
        print("Please edit it with your actual camera parameters and board settings.")
        
    calibrator = HandEyeCalibration(config_file)
    
    if os.path.exists(data_file):
        calibrator.process_data(data_file)
        calibrator.run_calibration()
    else:
        print(f"Data file {data_file} not found. Please provide a JSON file with image paths and poses.")
