import cv2
import numpy as np
import glob
import os
import argparse
import json
import datetime
import sys

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_blob_detector():
    """ Returns a tuned SimpleBlobDetector for circle grid detection """
    blobParams = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    blobParams.filterByArea = True
    blobParams.minArea = 50
    blobParams.maxArea = 100000
    
    # Filter by Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.7
    
    # Filter by Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.8
    
    # Filter by Inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.1
    
    # Filter by Color
    blobParams.filterByColor = True
    blobParams.blobColor = 0 # Black circles
    
    return cv2.SimpleBlobDetector_create(blobParams)

def calibrate(
    image_dir, 
    pattern_type="circles", 
    rows=7, 
    cols=7, 
    spacing=1.5, 
    output_file="calibration_result.json",
    debug_dir="debug_output",
    show_process=False
):
    """
    Main calibration function.
    
    Args:
        image_dir (str): Directory containing images.
        pattern_type (str): 'chessboard', 'circles', or 'asymmetric_circles'.
        rows (int): Number of internal corners/circles in height.
        cols (int): Number of internal corners/circles in width.
        spacing (float): Physical distance between points (mm usually).
        output_file (str): Path to save JSON result.
        debug_dir (str): Path to save debug images.
        show_process (bool): If True, show windows (requires GUI environment).
    """
    
    # Check directory
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        return False

    # Create debug directory
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # Prepare object points (3D world points)
    # The structure depends on the pattern type
    pattern_size = (cols, rows) # (width, height)
    
    if pattern_type == "chessboard" or pattern_type == "circles":
        # Regular grid
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= spacing
    elif pattern_type == "asymmetric_circles":
        # Asymmetric grid
        objp = np.zeros((rows * cols, 3), np.float32)
        for i in range(rows):
            for j in range(cols):
                objp[i*cols + j, 0] = (2*j + (i % 2)) * spacing
                objp[i*cols + j, 1] = i * spacing
                objp[i*cols + j, 2] = 0
    else:
        print(f"Error: Unknown pattern type '{pattern_type}'")
        return False

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Find images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not images:
        print(f"No images found in '{image_dir}'")
        return False

    print(f"Found {len(images)} images in '{image_dir}'. processing...")
    print(f"Configuration: Type={pattern_type}, Size={pattern_size}, Spacing={spacing}")

    blob_detector = get_blob_detector() if "circles" in pattern_type else None
    
    img_shape = None
    success_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read {fname}")
            continue
            
        if img_shape is None:
            img_shape = img.shape[:2][::-1] # (width, height)
        else:
            if img.shape[:2][::-1] != img_shape:
                print(f"Warning: Skipping {fname} due to size mismatch.")
                continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        found = False
        corners = None
        
        if pattern_type == "chessboard":
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if found:
                # Refine corners
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
                
        elif pattern_type == "circles":
            found, corners = cv2.findCirclesGrid(
                gray, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=blob_detector
            )
            
        elif pattern_type == "asymmetric_circles":
            found, corners = cv2.findCirclesGrid(
                gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blob_detector
            )

        filename = os.path.basename(fname)
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            success_count += 1
            print(f"[OK] {filename}")
            
            # Visualization
            cv2.drawChessboardCorners(img, pattern_size, corners, found)
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f"detected_{filename}"), img)
            if show_process:
                cv2.imshow('Detection', img)
                cv2.waitKey(100)
        else:
            print(f"[FAIL] {filename} - Pattern not found")

    if show_process:
        cv2.destroyAllWindows()

    if success_count == 0:
        print("Error: Pattern not found in any image.")
        return False

    print(f"\nCalibrating with {success_count} valid images...")
    
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
    except cv2.error as e:
        print(f"Calibration failed: {e}")
        return False

    # Calculate Re-projection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    total_error = mean_error / len(objpoints)

    print("\nCalibration Success!")
    print(f"Re-projection Error: {total_error:.4f} pixels (lower is better)")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)

    # Save to JSON
    result_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pattern_type": pattern_type,
        "pattern_size": pattern_size,
        "spacing_mm": spacing,
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "reprojection_error": total_error,
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "rvecs": rvecs, # Optional: might be too verbose
        "tvecs": tvecs  # Optional: might be too verbose
    }

    # Simplify rvecs/tvecs for json (list of lists)
    result_data["rvecs"] = [r.flatten().tolist() for r in rvecs]
    result_data["tvecs"] = [t.flatten().tolist() for t in tvecs]

    try:
        with open(output_file, 'w') as f:
            json.dump(result_data, f, cls=NumpyEncoder, indent=4)
        print(f"\nResults saved to '{output_file}'")
    except Exception as e:
        print(f"Failed to save JSON: {e}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCV Camera Calibration Tool")
    
    parser.add_argument("--dir", type=str, required=True, help="Path to image folder")
    parser.add_argument("--type", type=str, default="circles", choices=["chessboard", "circles", "asymmetric_circles"], help="Pattern type")
    parser.add_argument("--rows", type=int, default=7, help="Number of rows")
    parser.add_argument("--cols", type=int, default=7, help="Number of columns")
    parser.add_argument("--spacing", type=float, default=1.5, help="Physical spacing between points (mm)")
    parser.add_argument("--out", type=str, default="calibration_result.json", help="Output JSON filename")
    parser.add_argument("--debug", type=str, default="debug_output", help="Directory for debug images")
    parser.add_argument("--show", action="store_true", help="Show processing window (requires GUI)")

    args = parser.parse_args()
    
    calibrate(
        image_dir=args.dir,
        pattern_type=args.type,
        rows=args.rows,
        cols=args.cols,
        spacing=args.spacing,
        output_file=args.out,
        debug_dir=args.debug,
        show_process=args.show
    )
