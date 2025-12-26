import cv2
import numpy as np
import glob
import os
import argparse
import json
import datetime
import sys

# Try to import tkinter for folder selection
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

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

def select_folder_gui():
    """Opens a folder selection dialog"""
    if not HAS_TKINTER:
        print("Tkinter not found. Please enter path manually.")
        return None
    
    try:
        # Create a root window but hide it
        root = tk.Tk()
        root.withdraw()
        
        # Bring dialog to front (might help on some OS)
        root.lift()
        root.attributes('-topmost', True)
        
        print("Opening folder selection dialog...")
        folder_path = filedialog.askdirectory(title="Select Image Directory")
        
        root.destroy()
        return folder_path if folder_path else None
    except Exception as e:
        print(f"Error opening dialog: {e}")
        return None

def calibrate(
    image_dir=None, 
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
    """
    
    # 1. Handle image directory selection
    if not image_dir:
        print("No image directory provided via command line.")
        image_dir = select_folder_gui()
        
        # Fallback to console input
        if not image_dir:
            image_dir = input("Please enter image folder path: ").strip()
            # Remove quotes if user added them
            if (image_dir.startswith('"') and image_dir.endswith('"')) or \
               (image_dir.startswith("'") and image_dir.endswith("'")):
                image_dir = image_dir[1:-1]
    
    # 2. Validate directory
    if not image_dir or not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        return False

    print(f"Processing images from: {image_dir}")

    # Create debug directory
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # 3. Prepare object points (3D world points)
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

    # 4. Find images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not images:
        print(f"No images found in '{image_dir}'")
        return False

    print(f"Found {len(images)} images. Processing...")
    
    blob_detector = get_blob_detector() if "circles" in pattern_type else None
    
    img_shape = None
    success_count = 0

    # 5. Process each image
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        if img_shape is None:
            img_shape = img.shape[:2][::-1] # (width, height)
        else:
            if img.shape[:2][::-1] != img_shape:
                # print(f"Warning: Skipping {fname} due to size mismatch.")
                continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        found = False
        corners = None
        
        if pattern_type == "chessboard":
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if found:
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
            
            if debug_dir:
                cv2.drawChessboardCorners(img, pattern_size, corners, found)
                cv2.imwrite(os.path.join(debug_dir, f"detected_{filename}"), img)
            if show_process:
                cv2.imshow('Detection', img)
                cv2.waitKey(50)
        else:
            print(f"[FAIL] {filename}")

    if show_process:
        cv2.destroyAllWindows()

    if success_count == 0:
        print("Error: Pattern not found in any image.")
        return False

    print(f"\nCalibrating with {success_count} valid images...")
    
    # 6. Run Calibration
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
    except cv2.error as e:
        print(f"Calibration failed: {e}")
        return False

    # ret is the RMS error
    rms_error = ret

    print("\nCalibration Success!")
    print(f"RMS Error: {rms_error:.4f}")
    
    # 7. Save to JSON in specified format
    result_data = {
        "rms": rms_error,
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": [img_shape[0], img_shape[1]],
        "calibration_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(result_data, f, cls=NumpyEncoder, indent=4)
        print(f"\nResults saved to '{output_file}'")
    except Exception as e:
        print(f"Failed to save JSON: {e}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCV Camera Calibration Tool")
    
    # Made --dir optional
    parser.add_argument("--dir", type=str, help="Path to image folder (optional, will ask if empty)")
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
