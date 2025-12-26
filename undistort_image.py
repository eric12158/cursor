import cv2
import numpy as np
import argparse
import json
import os

def undistort(image_path, json_path, output_path=None):
    """
    Undistorts an image using camera parameters from a JSON file.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file '{json_path}' not found.")
        return

    # Load calibration data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    try:
        mtx = np.array(data["camera_matrix"])
        dist = np.array(data["dist_coeffs"])
    except KeyError as e:
        print(f"Error: Missing key {e} in JSON file.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    h, w = img.shape[:2]
    
    # Get optimal new camera matrix (preserves valid pixels)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image (optional, depends on preference)
    x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w] 
    
    if output_path is None:
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = f"undistorted_{name}{ext}"
    
    cv2.imwrite(output_path, dst)
    print(f"Undistorted image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undistort Image Tool")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--json", type=str, required=True, help="Path to calibration JSON file")
    parser.add_argument("--out", type=str, default=None, help="Path to output image")
    
    args = parser.parse_args()
    
    undistort(args.img, args.json, args.out)
