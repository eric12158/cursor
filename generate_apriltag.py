import cv2
import numpy as np
import argparse
import sys

def get_tag_family_dict(family_name):
    """
    Returns the OpenCV ArUco dictionary for the specified AprilTag family.
    """
    families = {
        'tag16h5': cv2.aruco.DICT_APRILTAG_16h5,
        'tag25h9': cv2.aruco.DICT_APRILTAG_25h9,
        'tag36h11': cv2.aruco.DICT_APRILTAG_36h11,
        'tag36h10': cv2.aruco.DICT_APRILTAG_36h10,
    }
    
    # Try to handle case-insensitivity
    family_name = family_name.lower()
    if family_name in families:
        return families[family_name]
    else:
        print(f"Error: Unsupported or unknown tag family '{family_name}'.")
        print(f"Supported families: {list(families.keys())}")
        return None

def generate_apriltag(tag_family, tag_id, size_pixels, output_path):
    """
    Generates an AprilTag image.
    
    Args:
        tag_family (str): The tag family (e.g., 'tag36h11').
        tag_id (int): The ID of the tag.
        size_pixels (int): The width/height of the output image in pixels.
        output_path (str): The file path to save the image.
    """
    
    print(f"Generating AprilTag: Family={tag_family}, ID={tag_id}, Size={size_pixels}px")
    
    # Get the dictionary
    aruco_dict_id = get_tag_family_dict(tag_family)
    if aruco_dict_id is None:
        return False

    try:
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        
        # Generate the marker
        # 1 means 1 marker bit border
        tag_image = np.zeros((size_pixels, size_pixels, 1), dtype="uint8")
        cv2.aruco.generateImageMarker(dictionary, tag_id, size_pixels, tag_image, 1)
        
        # Save the image
        cv2.imwrite(output_path, tag_image)
        print(f"Successfully saved tag to '{output_path}'")
        return True
        
    except Exception as e:
        print(f"Error generating tag: {e}")
        return False

if __name__ == "__main__":
    # --- Configuration ---
    # You can modify these default values directly or use command line arguments
    DEFAULT_FAMILY = "tag36h11"
    DEFAULT_ID = 0
    DEFAULT_SIZE = 500  # pixels
    DEFAULT_OUTPUT = "apriltag.png"

    parser = argparse.ArgumentParser(description="Generate AprilTag images.")
    parser.add_argument("--family", type=str, default=DEFAULT_FAMILY, help="Tag family (e.g., tag36h11, tag25h9, tag16h5)")
    parser.add_argument("--id", type=int, default=DEFAULT_ID, help="Tag ID")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Image size in pixels (square)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output filename")

    args = parser.parse_args()

    generate_apriltag(args.family, args.id, args.size, args.output)
