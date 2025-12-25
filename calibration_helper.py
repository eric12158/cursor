import argparse

def calculate_z_resolution(known_height_mm, pixel_shift):
    """
    Calculate the Z-axis pixel equivalence (mm per pixel shift).
    
    Args:
        known_height_mm (float): The physical height of the gauge block (e.g., 5.0 mm).
        pixel_shift (float): The measured displacement of the laser line in pixels.
        
    Returns:
        float: Z-axis resolution (mm/pixel).
    """
    if pixel_shift == 0:
        raise ValueError("Pixel shift cannot be zero.")
    
    z_res = known_height_mm / pixel_shift
    return z_res

def calculate_height(current_pixel_pos, zero_pixel_pos, z_resolution):
    """
    Calculate physical height from a given pixel position.
    
    Args:
        current_pixel_pos (float): The center position of the laser on the object.
        zero_pixel_pos (float): The center position of the laser on the reference plane.
        z_resolution (float): The calibration factor (mm/pixel).
        
    Returns:
        float: Calculated height in mm.
    """
    shift = abs(current_pixel_pos - zero_pixel_pos)
    return shift * z_resolution

def subpixel_centroid(pixel_values):
    """
    Demonstration of Gray-Scale Centroid (Center of Gravity) algorithm
    for extracting laser line position with sub-pixel accuracy.
    
    Args:
        pixel_values (list): List of intensity values for a column/row cross-section of the laser line.
        
    Returns:
        float: Sub-pixel index of the center.
    """
    # Sum of (index * intensity) / Sum of intensity
    weighted_sum = sum(i * val for i, val in enumerate(pixel_values))
    total_intensity = sum(pixel_values)
    
    if total_intensity == 0:
        return 0.0
        
    return weighted_sum / total_intensity

def main():
    print("--- 2.5D Camera Z-Calibration Helper ---")
    
    # Interactive mode example (can be replaced with args)
    try:
        print("\nStep 1: Calculate Z-Resolution (Calibration)")
        h_real = float(input("Enter known height of gauge block (mm): ") or 10.0)
        p_zero = float(input("Enter laser pixel position on BASE (pixel index): ") or 100.0)
        p_top = float(input("Enter laser pixel position on BLOCK (pixel index): ") or 150.0)
        
        pixel_shift = abs(p_top - p_zero)
        z_res = calculate_z_resolution(h_real, pixel_shift)
        
        print(f"\n[Result] Pixel Shift: {pixel_shift:.2f} pixels")
        print(f"[Result] Z-Axis Equivalence: {z_res:.5f} mm/pixel")
        print(f"Meaning: A shift of 1 pixel represents {z_res:.5f} mm of height.")
        
        print("\nStep 2: Simulation (Test Measurement)")
        p_meas = float(input("Enter a new measured laser pixel position: ") or 125.0)
        measured_height = calculate_height(p_meas, p_zero, z_res)
        print(f"[Result] Calculated Height: {measured_height:.3f} mm")
        
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()
