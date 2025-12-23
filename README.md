# 2D Camera to 2.5D (Laser Triangulation) Guide

This project demonstrates how to convert a standard 2D camera system into a 2.5D measurement system using **Laser Triangulation**.

## 1. Principle: How to get 2.5D from 2D?

To obtain depth (Z) information from a single 2D camera, the most common industrial method is **Laser Triangulation** (Sheet-of-Light).

1.  **Setup**: A laser line generator projects a straight line onto the object.
2.  **Angle**: The camera is mounted at an angle (alpha) relative to the laser plane.
3.  **Deformation**: When the laser line hits an object with height, the line appears "broken" or shifted in the camera image.
4.  **Calculation**: The amount of pixel shift ($ \Delta p $) of the laser line corresponds to the physical height ($ \Delta Z $).

## 2. Calculating Z Pixel Equivalence (Z当量)

You already know the X/Y pixel equivalence (e.g., mm/pixel). To find the Z equivalence ($ K_z $), you need the geometric relationship.

### Method A: Geometric Calculation (Theoretical)

If you know the angle $\theta$ between the Camera's Optical Axis and the Laser Plane:

$$ \Delta Z = \frac{\Delta p \cdot S_{sensor}}{\sin(\theta) \cdot M} $$

Where:
- $\Delta p$: Shift in pixels.
- $S_{sensor}$: Physical size of one pixel on the sensor.
- $\theta$: Angle between laser and camera view.
- $M$: Magnification (or use your known X/Y equivalence).

### Method B: Calibration Block (Empirical & Recommended)

The most accurate way is to measure a known standard (Gauge block).

1.  Place a flat reference surface (Zero plane). Record the laser line position ($ P_{zero} $).
2.  Place a block of known height $ H_{real} $ (e.g., 10mm).
3.  Record the new laser line position ($ P_{top} $).
4.  Calculate the shift: $ \Delta P = P_{top} - P_{zero} $ (in pixels).
5.  Calculate Z Equivalence:

$$ \text{Z_Res} = \frac{H_{real}}{\Delta P} \quad (\text{mm/pixel}) $$

## 3. Processing Pipeline

1.  **Capture Image**: Acquire image with laser line.
2.  **ROI Selection**: Focus on the laser line area.
3.  **Laser Extraction**:
    - Use thresholding to find the laser.
    - Calculate the **Center of Gravity (Gray-value centroid)** for sub-pixel accuracy.
    - Result: A set of (row, col) coordinates for the laser line.
4.  **Height Map Generation**:
    - Convert the "row" or "col" deviation to Z-height using `Z_Res`.
    - Stack these profiles as the object moves (scanning) to create a 2.5D Point Cloud or Depth Map.

## Usage of `calibration_helper.py`

Use the provided script to calculate your Z-resolution based on calibration data.

```bash
python calibration_helper.py
```
