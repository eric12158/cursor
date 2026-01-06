# 手眼标定程序 (Hand-Eye Calibration)

该程序用于进行机器人手眼标定（Eye-in-Hand 或 Eye-to-Hand），基于 OpenCV 实现。它可以替代 MATLAB 工具箱流程，允许用户通过配置文件输入相机内参，并直接处理图片和机械臂位姿数据。

## 功能特点

1.  **配置化**：相机内参、畸变系数、标定板参数均通过 `config.json` 配置。
2.  **通用性**：支持棋盘格 (Chessboard) 和圆点阵列 (Circles Grid) 两种标定板。
3.  **易操作**：只需将采集的图片和对应的机械臂位姿文件放入指定目录即可一键运行。
4.  **准确性**：使用 OpenCV 的 `solvePnP` 和 `calibrateHandEye` (默认 Tsai 方法) 算法，包含重投影误差验证步骤（通过计算标定板相对于基座的固定位置变异程度）。

## 使用步骤

### 1. 准备工作

确保安装了必要的 Python 库：
```bash
pip install opencv-python numpy
```

### 2. 配置文件 (`config.json`)

程序首次运行会自动生成 `config.json`。请根据实际情况修改以下参数：

*   **camera_intrinsics**: 填入你的相机内参 (`fx`, `fy`, `cx`, `cy`) 和畸变系数 (`k1`, `k2`, `p1`, `p2`, `k3`)。这些数据通常来自之前的相机标定步骤（如 MATLAB Calibrator 或 OpenCV 标定例程）。
*   **calibration_board**:
    *   `type`: "circles" (圆点) 或 "chessboard" (棋盘格).
    *   `rows`: 标点行数（棋盘格为内角点行数）.
    *   `cols`: 标点列数（棋盘格为内角点列数）.
    *   `spacing`: 点间距或格边长 (mm).
*   **data_source**:
    *   `images_dir`: 图片文件夹路径 (默认 `./images`).
    *   `poses_file`: 机械臂位姿文件路径 (默认 `./images/pos.txt`).

### 3. 数据准备

*   **图片**: 将采集到的标定板图片放入 `images` 文件夹。
*   **位姿**: 在 `images/pos.txt` 中填入对应的机械臂末端位姿。
    *   格式支持：`[x, y, z, rx, ry, rz]` 或 `x, y, z, rx, ry, rz` (逗号或空格分隔).
    *   **注意**: 图片和位姿必须一一对应（按文件名排序对应行号）。
    *   `rx, ry, rz` 的单位和旋转顺序需与程序中的 `pose_robot` 函数一致（默认假设为 RPY 欧拉角，单位视机器人控制器而定，通常为弧度或角度，请根据实际情况调整 `hand_eye_calibration_new.py` 中的 `pose_robot` 函数）。

### 4. 运行标定

```bash
python hand_eye_calibration_new.py
```

### 5. 查看结果

*   程序会在控制台输出标定过程日志。
*   标定结果（相机相对于末端的变换矩阵）将保存到 `cam2end.txt`。
*   控制台末尾会输出验证信息（标定板相对于基座的计算位置），如果每次计算结果一致（方差小），说明标定准确。

## 注意事项

*   **坐标系**: 请确认机械臂输出的旋转向量（rx, ry, rz）是旋转向量、欧拉角还是四元数。本程序默认为欧拉角 (RPY)。如果您的机器人使用不同格式（如轴角或四元数），请修改代码中的 `pose_robot` 函数。
*   **单位**: 确保输入的平移单位 (x, y, z) 和标定板间距单位一致（通常为 mm）。
