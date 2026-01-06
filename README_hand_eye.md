## 目的

`hand_eye_calibrate.py` 用于**离线**手眼标定（Eye-in-hand），保留 OpenCV `cv2.calibrateHandEye` 的算法调用，但把输入改成**可配置、可通用**的形式：

- **相机内参/畸变**：从 `hand_eye_config.json` 手动填写（不再依赖 MATLAB `.mat`）
- **标定板**：支持 `chessboard`（棋盘格）和 `circles`（圆点阵），参数与 `import sys.py` 一致
- **数据组织**：推荐 `calibration_data.json`（每张图绑定一组机械臂位姿），兼容旧版 `images/pos.txt`

输出：
- `cam2end.txt`：手眼结果 \(T_{camera\rightarrow gripper}\)（与原脚本一致）
- `handeye_report.json`：统计信息（样本数、平均重投影误差、标定板在基座系下的稳定性统计）

---

## 快速开始

1) 编辑 `hand_eye_config.json`，至少填写：
- `camera.fx/fy/cx/cy`
- `camera.dist`（没有就填 0）
- `calibration`（标定板类型/行列/间距）
- `dataset.pose_angle_unit` 与 `dataset.pose_is_base_to_gripper`

2) 准备数据（二选一）

### 方式 A（推荐）：`calibration_data.json`

文件是一个 JSON 数组，每项至少包含：

```json
{
  "img_path": "0001.jpg",
  "robot_pose": [x, y, z, rx, ry, rz]
}
```

- `img_path`：可以是相对 `dataset.images_dir` 的文件名（推荐），也可以是绝对路径
- `robot_pose`：6D 位姿
  - 平移单位：**mm**
  - 旋转：按 `dataset.pose_angle_unit` 解释（`rad` 或 `deg`）
  - 欧拉角顺序：**xyz**（等价于 \(R = R_z R_y R_x\)，与参考脚本一致）

### 方式 B（兼容旧数据）：`images/pos.txt`

- 图片放在 `dataset.images_dir` 里（默认 `./images`）
- 位姿放在 `dataset.pose_file`（默认 `./images/pos.txt`）
- **要求**：图片按文件名排序后的顺序，与 `pos.txt` 行号一一对应
- 每行格式示例：

```
[x,y,z,rx,ry,rz]
```

---

## 关键配置说明（保证“准确无误”的核心）

### 1) `dataset.pose_is_base_to_gripper`

OpenCV `calibrateHandEye` 需要输入：**T_gripper_to_base**（末端到基座）。

但多数机械臂返回的是：**T_base_to_gripper**（基座到末端）。

- 若你的 `robot_pose` 是“基座到末端”：设置 `true`（脚本会自动求逆）
- 若你的 `robot_pose` 已经是“末端到基座”：设置 `false`

> 这个选项是导致手眼结果“差很多/完全不对”的最常见原因。

### 2) `dataset.pose_angle_unit`

你的 `robot_pose` 旋转角如果是弧度：`rad`  
如果是角度（度）：`deg`

> 单位错了会导致旋转矩阵完全错误，结果必然不准。

### 3) `calibration.rows/cols/spacing`

与 `import sys.py` 保持一致：

- `type = chessboard`：
  - `rows/cols` 表示“格子数”
  - 内角点数为 `(rows-1) x (cols-1)`
  - `spacing` 为**方格边长**（mm）

- `type = circles`：
  - `rows/cols` 表示“圆点数”
  - `spacing` 为**圆心间距**（mm）

---

## 运行

```bash
python3 /workspace/hand_eye_calibrate.py --config /workspace/hand_eye_config.json
```

---

## 如何判断结果可靠

脚本会输出：
- **平均重投影误差**（像素，越小越好）
- 每组样本计算得到的 **base->board**（标定板在基座系下的位姿），理想情况下应当稳定
- `handeye_report.json` 中给出 base->board 平移的均值与标准差（std 越小越稳定）

