# 线束位置检测系统

## 概述

本系统使用**斑点分析（Blob Analysis）**方法动态检测图像中线束的5个位置。斑点分析是一种有效的图像处理方法，特别适用于检测具有特定特征的连通区域。

## 为什么选择斑点分析？

### 优势：
1. **鲁棒性强**：对光照变化、噪声有一定的抗干扰能力
2. **计算效率高**：算法简单，处理速度快，适合实时应用
3. **参数可调**：可以根据线束特征灵活调整检测参数
4. **无需训练**：不需要机器学习模型，直接基于图像特征

### 适用场景：
- 线束在图像中呈现为明显的连通区域
- 线束与背景有较好的对比度
- 需要快速检测多个位置点

## 检测流程

### 完整流程步骤：

```
1. 图像输入
   ↓
2. 图像预处理
   ├─ 灰度化转换
   ├─ 高斯模糊降噪
   ├─ 自适应阈值二值化
   └─ 形态学操作（去噪、连接）
   ↓
3. 斑点检测
   ├─ 设置检测参数（面积范围、形状特征等）
   ├─ 执行SimpleBlobDetector检测
   └─ 获取所有候选斑点
   ↓
4. 结果过滤与排序
   ├─ 按面积或置信度排序
   ├─ 选择前5个最可能的点
   └─ 验证位置合理性
   ↓
5. 位置提取
   ├─ 提取坐标信息
   └─ 返回5个位置点
   ↓
6. 结果可视化（可选）
   └─ 在图像上标记检测到的位置
```

## 使用方法

### 1. 安装依赖

```bash
pip install opencv-python numpy
```

### 2. 处理单张图像

```python
from wiring_harness_detection import process_image_file

# 检测图像中的线束位置
positions = process_image_file('input_image.jpg', 'output_result.jpg')

# 打印结果
for i, (x, y) in enumerate(positions):
    print(f"位置 {i+1}: ({x:.2f}, {y:.2f})")
```

### 3. 命令行使用

```bash
# 处理单张图像
python wiring_harness_detection.py input.jpg output.jpg

# 实时视频流检测
python wiring_harness_detection.py
```

### 4. 在代码中使用

```python
import cv2
from wiring_harness_detection import WiringHarnessDetector

# 读取图像
image = cv2.imread('your_image.jpg')

# 创建检测器
detector = WiringHarnessDetector()

# 检测位置
positions = detector.detect_harness_positions(image, expected_positions=5)

# 可视化结果
result = detector.visualize_results(image, positions)
cv2.imshow('Result', result)
cv2.waitKey(0)
```

## 参数调整指南

### 关键参数说明：

1. **minArea / maxArea**：控制检测的斑点面积范围
   - 如果线束较大，增加maxArea
   - 如果检测到太多小噪点，增加minArea

2. **自适应阈值参数**：
   - `blockSize`：当前为11，控制局部区域大小
   - `C`：当前为2，控制阈值偏移量

3. **形态学操作**：
   - `MORPH_CLOSE`：连接断开的线束
   - `MORPH_OPEN`：去除小噪点

### 调整示例：

```python
detector = WiringHarnessDetector()

# 调整面积范围
detector.params.minArea = 100  # 增大最小面积，过滤小噪点
detector.params.maxArea = 100000  # 增大最大面积，检测更大的线束

# 重新创建检测器
detector.detector = cv2.SimpleBlobDetector_create(detector.params)
```

## 替代方案

如果斑点分析效果不理想，可以考虑以下方法：

### 1. **轮廓检测（Contour Detection）**
```python
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 分析轮廓特征，找到线束位置
```

### 2. **模板匹配（Template Matching）**
- 适用于线束形状固定的情况
- 需要先准备模板图像

### 3. **特征点检测（Feature Detection）**
- SIFT、ORB、AKAZE等特征点检测器
- 适用于线束有独特特征的情况

### 4. **深度学习方法**
- YOLO、Faster R-CNN等目标检测模型
- 需要标注数据和训练

## 性能优化建议

1. **图像分辨率**：如果图像很大，可以先缩放再处理
2. **ROI设置**：如果知道线束大致位置，可以设置感兴趣区域
3. **多线程**：视频流处理可以使用多线程提高帧率

## 故障排除

### 问题1：检测不到任何位置
- 检查图像对比度是否足够
- 调整minArea参数，减小阈值
- 检查预处理步骤是否正确

### 问题2：检测到太多位置
- 增加minArea参数
- 调整形态学操作参数
- 添加额外的过滤条件

### 问题3：检测位置不准确
- 检查图像质量（模糊、噪声）
- 调整预处理参数
- 考虑使用更高级的检测方法

## 示例输出

检测结果格式：
```json
[
  [123.45, 234.56],
  [345.67, 456.78],
  [567.89, 678.90],
  [789.01, 890.12],
  [901.23, 123.45]
]
```

## 许可证

MIT License
