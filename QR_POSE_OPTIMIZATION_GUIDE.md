# 二维码姿态估计优化指南

## 问题分析

### 原始问题
同一位置拍摄的不同图片，旋转角度（Rx, Ry）误差很大：

| 图片 | Rx | Ry | Rz | 距离(mm) |
|------|----|----|----|----|
| 1 | 6.16° | 6.18° | 86.47° | 250.40 |
| 2 | 3.53° | -1.48° | 85.58° | 296.76 |
| 3 | -5.60° | -3.70° | 85.63° | 294.25 |

**观察到的问题：**
- Rx 变化范围：11.76° (-5.60° ~ 6.16°)
- Ry 变化范围：9.88° (-3.70° ~ 6.18°)
- Rz 相对稳定：约85-86°

### 原因分析

当二维码与相机平面**接近平行**时（Rz ≈ 85-86°），会出现以下问题：

1. **数值不稳定性**：小的像素误差会被放大为大的角度误差
2. **万向节死锁**：接近奇异点时，欧拉角表示不唯一
3. **角点检测误差**：整数像素级别的定位误差累积
4. **透视变换敏感**：平面物体的透视畸变影响

## 优化策略

### 1. 使用 SOLVEPNP_IPPE_SQUARE

**原理：**
- IPPE (Infinitesimal Plane-based Pose Estimation) 是专为平面物体设计的闭式解法
- IPPE_SQUARE 进一步针对**正方形**（如二维码）进行优化
- 直接求解，无迭代，避免局部最优

**优势：**
```python
# 传统方法（迭代）
cv2.solvePnP(..., flags=cv2.SOLVEPNP_ITERATIVE)  # 可能陷入局部最优

# 优化方法（闭式解）
cv2.solvePnP(..., flags=cv2.SOLVEPNP_IPPE_SQUARE)  # 稳定、快速
```

**适用条件：**
- ✅ 正方形平面目标（二维码完美适用）
- ✅ 必须使用4个角点（不能用5点）
- ✅ 角点顺序正确（TL, TR, BR, BL）

### 2. 增强图像预处理

**A. CLAHE 对比度增强**
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
```
- 提高局部对比度
- 改善不均匀光照条件下的角点检测

**B. 双边滤波**
```python
gray = cv2.bilateralFilter(gray, 5, 50, 50)
```
- 保留边缘的同时降噪
- 比高斯模糊更适合角点检测

### 3. 优化亚像素参数

**增大搜索窗口：**
```python
# 原版：11x11 窗口
winSize=(5, 5)  # 5*2+1 = 11

# 优化版：21x21 窗口
winSize=(10, 10)  # 10*2+1 = 21
```

**更严格的收敛条件：**
```python
# 原版：30次迭代，0.001精度
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 优化版：100次迭代，0.0001精度
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
```

### 4. 双方法对比选择

程序同时使用两种方法：

**方法1：IPPE_SQUARE（4点）**
- 专为正方形设计
- 更稳定，特别是接近平行时

**方法2：迭代法（5点）**
- 使用中心点增加约束
- 通用性更好

**自动选择：**
```python
if error1 <= error2:
    使用 IPPE_SQUARE 结果
else:
    使用 5点迭代结果
```

### 5. 重投影误差验证

计算并显示最终的重投影误差：
```python
final_error = np.mean(np.linalg.norm(检测点 - 重投影点, axis=1))
```

**经验阈值：**
- ✅ < 0.5 像素：优秀
- ⚠️ 0.5-1.0 像素：良好
- ❌ > 1.0 像素：需要检查

## 预期效果

### 角度稳定性提升

优化前（你的数据）：
- Rx 波动：±5-6°
- Ry 波动：±4-5°

优化后预期：
- Rx 波动：±0.5-1.5°（提升**3-5倍**）
- Ry 波动：±0.5-1.5°（提升**3-5倍**）
- Rz 波动：±0.3-0.8°（已经较稳定）

### 距离测量精度

优化前：
- 距离波动：250.40 ~ 296.76 mm（46mm 差异）

优化后预期：
- 距离波动：< 5-10 mm（提升**5-10倍**）

## 使用建议

### 1. 拍摄条件优化

**光照：**
- ✅ 均匀、充足的光照
- ❌ 避免强烈阴影和高光

**角度：**
- ✅ 尽量正对二维码（Rz 接近 90°）
- ⚠️ 如果必须倾斜，保持 Rz 在 60-120° 范围内
- ❌ 避免极端角度（< 30° 或 > 150°）

**距离：**
- ✅ 二维码占画面的 20-60%
- ❌ 避免太近（模糊）或太远（像素不足）

### 2. 多帧平均（可选）

如果需要更高稳定性，可以实现：

```python
# 伪代码
results = []
for frame in frames:
    rvec, tvec = detect_and_solve(frame)
    results.append((rvec, tvec))

# 平均旋转向量和平移向量
avg_rvec = np.mean([r for r, t in results], axis=0)
avg_tvec = np.mean([t for r, t in results], axis=0)
```

### 3. 卡尔曼滤波（高级）

对于视频流实时处理：

```python
# 使用卡尔曼滤波平滑结果
kalman = cv2.KalmanFilter(6, 6)  # 6维状态：rvec(3) + tvec(3)
prediction = kalman.predict()
estimation = kalman.correct(measurement)
```

## 验证方法

### 1. 重投影误差

查看日志中的"最终重投影误差"：
```
最终重投影误差: 0.2345 像素  ← 应该 < 0.5
```

### 2. 固定位置多次测试

将相机和二维码固定，拍摄10-20张图片：
```python
# 计算标准差
rx_std = np.std([rx1, rx2, rx3, ...])  # 应该 < 1.5°
ry_std = np.std([ry1, ry2, ry3, ...])  # 应该 < 1.5°
dist_std = np.std([d1, d2, d3, ...])   # 应该 < 5mm
```

### 3. 与真实值对比

使用精确测量工具（如激光测距仪）验证距离：
```
测量距离：300.0 mm
计算距离：298.5 mm
误差：1.5 mm (0.5%)  ← 良好
```

## 故障排查

### 问题1：角度仍然不稳定

**可能原因：**
1. 相机标定质量差 → 重新标定
2. 图像模糊 → 改善对焦
3. 二维码打印质量差 → 使用高质量打印

### 问题2：距离误差大

**可能原因：**
1. 二维码尺寸输入错误 → 精确测量实际尺寸
2. 分辨率不匹配 → 检查标定分辨率与当前图片是否一致
3. 畸变校正不足 → 使用更多标定图片

### 问题3：重投影误差大

**可能原因：**
1. 角点检测失败 → 改善光照和对比度
2. 镜头畸变严重 → 使用更好的镜头或重新标定
3. 二维码变形 → 确保二维码平整

## 技术参考

### OpenCV 文档
- [solvePnP](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
- [cornerSubPix](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e)

### 算法论文
- IPPE: "Infinitesimal Plane-Based Pose Estimation" (Collins & Bartoli, 2014)
- EPnP: "Accurate O(n) Solution to the PnP Problem" (Lepetit et al., 2009)

---

**版本：** V2.5  
**最后更新：** 2025-01-03
