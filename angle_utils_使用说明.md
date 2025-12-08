# angle_utils.py 使用说明

## 概述

`angle_utils.py` 是一个独立的角度工具函数库，提供角度规范化功能，可以在任何需要处理角度的项目中直接使用。

## 功能函数

### 1. `normalize_angle_deg(angle_deg)`

将角度规范化到 [-180, 180] 度范围。

**参数：**
- `angle_deg`: 角度（度），可以是任意值

**返回：**
- 规范化后的角度（度），范围 [-180, 180]

**示例：**
```python
from angle_utils import normalize_angle_deg

print(normalize_angle_deg(181))    # 输出: -179
print(normalize_angle_deg(-181))   # 输出: 179
print(normalize_angle_deg(360))    # 输出: 0
print(normalize_angle_deg(540))    # 输出: 180
```

---

### 2. `normalize_angle_rad(angle_rad)`

将角度规范化到 [-π, π] 弧度范围。

**参数：**
- `angle_rad`: 角度（弧度），可以是任意值

**返回：**
- 规范化后的角度（弧度），范围 [-π, π]

**示例：**
```python
import math
from angle_utils import normalize_angle_rad

print(normalize_angle_rad(math.pi + 0.1))      # 输出: -3.04159...
print(normalize_angle_rad(-math.pi - 0.1))     # 输出: 3.04159...
print(normalize_angle_rad(2 * math.pi))         # 输出: 0.0
```

---

### 3. `angle_diff_deg(angle1_deg, angle2_deg)`

计算两个角度之间的最短角度差（度），自动处理角度跨越 ±180° 的情况。

**参数：**
- `angle1_deg`: 第一个角度（度）
- `angle2_deg`: 第二个角度（度）

**返回：**
- 角度差（度），范围 [-180, 180]，表示从 `angle2` 到 `angle1` 的最短旋转角度

**示例：**
```python
from angle_utils import angle_diff_deg

# 处理角度跨越 ±180° 的情况
print(angle_diff_deg(179, -179))   # 输出: -2（而不是 358）
print(angle_diff_deg(-179, 179))    # 输出: 2（而不是 -358）

# 普通情况
print(angle_diff_deg(10, 350))      # 输出: 20
print(angle_diff_deg(350, 10))      # 输出: -20
```

---

## 使用方法

### 在其他项目中导入使用

```python
# 方法1：导入所有函数
from angle_utils import normalize_angle_deg, normalize_angle_rad, angle_diff_deg

# 方法2：导入整个模块
import angle_utils
angle = angle_utils.normalize_angle_deg(181)

# 方法3：导入并重命名
from angle_utils import normalize_angle_deg as norm_deg
angle = norm_deg(181)
```

### 在 qr_positioning.py 中的使用

```python
import math
from angle_utils import normalize_angle_deg, normalize_angle_rad, angle_diff_deg

# 在函数中使用
def getDrAndDl(pA, pB):
    # ...
    normalized_angle = normalize_angle_deg(pA[2])  # 规范化角度
    ddr2 = normalize_angle_rad(ddr2)              # 规范化弧度
    # ...

# 在主函数中使用
c = angle_diff_deg(a[2], point[2])  # 计算角度差，自动处理跨越±180°的情况
```

---

## 应用场景

1. **角度规范化**：将任意角度值规范化到标准范围
2. **角度差计算**：计算两个角度之间的最短角度差，避免跨越 ±180° 时的错误
3. **机器人/导航系统**：处理旋转角度和方向计算
4. **图形处理**：处理图像旋转角度
5. **游戏开发**：处理角色朝向和旋转

---

## 注意事项

1. **角度范围**：所有角度都规范化到 [-180, 180] 度或 [-π, π] 弧度
2. **角度差方向**：`angle_diff_deg(a, b)` 返回从 `b` 到 `a` 的最短角度差
3. **等价角度**：180° 和 -180° 在数学上等价，函数可能返回其中任意一个
4. **性能**：使用 while 循环实现，对于极端大的角度值可能需要多次迭代

---

## 测试

运行测试文件验证功能：

```bash
python3 angle_utils.py          # 运行内置测试
python3 test_angle_normalization.py  # 运行完整测试套件
```
