"""
线束位置检测系统
使用斑点分析（Blob Analysis）动态检测图像中线束的5个位置
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import json


class WiringHarnessDetector:
    """线束位置检测器"""
    
    def __init__(self):
        """初始化检测器"""
        # 创建SimpleBlobDetector参数
        self.params = cv2.SimpleBlobDetector_Params()
        
        # 设置斑点检测参数
        self.params.filterByArea = True
        self.params.minArea = 50  # 最小面积，根据实际情况调整
        self.params.maxArea = 50000  # 最大面积
        
        self.params.filterByCircularity = False  # 不按圆形度过滤
        self.params.filterByConvexity = False  # 不按凸性过滤
        self.params.filterByInertia = False  # 不按惯性比过滤
        
        # 颜色过滤（如果需要）
        self.params.filterByColor = False
        
        # 创建检测器
        self.detector = cv2.SimpleBlobDetector_create(self.params)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        包括：灰度化、降噪、二值化等
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值二值化（适用于光照不均匀的情况）
        binary = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # 形态学操作：去除小噪点，连接断开的线束
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def detect_blobs(self, binary_image: np.ndarray) -> List[cv2.KeyPoint]:
        """
        使用斑点分析检测线束位置
        """
        # 检测斑点
        keypoints = self.detector.detect(binary_image)
        return keypoints
    
    def filter_and_sort_keypoints(
        self, 
        keypoints: List[cv2.KeyPoint], 
        expected_count: int = 5
    ) -> List[cv2.KeyPoint]:
        """
        过滤和排序关键点，选择最可能的5个线束位置
        """
        if len(keypoints) == 0:
            return []
        
        # 按面积或响应强度排序（响应强度通常表示检测的置信度）
        sorted_kpts = sorted(keypoints, key=lambda x: x.size, reverse=True)
        
        # 如果检测到的点少于期望数量，返回所有点
        if len(sorted_kpts) <= expected_count:
            return sorted_kpts
        
        # 如果检测到的点多于期望数量，选择前N个
        return sorted_kpts[:expected_count]
    
    def detect_harness_positions(
        self, 
        image: np.ndarray, 
        expected_positions: int = 5
    ) -> List[Tuple[float, float]]:
        """
        主检测函数：检测线束的5个位置
        
        参数:
            image: 输入图像（BGR或灰度）
            expected_positions: 期望检测到的位置数量（默认5）
        
        返回:
            位置列表，每个位置是(x, y)坐标元组
        """
        # 1. 图像预处理
        binary = self.preprocess_image(image)
        
        # 2. 斑点检测
        keypoints = self.detect_blobs(binary)
        
        # 3. 过滤和排序
        filtered_keypoints = self.filter_and_sort_keypoints(
            keypoints, 
            expected_positions
        )
        
        # 4. 提取坐标
        positions = [(kp.pt[0], kp.pt[1]) for kp in filtered_keypoints]
        
        return positions
    
    def visualize_results(
        self, 
        image: np.ndarray, 
        positions: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        可视化检测结果
        """
        result_image = image.copy()
        
        # 在图像上标记检测到的位置
        for i, (x, y) in enumerate(positions):
            # 绘制圆圈标记
            cv2.circle(result_image, (int(x), int(y)), 10, (0, 255, 0), 2)
            # 添加编号
            cv2.putText(
                result_image, 
                str(i + 1), 
                (int(x) + 15, int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        return result_image


def process_image_file(
    image_path: str, 
    output_path: Optional[str] = None
) -> List[Tuple[float, float]]:
    """
    处理图像文件并返回检测到的位置
    
    参数:
        image_path: 输入图像路径
        output_path: 可选，输出结果图像路径
    
    返回:
        检测到的位置列表
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 创建检测器
    detector = WiringHarnessDetector()
    
    # 检测位置
    positions = detector.detect_harness_positions(image, expected_positions=5)
    
    # 可视化结果
    if output_path:
        result_image = detector.visualize_results(image, positions)
        cv2.imwrite(output_path, result_image)
        print(f"结果图像已保存到: {output_path}")
    
    return positions


def process_video_stream(
    camera_index: int = 0,
    save_results: bool = False
):
    """
    处理视频流（实时检测）
    
    参数:
        camera_index: 摄像头索引（默认0）
        save_results: 是否保存检测结果
    """
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开摄像头 {camera_index}")
    
    # 创建检测器
    detector = WiringHarnessDetector()
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测位置
            positions = detector.detect_harness_positions(frame, expected_positions=5)
            
            # 可视化结果
            result_frame = detector.visualize_results(frame, positions)
            
            # 显示结果
            cv2.imshow('线束位置检测', result_frame)
            
            # 打印检测到的位置
            if len(positions) > 0:
                print(f"帧 {frame_count}: 检测到 {len(positions)} 个位置")
                for i, (x, y) in enumerate(positions):
                    print(f"  位置 {i+1}: ({x:.2f}, {y:.2f})")
            
            # 保存结果（可选）
            if save_results and len(positions) == 5:
                cv2.imwrite(f"result_frame_{frame_count}.jpg", result_frame)
                with open(f"positions_{frame_count}.json", 'w') as f:
                    json.dump(positions, f, indent=2)
            
            frame_count += 1
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 处理单张图像
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "result.jpg"
        
        positions = process_image_file(image_path, output_path)
        print(f"检测到 {len(positions)} 个位置:")
        for i, (x, y) in enumerate(positions):
            print(f"位置 {i+1}: ({x:.2f}, {y:.2f})")
    else:
        # 实时视频流处理
        print("启动实时检测... 按'q'退出")
        process_video_stream()
