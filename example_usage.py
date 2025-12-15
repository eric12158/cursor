"""
线束位置检测使用示例
"""

import cv2
import numpy as np
from wiring_harness_detection import WiringHarnessDetector, process_image_file


def example_single_image():
    """示例1：处理单张图像"""
    print("=" * 50)
    print("示例1：处理单张图像")
    print("=" * 50)
    
    # 注意：需要替换为实际的图像路径
    image_path = "your_image.jpg"
    
    try:
        positions = process_image_file(image_path, "result.jpg")
        
        print(f"\n检测到 {len(positions)} 个位置:")
        for i, (x, y) in enumerate(positions):
            print(f"  位置 {i+1}: X={x:.2f}, Y={y:.2f}")
    
    except FileNotFoundError:
        print(f"错误：找不到图像文件 {image_path}")
        print("请将图像路径替换为实际的文件路径")


def example_custom_detector():
    """示例2：自定义检测器参数"""
    print("\n" + "=" * 50)
    print("示例2：自定义检测器参数")
    print("=" * 50)
    
    # 创建检测器
    detector = WiringHarnessDetector()
    
    # 自定义参数
    detector.params.minArea = 100  # 最小面积
    detector.params.maxArea = 50000  # 最大面积
    
    # 重新创建检测器
    detector.detector = cv2.SimpleBlobDetector_create(detector.params)
    
    # 读取图像
    image_path = "your_image.jpg"
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError
        
        # 检测位置
        positions = detector.detect_harness_positions(image, expected_positions=5)
        
        # 可视化
        result = detector.visualize_results(image, positions)
        cv2.imwrite("custom_result.jpg", result)
        
        print(f"检测完成，结果保存到 custom_result.jpg")
        print(f"检测到 {len(positions)} 个位置")
    
    except FileNotFoundError:
        print(f"错误：找不到图像文件 {image_path}")


def example_batch_processing():
    """示例3：批量处理多张图像"""
    print("\n" + "=" * 50)
    print("示例3：批量处理多张图像")
    print("=" * 50)
    
    import os
    import glob
    
    # 图像文件夹路径
    image_folder = "images/"
    output_folder = "results/"
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.png"))
    
    if not image_files:
        print(f"在 {image_folder} 中未找到图像文件")
        return
    
    detector = WiringHarnessDetector()
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"result_{filename}")
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            positions = detector.detect_harness_positions(image, expected_positions=5)
            result = detector.visualize_results(image, positions)
            
            cv2.imwrite(output_path, result)
            print(f"处理完成: {filename} -> 检测到 {len(positions)} 个位置")
        
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")


def example_create_test_image():
    """示例4：创建测试图像"""
    print("\n" + "=" * 50)
    print("示例4：创建测试图像")
    print("=" * 50)
    
    # 创建一个测试图像（白色背景，黑色线束）
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 模拟5个线束位置（黑色圆形）
    test_positions = [
        (150, 150),
        (400, 200),
        (650, 150),
        (200, 450),
        (600, 450)
    ]
    
    for x, y in test_positions:
        cv2.circle(image, (x, y), 30, (0, 0, 0), -1)
    
    # 保存测试图像
    cv2.imwrite("test_image.jpg", image)
    print("测试图像已创建: test_image.jpg")
    
    # 使用检测器检测
    detector = WiringHarnessDetector()
    positions = detector.detect_harness_positions(image, expected_positions=5)
    
    print(f"\n检测结果:")
    for i, (x, y) in enumerate(positions):
        print(f"  位置 {i+1}: ({x:.2f}, {y:.2f})")
    
    # 可视化结果
    result = detector.visualize_results(image, positions)
    cv2.imwrite("test_result.jpg", result)
    print("\n结果图像已保存: test_result.jpg")


def example_roi_detection():
    """示例5：在感兴趣区域（ROI）中检测"""
    print("\n" + "=" * 50)
    print("示例5：ROI区域检测")
    print("=" * 50)
    
    image_path = "your_image.jpg"
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError
        
        # 定义ROI区域（x, y, width, height）
        roi_x, roi_y, roi_w, roi_h = 100, 100, 500, 400
        roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # 在ROI中检测
        detector = WiringHarnessDetector()
        positions = detector.detect_harness_positions(roi, expected_positions=5)
        
        # 将ROI坐标转换为原图坐标
        global_positions = [(x + roi_x, y + roi_y) for x, y in positions]
        
        # 可视化（在原图上）
        result = detector.visualize_results(image, global_positions)
        cv2.rectangle(result, (roi_x, roi_y), 
                     (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        
        cv2.imwrite("roi_result.jpg", result)
        print(f"ROI检测完成，结果保存到 roi_result.jpg")
        print(f"检测到 {len(global_positions)} 个位置")
    
    except FileNotFoundError:
        print(f"错误：找不到图像文件 {image_path}")


if __name__ == "__main__":
    print("线束位置检测示例程序")
    print("\n请选择要运行的示例:")
    print("1. 处理单张图像")
    print("2. 自定义检测器参数")
    print("3. 批量处理多张图像")
    print("4. 创建测试图像（推荐先运行此示例）")
    print("5. ROI区域检测")
    
    choice = input("\n请输入选项 (1-5): ")
    
    if choice == "1":
        example_single_image()
    elif choice == "2":
        example_custom_detector()
    elif choice == "3":
        example_batch_processing()
    elif choice == "4":
        example_create_test_image()
    elif choice == "5":
        example_roi_detection()
    else:
        print("无效选项，运行示例4（创建测试图像）...")
        example_create_test_image()
