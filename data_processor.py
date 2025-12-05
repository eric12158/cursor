"""
数据处理和分析模块
用于NTC测试数据的后处理、滤波、统计分析等
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NTCDataProcessor:
    """NTC数据处理器"""
    
    def __init__(self, filter_enabled: bool = True, 
                 filter_type: str = "moving_average",
                 filter_window: int = 5,
                 outlier_threshold: float = 3.0):
        """
        初始化数据处理器
        
        Args:
            filter_enabled: 是否启用滤波
            filter_type: 滤波类型 ('moving_average', 'median', 'butterworth')
            filter_window: 滤波窗口大小
            outlier_threshold: 异常值检测阈值（标准差倍数）
        """
        self.filter_enabled = filter_enabled
        self.filter_type = filter_type
        self.filter_window = filter_window
        self.outlier_threshold = outlier_threshold
    
    def remove_outliers(self, data: np.ndarray, method: str = "zscore") -> Tuple[np.ndarray, np.ndarray]:
        """
        移除异常值
        
        Args:
            data: 输入数据
            method: 方法 ('zscore', 'iqr')
        
        Returns:
            cleaned_data: 清理后的数据
            outlier_mask: 异常值掩码
        """
        if method == "zscore":
            z_scores = np.abs((data - np.nanmean(data)) / np.nanstd(data))
            outlier_mask = z_scores > self.outlier_threshold
        elif method == "iqr":
            q1 = np.nanpercentile(data, 25)
            q3 = np.nanpercentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (data < lower_bound) | (data > upper_bound)
        else:
            outlier_mask = np.zeros_like(data, dtype=bool)
        
        cleaned_data = data.copy()
        cleaned_data[outlier_mask] = np.nan
        
        logger.info(f"检测到 {np.sum(outlier_mask)} 个异常值")
        return cleaned_data, outlier_mask
    
    def moving_average_filter(self, data: np.ndarray, window: int = None) -> np.ndarray:
        """移动平均滤波"""
        if window is None:
            window = self.filter_window
        
        # 处理NaN值
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) == 0:
            return data
        
        filtered = np.full_like(data, np.nan)
        
        # 对有效数据进行滤波
        valid_data = data[valid_mask]
        if len(valid_data) >= window:
            filtered_valid = np.convolve(valid_data, np.ones(window)/window, mode='same')
            filtered[valid_mask] = filtered_valid
        
        return filtered
    
    def median_filter(self, data: np.ndarray, window: int = None) -> np.ndarray:
        """中值滤波"""
        if window is None:
            window = self.filter_window
        
        try:
            filtered = signal.medfilt(data, kernel_size=window)
            return filtered
        except:
            return data
    
    def butterworth_filter(self, data: np.ndarray, 
                          cutoff: float = 5.0, 
                          fs: float = 50.0, 
                          order: int = 4) -> np.ndarray:
        """
        巴特沃斯低通滤波
        
        Args:
            data: 输入数据
            cutoff: 截止频率 (Hz)
            fs: 采样频率 (Hz)
            order: 滤波器阶数
        """
        try:
            nyquist = fs / 2
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            filtered = signal.filtfilt(b, a, data)
            return filtered
        except:
            return data
    
    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """应用滤波"""
        if not self.filter_enabled:
            return data
        
        if self.filter_type == "moving_average":
            return self.moving_average_filter(data)
        elif self.filter_type == "median":
            return self.median_filter(data)
        elif self.filter_type == "butterworth":
            return self.butterworth_filter(data)
        else:
            logger.warning(f"未知的滤波类型: {self.filter_type}")
            return data
    
    def calculate_statistics(self, data: np.ndarray) -> Dict:
        """计算统计信息"""
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return {}
        
        stats = {
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'range': np.max(valid_data) - np.min(valid_data),
            'median': np.median(valid_data),
            'q25': np.percentile(valid_data, 25),
            'q75': np.percentile(valid_data, 75),
            'iqr': np.percentile(valid_data, 75) - np.percentile(valid_data, 25),
            'samples': len(valid_data),
            'valid_ratio': len(valid_data) / len(data)
        }
        
        return stats
    
    def calculate_derivative(self, data: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """计算导数（变化率）"""
        if len(data) < 2:
            return np.array([])
        
        dt = np.diff(timestamps)
        ddata = np.diff(data)
        
        # 避免除零
        dt = np.where(dt == 0, np.nan, dt)
        derivative = ddata / dt
        
        return derivative
    
    def detect_steady_state(self, data: np.ndarray, 
                           window: int = 50,
                           threshold: float = 0.1) -> Tuple[int, bool]:
        """
        检测稳态
        
        Args:
            data: 数据数组
            window: 检测窗口大小
            threshold: 稳定性阈值（标准差）
        
        Returns:
            steady_index: 达到稳态的索引
            is_steady: 是否达到稳态
        """
        if len(data) < window:
            return -1, False
        
        for i in range(window, len(data)):
            window_data = data[i-window:i]
            if np.nanstd(window_data) < threshold:
                return i, True
        
        return -1, False
    
    def process_channel_data(self, data: np.ndarray, 
                            timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        处理单个通道的完整数据
        
        Args:
            data: 原始数据
            timestamps: 时间戳（可选）
        
        Returns:
            处理结果字典
        """
        result = {
            'raw': data.copy(),
            'cleaned': None,
            'filtered': None,
            'statistics': {},
            'steady_state': {}
        }
        
        # 移除异常值
        cleaned, outlier_mask = self.remove_outliers(data)
        result['cleaned'] = cleaned
        result['outlier_count'] = int(np.sum(outlier_mask))
        
        # 应用滤波
        filtered = self.apply_filter(cleaned)
        result['filtered'] = filtered
        
        # 计算统计信息（使用滤波后的数据）
        result['statistics'] = self.calculate_statistics(filtered)
        
        # 检测稳态
        if timestamps is not None and len(filtered) > 50:
            steady_idx, is_steady = self.detect_steady_state(filtered)
            result['steady_state'] = {
                'index': int(steady_idx),
                'is_steady': is_steady,
                'time_to_steady': timestamps[steady_idx] if steady_idx >= 0 else None
            }
        
        # 计算变化率
        if timestamps is not None and len(filtered) > 1:
            derivative = self.calculate_derivative(filtered, timestamps)
            result['derivative'] = derivative.tolist()
            result['max_rate'] = float(np.nanmax(np.abs(derivative))) if len(derivative) > 0 else None
        
        return result
    
    def compare_channels(self, channel_data: Dict[str, np.ndarray]) -> Dict:
        """
        比较多个通道的数据
        
        Args:
            channel_data: 通道数据字典 {channel_name: data_array}
        
        Returns:
            比较结果
        """
        comparison = {
            'channel_stats': {},
            'cross_correlation': {},
            'mean_difference': {},
            'std_difference': {}
        }
        
        channel_names = list(channel_data.keys())
        
        # 计算各通道统计
        for ch_name, data in channel_data.items():
            comparison['channel_stats'][ch_name] = self.calculate_statistics(data)
        
        # 计算通道间相关性
        for i, ch1 in enumerate(channel_names):
            for ch2 in channel_names[i+1:]:
                data1 = channel_data[ch1]
                data2 = channel_data[ch2]
                
                # 找到共同的有效数据点
                valid_mask = ~(np.isnan(data1) | np.isnan(data2))
                if np.sum(valid_mask) > 10:
                    valid1 = data1[valid_mask]
                    valid2 = data2[valid_mask]
                    
                    # 计算相关系数
                    corr = np.corrcoef(valid1, valid2)[0, 1]
                    comparison['cross_correlation'][f"{ch1}_vs_{ch2}"] = float(corr)
                    
                    # 计算均值差和标准差差
                    mean_diff = np.mean(valid1) - np.mean(valid2)
                    std_diff = np.std(valid1) - np.std(valid2)
                    comparison['mean_difference'][f"{ch1}_vs_{ch2}"] = float(mean_diff)
                    comparison['std_difference'][f"{ch1}_vs_{ch2}"] = float(std_diff)
        
        return comparison
