"""
NTC温度传感器校准工具
支持Beta方程和Steinhart-Hart方程
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NTCCalibration:
    """NTC校准类"""
    
    @staticmethod
    def beta_equation(resistance: np.ndarray, r0: float, t0: float, beta: float) -> np.ndarray:
        """
        Beta方程：1/T = 1/T0 + (1/Beta) * ln(R/R0)
        
        Args:
            resistance: 电阻值（Ω）
            r0: 参考温度下的电阻值（Ω）
            t0: 参考温度（°C）
            beta: Beta值（K）
        
        Returns:
            temperature: 温度（°C）
        """
        t0_kelvin = t0 + 273.15
        resistance = np.maximum(resistance, 0.001)
        ratio = np.maximum(resistance / r0, 0.001)
        
        inv_t = 1.0 / t0_kelvin + (1.0 / beta) * np.log(ratio)
        temperature_kelvin = 1.0 / inv_t
        temperature = temperature_kelvin - 273.15
        
        return temperature
    
    @staticmethod
    def steinhart_hart_equation(resistance: np.ndarray, 
                                a: float, b: float, c: float) -> np.ndarray:
        """
        Steinhart-Hart方程：1/T = A + B*ln(R) + C*(ln(R))³
        更精确，但需要3个校准点
        
        Args:
            resistance: 电阻值（Ω）
            a: Steinhart-Hart系数A
            b: Steinhart-Hart系数B
            c: Steinhart-Hart系数C
        
        Returns:
            temperature: 温度（°C）
        """
        resistance = np.maximum(resistance, 0.001)
        ln_r = np.log(resistance)
        
        inv_t = a + b * ln_r + c * ln_r ** 3
        temperature_kelvin = 1.0 / inv_t
        temperature = temperature_kelvin - 273.15
        
        return temperature
    
    @staticmethod
    def calculate_steinhart_hart_coefficients(
        temperatures: List[float],
        resistances: List[float]
    ) -> Tuple[float, float, float]:
        """
        从校准点计算Steinhart-Hart系数
        需要至少3个校准点，建议使用更多点以提高精度
        
        Args:
            temperatures: 温度列表（°C）
            resistances: 对应的电阻值列表（Ω）
        
        Returns:
            (a, b, c): Steinhart-Hart系数
        """
        if len(temperatures) < 3:
            raise ValueError("至少需要3个校准点")
        
        if len(temperatures) != len(resistances):
            raise ValueError("温度和电阻值数量必须相同")
        
        # 转换为开尔文
        t_kelvin = np.array(temperatures) + 273.15
        r = np.array(resistances)
        
        # 构建矩阵方程
        # 1/T = A + B*ln(R) + C*(ln(R))³
        ln_r = np.log(r)
        y = 1.0 / t_kelvin
        x = np.column_stack([np.ones_like(ln_r), ln_r, ln_r ** 3])
        
        # 最小二乘法求解
        coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        a, b, c = coeffs
        
        logger.info(f"Steinhart-Hart系数: A={a:.10e}, B={b:.10e}, C={c:.10e}")
        return float(a), float(b), float(c)
    
    @staticmethod
    def calculate_beta_from_points(
        t1: float, r1: float,
        t2: float, r2: float
    ) -> float:
        """
        从两个校准点计算Beta值
        
        Args:
            t1, t2: 温度（°C）
            r1, r2: 对应的电阻值（Ω）
        
        Returns:
            beta: Beta值（K）
        """
        t1_k = t1 + 273.15
        t2_k = t2 + 273.15
        
        beta = np.log(r2 / r1) / (1.0 / t1_k - 1.0 / t2_k)
        
        logger.info(f"从校准点计算的Beta值: {beta:.2f} K")
        return float(beta)
    
    @staticmethod
    def compare_equations(
        resistance: np.ndarray,
        r0: float, t0: float, beta: float,
        sh_a: Optional[float] = None,
        sh_b: Optional[float] = None,
        sh_c: Optional[float] = None
    ) -> dict:
        """
        比较Beta方程和Steinhart-Hart方程的结果
        
        Returns:
            比较结果字典
        """
        temp_beta = NTCCalibration.beta_equation(resistance, r0, t0, beta)
        
        result = {
            'beta_temperature': temp_beta,
            'beta_mean': float(np.mean(temp_beta)),
            'beta_std': float(np.std(temp_beta))
        }
        
        if sh_a is not None and sh_b is not None and sh_c is not None:
            temp_sh = NTCCalibration.steinhart_hart_equation(resistance, sh_a, sh_b, sh_c)
            result['steinhart_hart_temperature'] = temp_sh
            result['steinhart_hart_mean'] = float(np.mean(temp_sh))
            result['steinhart_hart_std'] = float(np.std(temp_sh))
            result['difference'] = temp_sh - temp_beta
            result['max_difference'] = float(np.max(np.abs(result['difference'])))
            result['mean_difference'] = float(np.mean(result['difference']))
        
        return result


# 示例使用
if __name__ == "__main__":
    # 示例：校准一个10kΩ@25°C的NTC传感器
    # 假设已知几个校准点
    calibration_points = {
        'temperatures': [0, 25, 50, 75, 100],  # °C
        'resistances': [32650, 10000, 3600, 1500, 680]  # Ω (示例值)
    }
    
    # 计算Steinhart-Hart系数
    sh_coeffs = NTCCalibration.calculate_steinhart_hart_coefficients(
        calibration_points['temperatures'],
        calibration_points['resistances']
    )
    
    print(f"Steinhart-Hart系数: A={sh_coeffs[0]:.10e}, "
          f"B={sh_coeffs[1]:.10e}, C={sh_coeffs[2]:.10e}")
    
    # 计算Beta值（使用25°C和50°C点）
    beta = NTCCalibration.calculate_beta_from_points(
        25, 10000,
        50, 3600
    )
    print(f"Beta值: {beta:.2f} K")
    
    # 测试转换
    test_resistance = np.array([10000, 5000, 2000, 1000])
    temp_beta = NTCCalibration.beta_equation(test_resistance, 10000, 25, beta)
    temp_sh = NTCCalibration.steinhart_hart_equation(
        test_resistance, sh_coeffs[0], sh_coeffs[1], sh_coeffs[2]
    )
    
    print("\n电阻值 -> 温度转换对比:")
    print("电阻(Ω)  Beta方程(°C)  Steinhart-Hart(°C)  差值(°C)")
    for r, tb, ts in zip(test_resistance, temp_beta, temp_sh):
        print(f"{r:8.0f}  {tb:12.2f}  {ts:18.2f}  {ts-tb:10.3f}")
