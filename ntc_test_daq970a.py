"""
NTC温度传感器测试脚本
使用Keysight DAQ970A多通道数据采集仪表 + DAQM909A 4通道同步采样模块

测试参数：
- 测试时长：15秒
- 采样频率：50Hz
- 通道配置：DAQM909A 4通道同步采样

作者：Auto
日期：2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from typing import List, Tuple, Optional, Dict
import logging
from data_processor import NTCDataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ntc_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NTC_DAQ970A_Test:
    """NTC测试类，使用DAQ970A+DAQM909A进行数据采集"""
    
    def __init__(self, config_file: str = "config.json", resource_name: Optional[str] = None):
        """
        初始化测试系统
        
        Args:
            config_file: 配置文件路径
            resource_name: VISA资源名称（可选，会覆盖配置文件中的设置）
        """
        # 加载配置
        self.config = self.load_config(config_file)
        
        # 设置参数
        self.resource_name = resource_name or self.config['instrument']['resource_name']
        self.instrument = None
        self.test_duration = self.config['test_parameters']['duration']
        self.sample_rate = self.config['test_parameters']['sample_rate']
        self.total_samples = int(self.test_duration * self.sample_rate)
        self.channels = self.config['test_parameters']['channels']
        self.ntc_params = self.config['ntc_parameters']
        self.data = {}
        self.timestamps = None
        
        # 初始化数据处理器
        dp_config = self.config.get('data_processing', {})
        self.data_processor = NTCDataProcessor(
            filter_enabled=dp_config.get('filter_enabled', True),
            filter_type=dp_config.get('filter_type', 'moving_average'),
            filter_window=dp_config.get('filter_window', 5),
            outlier_threshold=dp_config.get('outlier_threshold', 3.0)
        )
        
        # 输出配置
        self.output_config = self.config.get('output', {})
        if self.output_config.get('output_directory'):
            os.makedirs(self.output_config['output_directory'], exist_ok=True)
    
    def load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"已加载配置文件: {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_file} 不存在，使用默认配置")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}，使用默认配置")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'instrument': {
                'resource_name': 'TCPIP0::192.168.1.100::inst0::INSTR',
                'timeout': 20000
            },
            'test_parameters': {
                'duration': 15.0,
                'sample_rate': 50.0,
                'channels': [101, 102, 103, 104]
            },
            'ntc_parameters': {
                'r0': 10000.0,
                't0': 25.0,
                'beta': 3950.0
            },
            'data_processing': {
                'filter_enabled': True,
                'filter_type': 'moving_average',
                'filter_window': 5,
                'outlier_threshold': 3.0
            },
            'output': {
                'save_data': True,
                'save_plot': True,
                'output_directory': './results'
            }
        }
        
    def connect(self):
        """连接仪器"""
        try:
            import pyvisa
            rm = pyvisa.ResourceManager()
            self.instrument = rm.open_resource(self.resource_name)
            timeout = self.config['instrument'].get('timeout', 20000)
            self.instrument.timeout = timeout
            logger.info(f"成功连接到仪器: {self.resource_name}")
            
            # 查询仪器ID
            idn = self.instrument.query("*IDN?")
            logger.info(f"仪器信息: {idn.strip()}")
            return True
        except ImportError:
            logger.error("请安装pyvisa库: pip install pyvisa pyvisa-py")
            return False
        except Exception as e:
            logger.error(f"连接仪器失败: {e}")
            return False
    
    def configure_channels(self):
        """配置DAQM909A的4个通道"""
        try:
            # 配置通道为温度测量模式（NTC）
            # DAQM909A通道编号：101-104
            for channel in self.channels:
                # 设置为温度测量，使用NTC传感器
                # 根据实际NTC参数调整，这里假设使用10kΩ NTC
                self.instrument.write(f"CONF:TEMP RTD,PT100,{channel}")
                # 或者使用电阻测量模式，然后转换为温度
                # self.instrument.write(f"CONF:RES 10000,{channel}")
                
                # 设置采样速率
                self.instrument.write(f"SENS:TEMP:NPLC 0.02,{channel}")  # 对应50Hz采样率
                
            logger.info(f"已配置通道: {self.channels}")
            return True
        except Exception as e:
            logger.error(f"配置通道失败: {e}")
            return False
    
    def configure_scan_list(self):
        """配置扫描列表，使用同步采样"""
        try:
            # 创建扫描列表，包含所有4个通道
            channel_list = ",".join([str(ch) for ch in self.channels])
            self.instrument.write(f"ROUT:SCAN (@{channel_list})")
            
            # 设置扫描间隔（对应50Hz采样率，间隔20ms）
            scan_interval = 1.0 / self.sample_rate  # 0.02秒
            self.instrument.write(f"ROUT:SCAN:INT {scan_interval}")
            
            logger.info(f"扫描列表已配置: {channel_list}")
            return True
        except Exception as e:
            logger.error(f"配置扫描列表失败: {e}")
            return False
    
    def start_measurement(self) -> Tuple[np.ndarray, dict]:
        """
        开始测量
        
        Returns:
            timestamps: 时间戳数组
            data: 各通道数据字典
        """
        try:
            logger.info(f"开始测量，时长: {self.test_duration}秒，采样率: {self.sample_rate}Hz")
            
            # 初始化数据存储
            data = {ch: [] for ch in self.channels}
            timestamps = []
            
            # 开始扫描
            self.instrument.write("INIT")
            
            start_time = time.time()
            sample_count = 0
            
            while (time.time() - start_time) < self.test_duration:
                # 读取扫描数据
                try:
                    # 使用FETCH?读取最新数据
                    readings = self.instrument.query("FETCH?")
                    
                    # 解析数据（格式可能为：101,value1,102,value2,...）
                    values = readings.strip().split(',')
                    
                    # 提取各通道数据
                    for i, channel in enumerate(self.channels):
                        if i * 2 + 1 < len(values):
                            try:
                                value = float(values[i * 2 + 1])
                                data[channel].append(value)
                            except (ValueError, IndexError):
                                data[channel].append(np.nan)
                    
                    current_time = time.time() - start_time
                    timestamps.append(current_time)
                    sample_count += 1
                    
                    # 控制采样率
                    time.sleep(1.0 / self.sample_rate)
                    
                except Exception as e:
                    logger.warning(f"读取数据时出错: {e}")
                    continue
            
            # 转换为numpy数组
            self.timestamps = np.array(timestamps)
            for ch in self.channels:
                data[ch] = np.array(data[ch])
            
            self.data = data
            
            logger.info(f"测量完成，共采集 {sample_count} 个数据点")
            return self.timestamps, self.data
            
        except Exception as e:
            logger.error(f"测量过程出错: {e}")
            return None, {}
    
    def convert_resistance_to_temperature(self, resistance: np.ndarray, 
                                         r0: Optional[float] = None, 
                                         t0: Optional[float] = None,
                                         beta: Optional[float] = None) -> np.ndarray:
        """
        将NTC电阻值转换为温度值（使用Beta方程）
        
        Args:
            resistance: 电阻值数组（Ω）
            r0: 参考温度下的电阻值（Ω），默认从配置文件读取
            t0: 参考温度（°C），默认从配置文件读取
            beta: Beta值（K），默认从配置文件读取
        
        Returns:
            temperature: 温度数组（°C）
        """
        # 使用配置文件中的参数
        r0 = r0 or self.ntc_params['r0']
        t0 = t0 or self.ntc_params['t0']
        beta = beta or self.ntc_params['beta']
        
        # Beta方程: 1/T = 1/T0 + (1/Beta) * ln(R/R0)
        # T0 = t0 + 273.15 (转换为开尔文)
        t0_kelvin = t0 + 273.15
        
        # 避免除零和对数负数
        resistance = np.maximum(resistance, 0.001)
        ratio = resistance / r0
        ratio = np.maximum(ratio, 0.001)
        
        # 计算温度（开尔文）
        inv_t = 1.0 / t0_kelvin + (1.0 / beta) * np.log(ratio)
        temperature_kelvin = 1.0 / inv_t
        
        # 转换为摄氏度
        temperature = temperature_kelvin - 273.15
        
        return temperature
    
    def analyze_data(self) -> dict:
        """分析采集的数据（使用数据处理器）"""
        if not self.data:
            logger.warning("没有数据可分析")
            return {}
        
        analysis = {}
        processed_data = {}
        
        for channel in self.channels:
            if channel in self.data and len(self.data[channel]) > 0:
                values = self.data[channel]
                
                # 如果数据是电阻值，转换为温度
                if np.mean(values[~np.isnan(values)]) > 100:  # 可能是电阻值
                    temperatures = self.convert_resistance_to_temperature(values)
                else:  # 可能已经是温度值
                    temperatures = values
                
                # 使用数据处理器进行完整处理
                processed = self.data_processor.process_channel_data(
                    temperatures, self.timestamps
                )
                processed_data[channel] = processed
                
                # 提取统计信息
                if processed['statistics']:
                    analysis[channel] = processed['statistics']
        
        # 通道间比较
        if len(processed_data) > 1:
            channel_data = {str(ch): processed_data[ch]['filtered'] 
                          for ch in processed_data.keys() 
                          if processed_data[ch]['filtered'] is not None}
            if channel_data:
                comparison = self.data_processor.compare_channels(channel_data)
                analysis['channel_comparison'] = comparison
        
        return analysis
    
    def plot_results(self, save_path: Optional[str] = None):
        """绘制测试结果（包含原始数据和处理后数据）"""
        if not self.data or self.timestamps is None:
            logger.warning("没有数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'NTC测试结果 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                     fontsize=14, fontweight='bold')
        
        for idx, channel in enumerate(self.channels):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            if channel in self.data and len(self.data[channel]) > 0:
                values = self.data[channel]
                valid_mask = ~np.isnan(values)
                
                if np.sum(valid_mask) > 0:
                    # 转换为温度（如果需要）
                    if np.mean(values[valid_mask]) > 100:
                        temperatures = self.convert_resistance_to_temperature(values[valid_mask])
                        timestamps_valid = self.timestamps[valid_mask]
                    else:
                        temperatures = values[valid_mask]
                        timestamps_valid = self.timestamps[valid_mask]
                    
                    # 处理数据
                    processed = self.data_processor.process_channel_data(
                        temperatures, timestamps_valid
                    )
                    
                    # 绘制原始数据（浅色）
                    ax.plot(timestamps_valid, temperatures, 'lightblue', 
                           linewidth=0.5, alpha=0.5, label='原始数据')
                    
                    # 绘制滤波后的数据（深色）
                    if processed['filtered'] is not None:
                        filtered_valid = processed['filtered'][~np.isnan(processed['filtered'])]
                        if len(filtered_valid) > 0:
                            ax.plot(timestamps_valid[:len(filtered_valid)], filtered_valid, 
                                   'b-', linewidth=1.5, label='滤波后数据')
                    
                    ax.set_xlabel('时间 (秒)')
                    ax.set_ylabel('温度 (°C)')
                    ax.set_title(f'通道 {channel} - 温度曲线')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)
                    
                    # 添加统计信息
                    if processed['statistics']:
                        stats = processed['statistics']
                        info_text = (f'均值: {stats["mean"]:.2f}°C\n'
                                   f'标准差: {stats["std"]:.3f}°C\n'
                                   f'范围: {stats["min"]:.2f}~{stats["max"]:.2f}°C')
                        ax.text(0.02, 0.98, info_text,
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                               fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存至: {save_path}")
        
        plt.show()
    
    def save_data(self, filename: Optional[str] = None):
        """保存测试数据"""
        if not self.output_config.get('save_data', True):
            logger.info("数据保存已禁用")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_config.get('output_directory', '.')
            filename = os.path.join(output_dir, f"ntc_test_data_{timestamp}.json")
        
        save_data = {
            'test_info': {
                'duration': self.test_duration,
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'ntc_parameters': self.ntc_params,
                'timestamp': datetime.now().isoformat(),
                'instrument': self.resource_name
            },
            'raw_data': {},
            'processed_data': {}
        }
        
        # 保存原始数据
        for channel in self.channels:
            if channel in self.data:
                save_data['raw_data'][str(channel)] = self.data[channel].tolist()
        
        if self.timestamps is not None:
            save_data['timestamps'] = self.timestamps.tolist()
        
        # 处理并保存处理后的数据
        for channel in self.channels:
            if channel in self.data and len(self.data[channel]) > 0:
                values = self.data[channel]
                # 转换为温度
                if np.mean(values[~np.isnan(values)]) > 100:
                    temperatures = self.convert_resistance_to_temperature(values)
                else:
                    temperatures = values
                
                processed = self.data_processor.process_channel_data(
                    temperatures, self.timestamps
                )
                # 只保存关键数据
                save_data['processed_data'][str(channel)] = {
                    'filtered': processed['filtered'].tolist() if processed['filtered'] is not None else None,
                    'statistics': processed['statistics'],
                    'outlier_count': processed.get('outlier_count', 0)
                }
        
        # 添加分析结果
        analysis = self.analyze_data()
        save_data['analysis'] = analysis
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据已保存至: {filename}")
        return filename
    
    def disconnect(self):
        """断开连接"""
        if self.instrument:
            try:
                self.instrument.close()
                logger.info("已断开仪器连接")
            except:
                pass


def main():
    """主函数"""
    # 创建测试实例（从配置文件加载）
    test = NTC_DAQ970A_Test(config_file="config.json")
    
    try:
        # 连接仪器
        if not test.connect():
            logger.error("无法连接仪器，请检查连接和IP地址")
            return
        
        # 配置通道
        if not test.configure_channels():
            logger.error("通道配置失败")
            return
        
        # 配置扫描列表
        if not test.configure_scan_list():
            logger.error("扫描列表配置失败")
            return
        
        # 开始测量
        timestamps, data = test.start_measurement()
        
        if timestamps is not None and data:
            # 分析数据
            analysis = test.analyze_data()
            logger.info("数据分析结果:")
            for ch, stats in analysis.items():
                logger.info(f"通道 {ch}: 均值={stats['mean']:.2f}°C, "
                          f"标准差={stats['std']:.3f}°C, "
                          f"范围={stats['min']:.2f}~{stats['max']:.2f}°C")
            
            # 绘制结果
            if test.output_config.get('save_plot', True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = test.output_config.get('output_directory', '.')
                plot_file = os.path.join(output_dir, f"ntc_test_plot_{timestamp}.png")
                test.plot_results(save_path=plot_file)
            else:
                test.plot_results()
            
            # 保存数据
            test.save_data()
        else:
            logger.error("数据采集失败")
    
    except KeyboardInterrupt:
        logger.info("用户中断测试")
    except Exception as e:
        logger.error(f"测试过程出错: {e}", exc_info=True)
    finally:
        # 断开连接
        test.disconnect()


if __name__ == "__main__":
    main()
