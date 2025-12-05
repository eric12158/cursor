"""
NTC测试使用示例
演示如何使用测试脚本进行NTC温度传感器测试
"""

from ntc_test_daq970a import NTC_DAQ970A_Test
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_test():
    """基本测试示例"""
    logger.info("=== 基本测试示例 ===")
    
    # 创建测试实例（使用配置文件）
    test = NTC_DAQ970A_Test(config_file="config.json")
    
    try:
        # 连接仪器
        if not test.connect():
            logger.error("无法连接仪器")
            return
        
        # 配置通道
        test.configure_channels()
        
        # 配置扫描列表
        test.configure_scan_list()
        
        # 开始测量
        timestamps, data = test.start_measurement()
        
        if timestamps is not None:
            # 分析数据
            analysis = test.analyze_data()
            logger.info("测试完成")
            
            # 保存结果
            test.save_data()
            test.plot_results()
        
    except Exception as e:
        logger.error(f"测试出错: {e}")
    finally:
        test.disconnect()


def example_custom_config():
    """自定义配置示例"""
    logger.info("=== 自定义配置示例 ===")
    
    # 使用自定义资源名称
    test = NTC_DAQ970A_Test(
        config_file="config.json",
        resource_name="USB0::0x2A8D::0x1301::MY12345678::INSTR"  # USB连接
    )
    
    # 修改测试参数
    test.test_duration = 30.0  # 30秒测试
    test.sample_rate = 100.0    # 100Hz采样率
    
    # 执行测试...
    # (类似上面的流程)


def example_batch_test():
    """批量测试示例"""
    logger.info("=== 批量测试示例 ===")
    
    # 测试多个传感器
    test_configs = [
        {"channels": [101, 102], "duration": 15.0},
        {"channels": [103, 104], "duration": 15.0},
    ]
    
    for i, config in enumerate(test_configs):
        logger.info(f"开始第 {i+1} 个测试...")
        
        test = NTC_DAQ970A_Test(config_file="config.json")
        test.channels = config["channels"]
        test.test_duration = config["duration"]
        
        try:
            if test.connect():
                test.configure_channels()
                test.configure_scan_list()
                timestamps, data = test.start_measurement()
                
                if timestamps is not None:
                    test.save_data(f"batch_test_{i+1}.json")
                    logger.info(f"第 {i+1} 个测试完成")
        except Exception as e:
            logger.error(f"第 {i+1} 个测试出错: {e}")
        finally:
            test.disconnect()


if __name__ == "__main__":
    # 运行基本测试
    example_basic_test()
    
    # 取消注释以运行其他示例
    # example_custom_config()
    # example_batch_test()
