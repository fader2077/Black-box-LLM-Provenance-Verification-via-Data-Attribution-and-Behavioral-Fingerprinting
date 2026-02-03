"""
全自动测试脚本 - 使用超稳健策略完成所有测试
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


class AutomatedTestRunner:
    """自动化测试运行器"""
    
    def __init__(self):
        self.results = []
        self.ultra_robust_script = "experiments/ultra_robust_extraction.py"
    
    def run_extraction(
        self,
        model_name: str,
        engine: str,
        num_probes: int,
        output_file: str,
        probes_per_session: int = 3,
        rest_time: int = 4
    ) -> bool:
        """
        运行指纹提取
        
        Returns:
            是否成功
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始提取: {model_name}")
        logger.info(f"{'='*60}")
        
        cmd = [
            "python", self.ultra_robust_script,
            "--model", model_name,
            "--engine", engine,
            "--num-probes", str(num_probes),
            "--probes-per-session", str(probes_per_session),
            "--rest-time", str(rest_time),
            "--device", "cuda",
            "--output", output_file
        ]
        
        try:
            logger.info(f"命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                timeout=num_probes * 20  # 每个探针最多20秒
            )
            
            if result.returncode == 0:
                logger.success(f"✓ {model_name} 提取成功")
                
                # 验证输出文件
                if Path(output_file).exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        fp = json.load(f)
                        logger.info(f"  维度: {fp['logit_fingerprint']['dimension']}")
                        logger.info(f"  成功率: {fp['extraction_stats']['success_rate']:.2%}")
                    return True
                else:
                    logger.error(f"输出文件不存在: {output_file}")
                    return False
            else:
                logger.error(f"✗ {model_name} 提取失败 (退出码: {result.returncode})")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {model_name} 超时")
            return False
        except Exception as e:
            logger.error(f"✗ {model_name} 异常: {e}")
            return False
    
    def run_similarity_analysis(self) -> bool:
        """运行相似度分析"""
        logger.info(f"\n{'='*60}")
        logger.info("运行相似度分析")
        logger.info(f"{'='*60}")
        
        try:
            result = subprocess.run(
                ["python", "quick_similarity_analysis.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.success("✓ 相似度分析完成")
                # 打印分析结果
                print(result.stdout)
                return True
            else:
                logger.error(f"✗ 相似度分析失败")
                print(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"✗ 相似度分析异常: {e}")
            return False
    
    def check_gpu_usage(self):
        """检查GPU使用情况"""
        logger.info("\n检查GPU使用情况...")
        
        try:
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout
                logger.info("Ollama状态:")
                for line in output.strip().split('\n'):
                    logger.info(f"  {line}")
                
                # 检查是否使用GPU
                if "GPU" in output:
                    logger.success("✓ 确认使用GPU")
                    return True
                else:
                    logger.warning("⚠ 未检测到GPU使用")
                    return False
            else:
                logger.warning("无法检查Ollama状态")
                return False
                
        except Exception as e:
            logger.error(f"GPU检查失败: {e}")
            return False
    
    def run_comprehensive_test(self):
        """运行全面测试"""
        logger.info("\n" + "="*60)
        logger.info("开始全面测试流程")
        logger.info("="*60)
        
        # 测试配置
        test_configs = [
            {
                "name": "llama3.2:3b",
                "engine": "ollama",
                "num_probes": 30,  # 使用较少的探针以确保完成
                "output": "data/anchor_models/llama3_2_3b_fingerprint.json",
                "description": "Llama锚点模型"
            },
            {
                "name": "deepseek-r1:8b",
                "engine": "ollama",
                "num_probes": 30,
                "output": "results/deepseek_r1_8b_fingerprint.json",
                "description": "DeepSeek-R1测试模型"
            },
        ]
        
        # 执行提取
        for config in test_configs:
            logger.info(f"\n处理: {config['description']}")
            
            success = self.run_extraction(
                model_name=config['name'],
                engine=config['engine'],
                num_probes=config['num_probes'],
                output_file=config['output'],
                probes_per_session=3,  # 每3个探针重新加载
                rest_time=4  # 每个探针间休息4秒
            )
            
            self.results.append({
                "model": config['name'],
                "description": config['description'],
                "success": success
            })
            
            if success:
                # 检查GPU使用
                self.check_gpu_usage()
            
            # 任务间休息
            logger.info("任务间休息10秒...")
            time.sleep(10)
        
        # 运行相似度分析
        if all(r['success'] for r in self.results):
            logger.info("\n所有提取完成，开始相似度分析...")
            self.run_similarity_analysis()
        else:
            logger.warning("\n部分提取失败，跳过相似度分析")
        
        # 打印摘要
        self.print_summary()
    
    def print_summary(self):
        """打印测试摘要"""
        logger.info("\n" + "="*60)
        logger.info("测试摘要")
        logger.info("="*60)
        
        for result in self.results:
            status = "✓" if result['success'] else "✗"
            logger.info(f"{status} {result['model']}: {result['description']}")
        
        success_count = sum(1 for r in self.results if r['success'])
        total_count = len(self.results)
        
        logger.info(f"\n成功: {success_count}/{total_count}")
        
        if success_count == total_count:
            logger.success("\n✓ 所有测试通过!")
        else:
            logger.warning(f"\n⚠ {total_count - success_count} 个测试失败")


def main():
    runner = AutomatedTestRunner()
    
    try:
        runner.run_comprehensive_test()
    except KeyboardInterrupt:
        logger.warning("\n用户中断测试")
    except Exception as e:
        logger.error(f"\n测试异常: {e}")
        raise


if __name__ == "__main__":
    main()
