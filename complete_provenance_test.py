"""
完整谱系判定测试 - DeepSeek-R1-Distill-Llama-8B
测试目标：判定模型是属于 Llama 还是 DeepSeek 家族
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


class ComprehensiveProvenanceTest:
    """完整谱系判定测试"""
    
    def __init__(self):
        self.ultra_robust = "experiments/ultra_robust_extraction.py"
        self.results = []
    
    def run_extraction(self, model_name: str, output_file: str, num_probes: int = 1500) -> bool:
        """运行指纹提取"""
        logger.info(f"\n{'='*70}")
        logger.info(f"提取指纹: {model_name}")
        logger.info(f"{'='*70}")
        
        # 检查是否已存在
        if Path(output_file).exists():
            logger.info(f"✓ 指纹文件已存在: {output_file}")
            return True
        
        cmd = [
            "python", self.ultra_robust,
            "--model", model_name,
            "--engine", "ollama",
            "--num-probes", str(num_probes),
            "--probes-per-session", "3",
            "--rest-time", "4",
            "--device", "cuda",
            "--output", output_file
        ]
        
        try:
            logger.info(f"运行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=False, timeout=num_probes * 20)
            
            if result.returncode == 0 and Path(output_file).exists():
                logger.success(f"✓ {model_name} 提取成功")
                return True
            else:
                logger.error(f"✗ {model_name} 提取失败")
                return False
                
        except Exception as e:
            logger.error(f"✗ 提取异常: {e}")
            return False
    
    def check_anchors(self) -> dict:
        """检查锚点模型状态"""
        logger.info("\n检查锚点模型...")
        
        anchors = {
            "gpt": {
                "gpt2": "data/anchor_models_transformers/gpt2_fingerprint.json",
                "gpt2-medium": "data/anchor_models_transformers/gpt2_medium_fingerprint.json"
            },
            "deepseek": {
                "deepseek-r1:7b": "data/anchor_models/deepseek-r1_7b_fingerprint.json"
            },
            "llama": {
                "llama3.2:3b": "data/anchor_models/llama3_2_3b_fingerprint.json"
            }
        }
        
        status = {}
        for family, models in anchors.items():
            status[family] = {}
            for name, path in models.items():
                exists = Path(path).exists()
                status[family][name] = {
                    "path": path,
                    "exists": exists
                }
                
                symbol = "✓" if exists else "✗"
                logger.info(f"  {symbol} {family:10} {name:20} {path}")
        
        return status
    
    def ensure_llama_anchor(self) -> bool:
        """确保 Llama 锚点存在"""
        llama_fp = "data/anchor_models/llama3_2_3b_fingerprint.json"
        
        if Path(llama_fp).exists():
            logger.success("✓ Llama 锚点已存在")
            return True
        
        logger.info("提取 Llama 锚点...")
        return self.run_extraction("llama3.2:3b", llama_fp, num_probes=30)
    
    def extract_target_model(self) -> bool:
        """提取目标模型指纹"""
        target_model = "deepseek-r1:8b-llama-distill-q4_K_M"
        output_file = "results/deepseek_r1_distill_llama_8b_fingerprint.json"
        
        logger.info(f"\n{'='*70}")
        logger.info("提取目标模型: DeepSeek-R1-Distill-Llama-8B")
        logger.info(f"{'='*70}")
        
        return self.run_extraction(target_model, output_file, num_probes=30)
    
    def run_similarity_analysis(self) -> dict:
        """运行相似度分析"""
        logger.info(f"\n{'='*70}")
        logger.info("相似度分析")
        logger.info(f"{'='*70}")
        
        try:
            # 使用环境变量设置编码
            import os
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                ["python", "quick_similarity_analysis.py"],
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                logger.success("✓ 相似度分析完成")
                print(result.stdout)
                
                # 解析结果
                result_file = Path("results/quick_analysis_result.json")
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                        
            else:
                logger.error("✗ 相似度分析失败")
                print(result.stderr)
                
        except Exception as e:
            logger.error(f"✗ 分析异常: {e}")
        
        return None
    
    def verify_gpu_usage(self):
        """验证GPU使用"""
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
                
                if "GPU" in output:
                    logger.success("✓ 确认使用GPU")
                    return True
                else:
                    logger.warning("⚠ 未检测到GPU使用")
                    
        except Exception as e:
            logger.error(f"GPU检查失败: {e}")
        
        return False
    
    def generate_final_report(self, similarity_results: dict):
        """生成最终报告"""
        logger.info(f"\n{'='*70}")
        logger.info("最终测试报告")
        logger.info(f"{'='*70}")
        
        if not similarity_results:
            logger.error("无相似度结果，无法生成报告")
            return
        
        # 提取关键信息
        target = similarity_results.get('target_model', 'unknown')
        similarities = similarity_results.get('similarities', [])
        
        logger.info(f"\n目标模型: {target}")
        logger.info(f"\n相似度排名:")
        
        for i, sim in enumerate(similarities[:5], 1):
            model = sim.get('model', 'unknown')
            family = sim.get('family', 'unknown')
            score = sim.get('similarity', 0)
            
            medal = ["🥇", "🥈", "🥉", "", ""][i-1] if i <= 3 else ""
            logger.info(f"{i}. {model:25} [{family:10}] {score:.4f} {medal}")
        
        # 类别统计
        category_avg = similarity_results.get('category_average', {})
        if category_avg:
            logger.info(f"\n类别平均相似度:")
            
            sorted_cats = sorted(category_avg.items(), key=lambda x: x[1], reverse=True)
            for family, avg_score in sorted_cats:
                logger.info(f"  {family:15} {avg_score:.4f}")
        
        # 判定结论
        if similarities:
            top_model = similarities[0]
            top_family = top_model.get('family', 'unknown')
            top_score = top_model.get('similarity', 0)
            
            logger.info(f"\n{'='*70}")
            logger.success(f"✅ 判定结论: DeepSeek-R1-Distill-Llama-8B 属于 {top_family.upper()} 家族")
            logger.info(f"   最高相似度: {top_model.get('model')} ({top_score:.4f})")
            logger.info(f"{'='*70}")
    
    def run_comprehensive_test(self):
        """运行完整测试"""
        logger.info("\n" + "="*70)
        logger.info("DeepSeek-R1-Distill-Llama-8B 完整谱系判定测试")
        logger.info("="*70)
        
        # 1. 检查锚点状态
        anchor_status = self.check_anchors()
        
        # 2. 确保 Llama 锚点存在（关键！）
        if not self.ensure_llama_anchor():
            logger.error("✗ 无法获取 Llama 锚点，测试中止")
            return False
        
        # 3. 提取目标模型指纹
        if not self.extract_target_model():
            logger.error("✗ 目标模型提取失败，测试中止")
            return False
        
        # 短暂休息
        logger.info("\n休息10秒...")
        time.sleep(10)
        
        # 4. 运行相似度分析
        similarity_results = self.run_similarity_analysis()
        
        # 5. 验证GPU使用
        self.verify_gpu_usage()
        
        # 6. 生成最终报告
        if similarity_results:
            self.generate_final_report(similarity_results)
        
        return True


def main():
    tester = ComprehensiveProvenanceTest()
    
    try:
        success = tester.run_comprehensive_test()
        
        if success:
            logger.success("\n✓ 所有测试完成!")
        else:
            logger.error("\n✗ 测试未完全通过")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n用户中断测试")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n测试异常: {e}")
        raise


if __name__ == "__main__":
    main()
