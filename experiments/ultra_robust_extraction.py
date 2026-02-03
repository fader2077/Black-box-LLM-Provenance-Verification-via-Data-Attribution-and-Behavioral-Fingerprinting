"""
超稳健的指纹提取工具
专门针对Ollama KeyboardInterrupt问题的解决方案
"""

import sys
import json
import time
import argparse
import signal
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.unified_loader import load_model
from src.fingerprint.logit_extractor import LogitExtractor
import numpy as np


class UltraRobustExtractor:
    """超稳健提取器 - 针对KeyboardInterrupt优化"""
    
    def __init__(
        self,
        model_name: str,
        engine: str,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints"
    ):
        self.model_name = model_name
        self.engine = engine
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 不在初始化时加载模型，而是在每个批次前加载
        self.model = None
        self.logit_extractor = None
        
        # 信号处理
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        logger.warning("接收到中断信号，准备保存进度...")
        self.interrupted = True
    
    def _ensure_model_loaded(self):
        """确保模型已加载，如果未加载则加载"""
        if self.model is None:
            logger.info(f"加载模型: {self.model_name}")
            self.model = load_model(self.model_name, self.engine, self.device)
            self.logit_extractor = LogitExtractor(self.model)
            time.sleep(2)  # 等待模型稳定
    
    def _reload_model(self):
        """重新加载模型以刷新连接"""
        logger.info("重新加载模型以刷新连接...")
        
        # 清理旧模型
        if self.model is not None:
            del self.model
            del self.logit_extractor
            self.model = None
            self.logit_extractor = None
        
        # 强制垃圾回收
        gc.collect()
        time.sleep(3)
        
        # 重新加载
        self._ensure_model_loaded()
    
    def extract_single_probe(self, prompt: str, probe_id: int, max_retries: int = 3) -> Optional[Dict]:
        """提取单个探针，带有重试机制"""
        for attempt in range(max_retries):
            try:
                self._ensure_model_loaded()
                
                # 提取概率分布
                probs = self.logit_extractor.extract_token_probabilities(prompt)
                
                # 提取特征向量
                features = self._extract_features(probs)
                
                return {
                    'probe_id': probe_id,
                    'features': features.tolist() if isinstance(features, np.ndarray) else features,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                logger.warning(f"探针 {probe_id} 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # 如果是连接错误，尝试重新加载模型
                    if "connection" in str(e).lower() or "timeout" in str(e).lower():
                        logger.info("检测到连接问题，重新加载模型...")
                        self._reload_model()
                    else:
                        time.sleep(5)
                else:
                    logger.error(f"放弃探针 {probe_id}")
                    return None
        
        return None
    
    def extract_fingerprint(
        self,
        probes: List[Dict],
        output_file: str,
        probes_per_session: int = 3,  # 每3个探针重新加载模型
        rest_between_probes: int = 3  # 探针间休息3秒
    ) -> Dict:
        """
        提取指纹 - 超保守策略
        
        Args:
            probes: 探针列表
            output_file: 输出文件路径
            probes_per_session: 每个会话处理的探针数（之后重新加载模型）
            rest_between_probes: 探针之间的休息时间（秒）
        """
        checkpoint_file = self.checkpoint_dir / f"{self.model_name.replace(':', '_')}_checkpoint.json"
        
        # 尝试恢复检查点
        start_idx = 0
        partial_results = []
        
        if checkpoint_file.exists():
            logger.info(f"发现检查点文件: {checkpoint_file}")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                start_idx = checkpoint_data['last_processed_index'] + 1
                partial_results = checkpoint_data['partial_results']
                logger.info(f"从索引 {start_idx} 恢复 (已完成 {len(partial_results)} 个探针)")
        
        logger.info(f"开始提取指纹: {len(probes)} 个探针")
        logger.info(f"策略: 每 {probes_per_session} 个探针重新加载模型")
        logger.info(f"探针间休息: {rest_between_probes} 秒")
        
        # 逐个处理探针
        for idx in range(start_idx, len(probes)):
            if self.interrupted:
                logger.warning("检测到中断标志，保存并退出...")
                self._save_checkpoint(checkpoint_file, idx - 1, partial_results)
                break
            
            probe = probes[idx]
            prompt = probe.get('prompt', '')
            
            logger.info(f"\n[{idx + 1}/{len(probes)}] 处理探针 {idx}")
            
            # 每N个探针重新加载模型
            if idx > 0 and (idx - start_idx) % probes_per_session == 0:
                logger.info(f"已处理 {probes_per_session} 个探针，重新加载模型...")
                self._reload_model()
            
            # 提取单个探针
            result = self.extract_single_probe(prompt, idx)
            
            if result:
                partial_results.append(result)
                logger.success(f"✓ 探针 {idx} 成功")
            else:
                # 使用零向量作为失败探针的特征
                partial_results.append({
                    'probe_id': idx,
                    'features': [0.0] * 20,
                    'error': 'extraction_failed'
                })
                logger.error(f"✗ 探针 {idx} 失败（使用零向量）")
            
            # 每个探针后保存检查点
            self._save_checkpoint(checkpoint_file, idx, partial_results)
            
            # 休息
            if idx < len(probes) - 1:
                logger.info(f"休息 {rest_between_probes} 秒...")
                time.sleep(rest_between_probes)
        
        # 聚合结果
        logger.info("\n聚合指纹特征...")
        fingerprint = self._aggregate_results(partial_results, self.model_name)
        
        # 保存最终结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f, ensure_ascii=False, indent=2)
        
        logger.success(f"\n指纹已保存: {output_path}")
        
        # 删除检查点
        if checkpoint_file.exists() and not self.interrupted:
            checkpoint_file.unlink()
            logger.info("删除检查点文件")
        
        return fingerprint
    
    def _extract_features(self, probs: Dict) -> np.ndarray:
        """从概率分布中提取特征向量"""
        if not probs or 'token_probs' not in probs:
            return np.zeros(20)
        
        token_probs = probs['token_probs']
        
        # 提取top tokens的概率
        probs_list = [p for _, p in token_probs[:20]]
        
        # 填充到固定长度
        while len(probs_list) < 20:
            probs_list.append(0.0)
        
        return np.array(probs_list[:20])
    
    def _aggregate_results(self, results: List[Dict], model_id: str) -> Dict:
        """聚合部分结果为完整指纹"""
        # 提取所有特征向量
        feature_vectors = []
        failed_count = 0
        
        for result in results:
            if 'error' in result:
                failed_count += 1
            else:
                feature_vectors.append(result['features'])
        
        if not feature_vectors:
            logger.error("没有有效的特征向量")
            return None
        
        # 连接所有特征
        all_features = np.concatenate(feature_vectors)
        
        fingerprint = {
            "model_name": model_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "logit_fingerprint": {
                "vector": all_features.tolist(),
                "dimension": len(all_features),
                "stats": {
                    "mean": float(np.mean(all_features)),
                    "std": float(np.std(all_features)),
                    "min": float(np.min(all_features)),
                    "max": float(np.max(all_features)),
                }
            },
            "extraction_stats": {
                "total_probes": len(results),
                "successful_probes": len(feature_vectors),
                "failed_probes": failed_count,
                "success_rate": len(feature_vectors) / len(results) if results else 0
            }
        }
        
        return fingerprint
    
    def _save_checkpoint(self, filepath: Path, last_index: int, results: List[Dict]):
        """保存检查点"""
        checkpoint_data = {
            'model_name': self.model_name,
            'last_processed_index': last_index,
            'partial_results': results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检查点已保存: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="超稳健指纹提取工具")
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--engine", default="ollama", help="推理引擎")
    parser.add_argument("--num-probes", type=int, default=100, help="探针数量")
    parser.add_argument("--probes-per-session", type=int, default=3, help="每个会话处理的探针数")
    parser.add_argument("--rest-time", type=int, default=3, help="探针间休息时间（秒）")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--output", required=True, help="输出文件路径")
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG"
    )
    
    # 加载探针数据
    probe_files = [
        "data/probes/political_probes.json",
        "data/probes/linguistic_probes.json", 
        "data/probes/memorization_probes.json"
    ]
    
    all_probes = []
    for probe_file in probe_files:
        if Path(probe_file).exists():
            with open(probe_file, 'r', encoding='utf-8') as f:
                probes = json.load(f)
                all_probes.extend(probes[:args.num_probes // 3])
    
    logger.info(f"加载了 {len(all_probes)} 个探针")
    
    # 创建提取器
    extractor = UltraRobustExtractor(
        model_name=args.model,
        engine=args.engine,
        device=args.device
    )
    
    # 提取指纹
    try:
        fingerprint = extractor.extract_fingerprint(
            probes=all_probes,
            output_file=args.output,
            probes_per_session=args.probes_per_session,
            rest_between_probes=args.rest_time
        )
        
        if fingerprint:
            logger.success(f"\n✓ 提取完成")
            logger.info(f"  成功率: {fingerprint['extraction_stats']['success_rate']:.2%}")
            logger.info(f"  维度: {fingerprint['logit_fingerprint']['dimension']}")
        
    except KeyboardInterrupt:
        logger.warning("\n用户中断提取过程")
        logger.info("可以使用相同命令恢复提取")
    
    except Exception as e:
        logger.error(f"提取失败: {e}")
        raise


if __name__ == "__main__":
    main()
