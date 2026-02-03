"""
稳健的指纹提取工具
带有自动重试、错误恢复和检查点保存功能
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.unified_loader import load_model
from src.fingerprint.logit_extractor import LogitExtractor
from src.fingerprint.refusal_detector import RefusalDetector
import numpy as np


class RobustFingerprintExtractor:
    """带有错误恢复的稳健指纹提取器"""
    
    def __init__(
        self,
        model,
        batch_size: int = 5,
        max_retries: int = 3,
        retry_delay: int = 5,
        checkpoint_interval: int = 10,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logit_extractor = LogitExtractor(model)
        self.refusal_detector = RefusalDetector()
    
    def extract_with_retry(
        self,
        probes: List[Dict],
        model_id: str,
        resume_from_checkpoint: bool = True
    ) -> Dict:
        """
        提取指纹，带有自动重试和检查点恢复
        
        Args:
            probes: 探针列表
            model_id: 模型标识符
            resume_from_checkpoint: 是否从检查点恢复
        
        Returns:
            完整的指纹字典
        """
        checkpoint_file = self.checkpoint_dir / f"{model_id.replace(':', '_')}_checkpoint.json"
        
        # 尝试恢复检查点
        start_idx = 0
        partial_results = []
        
        if resume_from_checkpoint and checkpoint_file.exists():
            logger.info(f"发现检查点文件: {checkpoint_file}")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                start_idx = checkpoint_data['last_processed_index'] + 1
                partial_results = checkpoint_data['partial_results']
                logger.info(f"从索引 {start_idx} 恢复 (已完成 {len(partial_results)} 个探针)")
        
        logger.info(f"开始提取指纹: {len(probes)} 个探针")
        logger.info(f"批处理大小: {self.batch_size}")
        logger.info(f"最大重试次数: {self.max_retries}")
        
        # 分批处理
        total_batches = (len(probes) - start_idx + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(total_batches):
            batch_start = start_idx + batch_num * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(probes))
            batch = probes[batch_start:batch_end]
            
            logger.info(f"\n处理批次 {batch_num + 1}/{total_batches} (探针 {batch_start}-{batch_end-1})")
            
            # 处理批次中的每个探针
            for idx, probe in enumerate(batch):
                global_idx = batch_start + idx
                prompt = probe.get('prompt', '')
                
                success = False
                for attempt in range(self.max_retries):
                    try:
                        # 提取概率分布
                        probs = self.logit_extractor.extract_token_probabilities(prompt)
                        
                        # 提取特征向量
                        features = self._extract_features(probs)
                        
                        partial_results.append({
                            'probe_id': probe.get('id', global_idx),
                            'features': features.tolist() if isinstance(features, np.ndarray) else features
                        })
                        
                        logger.debug(f"  ✓ 探针 {global_idx} 成功")
                        success = True
                        break
                        
                    except KeyboardInterrupt:
                        logger.warning(f"  检测到中断，保存检查点...")
                        self._save_checkpoint(checkpoint_file, global_idx - 1, partial_results)
                        raise
                        
                    except Exception as e:
                        logger.warning(f"  ✗ 探针 {global_idx} 失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                        else:
                            logger.error(f"  放弃探针 {global_idx}")
                            # 使用零向量作为失败探针的特征
                            partial_results.append({
                                'probe_id': probe.get('id', global_idx),
                                'features': [0.0] * 20,  # 默认特征维度
                                'error': str(e)
                            })
                
                # 定期保存检查点
                if (global_idx + 1) % self.checkpoint_interval == 0:
                    logger.info(f"  保存检查点 (已完成 {global_idx + 1} 个探针)")
                    self._save_checkpoint(checkpoint_file, global_idx, partial_results)
            
            # 批次间短暂休息
            if batch_num < total_batches - 1:
                logger.info(f"批次完成，休息 2 秒...")
                time.sleep(2)
        
        # 聚合结果
        logger.info("\n聚合指纹特征...")
        fingerprint = self._aggregate_results(partial_results, model_id)
        
        # 删除检查点
        if checkpoint_file.exists():
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
        for result in results:
            if 'error' not in result:
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
            "refusal_fingerprint": None,  # 简化版本暂不提取
            "extraction_stats": {
                "total_probes": len(results),
                "successful_probes": len(feature_vectors),
                "failed_probes": len(results) - len(feature_vectors)
            }
        }
        
        return fingerprint
    
    def _save_checkpoint(self, filepath: Path, last_index: int, results: List[Dict]):
        """保存检查点"""
        checkpoint_data = {
            'last_processed_index': last_index,
            'partial_results': results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="稳健的指纹提取")
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--engine", default="ollama", help="推理引擎")
    parser.add_argument("--num-probes", type=int, default=438, help="使用的探针数量")
    parser.add_argument("--batch-size", type=int, default=5, help="批处理大小")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--no-resume", action="store_true", help="不从检查点恢复")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("稳健指纹提取工具")
    logger.info("=" * 80)
    logger.info(f"模型: {args.model}")
    logger.info(f"引擎: {args.engine}")
    logger.info(f"设备: {args.device}")
    logger.info(f"探针数量: {args.num_probes}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info("=" * 80)
    
    # 加载探针
    logger.info("\n[1/3] 加载探针...")
    probes_path = Path("data/probes/all_probes.json")
    
    if not probes_path.exists():
        logger.error("探针文件不存在")
        return
    
    with open(probes_path, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
    
    all_probes = []
    for probe_type, probes in probes_data.items():
        all_probes.extend(probes)
    
    selected_probes = all_probes[:args.num_probes]
    logger.info(f"✓ 已加载 {len(selected_probes)} 个探针")
    
    # 加载模型
    logger.info(f"\n[2/3] 加载模型: {args.model}...")
    model = load_model(
        model_name=args.model,
        engine=args.engine,
        device=args.device
    )
    logger.info("✓ 模型加载成功")
    
    # 提取指纹
    logger.info("\n[3/3] 提取指纹 (带错误恢复)...")
    
    extractor = RobustFingerprintExtractor(
        model=model,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    )
    
    try:
        fingerprint = extractor.extract_with_retry(
            probes=selected_probes,
            model_id=args.model,
            resume_from_checkpoint=not args.no_resume
        )
        
        if fingerprint:
            # 保存结果
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = Path(f"results/{args.model.replace(':', '_')}_fingerprint.json")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, indent=2, ensure_ascii=False)
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ 指纹提取完成")
            logger.info(f"保存到: {output_path}")
            logger.info(f"特征维度: {fingerprint['logit_fingerprint']['dimension']}")
            stats = fingerprint['extraction_stats']
            logger.info(f"成功: {stats['successful_probes']}/{stats['total_probes']}")
            logger.info("=" * 80)
        else:
            logger.error("指纹提取失败")
    
    except KeyboardInterrupt:
        logger.warning("\n检测到中断，检查点已保存")
        logger.info("使用相同命令重新运行以继续")
    except Exception as e:
        logger.error(f"提取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
