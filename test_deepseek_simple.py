# -*- coding: utf-8 -*-
"""
简化的DeepSeek-R1测试（避免进度条问题）
"""
import sys
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 禁用tqdm进度条以避免KeyboardInterrupt问题
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def main():
    import torch
    from src.fingerprint import extract_fingerprint
    from src.attribution.similarity import SimilarityCalculator
    from src.attribution.anchor_models import AnchorModelsDatabase
    from src.utils.unified_loader import load_model
    
    logger.info("="*80)
    logger.info("DeepSeek-R1-Distill-Llama-8B 简化测试")
    logger.info("="*80)
    
    # 检查GPU
    logger.info(f"\nGPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载探针（使用前100个）
    logger.info("\n[1/4] 加载探针...")
    probe_file = Path("data/probes/all_probes.json")
    with open(probe_file, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
    
    probes = []
    for category, items in probes_data.items():
        if isinstance(items, list):
            probes.extend(items)
    
    # 只使用100个探针加速测试
    probes = probes[:100]
    logger.success(f"✓ 已加载 {len(probes)} 个探针")
    
    # 加载模型
    logger.info("\n[2/4] 加载模型到GPU...")
    logger.info("（此步骤可能需要几分钟，请耐心等待...）")
    
    model = load_model(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        engine="transformers",
        device="cuda"
    )
    logger.success("✓ 模型加载成功")
    
    # 提取指纹
    logger.info(f"\n[3/4] 提取指纹（{len(probes)} 个探针）...")
    fingerprint = extract_fingerprint(model, probes=probes)
    logger.success(f"✓ 指纹提取完成，维度: {len(fingerprint['logit_fingerprint']['vector'])}")
    
    # 计算相似度
    logger.info("\n[4/4] 计算与锚点的相似度...")
    db = AnchorModelsDatabase()
    anchors = db.list_all_anchors()
    
    calculator = SimilarityCalculator()
    results = {}
    
    for anchor_name, anchor_info in anchors.items():
        if anchor_info['has_fingerprint']:
            anchor_fp = db.load_fingerprint(anchor_name)
            if anchor_fp:
                sim = calculator.calculate_fingerprint_similarity(fingerprint, anchor_fp)
                results[anchor_name] = sim['overall_similarity']
                logger.info(f"  {anchor_name:30s}: {sim['overall_similarity']:.4f}")
    
    # 结果分析
    logger.info("\n" + "="*80)
    logger.info("结果分析")
    logger.info("="*80)
    
    if results:
        best = max(results.items(), key=lambda x: x[1])
        logger.success(f"\n最相似: {best[0]} ({best[1]:.4f})")
        
        # 分类
        deepseek_avg = sum(v for k, v in results.items() if 'deepseek' in k.lower()) / max(1, sum(1 for k in results if 'deepseek' in k.lower()))
        llama_avg = sum(v for k, v in results.items() if 'llama' in k.lower()) / max(1, sum(1 for k in results if 'llama' in k.lower()))
        gpt_avg = sum(v for k, v in results.items() if 'gpt' in k.lower()) / max(1, sum(1 for k in results if 'gpt' in k.lower()))
        
        logger.info(f"\n平均相似度:")
        logger.info(f"  DeepSeek: {deepseek_avg:.4f}")
        logger.info(f"  Llama: {llama_avg:.4f}")
        logger.info(f"  GPT: {gpt_avg:.4f}")
        
        # 保存结果
        result_file = Path("results/deepseek_r1_simple_test.json")
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "probes": len(probes),
                "results": results,
                "best_match": best[0],
                "averages": {
                    "deepseek": deepseek_avg,
                    "llama": llama_avg,
                    "gpt": gpt_avg
                }
            }, f, indent=2, ensure_ascii=False)
        logger.success(f"\n结果已保存: {result_file}")
    
    logger.info("\n" + "="*80)
    logger.success("测试完成！")
    logger.info("="*80)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
