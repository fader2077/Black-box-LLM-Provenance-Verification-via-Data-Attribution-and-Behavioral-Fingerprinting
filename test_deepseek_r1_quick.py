# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Distill-Llama-8B 快速测试（使用少量探针）
"""
import sys
import json
from pathlib import Path
from loguru import logger

# 配置 logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def main():
    from src.fingerprint import extract_fingerprint
    from src.attribution.similarity import SimilarityCalculator
    from src.utils.unified_loader import load_model
    from src.attribution.anchor_models import AnchorModelsDatabase
    
    logger.info("="*80)
    logger.info("DeepSeek-R1-Distill-Llama-8B 快速测试")
    logger.info("="*80)
    
    # 加载探针（只使用前50个）
    probe_file = Path("data/probes/all_probes.json")
    with open(probe_file, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
    
    probes = []
    for category, items in probes_data.items():
        if isinstance(items, list):
            probes.extend(items)
    
    # 只使用前50个探针
    test_probes = probes[:50]
    logger.info(f"使用 {len(test_probes)} 个探针进行快速测试")
    
    # 加载模型
    logger.info("加载 DeepSeek-R1-Distill-Llama-8B...")
    model = load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", engine="transformers")
    logger.success("模型加载成功")
    
    # 提取指纹
    logger.info("提取指纹...")
    fingerprint = extract_fingerprint(model, probes=test_probes)
    logger.success(f"指纹提取成功，向量长度: {len(fingerprint['logit_fingerprint']['vector'])}")
    
    # 加载锚点模型
    logger.info("加载锚点模型...")
    db = AnchorModelsDatabase()
    anchors = db.list_all_anchors()
    
    # 计算相似度
    calculator = SimilarityCalculator()
    logger.info("\n计算与锚点模型的相似度...")
    
    results = {}
    for anchor_name, anchor_info in anchors.items():
        if anchor_info['has_fingerprint']:
            anchor_fp = db.load_fingerprint(anchor_name)
            if anchor_fp:
                similarity = calculator.calculate_fingerprint_similarity(fingerprint, anchor_fp)
                results[anchor_name] = similarity['overall_similarity']
                logger.info(f"  {anchor_name:30s}: {similarity['overall_similarity']:.4f}")
    
    # 分析结果
    logger.info("\n" + "="*80)
    logger.info("分析结果")
    logger.info("="*80)
    
    if results:
        best_match = max(results.items(), key=lambda x: x[1])
        logger.success(f"\n最相似的锚点模型: {best_match[0]} (相似度: {best_match[1]:.4f})")
        
        # 分类统计
        deepseek_scores = {k: v for k, v in results.items() if 'deepseek' in k.lower()}
        llama_scores = {k: v for k, v in results.items() if 'llama' in k.lower()}
        gpt_scores = {k: v for k, v in results.items() if 'gpt' in k.lower()}
        
        if deepseek_scores:
            avg_deepseek = sum(deepseek_scores.values()) / len(deepseek_scores)
            logger.info(f"\nDeepSeek 系列平均相似度: {avg_deepseek:.4f}")
        
        if llama_scores:
            avg_llama = sum(llama_scores.values()) / len(llama_scores)
            logger.info(f"Llama 系列平均相似度: {avg_llama:.4f}")
        
        if gpt_scores:
            avg_gpt = sum(gpt_scores.values()) / len(gpt_scores)
            logger.info(f"GPT 系列平均相似度: {avg_gpt:.4f}")
    
    # 保存结果
    result_file = Path("results/deepseek_r1_quick_test.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "probe_count": len(test_probes),
            "similarity_scores": results,
            "best_match": best_match[0] if results else None,
        }, f, indent=2, ensure_ascii=False)
    
    logger.success(f"\n结果已保存到: {result_file}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("\n用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
