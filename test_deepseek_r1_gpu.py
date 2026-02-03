# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Distill-Llama-8B GPU测试
强制使用GPU进行评估
"""
import sys
import json
import torch
from pathlib import Path
from loguru import logger

# 配置 logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("deepseek_r1_gpu_test.log", level="DEBUG", encoding="utf-8")

def check_gpu():
    """检查GPU可用性"""
    logger.info("="*80)
    logger.info("检查GPU状态")
    logger.info("="*80)
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用！此测试需要GPU。")
        logger.error("请确保：")
        logger.error("  1. 安装了支持CUDA的PyTorch")
        logger.error("  2. 系统有可用的NVIDIA GPU")
        logger.error("  3. 已安装CUDA驱动")
        return False
    
    logger.success(f"✓ CUDA可用")
    logger.info(f"  GPU数量: {torch.cuda.device_count()}")
    logger.info(f"  当前设备: {torch.cuda.current_device()}")
    logger.info(f"  设备名称: {torch.cuda.get_device_name(0)}")
    logger.info(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return True

def main():
    # 检查GPU
    if not check_gpu():
        return 1
    
    logger.info("\n" + "="*80)
    logger.info("DeepSeek-R1-Distill-Llama-8B 评估（GPU加速）")
    logger.info("="*80)
    
    from src.fingerprint import extract_fingerprint
    from src.attribution.similarity import SimilarityCalculator
    from src.attribution.anchor_models import AnchorModelsDatabase
    
    # 强制使用transformers + GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 动态导入以确保使用GPU
    from src.utils.unified_loader import load_model
    
    # 加载探针
    logger.info("\n[1/5] 加载探针数据...")
    probe_file = Path("data/probes/all_probes.json")
    with open(probe_file, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
    
    probes = []
    for category, items in probes_data.items():
        if isinstance(items, list):
            probes.extend(items)
    
    logger.success(f"✓ 已加载 {len(probes)} 个探针")
    
    # 加载模型（GPU）
    logger.info("\n[2/5] 加载 DeepSeek-R1-Distill-Llama-8B 到 GPU...")
    model = load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", engine="transformers")
    
    # 验证模型在GPU上
    if hasattr(model, 'loader') and hasattr(model.loader, 'device'):
        device = model.loader.device
        logger.success(f"✓ 模型已加载到: {device}")
        
        if device == "cpu":
            logger.error("❌ 模型在CPU上！需要GPU。")
            return 1
    
    # 提取指纹
    logger.info(f"\n[3/5] 提取指纹（使用 {len(probes)} 个探针）...")
    logger.info("此过程在GPU上会快很多，请稍候...")
    
    try:
        fingerprint = extract_fingerprint(model, probes=probes)
        logger.success(f"✓ 指纹提取成功")
        logger.info(f"  向量维度: {len(fingerprint['logit_fingerprint']['vector'])}")
    except Exception as e:
        logger.error(f"❌ 指纹提取失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 加载锚点模型
    logger.info("\n[4/5] 加载锚点模型并计算相似度...")
    db = AnchorModelsDatabase()
    anchors = db.list_all_anchors()
    
    logger.info(f"可用锚点: {len(anchors)} 个")
    for name, info in anchors.items():
        has_fp = "✓" if info['has_fingerprint'] else "✗"
        logger.info(f"  {has_fp} {name}")
    
    # 计算相似度
    calculator = SimilarityCalculator()
    logger.info("\n计算相似度...")
    
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
    logger.info("相似度分析结果")
    logger.info("="*80)
    
    if not results:
        logger.error("❌ 没有计算出相似度结果")
        return 1
    
    # 找出最相似的
    best_match = max(results.items(), key=lambda x: x[1])
    logger.success(f"\n最相似的锚点: {best_match[0]}")
    logger.success(f"相似度: {best_match[1]:.4f} ({best_match[1]*100:.2f}%)")
    
    # 分类统计
    deepseek_scores = {k: v for k, v in results.items() if 'deepseek' in k.lower()}
    llama_scores = {k: v for k, v in results.items() if 'llama' in k.lower()}
    gpt_scores = {k: v for k, v in results.items() if 'gpt' in k.lower()}
    
    logger.info("\n分类统计:")
    
    if deepseek_scores:
        avg_deepseek = sum(deepseek_scores.values()) / len(deepseek_scores)
        logger.info(f"\nDeepSeek 系列:")
        for model, score in deepseek_scores.items():
            logger.info(f"  {model:30s}: {score:.4f}")
        logger.info(f"  平均相似度: {avg_deepseek:.4f}")
    
    if llama_scores:
        avg_llama = sum(llama_scores.values()) / len(llama_scores)
        logger.info(f"\nLlama 系列:")
        for model, score in llama_scores.items():
            logger.info(f"  {model:30s}: {score:.4f}")
        logger.info(f"  平均相似度: {avg_llama:.4f}")
    
    if gpt_scores:
        avg_gpt = sum(gpt_scores.values()) / len(gpt_scores)
        logger.info(f"\nGPT 系列:")
        for model, score in gpt_scores.items():
            logger.info(f"  {model:30s}: {score:.4f}")
        logger.info(f"  平均相似度: {avg_gpt:.4f}")
    
    # 结论
    logger.info("\n" + "="*80)
    logger.info("结论")
    logger.info("="*80)
    
    if deepseek_scores and llama_scores:
        if avg_deepseek > avg_llama:
            diff = avg_deepseek - avg_llama
            logger.success(f"✓ DeepSeek-R1-Distill-Llama-8B 更接近 DeepSeek 系列")
            logger.info(f"  平均相似度差异: {diff:.4f} ({diff*100:.2f}%)")
        else:
            diff = avg_llama - avg_deepseek
            logger.success(f"✓ DeepSeek-R1-Distill-Llama-8B 更接近 Llama 系列")
            logger.info(f"  平均相似度差异: {diff:.4f} ({diff*100:.2f}%)")
    elif deepseek_scores:
        logger.info(f"DeepSeek-R1-Distill-Llama-8B 与 DeepSeek 系列的平均相似度: {avg_deepseek:.4f}")
    elif llama_scores:
        logger.info(f"DeepSeek-R1-Distill-Llama-8B 与 Llama 系列的平均相似度: {avg_llama:.4f}")
    else:
        logger.warning("⚠ 缺少 Llama 或 DeepSeek 锚点，无法进行对比")
    
    # 保存结果
    logger.info("\n[5/5] 保存结果...")
    result_file = Path("results/deepseek_r1_distill_llama_8b_evaluation.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "device": "cuda",
        "probe_count": len(probes),
        "similarity_scores": results,
        "best_match": {
            "model": best_match[0],
            "score": best_match[1]
        },
        "category_averages": {
            "deepseek": avg_deepseek if deepseek_scores else None,
            "llama": avg_llama if llama_scores else None,
            "gpt": avg_gpt if gpt_scores else None,
        }
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logger.success(f"✓ 结果已保存到: {result_file}")
    
    logger.info("\n" + "="*80)
    logger.success("🎉 测试完成！")
    logger.info("="*80)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("\n⚠ 用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
