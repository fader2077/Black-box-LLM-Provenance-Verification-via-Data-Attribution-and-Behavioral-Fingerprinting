# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Distill-Llama-8B 完整测试
测试该模型与 Llama 和 DeepSeek 的相似度
"""
import sys
import json
from pathlib import Path
from loguru import logger

# 配置 logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("deepseek_r1_test.log", level="DEBUG", encoding="utf-8")

def main():
    logger.info("="*80)
    logger.info("DeepSeek-R1-Distill-Llama-8B 相似度分析")
    logger.info("="*80)
    
    # 查找最新的评估结果
    results_dir = Path("results")
    pattern = "evaluation_deepseek-ai_DeepSeek-R1-Distill-Llama-8B*.json"
    
    result_files = list(results_dir.glob(pattern))
    if not result_files:
        logger.error("未找到评估结果文件")
        logger.info("请先运行: python experiments/full_evaluation.py --target-model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --engine transformers")
        return 1
    
    # 使用最新的结果文件
    result_file = sorted(result_files)[-1]
    logger.info(f"加载结果: {result_file}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 提取相似度信息
    logger.info("\n" + "="*80)
    logger.info("相似度分析结果")
    logger.info("="*80)
    
    best_match = results.get('best_match', {})
    logger.info(f"\n最佳匹配:")
    logger.info(f"  模型: {best_match.get('model_name')}")
    logger.info(f"  相似度: {best_match.get('similarity_score', 0):.4f}")
    logger.info(f"  来源: {best_match.get('source')}")
    logger.info(f"  类别: {best_match.get('category')}")
    
    # 显示所有锚点模型的相似度
    logger.info(f"\n所有锚点模型相似度:")
    similarity_scores = results.get('similarity_scores', {})
    
    # 分类统计
    llama_scores = {}
    deepseek_scores = {}
    other_scores = {}
    
    for model_name, score in similarity_scores.items():
        logger.info(f"  {model_name:30s} : {score:.4f}")
        
        # 分类
        if 'llama' in model_name.lower():
            llama_scores[model_name] = score
        elif 'deepseek' in model_name.lower():
            deepseek_scores[model_name] = score
        else:
            other_scores[model_name] = score
    
    # 分析结论
    logger.info("\n" + "="*80)
    logger.info("分析结论")
    logger.info("="*80)
    
    if llama_scores:
        avg_llama = sum(llama_scores.values()) / len(llama_scores)
        logger.info(f"\nLlama 系列平均相似度: {avg_llama:.4f}")
        for model, score in llama_scores.items():
            logger.info(f"  {model}: {score:.4f}")
    
    if deepseek_scores:
        avg_deepseek = sum(deepseek_scores.values()) / len(deepseek_scores)
        logger.info(f"\nDeepSeek 系列平均相似度: {avg_deepseek:.4f}")
        for model, score in deepseek_scores.items():
            logger.info(f"  {model}: {score:.4f}")
    
    # 判断更接近哪个
    logger.info("\n" + "-"*80)
    if llama_scores and deepseek_scores:
        if avg_llama > avg_deepseek:
            diff = avg_llama - avg_deepseek
            logger.info(f"✓ DeepSeek-R1-Distill-Llama-8B 更接近 Llama 系列")
            logger.info(f"  平均相似度差异: {diff:.4f} ({diff*100:.2f}%)")
        else:
            diff = avg_deepseek - avg_llama
            logger.info(f"✓ DeepSeek-R1-Distill-Llama-8B 更接近 DeepSeek 系列")
            logger.info(f"  平均相似度差异: {diff:.4f} ({diff*100:.2f}%)")
    
    # 风险评估
    risk_assessment = results.get('risk_assessment', {})
    logger.info(f"\n风险评估:")
    logger.info(f"  风险等级: {risk_assessment.get('risk_level')}")
    logger.info(f"  判定结果: {risk_assessment.get('verdict')}")
    logger.info(f"  置信度: {risk_assessment.get('confidence', 0):.4f}")
    
    logger.info("\n" + "="*80)
    logger.success("分析完成！")
    logger.info("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
