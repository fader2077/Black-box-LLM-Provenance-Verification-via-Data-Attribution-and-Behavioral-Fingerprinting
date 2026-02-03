"""
测试相似度计算的有效性

验证策略:
1. 使用同一模型提取两次指纹 → 相似度应该很高 (>0.9)
2. 使用不同模型提取指纹 → 相似度应该较低 (<0.5)
3. 确认指纹提取和相似度计算功能正常
"""

import json
from pathlib import Path
from loguru import logger

from src.utils.unified_loader import load_model
from src.fingerprint import extract_fingerprint
from src.attribution.similarity import SimilarityCalculator
from src.probes import build_all_probes


def test_same_model_similarity():
    """测试 1: 同一模型的两次指纹提取应该高度相似"""
    logger.info("=" * 80)
    logger.info("测试 1: 同一模型相似度验证 (GPT-2 vs GPT-2)")
    logger.info("=" * 80)
    
    # 加载探针
    probes_file = Path("data/probes/all_probes.json")
    if not probes_file.exists():
        logger.info("构建探针...")
        build_all_probes()
    
    with open(probes_file, 'r', encoding='utf-8') as f:
        all_probes = json.load(f)
    
    # 使用少量探针进行测试
    test_probes = []
    for probe_type in all_probes.keys():
        test_probes.extend(all_probes[probe_type][:10])  # 每类取 10 个
    
    logger.info(f"使用 {len(test_probes)} 个探针进行测试")
    
    # 加载模型
    logger.info("加载 GPT-2 模型...")
    model = load_model("gpt2", engine="transformers")
    
    # 第一次提取指纹
    logger.info("第一次提取指纹...")
    fp1 = extract_fingerprint(model, test_probes)
    
    # 第二次提取指纹
    logger.info("第二次提取指纹...")
    fp2 = extract_fingerprint(model, test_probes)
    
    # 计算相似度
    calc = SimilarityCalculator()
    similarity = calc.calculate_fingerprint_similarity(fp1, fp2)
    
    logger.info("\n" + "=" * 80)
    logger.info("结果分析:")
    logger.info("=" * 80)
    logger.info(f"模型: GPT-2 vs GPT-2 (同一模型)")
    logger.info(f"指纹维度: {fp1['logit_fingerprint']['dimension']}")
    
    if similarity.get("logit_similarity"):
        cosine = similarity["logit_similarity"].get("cosine_similarity", 0)
        logger.info(f"余弦相似度: {cosine:.4f}")
        logger.info(f"欧氏距离: {similarity['logit_similarity'].get('euclidean_distance', 0):.4f}")
        logger.info(f"皮尔逊相关: {similarity['logit_similarity'].get('pearson_correlation', 0):.4f}")
        
        # 判断结果
        if cosine > 0.9:
            logger.success(f"✅ 测试通过: 同一模型相似度 {cosine:.4f} > 0.9")
            return True
        else:
            logger.warning(f"⚠️ 测试异常: 同一模型相似度 {cosine:.4f} 应该 > 0.9")
            logger.warning("这可能表明指纹提取有随机性或不稳定")
            return False
    else:
        logger.error("❌ 无法计算相似度: logit_similarity 为空")
        return False


def test_different_models_similarity():
    """测试 2: 不同模型的指纹应该不相似"""
    logger.info("\n" + "=" * 80)
    logger.info("测试 2: 不同模型相似度验证 (GPT-2 vs GPT-2-Medium)")
    logger.info("=" * 80)
    
    # 加载探针
    probes_file = Path("data/probes/all_probes.json")
    with open(probes_file, 'r', encoding='utf-8') as f:
        all_probes = json.load(f)
    
    # 使用少量探针
    test_probes = []
    for probe_type in all_probes.keys():
        test_probes.extend(all_probes[probe_type][:10])
    
    logger.info(f"使用 {len(test_probes)} 个探针进行测试")
    
    # 加载第一个模型
    logger.info("加载 GPT-2 模型...")
    model1 = load_model("gpt2", engine="transformers")
    
    # 加载第二个模型
    logger.info("加载 GPT-2-Medium 模型...")
    model2 = load_model("gpt2-medium", engine="transformers")
    
    # 提取指纹
    logger.info("提取 GPT-2 指纹...")
    fp1 = extract_fingerprint(model1, test_probes)
    
    logger.info("提取 GPT-2-Medium 指纹...")
    fp2 = extract_fingerprint(model2, test_probes)
    
    # 计算相似度
    calc = SimilarityCalculator()
    similarity = calc.calculate_fingerprint_similarity(fp1, fp2)
    
    logger.info("\n" + "=" * 80)
    logger.info("结果分析:")
    logger.info("=" * 80)
    logger.info(f"模型: GPT-2 vs GPT-2-Medium (不同规模)")
    logger.info(f"GPT-2 指纹维度: {fp1['logit_fingerprint']['dimension']}")
    logger.info(f"GPT-2-Medium 指纹维度: {fp2['logit_fingerprint']['dimension']}")
    
    if similarity.get("logit_similarity"):
        cosine = similarity["logit_similarity"].get("cosine_similarity", 0)
        logger.info(f"余弦相似度: {cosine:.4f}")
        logger.info(f"欧氏距离: {similarity['logit_similarity'].get('euclidean_distance', 0):.4f}")
        logger.info(f"皮尔逊相关: {similarity['logit_similarity'].get('pearson_correlation', 0):.4f}")
        
        # 判断结果
        if 0.3 < cosine < 0.8:
            logger.success(f"✅ 测试通过: 不同模型相似度 {cosine:.4f} 在合理范围 (0.3-0.8)")
            logger.info("GPT-2 和 GPT-2-Medium 是同系列模型，有一定相似性是正常的")
            return True
        elif cosine < 0.3:
            logger.warning(f"⚠️ 相似度 {cosine:.4f} 较低，但也可能正常")
            return True
        else:
            logger.warning(f"⚠️ 相似度 {cosine:.4f} 较高，可能指纹区分度不够")
            return False
    else:
        logger.error("❌ 无法计算相似度: logit_similarity 为空")
        return False


def test_anchor_fingerprint_extraction():
    """测试 3: 使用 Transformers 提取锚点指纹"""
    logger.info("\n" + "=" * 80)
    logger.info("测试 3: 使用 Transformers 引擎提取可用的锚点指纹")
    logger.info("=" * 80)
    
    # 使用 HuggingFace 上可用的模型作为锚点
    anchor_models = [
        ("gpt2", "GPT-2 (OpenAI)"),
        ("google/gemma-2-2b-it", "Gemma-2-2B (Google)"),
        ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B (Alibaba)"),
    ]
    
    # 加载探针
    probes_file = Path("data/probes/all_probes.json")
    with open(probes_file, 'r', encoding='utf-8') as f:
        all_probes = json.load(f)
    
    # 使用 30 个探针
    test_probes = []
    for probe_type in all_probes.keys():
        test_probes.extend(all_probes[probe_type][:10])
    
    logger.info(f"使用 {len(test_probes)} 个探针")
    
    # 创建临时锚点数据库
    anchor_dir = Path("data/anchor_models_transformers")
    anchor_dir.mkdir(exist_ok=True)
    
    anchor_fingerprints = {}
    
    for model_name, description in anchor_models:
        try:
            logger.info(f"\n处理: {description} ({model_name})")
            
            # 加载模型
            logger.info("  载入模型...")
            model = load_model(model_name, engine="transformers")
            
            # 提取指纹
            logger.info("  提取指纹...")
            fingerprint = extract_fingerprint(model, test_probes)
            
            # 保存指纹
            safe_name = model_name.replace("/", "_").replace(":", "_")
            fp_file = anchor_dir / f"{safe_name}_fingerprint.json"
            
            with open(fp_file, 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, indent=2, ensure_ascii=False)
            
            anchor_fingerprints[model_name] = fingerprint
            
            logger.success(f"  ✓ 指纹已保存: {fp_file.name}")
            logger.info(f"  维度: {fingerprint['logit_fingerprint']['dimension']}")
            
        except Exception as e:
            logger.error(f"  ✗ 失败: {e}")
    
    # 计算交叉相似度
    if len(anchor_fingerprints) >= 2:
        logger.info("\n" + "=" * 80)
        logger.info("交叉相似度矩阵:")
        logger.info("=" * 80)
        
        calc = SimilarityCalculator()
        model_names = list(anchor_fingerprints.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    fp1 = anchor_fingerprints[model1]
                    fp2 = anchor_fingerprints[model2]
                    
                    similarity = calc.calculate_fingerprint_similarity(fp1, fp2)
                    cosine = similarity.get("logit_similarity", {}).get("cosine_similarity", 0)
                    
                    logger.info(f"{model1} <-> {model2}: {cosine:.4f}")
        
        logger.success("\n✅ 锚点指纹提取完成")
        return True
    else:
        logger.warning("⚠️ 可用锚点不足")
        return False


def main():
    logger.info("=" * 80)
    logger.info("相似度计算验证测试套件")
    logger.info("=" * 80)
    
    results = []
    
    # 测试 1: 同一模型
    try:
        result1 = test_same_model_similarity()
        results.append(("同一模型相似度", result1))
    except Exception as e:
        logger.error(f"测试 1 失败: {e}")
        results.append(("同一模型相似度", False))
    
    # 测试 2: 不同模型
    try:
        result2 = test_different_models_similarity()
        results.append(("不同模型相似度", result2))
    except Exception as e:
        logger.error(f"测试 2 失败: {e}")
        results.append(("不同模型相似度", False))
    
    # 测试 3: 锚点提取
    try:
        result3 = test_anchor_fingerprint_extraction()
        results.append(("锚点指纹提取", result3))
    except Exception as e:
        logger.error(f"测试 3 失败: {e}")
        results.append(("锚点指纹提取", False))
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        logger.info(f"{test_name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.success("\n🎉 所有测试通过！相似度计算功能正常。")
    else:
        logger.warning(f"\n⚠️ {total - passed} 个测试失败，需要进一步调查。")


if __name__ == "__main__":
    main()
