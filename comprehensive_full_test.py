# -*- coding: utf-8 -*-
"""
全面系统测试
测试所有核心功能和工作流程
"""
import sys
import json
from pathlib import Path
from loguru import logger

# 配置 logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("comprehensive_test.log", level="DEBUG", encoding="utf-8")

def test_probe_loading():
    """测试探针加载"""
    logger.info("\n" + "="*80)
    logger.info("测试 1: 探针加载")
    logger.info("="*80)
    
    try:
        from src.probes import ProbeGenerator
        
        # 加载缓存的探针
        probe_file = Path("data/probes/all_probes.json")
        if not probe_file.exists():
            logger.error(f"探针文件不存在: {probe_file}")
            return False
        
        with open(probe_file, 'r', encoding='utf-8') as f:
            probes = json.load(f)
        
        logger.info(f"✓ 成功加载 {len(probes)} 个探针")
        
        # 检查 probe_type 字段
        probes_with_type = [p for p in probes if 'probe_type' in p]
        logger.info(f"✓ 有 probe_type 字段的探针: {len(probes_with_type)}/{len(probes)}")
        
        # 统计各类型探针数量
        type_counts = {}
        for probe in probes:
            probe_type = probe.get('probe_type', 'unknown')
            type_counts[probe_type] = type_counts.get(probe_type, 0) + 1
        
        logger.info("探针类型分布:")
        for ptype, count in type_counts.items():
            logger.info(f"  {ptype:30s}: {count:4d}")
        
        return len(probes) == 438 and len(probes_with_type) >= 400
        
    except Exception as e:
        logger.error(f"✗ 探针加载测试失败: {e}")
        return False

def test_anchor_database():
    """测试锚点模型数据库"""
    logger.info("\n" + "="*80)
    logger.info("测试 2: 锚点模型数据库")
    logger.info("="*80)
    
    try:
        from src.attribution.anchor_models import AnchorDatabase
        
        db = AnchorDatabase()
        anchors = db.list_anchors()
        
        logger.info(f"✓ 数据库中有 {len(anchors)} 个锚点模型")
        
        for anchor in anchors:
            logger.info(f"  {anchor['model_name']:30s} - {anchor['source']:15s} - {anchor.get('category', 'unknown')}")
        
        # 检查是否有指纹
        with_fingerprints = [a for a in anchors if 'fingerprint' in a]
        logger.info(f"✓ 有指纹的锚点: {len(with_fingerprints)}/{len(anchors)}")
        
        return len(anchors) >= 3
        
    except Exception as e:
        logger.error(f"✗ 锚点数据库测试失败: {e}")
        return False

def test_similarity_calculation():
    """测试相似度计算"""
    logger.info("\n" + "="*80)
    logger.info("测试 3: 相似度计算逻辑")
    logger.info("="*80)
    
    try:
        from src.attribution.similarity import compare_fingerprints
        
        # 创建测试指纹（只有 logit，没有 refusal）
        fp1 = {
            "logit_fingerprint": {
                "vector": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
        fp2 = {
            "logit_fingerprint": {
                "vector": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
        
        result = compare_fingerprints(fp1, fp2)
        
        logger.info(f"Logit 相似度: {result['logit_similarity']['ensemble_score']:.4f}")
        logger.info(f"整体相似度: {result['overall_similarity']:.4f}")
        
        # 验证：当只有 logit 时，overall_similarity 应该等于 logit_similarity
        logit_score = result['logit_similarity']['ensemble_score']
        overall_score = result['overall_similarity']
        
        if abs(logit_score - overall_score) < 0.0001:
            logger.success("✓ 相似度计算逻辑正确（无 refusal 时使用 logit 分数）")
            return True
        else:
            logger.error(f"✗ 相似度计算错误: logit={logit_score}, overall={overall_score}")
            return False
        
    except Exception as e:
        logger.error(f"✗ 相似度计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载"""
    logger.info("\n" + "="*80)
    logger.info("测试 4: 模型加载（Transformers）")
    logger.info("="*80)
    
    try:
        from src.utils.unified_loader import load_model
        
        # 测试 GPT-2（轻量级模型）
        logger.info("加载 GPT-2 模型...")
        model, tokenizer = load_model("gpt2", engine="transformers")
        
        logger.success("✓ 模型加载成功")
        
        # 测试推理
        logger.info("测试推理...")
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model(**inputs)
        
        logger.info(f"✓ 推理成功，输出形状: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 模型加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fingerprint_extraction():
    """测试指纹提取"""
    logger.info("\n" + "="*80)
    logger.info("测试 5: 指纹提取")
    logger.info("="*80)
    
    try:
        from src.fingerprint import extract_fingerprint
        from src.utils.unified_loader import load_model
        
        # 加载模型
        logger.info("加载 GPT-2 模型...")
        model, tokenizer = load_model("gpt2", engine="transformers")
        
        # 加载少量探针进行测试
        probe_file = Path("data/probes/all_probes.json")
        with open(probe_file, 'r', encoding='utf-8') as f:
            all_probes = json.load(f)
        
        # 只使用前 10 个探针
        test_probes = all_probes[:10]
        logger.info(f"使用 {len(test_probes)} 个探针进行测试...")
        
        # 提取指纹
        fingerprint = extract_fingerprint(model, tokenizer, test_probes, engine="transformers")
        
        logger.info(f"✓ 指纹提取成功")
        logger.info(f"  Logit 特征数: {len(fingerprint.get('logit_fingerprint', {}).get('vector', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 指纹提取测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attribution_flow():
    """测试完整的溯源流程"""
    logger.info("\n" + "="*80)
    logger.info("测试 6: 完整溯源流程（GPT-2 自我对比）")
    logger.info("="*80)
    
    try:
        from src.fingerprint import extract_fingerprint
        from src.attribution.similarity import compare_fingerprints
        from src.utils.unified_loader import load_model
        
        # 加载模型
        logger.info("加载 GPT-2 模型...")
        model, tokenizer = load_model("gpt2", engine="transformers")
        
        # 加载探针
        probe_file = Path("data/probes/all_probes.json")
        with open(probe_file, 'r', encoding='utf-8') as f:
            all_probes = json.load(f)
        
        # 使用 30 个探针
        test_probes = all_probes[:30]
        logger.info(f"使用 {len(test_probes)} 个探针...")
        
        # 提取指纹
        logger.info("提取指纹...")
        fp1 = extract_fingerprint(model, tokenizer, test_probes, engine="transformers")
        fp2 = extract_fingerprint(model, tokenizer, test_probes, engine="transformers")
        
        # 计算相似度
        logger.info("计算相似度...")
        result = compare_fingerprints(fp1, fp2)
        
        overall_sim = result['overall_similarity']
        logger.info(f"✓ 整体相似度: {overall_sim:.4f}")
        
        # GPT-2 对比自己应该是 1.0
        if overall_sim >= 0.99:
            logger.success("✓ GPT-2 自我对比相似度正确（>= 0.99）")
            return True
        else:
            logger.warning(f"⚠ GPT-2 自我对比相似度偏低: {overall_sim:.4f}")
            return False
        
    except Exception as e:
        logger.error(f"✗ 溯源流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("="*80)
    logger.info("全面系统测试")
    logger.info("="*80)
    
    tests = [
        ("探针加载", test_probe_loading),
        ("锚点数据库", test_anchor_database),
        ("相似度计算", test_similarity_calculation),
        ("模型加载", test_model_loading),
        ("指纹提取", test_fingerprint_extraction),
        ("完整溯源流程", test_attribution_flow),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"测试 '{test_name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 汇总结果
    logger.info("\n" + "="*80)
    logger.info("测试结果汇总")
    logger.info("="*80)
    
    passed = 0
    failed = 0
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:10s} - {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("-"*80)
    logger.info(f"通过: {passed}/{len(tests)}")
    logger.info(f"失败: {failed}/{len(tests)}")
    
    if failed == 0:
        logger.success("\n🎉 所有测试通过！系统运行正常。")
        return 0
    else:
        logger.error(f"\n❌ 有 {failed} 个测试失败，需要修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
