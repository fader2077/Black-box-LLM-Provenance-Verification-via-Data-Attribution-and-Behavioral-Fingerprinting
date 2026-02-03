#!/usr/bin/env python3
"""
综合系统测试
测试整个 LLM 溯源系统的所有组件
"""

import sys
import json
from pathlib import Path
from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.probes import build_all_probes
from src.fingerprint import extract_fingerprint
from src.attribution import trace_provenance, AnchorModelsDatabase
from src.utils.unified_loader import load_model


def test_1_probe_system():
    """测试 1：探针系统"""
    logger.info("\n" + "="*80)
    logger.info("测试 1: 探针系统")
    logger.info("="*80)
    
    try:
        # 加载探针
        probes_path = project_root / "data" / "probes" / "all_probes.json"
        
        if not probes_path.exists():
            logger.info("探针文件不存在，构建新的探针...")
            probes_data = build_all_probes()
        else:
            logger.info(f"加载探针: {probes_path}")
            with open(probes_path, 'r', encoding='utf-8') as f:
                probes_data = json.load(f)
        
        # 统计探针
        total = 0
        for probe_type, probes in probes_data.items():
            count = len(probes)
            total += count
            logger.info(f"  {probe_type}: {count} 个探针")
        
        logger.success(f"✅ 测试通过：总共 {total} 个探针")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_model_loading():
    """测试 2：模型加载"""
    logger.info("\n" + "="*80)
    logger.info("测试 2: 模型加载")
    logger.info("="*80)
    
    test_cases = [
        ("gpt2", "transformers"),
        ("deepseek-r1:7b", "ollama"),
    ]
    
    results = []
    for model_name, engine in test_cases:
        try:
            logger.info(f"\n测试加载: {model_name} (引擎: {engine})")
            model = load_model(model_name, engine=engine)
            logger.success(f"  ✅ {model_name} 加载成功")
            results.append(True)
        except Exception as e:
            logger.error(f"  ❌ {model_name} 加载失败: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    logger.info(f"\n成功率: {success_rate:.1f}% ({sum(results)}/{len(results)})")
    
    if success_rate >= 50:
        logger.success("✅ 测试通过（至少 50% 成功）")
        return True
    else:
        logger.error("❌ 测试失败（成功率 < 50%）")
        return False


def test_3_fingerprint_extraction():
    """测试 3：指纹提取"""
    logger.info("\n" + "="*80)
    logger.info("测试 3: 指纹提取")
    logger.info("="*80)
    
    try:
        # 加载模型
        logger.info("加载 GPT-2 模型...")
        model = load_model("gpt2", engine="transformers")
        
        # 加载少量探针用于测试
        logger.info("加载探针...")
        probes_path = project_root / "data" / "probes" / "all_probes.json"
        with open(probes_path, 'r', encoding='utf-8') as f:
            probes_data = json.load(f)
        
        all_probes = []
        for probe_type, probes in probes_data.items():
            all_probes.extend(probes[:10])  # 每个类型取 10 个
        
        logger.info(f"使用 {len(all_probes)} 个探针进行测试...")
        
        # 提取指纹
        fingerprint = extract_fingerprint(
            model_interface=model,
            probes=all_probes,
            include_logit=True,
            include_refusal=False
        )
        
        # 验证指纹
        assert fingerprint["logit_fingerprint"] is not None
        fp_vector = fingerprint["logit_fingerprint"]["vector"]
        fp_dim = len(fp_vector)
        
        logger.info(f"指纹维度: {fp_dim}")
        logger.info(f"指纹统计:")
        logger.info(f"  均值: {fingerprint['logit_fingerprint']['stats']['mean']:.4f}")
        logger.info(f"  标准差: {fingerprint['logit_fingerprint']['stats']['std']:.4f}")
        
        # 检查指纹是否有效
        if all(v == 0.0 for v in fp_vector):
            raise ValueError("指纹全是 0，提取失败")
        
        logger.success("✅ 测试通过：指纹提取成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_anchor_database():
    """测试 4：锚点数据库"""
    logger.info("\n" + "="*80)
    logger.info("测试 4: 锚点数据库")
    logger.info("="*80)
    
    try:
        # 加载数据库
        db_path = str(project_root / "data" / "anchor_models")
        db = AnchorModelsDatabase(db_path)
        
        summary = db.export_database_summary()
        
        logger.info(f"总锚点数: {summary['total_anchors']}")
        logger.info(f"已有指纹: {summary['with_fingerprint']}")
        logger.info(f"缺少指纹: {summary['without_fingerprint']}")
        
        logger.info("\n锚点列表:")
        for name in summary['anchor_models']:
            logger.info(f"  • {name}")
        
        if summary['with_fingerprint'] >= 2:
            logger.success(f"✅ 测试通过：至少有 {summary['with_fingerprint']} 个锚点有指纹")
            return True
        else:
            logger.warning(f"⚠️  测试警告：只有 {summary['with_fingerprint']} 个锚点有指纹")
            logger.info("提示：运行 rebuild_all_anchors.py 提取锚点指纹")
            return False
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_similarity_calculation():
    """测试 5：相似度计算"""
    logger.info("\n" + "="*80)
    logger.info("测试 5: 相似度计算（同一模型）")
    logger.info("="*80)
    
    try:
        # 加载模型
        logger.info("加载 GPT-2 模型...")
        model = load_model("gpt2", engine="transformers")
        
        # 加载探针
        probes_path = project_root / "data" / "probes" / "all_probes.json"
        with open(probes_path, 'r', encoding='utf-8') as f:
            probes_data = json.load(f)
        
        all_probes = []
        for probe_type, probes in probes_data.items():
            all_probes.extend(probes[:10])
        
        logger.info(f"提取第一次指纹（{len(all_probes)} 探针）...")
        fp1 = extract_fingerprint(
            model_interface=model,
            probes=all_probes,
            include_logit=True,
            include_refusal=False
        )
        
        logger.info("提取第二次指纹...")
        fp2 = extract_fingerprint(
            model_interface=model,
            probes=all_probes,
            include_logit=True,
            include_refusal=False
        )
        
        # 计算相似度
        from src.attribution.similarity import cosine_similarity
        import numpy as np
        
        v1 = np.array(fp1["logit_fingerprint"]["vector"])
        v2 = np.array(fp2["logit_fingerprint"]["vector"])
        
        similarity = cosine_similarity(v1, v2)
        
        logger.info(f"相似度: {similarity:.4f}")
        
        if similarity > 0.95:
            logger.success(f"✅ 测试通过：同一模型相似度 {similarity:.4f} > 0.95")
            return True
        else:
            logger.warning(f"⚠️  相似度 {similarity:.4f} < 0.95，可能存在问题")
            return False
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_end_to_end_provenance():
    """测试 6：端到端溯源"""
    logger.info("\n" + "="*80)
    logger.info("测试 6: 端到端溯源分析")
    logger.info("="*80)
    
    try:
        # 加载模型
        logger.info("加载测试模型...")
        model = load_model("gpt2", engine="transformers")
        
        # 加载探针
        probes_path = project_root / "data" / "probes" / "all_probes.json"
        with open(probes_path, 'r', encoding='utf-8') as f:
            probes_data = json.load(f)
        
        all_probes = []
        for probe_type, probes in probes_data.items():
            all_probes.extend(probes[:20])  # 每类 20 个
        
        logger.info(f"提取指纹（{len(all_probes)} 探针）...")
        fingerprint = extract_fingerprint(
            model_interface=model,
            probes=all_probes,
            include_logit=True,
            include_refusal=False
        )
        
        # 执行溯源
        logger.info("执行溯源分析...")
        db_path = str(project_root / "data" / "anchor_models")
        db = AnchorModelsDatabase(db_path)
        
        result = trace_provenance(
            target_fingerprint=fingerprint,
            anchor_db=db
        )
        
        logger.info("\n溯源结果:")
        logger.info(f"  最佳匹配: {result['best_match']['anchor_name']}")
        logger.info(f"  相似度: {result['best_match']['similarity']:.2%}")
        logger.info(f"  风险等级: {result['risk_assessment']['risk_level']}")
        
        logger.success("✅ 测试通过：端到端溯源完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("="*80)
    logger.info("🔬 LLM 溯源系统 - 综合测试")
    logger.info("="*80)
    
    tests = [
        ("探针系统", test_1_probe_system),
        ("模型加载", test_2_model_loading),
        ("指纹提取", test_3_fingerprint_extraction),
        ("锚点数据库", test_4_anchor_database),
        ("相似度计算", test_5_similarity_calculation),
        ("端到端溯源", test_6_end_to_end_provenance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"测试 '{name}' 发生异常: {e}")
            results.append((name, False))
    
    # 总结
    logger.info("\n" + "="*80)
    logger.info("📊 测试结果总结")
    logger.info("="*80)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    success_rate = passed / total * 100
    
    logger.info(f"\n总体成功率: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate == 100:
        logger.success("\n🎉 所有测试通过！系统运行正常！")
    elif success_rate >= 80:
        logger.warning("\n⚠️  大部分测试通过，但有些问题需要修复")
    else:
        logger.error("\n❌ 多个测试失败，系统需要修复")
    
    return success_rate >= 80


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
