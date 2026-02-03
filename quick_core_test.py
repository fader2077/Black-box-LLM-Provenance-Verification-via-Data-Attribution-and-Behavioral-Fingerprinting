# -*- coding: utf-8 -*-
"""
快速核心功能测试（不需要加载大模型）
"""
import sys
import json
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

def test_1_probe_loading():
    """测试探针加载"""
    print("\n" + "="*80)
    print("测试 1: 探针加载")
    print("="*80)
    
    try:
        probe_file = Path("data/probes/all_probes.json")
        if not probe_file.exists():
            print(f"✗ 探针文件不存在: {probe_file}")
            return False
        
        with open(probe_file, 'r', encoding='utf-8') as f:
            probes_data = json.load(f)
        
        # 如果是字典，展开为列表
        if isinstance(probes_data, dict):
            probes = []
            for category, items in probes_data.items():
                if isinstance(items, list):
                    probes.extend(items)
        else:
            probes = probes_data
        
        print(f"✓ 成功加载 {len(probes)} 个探针")
        
        # 检查 probe_type 字段
        probes_with_type = [p for p in probes if isinstance(p, dict) and 'probe_type' in p]
        print(f"✓ 有 probe_type 字段的探针: {len(probes_with_type)}/{len(probes)}")
        
        # 统计各类型
        type_counts = {}
        for probe in probes:
            if isinstance(probe, dict):
                ptype = probe.get('probe_type', probe.get('type', 'unknown'))
                type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        print("\n探针类型分布:")
        for ptype, count in sorted(type_counts.items()):
            print(f"  {ptype:30s}: {count:4d}")
        
        return len(probes) >= 400  # 允许一些误差
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_2_similarity_logic():
    """测试相似度计算逻辑"""
    print("\n" + "="*80)
    print("测试 2: 相似度计算逻辑")
    print("="*80)
    
    try:
        from src.attribution.similarity import SimilarityCalculator
        
        # 创建计算器实例
        calculator = SimilarityCalculator()
        
        # 测试 1: 只有 logit 指纹
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
        
        result = calculator.calculate_fingerprint_similarity(fp1, fp2)
        logit_score = result['logit_similarity']['ensemble_score']
        overall_score = result['overall_similarity']
        
        print(f"Logit 相似度: {logit_score:.4f}")
        print(f"整体相似度: {overall_score:.4f}")
        
        if abs(logit_score - overall_score) < 0.0001:
            print("✓ 相似度计算逻辑正确（无 refusal 时使用 logit 分数）")
            return True
        else:
            print(f"✗ 相似度计算错误: logit={logit_score}, overall={overall_score}")
            return False
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3_anchor_database():
    """测试锚点模型数据库"""
    print("\n" + "="*80)
    print("测试 3: 锚点模型数据库")
    print("="*80)
    
    try:
        from src.attribution.anchor_models import AnchorModelsDatabase
        
        db = AnchorModelsDatabase()
        anchors = db.list_all_anchors()
        
        print(f"✓ 数据库中有 {len(anchors)} 个锚点模型")
        
        for model_name, info in anchors.items():
            has_fp = "✓" if info.get('has_fingerprint') else "✗"
            source = info.get('source', 'unknown')
            print(f"  {has_fp} {model_name:30s} - {source:15s}")
        
        with_fingerprints = [name for name, info in anchors.items() if info.get('has_fingerprint')]
        print(f"\n✓ 有指纹的锚点: {len(with_fingerprints)}/{len(anchors)}")
        
        return len(anchors) >= 3
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_4_unified_loader():
    """测试统一加载器"""
    print("\n" + "="*80)
    print("测试 4: 统一加载器")
    print("="*80)
    
    try:
        from src.utils.unified_loader import load_model
        
        # 只检查函数可以调用，不实际加载模型
        print("✓ unified_loader 模块导入成功")
        print(f"✓ load_model 函数可用: {callable(load_model)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("快速核心功能测试")
    print("="*80)
    
    tests = [
        ("探针加载", test_1_probe_loading),
        ("相似度计算逻辑", test_2_similarity_logic),
        ("锚点数据库", test_3_anchor_database),
        ("统一加载器", test_4_unified_loader),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ 测试 '{test_name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 汇总
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r)
    failed = len(results) - passed
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} - {test_name}")
    
    print("-"*80)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有核心功能测试通过！")
        return 0
    else:
        print(f"\n❌ 有 {failed} 个测试失败。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
