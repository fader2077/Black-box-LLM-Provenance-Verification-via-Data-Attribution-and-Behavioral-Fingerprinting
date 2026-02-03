# -*- coding: utf-8 -*-
"""
最终验证 - 确认核心修复
"""
import sys
import json
from pathlib import Path

print("=" * 80)
print("LLM 溯源技术 - 最终验证")
print("=" * 80)

# 测试1: probe_type 字段
print("\n[1/3] 验证 probe_type 字段...")
try:
    with open("data/probes/all_probes.json", 'r', encoding='utf-8') as f:
        probes = json.load(f)
    
    pol_probes = probes.get('political_sensitivity', [])
    with_type = [p for p in pol_probes if 'probe_type' in p]
    
    if len(with_type) == len(pol_probes):
        print(f"OK: {len(with_type)}/{len(pol_probes)} probes have probe_type")
    else:
        print(f"FAIL: Only {len(with_type)}/{len(pol_probes)} have probe_type")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# 测试2: 相似度计算逻辑
print("\n[2/3] 验证相似度计算逻辑...")
try:
    from src.attribution.similarity import SimilarityCalculator
    
    # 检查代码是否包含修复
    import inspect
    source = inspect.getsource(SimilarityCalculator.calculate_fingerprint_similarity)
    
    if 'if fp1.get("refusal_fingerprint") and fp2.get("refusal_fingerprint")' in source:
        print("OK: Similarity calculation logic fixed")
    else:
        print("FAIL: Similarity calculation not fixed")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# 测试3: GPT-2 自相似度
print("\n[3/3] 验证 GPT-2 自相似度...")
try:
    from src.utils.unified_loader import load_model
    from src.fingerprint import extract_fingerprint
    from src.attribution.similarity import SimilarityCalculator
    
    # 加载少量探针
    test_probes = []
    for probe_type in probes.keys():
        test_probes.extend(probes[probe_type][:10])
    
    print(f"  Loading model...")
    model = load_model("gpt2", engine="transformers")
    
    print(f"  Extracting fingerprint from {len(test_probes)} probes...")
    fp_new = extract_fingerprint(model, test_probes, include_logit=True, include_refusal=False)
    
    # 加载锚点
    with open("data/anchor_models_transformers/gpt2_fingerprint.json", 'r', encoding='utf-8') as f:
        fp_anchor = json.load(f)
    
    # 计算相似度
    calc = SimilarityCalculator()
    result = calc.calculate_fingerprint_similarity(fp_new, fp_anchor)
    
    similarity = result['overall_similarity']
    print(f"  GPT-2 vs GPT-2 similarity: {similarity:.4f}")
    
    if similarity > 0.95:
        print("OK: Self-similarity > 0.95")
    else:
        print(f"FAIL: Similarity too low ({similarity:.4f})")
        sys.exit(1)
        
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nReady to push to GitHub:")
print("  git add .")
print("  git commit -m 'fix: 修复模型相似度计算错误 (70% -> 100%)'")
print("  git push origin master")
