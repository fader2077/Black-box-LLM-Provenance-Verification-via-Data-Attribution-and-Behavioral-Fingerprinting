"""快速测试 - 验证 GPT-2 自相似度"""
import json
import numpy as np
from pathlib import Path
from src.utils.unified_loader import load_model
from src.fingerprint import extract_fingerprint
from src.attribution.similarity import SimilarityCalculator

# 1. 加载少量探针（只用 30 个测试）
with open("data/probes/all_probes.json", 'r', encoding='utf-8') as f:
    all_probes_dict = json.load(f)

test_probes = []
for probe_type, probes in all_probes_dict.items():
    test_probes.extend(probes[:10])  # 每个类型取 10 个

print(f"使用 {len(test_probes)} 个探针进行测试")

# 2. 加载 GPT-2 模型
print("正在加载 GPT-2...")
model = load_model("gpt2", engine="transformers")

# 3. 提取指纹
print("正在提取指纹...")
fp_new = extract_fingerprint(model, test_probes, include_logit=True, include_refusal=False)

# 4. 加载锚点 GPT-2 指纹
anchor_fp_path = Path("data/anchor_models_transformers/gpt2_fingerprint.json")
with open(anchor_fp_path, 'r', encoding='utf-8') as f:
    fp_anchor = json.load(f)

# 5. 计算相似度
print("\n正在计算相似度...")
calc = SimilarityCalculator()
result = calc.calculate_fingerprint_similarity(fp_new, fp_anchor)

print(f"\n✅ 结果:")
print(f"  Cosine 相似度: {result['logit_similarity']['cosine_similarity']:.4f}")
print(f"  Pearson 相关: {result['logit_similarity']['pearson_correlation']:.4f}")
print(f"  KL 相似度: {result['logit_similarity']['kl_similarity']:.4f}")
print(f"  Ensemble 分数: {result['logit_similarity']['ensemble_score']:.4f}")
print(f"  整体相似度: {result['overall_similarity']:.4f}")

# 6. 比较指纹向量
vec_new = np.array(fp_new["logit_fingerprint"]["vector"])
vec_anchor = np.array(fp_anchor["logit_fingerprint"]["vector"])

print(f"\n指纹统计:")
print(f"  新指纹维度: {len(vec_new)}")
print(f"  锚点指纹维度: {len(vec_anchor)}")
print(f"  新指纹前5个值: {vec_new[:5]}")
print(f"  锚点指纹前5个值: {vec_anchor[:5]}")

# 7. 诊断
if len(vec_new) != len(vec_anchor):
    print(f"\n⚠️  警告：维度不匹配！")
    print(f"  新指纹使用了 {len(test_probes)} 个探针")
    print(f"  锚点指纹可能使用了不同数量的探针")
else:
    unique_values = len(set(vec_new[:100]))
    print(f"\n  前100个值的唯一值数量: {unique_values}")
    if unique_values < 10:
        print(f"  ⚠️  警告：重复值过多，可能使用了伪概率！")
    else:
        print(f"  ✅  指纹向量看起来正常")
