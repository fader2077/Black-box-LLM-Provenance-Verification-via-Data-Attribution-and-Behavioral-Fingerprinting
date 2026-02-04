"""
使用现有有效锚点测试目标模型
仅使用有真实logits的锚点: GPT2, GPT2-Medium, DeepSeek-R1:7b
"""
import json
from pathlib import Path
from loguru import logger
import numpy as np

def load_fingerprint(fp_path):
    """加载指纹文件"""
    with open(fp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['logit_fingerprint']['vector'])

def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    # 对齐维度
    min_len = min(len(v1), len(v2))
    v1 = v1[:min_len]
    v2 = v2[:min_len]
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)

# 有效锚点（有真实logits）
anchors = {
    "gpt2": {
        "path": "data/anchor_models/gpt2_fingerprint.json",
        "family": "gpt",
        "source": "openai"
    },
    "gpt2-medium": {
        "path": "data/anchor_models/gpt2_medium_fingerprint.json",
        "family": "gpt",
        "source": "openai"
    },
    "deepseek-r1:7b": {
        "path": "data/anchor_models/deepseek_r1_7b_fingerprint.json",
        "family": "deepseek",
        "source": "china"
    }
}

# 目标模型
target_model = "deepseek-r1:8b-llama-distill"
target_path = "results/deepseek-r1_8b-llama-distill-q4_K_M_fingerprint.json"

logger.info("=" * 70)
logger.info("使用有效锚点测试目标模型")
logger.info("=" * 70)

# 检查目标指纹是否存在
if not Path(target_path).exists():
    logger.error(f"目标指纹不存在: {target_path}")
    logger.info("请先运行: python experiments/full_evaluation.py --target-model deepseek-r1:8b-llama-distill-q4_K_M --engine ollama")
    exit(1)

# 加载目标指纹
logger.info(f"\n加载目标模型指纹: {target_model}")
target_fp = load_fingerprint(target_path)
logger.info(f"  维度: {len(target_fp)}")
logger.info(f"  范围: [{target_fp.min():.3f}, {target_fp.max():.3f}]")

# 检查目标指纹是否有效
if target_fp.max() == 0 and target_fp.min() == 0:
    logger.error("⚠️ 目标指纹全为0（启发式特征），无法进行有效比较")
    logger.info("建议使用 transformers 引擎重新提取")
    exit(1)

# 计算与各锚点的相似度
logger.info("\n" + "=" * 70)
logger.info("相似度分析")
logger.info("=" * 70)

similarities = []

for anchor_name, anchor_info in anchors.items():
    anchor_path = Path(anchor_info['path'])
    
    if not anchor_path.exists():
        logger.warning(f"锚点指纹不存在: {anchor_path}")
        continue
    
    # 加载锚点指纹
    anchor_fp = load_fingerprint(anchor_path)
    
    # 检查锚点指纹是否有效
    if anchor_fp.max() == 0 and anchor_fp.min() == 0:
        logger.warning(f"⚠️ {anchor_name} 指纹全为0，跳过")
        continue
    
    # 计算相似度
    similarity = cosine_similarity(target_fp, anchor_fp)
    
    similarities.append({
        "anchor": anchor_name,
        "family": anchor_info['family'],
        "source": anchor_info['source'],
        "similarity": similarity
    })
    
    logger.info(f"\n{anchor_name:25} [{anchor_info['family']:10}]")
    logger.info(f"  相似度: {similarity:.4f}")
    logger.info(f"  锚点维度: {len(anchor_fp)}")

# 排序并显示结果
logger.info("\n" + "=" * 70)
logger.info("最终结果")
logger.info("=" * 70)

similarities.sort(key=lambda x: x['similarity'], reverse=True)

print("\n相似度排名:")
for i, sim in enumerate(similarities, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
    print(f"{i}. {sim['anchor']:25} [{sim['family']:10}]  {sim['similarity']:.4f}  {emoji}")

# 判定
if similarities:
    best = similarities[0]
    second_best = similarities[1] if len(similarities) > 1 else None
    
    print(f"\n结论:")
    if second_best:
        diff_pct = (best['similarity'] - second_best['similarity']) / second_best['similarity'] * 100
        print(f"  {target_model} 与 {best['family']} 家族相似度最高")
        print(f"  比第二名高 {diff_pct:.2f}%")
    else:
        print(f"  {target_model} 与 {best['family']} 家族最相似")
    
    print(f"\n  最高相似度: {best['similarity']:.4f}")
    print(f"  家族: {best['family']}")
    print(f"  来源: {best['source']}")
    
    # 置信度评估
    if best['similarity'] > 0.8:
        confidence = "极高"
    elif best['similarity'] > 0.6:
        confidence = "高"
    elif best['similarity'] > 0.4:
        confidence = "中等"
    else:
        confidence = "低"
    
    print(f"  置信度: {confidence}")

logger.info("\n" + "=" * 70)
logger.info("注意事项")
logger.info("=" * 70)
logger.info("✅ 此次测试使用了有真实logits的锚点")
logger.info("⚠️ 未包含Llama系列锚点（Ollama不支持logprobs）")
logger.info("💡 若需更准确结果，建议使用HuggingFace transformers引擎")
