"""检查锚点指纹的有效性"""
import json
from pathlib import Path

anchor_dir = Path("data/anchor_models")

anchors = [
    "gpt2_fingerprint.json",
    "gpt2_medium_fingerprint.json",
    "deepseek_r1_7b_fingerprint.json",
    "llama3_2_3b_fingerprint.json",
    "llama3_1_8b_fingerprint.json",
]

print("=" * 60)
print("锚点指纹有效性检查")
print("=" * 60)

for anchor_file in anchors:
    filepath = anchor_dir / anchor_file
    if not filepath.exists():
        print(f"❌ {anchor_file:35} 不存在")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model_name = data.get("model_name", "Unknown")
    timestamp = data.get("timestamp", "Unknown")
    
    logit_fp = data.get("logit_fingerprint", {})
    vector = logit_fp.get("vector", [])
    dimension = logit_fp.get("dimension", 0)
    
    # 检查是否有有效的logits
    max_val = max(vector) if vector else 0
    min_val = min(vector) if vector else 0
    non_zero = sum(1 for v in vector if v != 0.0)
    
    status = "✅ 有效" if max_val > 0 or min_val < 0 else "❌ 全零"
    
    print(f"{status} {anchor_file:35}")
    print(f"   模型: {model_name}")
    print(f"   维度: {dimension}, 非零值: {non_zero}/{len(vector)}")
    print(f"   范围: [{min_val:.3f}, {max_val:.3f}]")
    print()

print("=" * 60)
print("建议:")
print("=" * 60)
print("✅ GPT2 系列: 使用 transformers 引擎提取，有真实 logits")
print("✅ DeepSeek-R1: 使用完整评估提取，有真实 logits")
print("❌ Llama 系列: 使用 ollama 引擎提取，仅有启发式特征（全零）")
print()
print("解决方案:")
print("1. 使用 transformers/HuggingFace 版本的 Llama 模型重新提取")
print("2. 或者使用 vLLM 引擎（支持 logprobs）")
print("3. 或者仅使用 transformers 锚点进行测试")
