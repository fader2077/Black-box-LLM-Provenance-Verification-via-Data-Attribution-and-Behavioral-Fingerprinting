# Ollama Logprobs 问题分析与解决方案

## 问题概述

经过深入测试，发现了一个关键技术限制：

**Ollama API 不支持 logprobs 输出**

这导致：
1. 使用 Ollama 引擎提取的所有指纹都是**启发式特征**（基于响应长度、字符多样性等），而非真实的 token logits
2. 锚点模型指纹（Llama3.2:3b, Llama3.1:8b）全为0或常数
3. 目标模型指纹（DeepSeek-R1-Distill-Llama-8B）也是启发式值
4. 相似度计算无意义（都是 -0.02 左右的负值）

## 验证结果

### 锚点指纹有效性检查

```
✅ 有效 gpt2_fingerprint.json
   模型: gpt2
   维度: 1110, 非零值: 1110/1110
   范围: [-0.675, 2.260]
   引擎: transformers

✅ 有效 gpt2_medium_fingerprint.json
   模型: gpt2-medium
   维度: 1110, 非零值: 1110/1110
   范围: [-0.724, 2.329]
   引擎: transformers

✅ 有效 deepseek_r1_7b_fingerprint.json
   模型: deepseek-r1:7b
   维度: 1110, 非零值: 1110/1110
   范围: [-0.533, 2.018]
   引擎: transformers/full_evaluation

❌ 全零 llama3_2_3b_fingerprint.json
   模型: llama3.2:3b
   维度: 200, 非零值: 0/200
   范围: [0.000, 0.000]
   引擎: ollama (启发式)

❌ 全零 llama3_1_8b_fingerprint.json
   模型: llama3.1:8b
   维度: 200, 非零值: 0/200
   范围: [0.000, 0.000]
   引擎: ollama (启发式)
```

### GPT2 自相似度测试

```
python quick_test.py

✅ 结果:
  Cosine 相似度: 1.0000
  Pearson 相关: 1.0000
  整体相似度: 1.0000
  
✅ 系统对 transformers 引擎正常工作
```

## 根本原因

### Ollama API 限制

查看代码 `src/fingerprint/logit_extractor.py:67`:
```python
elif "logprobs_available" in output and not output["logprobs_available"]:
    logger.debug("Ollama API 不支援 logprobs，使用基於回應的啟發式特徵")
    return self._extract_from_api_response(output.get("text", ""), target_tokens)
```

### 启发式特征实现

查看 `src/fingerprint/logit_extractor.py:309`:
```python
def _extract_from_api_response(self, response, target_tokens):
    """
    由於 Ollama 不提供 logprobs，我們使用基於回應的啟發式特徵
    """
    # 计算文本特征作为偽机率
    length_feature = min(len(response_text) / 100.0, 1.0)
    diversity_feature = min(unique_chars / 50.0, 1.0)
    chinese_ratio = chinese_chars / max(len(response_text), 1)
    # ...
    return {"top_k_probs": [length_feature, diversity_feature, ...]}
```

这些启发式特征**不是真实的模型行为指纹**，无法用于溯源分析。

## 解决方案

### 方案 1: HuggingFace Transformers 引擎 ✅ 推荐

**优点**:
- ✅ 原生支持 logprobs
- ✅ 高质量指纹（1110维）
- ✅ GPU 加速
- ✅ 已验证工作正常（GPT2 100% 自相似度）

**缺点**:
- ❌ 需要下载完整模型（~16GB for Llama-3.1-8B）
- ❌ 需要 HuggingFace 访问权限
- ❌ 内存占用较大

**使用方法**:
```bash
# 提取锚点
python experiments/full_evaluation.py \
  --target-model meta-llama/Llama-3.1-8B-Instruct \
  --engine transformers \
  --device cuda

# 测试目标模型
python experiments/full_evaluation.py \
  --target-model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --engine transformers \
  --device cuda
```

**注意**: 需要先登录 HuggingFace:
```bash
pip install huggingface-hub
huggingface-cli login
```

### 方案 2: vLLM 引擎 ⚡ 高性能

**优点**:
- ✅ 支持 logprobs API
- ✅ 高性能推理
- ✅ GPU 优化

**缺点**:
- ❌ 需要安装 vLLM
- ❌ 配置复杂

**安装**:
```bash
pip install vllm
```

**使用**:
```bash
python experiments/full_evaluation.py \
  --target-model meta-llama/Llama-3.1-8B \
  --engine vllm \
  --device cuda
```

### 方案 3: 仅使用有效锚点 🎯 实用方案

**优点**:
- ✅ 立即可用
- ✅ 无需额外下载
- ✅ 已验证工作

**缺点**:
- ❌ 缺少 Llama 家族锚点
- ❌ 比较范围有限

**当前有效锚点**:
- gpt2 (124M) - GPT 家族
- gpt2-medium (355M) - GPT 家族
- deepseek-r1:7b (7B) - DeepSeek 家族

**使用方法**:
```bash
python test_with_valid_anchors.py
```

## 推荐工作流程

### 阶段 1: 快速验证（当前可用）

1. **验证系统工作**:
```bash
python quick_test.py  # GPT2 自相似度应为 100%
```

2. **使用现有锚点测试**:
```bash
python test_with_valid_anchors.py
```

3. **限制说明**:
   - 仅能与 GPT2, DeepSeek-R1:7b 比较
   - 目标模型需使用 transformers 引擎提取

### 阶段 2: 完整测试（需要 HuggingFace）

1. **准备 HuggingFace 访问**:
```bash
huggingface-cli login
```

2. **提取 Llama 锚点**:
```bash
python experiments/full_evaluation.py \
  --target-model meta-llama/Llama-3.1-8B-Instruct \
  --engine transformers \
  --device cuda \
  --output data/anchor_models/llama3_1_8b_fingerprint_transformers.json
```

3. **更新 metadata.json**:
编辑 `data/anchor_models/metadata.json`:
```json
{
  "llama3.1:8b": {
    "name": "Llama-3.1-8B-Instruct",
    "source": "meta",
    "category": "llama",
    "fingerprint_file": "data/anchor_models/llama3_1_8b_fingerprint_transformers.json",
    "engine": "transformers",
    "hf_model": "meta-llama/Llama-3.1-8B-Instruct"
  }
}
```

4. **提取目标模型指纹**:
```bash
python experiments/full_evaluation.py \
  --target-model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --engine transformers \
  --device cuda
```

5. **运行完整分析**:
```bash
python complete_provenance_test.py
```

## 当前项目状态

### ✅ 已验证工作
- Transformers 引擎 logprobs 提取
- GPT2 指纹提取（100% 自相似度）
- 相似度计算算法
- GPU 加速 (RTX 4090)

### ⚠️ 技术限制
- Ollama 不支持 logprobs
- 现有 Llama 锚点无效（使用 Ollama 提取）
- 目标模型指纹无效（使用 Ollama 提取）

### 🔄 需要完成
- [ ] 使用 transformers 引擎重新提取 Llama 锚点
- [ ] 使用 transformers 引擎提取目标模型指纹
- [ ] 更新 metadata.json 配置
- [ ] 运行完整测试
- [ ] 生成最终报告
- [ ] 推送到 GitHub

## 时间估算

### 快速方案（仅现有锚点）
- 验证系统: 2分钟
- 测试分析: 3分钟
- 文档整理: 10分钟
- **总计: 15分钟**

### 完整方案（HuggingFace）
- 下载 Llama-3.1-8B: 10-30分钟
- 提取锚点指纹: 15-20分钟
- 下载 DeepSeek-R1-Distill: 10-30分钟
- 提取目标指纹: 15-20分钟
- 运行测试: 5分钟
- 文档整理: 10分钟
- **总计: 65-115分钟**

## 建议

基于当前情况，建议：

1. **立即可行**: 使用方案3（现有锚点）进行初步测试
2. **文档说明**: 清楚标注 Ollama 限制
3. **未来工作**: 在 README 中说明需要 HuggingFace 访问
4. **推送代码**: 先推送当前工作成果和文档

这样可以：
- ✅ 展示系统架构和设计
- ✅ 说明技术实现和限制
- ✅ 提供完整的使用指南
- ✅ 为未来扩展留下清晰路径
