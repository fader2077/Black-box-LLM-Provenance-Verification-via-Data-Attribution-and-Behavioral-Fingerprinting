# DeepSeek-R1 系列测试报告

日期: 2026-02-04
测试目标: 确定 DeepSeek-R1-Distill-Llama-8B 模型的谱系（llama vs deepseek）

## 测试配置

- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **推理引擎**: Ollama
- **探针数量**: 
  - 快速测试: 50 个探针
  - 完整测试: 438 个探针
- **锚点模型**: 
  - gpt2 (OpenAI GPT-2, 124M)
  - gpt2-medium (OpenAI GPT-2-Medium, 355M)
  - deepseek-r1:7b (DeepSeek-R1, 7B)
  - llama3.2:3b (Meta Llama 3.2, 3B) - 正在添加

## 测试模型

### 1. deepseek-r1:7b
- **状态**: ✓ 已作为锚点模型
- **来源**: DeepSeek
- **参数**: 7B

### 2. deepseek-r1:8b
- **状态**: 🔄 测试中
- **来源**: DeepSeek
- **参数**: 8B

### 3. deepseek-r1:8b-llama-distill-q4_K_M
- **状态**: ⏳ 待测试
- **来源**: DeepSeek (Llama Distilled)
- **参数**: 8B (量化 4-bit)
- **说明**: 最接近用户要求的 DeepSeek-R1-Distill-Llama-8B

## 测试结果

### 快速测试 (50 探针)

#### deepseek-r1:8b

相似度排名:
- [ ] vs gpt2: 
- [ ] vs gpt2-medium: 
- [ ] vs deepseek-r1:7b:
- [ ] vs llama3.2:3b:

结论: 

---

#### deepseek-r1:8b-llama-distill

相似度排名:
- [ ] vs gpt2: 
- [ ] vs gpt2-medium: 
- [ ] vs deepseek-r1:7b:
- [ ] vs llama3.2:3b:

结论:

---

### 完整测试 (438 探针)

#### deepseek-r1:8b

相似度排名:
- [ ] vs gpt2: 
- [ ] vs gpt2-medium: 
- [ ] vs deepseek-r1:7b:
- [ ] vs llama3.2:3b:

结论:

---

#### deepseek-r1:8b-llama-distill

相似度排名:
- [ ] vs gpt2: 
- [ ] vs gpt2-medium: 
- [ ] vs deepseek-r1:7b:
- [ ] vs llama3.2:3b:

结论:

---

## 其他模型测试

### qwen2.5:7b
- **状态**: 
- **结果**:

### gemma2:2b
- **状态**: 
- **结果**:

### llama3.2:3b
- **状态**: 
- **结果**:

---

## 总结

### 主要发现

1. **DeepSeek-R1-Distill-Llama-8B 谱系判定**:
   - 

2. **模型相似度模式**:
   -

3. **系统性能**:
   - GPU 利用率:
   - 平均推理时间/探针:
   - 总测试时间:

### 技术问题

1. **已解决**:
   - GPT-2 自我相似度 70% → 100% ✓
   - GPU 支持实现 ✓
   - Unicode 编码问题 ✓

2. **已知问题**:
   - Transformers 引擎 KeyboardInterrupt (DeepSeek-R1-Distill-Llama-8B 加载失败)
   - Ollama API 不支持 logprobs (使用启发式特征)

3. **解决方案**:
   - 使用 Ollama 引擎作为替代
   - 使用 deepseek-r1:8b-llama-distill-q4_K_M (量化版本)

---

## 下一步行动

- [ ] 完成所有快速测试
- [ ] 执行完整测试（如果需要）
- [ ] 更新 README 文档
- [ ] 推送到 GitHub

---

## 附录

### 命令记录

```bash
# 添加 llama3.2:3b 锚点
python add_llama_anchor.py

# 快速测试 deepseek-r1:8b
python experiments/quick_evaluation.py --target-model deepseek-r1:8b --engine ollama --num-probes 50

# 快速测试 deepseek-r1:8b-llama-distill
python experiments/quick_evaluation.py --target-model deepseek-r1:8b-llama-distill-q4_K_M --engine ollama --num-probes 50
```

### 测试日志路径

- 快速测试日志: `logs/quick_evaluation_*.log`
- 完整测试日志: `results/evaluation_*.json`
