# 锚点模型配置说明

本文档说明如何添加、修改或删除锚点模型。

---

## 锚点模型配置位置

锚点模型配置在以下文件中：

### 1. `data/anchor_models/metadata.json`

这是**主要配置文件**，定义了所有锚点模型的元数据。

**文件路径**: `data/anchor_models/metadata.json`

**结构示例**:
```json
{
  "anchors": [
    {
      "name": "gpt2",
      "family": "gpt",
      "source": "openai",
      "fingerprint_file": "data/anchor_models/gpt2_fingerprint.json",
      "description": "GPT-2 base model",
      "size": "124M parameters"
    },
    {
      "name": "deepseek-r1:7b",
      "family": "deepseek",
      "source": "china",
      "fingerprint_file": "data/anchor_models/deepseek_r1_7b_fingerprint.json",
      "description": "DeepSeek-R1 7B model",
      "size": "7B parameters"
    }
  ],
  "last_updated": "2026-02-04T07:00:00",
  "version": "1.0"
}
```

### 2. `src/attribution/anchor_models.py`

这个文件中的 `_load_database()` 函数从 `metadata.json` 读取配置。

**关键代码**:
```python
def _load_database(db_path: str) -> List[Dict]:
    """从数据库目录加载所有锚点模型"""
    metadata_file = Path(db_path) / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('anchors', [])
```

---

## 如何添加新锚点模型

### 步骤 1: 提取指纹

使用超稳健提取工具提取新模型的指纹：

```bash
# 示例：添加 Llama-3.1-8B 锚点
python experiments/ultra_robust_extraction.py \
  --model llama3.1:8b \
  --engine ollama \
  --num-probes 30 \
  --probes-per-session 3 \
  --rest-time 4 \
  --device cuda \
  --output data/anchor_models/llama3_1_8b_fingerprint.json
```

**参数说明**:
- `--model`: 模型名称（Ollama模型或HuggingFace路径）
- `--engine`: 推理引擎（`ollama` 或 `transformers`）
- `--num-probes`: 探针数量（建议30-100）
- `--probes-per-session`: 每N个探针重新加载模型（建议3）
- `--rest-time`: 探针间休息时间（秒）
- `--device`: 设备（`cuda` 或 `cpu`）
- `--output`: 输出指纹文件路径

### 步骤 2: 更新 metadata.json

编辑 `data/anchor_models/metadata.json`，在 `anchors` 数组中添加新条目：

```json
{
  "anchors": [
    // ... 现有锚点 ...
    {
      "name": "llama3.1:8b",
      "family": "llama",
      "source": "meta",
      "fingerprint_file": "data/anchor_models/llama3_1_8b_fingerprint.json",
      "description": "Llama-3.1-8B instruction tuned model",
      "size": "8B parameters"
    }
  ],
  "last_updated": "2026-02-04T08:20:00",
  "version": "1.1"
}
```

**字段说明**:
- `name`: 模型显示名称
- `family`: 模型家族（`llama`, `deepseek`, `gpt`, `qwen` 等）
- `source`: 来源组织（`meta`, `china`, `openai` 等）
- `fingerprint_file`: 指纹文件的相对或绝对路径
- `description`: 模型描述（可选）
- `size`: 参数量（可选）

### 步骤 3: 验证配置

运行测试以确保新锚点正确加载：

```bash
python -c "
from src.attribution.anchor_models import load_anchor_database
anchors = load_anchor_database('data/anchor_models')
for anchor in anchors:
    print(f'✓ {anchor[\"name\"]:20} {anchor[\"family\"]:10} {anchor[\"source\"]}')
"
```

---

## 如何删除锚点模型

### 方法 1: 从 metadata.json 中移除

编辑 `data/anchor_models/metadata.json`，删除对应的锚点条目：

```json
{
  "anchors": [
    // 删除不需要的锚点条目
    {
      "name": "qwen2.5:7b",  // <-- 删除这整个对象
      ...
    }
  ]
}
```

### 方法 2: 删除指纹文件（可选）

如果要彻底清理，也删除对应的指纹文件：

```bash
rm data/anchor_models/qwen2_5_7b_fingerprint.json
```

---

## 如何修改锚点属性

直接编辑 `metadata.json` 中的相应字段：

**示例：更改家族分类**
```json
{
  "name": "deepseek-r1:7b",
  "family": "deepseek",  // 修改此字段
  "source": "china",     // 或修改此字段
  ...
}
```

---

## 锚点模型命名规范

### 文件命名
- **指纹文件**: `{model_name}_fingerprint.json`
- **示例**: 
  - `gpt2_fingerprint.json`
  - `llama3_1_8b_fingerprint.json`
  - `deepseek_r1_7b_fingerprint.json`

### 模型名称
- 使用小写字母和下划线
- Ollama模型保留原名格式（如 `llama3.1:8b`）
- HuggingFace模型简化名称（如 `gpt2-medium` → `gpt2_medium`）

---

## 完整示例：添加 Qwen-2.5-7B 锚点

### 1. 提取指纹
```bash
python experiments/ultra_robust_extraction.py \
  --model qwen2.5:7b \
  --engine ollama \
  --num-probes 30 \
  --device cuda \
  --output data/anchor_models/qwen2_5_7b_fingerprint.json
```

### 2. 更新配置
编辑 `data/anchor_models/metadata.json`:
```json
{
  "anchors": [
    {
      "name": "qwen2.5:7b",
      "family": "qwen",
      "source": "china",
      "fingerprint_file": "data/anchor_models/qwen2_5_7b_fingerprint.json",
      "description": "Qwen-2.5-7B instruction tuned model"
    }
  ]
}
```

### 3. 测试
```bash
python experiments/full_evaluation.py \
  --target-model deepseek-r1:8b \
  --engine ollama
```

---

## 当前锚点模型列表

| 模型名称 | 家族 | 来源 | 指纹文件 | 状态 |
|---------|------|------|---------|------|
| gpt2 | gpt | openai | gpt2_fingerprint.json | ✅ |
| gpt2-medium | gpt | openai | gpt2_medium_fingerprint.json | ⚠️ 缺失 |
| deepseek-r1:7b | deepseek | china | deepseek_r1_7b_fingerprint.json | ✅ |
| llama3.2:3b | llama | meta | llama3_2_3b_fingerprint.json | ✅ |
| llama3.1:8b | llama | meta | llama3_1_8b_fingerprint.json | 🔄 提取中 |

---

## 故障排除

### 问题 1: "未找到锚点模型"

**原因**: `metadata.json` 路径错误或格式不正确

**解决**: 检查文件格式是否为有效 JSON，路径是否正确

### 问题 2: 相似度异常低（< 20%）

**原因**: 
- 探针数量不足
- 指纹维度不匹配
- 使用了不同的提取方法

**解决**: 
- 确保所有锚点使用相同的探针数量
- 使用 `ultra_robust_extraction.py` 统一提取
- 建议至少30个探针

### 问题 3: "指纹文件不存在"

**原因**: 指纹文件未生成或路径错误

**解决**: 
1. 检查 `fingerprint_file` 路径是否正确
2. 重新运行提取命令
3. 验证文件是否存在：`ls data/anchor_models/*_fingerprint.json`

---

## 技术细节

### 指纹文件格式

```json
{
  "model_name": "llama3.1:8b",
  "timestamp": "2026-02-04 08:20:00",
  "logit_fingerprint": {
    "vector": [0.1, 0.2, ..., 0.5],  // 长度 = num_probes × 20
    "dimension": 200,
    "stats": {
      "mean": 0.05,
      "std": 0.02,
      "min": 0.0,
      "max": 0.15
    }
  },
  "extraction_stats": {
    "total_probes": 10,
    "successful_probes": 10,
    "failed_probes": 0,
    "success_rate": 1.0
  }
}
```

### 相似度计算方法

系统使用多种相似度度量的平均值：
- Cosine 相似度
- Pearson 相关系数
- 欧几里得距离（归一化）

**注意**: 
- `full_evaluation.py` 使用完整1110维指纹（438探针 × ~20维/探针）
- `ultra_robust_extraction.py` 使用可配置的探针数（如30探针 = 600维）
- 维度不匹配时会自动零填充对齐

---

## 最佳实践

1. **统一提取方法**: 所有锚点使用相同的提取工具和参数
2. **足够的探针**: 至少30个探针，推荐50-100个
3. **定期更新**: 模型更新时重新提取指纹
4. **备份配置**: 修改前备份 `metadata.json`
5. **验证完整性**: 提取后运行测试验证

---

## 相关文件

- 锚点配置: `data/anchor_models/metadata.json`
- 锚点加载: `src/attribution/anchor_models.py`
- 提取工具: `experiments/ultra_robust_extraction.py`
- 完整评估: `experiments/full_evaluation.py`
- 快速分析: `quick_similarity_analysis.py`

---

**最后更新**: 2026年2月4日  
**版本**: 1.0
