# 基於資料歸因與行為指紋的黑盒大型語言模型溯源技術研究

Black-box LLM Provenance Verification via Data Attribution and Behavioral Fingerprinting

> ⚠️ **重要修复**: 2026年2月4日修复了相似度计算错误（70% → 100%）。详见 [BUGFIX_20260204.md](BUGFIX_20260204.md)

## 快速開始

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 運行評估

**重要**: 指定正確的推理引擎

```bash
# ✅ 正確: HuggingFace 模型使用 transformers 引擎
python experiments/full_evaluation.py --target-model gpt2 --engine transformers

# ✅ 正確: Ollama 模型使用 ollama 引擎  
python experiments/full_evaluation.py --target-model qwen2.5:7b --engine ollama

# ❌ 錯誤: 未指定引擎（默認 ollama，會導致 HuggingFace 模型失敗）
python experiments/full_evaluation.py --target-model gpt2
```

### 快速驗證

測試 GPT-2 自相似度（應為 100%）:
```bash
$env:PYTHONIOENCODING="utf-8"  # Windows 中文系統需要
python quick_test.py
```

預期輸出:
```
✅ 結果:
  Cosine 相似度: 1.0000
  Pearson 相關: 1.0000
  整體相似度: 1.0000
```

## 研究目標

在不接觸模型權重與訓練細節的前提下（Black-box assumption），針對地端部署的 LLM，透過設計特定的**資料歸因探針（Attribution Probes）**，提取模型在特定語義空間下的**行為指紋**，藉此判定該模型是否源自特定的基礎模型家族。

## 核心理論框架

若模型源自特定分佈（例如包含大量簡體中文政治審查、特定價值觀的語料），即使經過微調或對齊，其在特定特徵空間中的條件機率分佈仍會保留**殘留特徵（Residual Features）**。

## 方法論架構

### 第一階段：構建具有區辨力的「歸因探針資料集」
- **政治敏感性探針 (Political Sensitivity Probes)**
- **語言習慣探針 (Linguistic Shibboleths)**
- **記憶化探針 (Memorization Probes)**

### 第二階段：黑盒指紋提取
- **Logit 分佈指紋 (Logit-based Fingerprint)**
- **拒絕響應指紋 (Refusal Response Fingerprint)**

### 第三階段：資料歸因與相似度分析
- **錨點模型對比 (Anchor Model Comparison)**
- **計算歸因分數 (Attribution Score)**

## 專案結構

```
thesis/
├── src/
│   ├── probes/              # 探針數據集構建模組
│   │   ├── political_probes.py
│   │   ├── linguistic_probes.py
│   │   └── memorization_probes.py
│   ├── fingerprint/         # 指紋提取模組
│   │   ├── logit_extractor.py
│   │   └── refusal_detector.py
│   ├── attribution/         # 歸因分析模組
│   │   ├── similarity.py
│   │   └── anchor_models.py
│   └── utils/               # 工具函數
│       ├── model_loader.py
│       └── metrics.py
├── data/                    # 數據目錄
│   ├── probes/              # 探針數據集
│   ├── fingerprints/        # 提取的指紋
│   └── anchor_models/       # 錨點模型指紋
├── experiments/             # 實驗腳本
│   ├── pilot_study.py       # 初步實驗
│   ├── full_evaluation.py   # 完整評估
│   └── visualization.py     # 結果視覺化
├── configs/                 # 配置文件
│   └── default_config.yaml
├── requirements.txt
└── README.md
```

## 安裝與環境設定

```bash
# 安裝依賴
pip install -r requirements.txt

# 下載 Ollama 模型 (作為錨點模型)
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
ollama pull deepseek-r1:7b
```

## 快速開始

### 1. 構建探針數據集

```python
from src.probes import build_all_probes

# 構建完整的探針數據集
probes = build_all_probes(output_dir='data/probes')
```

### 2. 提取模型指紋

```python
from src.fingerprint import extract_fingerprint

# 對待測模型提取指紋
fingerprint = extract_fingerprint(
    model_name='target_model',
    probes_path='data/probes/all_probes.json'
)
```

### 3. 歸因分析

```python
from src.attribution import trace_provenance

# 進行溯源分析
results = trace_provenance(
    target_fingerprint=fingerprint,
    anchor_db_path='data/anchor_models'
)
```

## 使用案例

### Pilot Study: Qwen vs Llama 對比實驗

```bash
python experiments/pilot_study.py --models qwen2.5:7b llama3.2:3b
```

此實驗會：
1. 使用50個兩岸差異Prompt測試兩個模型
2. 提取Log-probability分佈
3. 生成t-SNE視覺化分佈圖

### 完整評估流程

```bash
python experiments/full_evaluation.py --target-model <model_name> --output report.json
```

輸出範例：
```json
{
  "model": "unknown_model",
  "attribution_scores": {
    "qwen2.5:7b": 0.85,
    "llama3.2:3b": 0.32,
    "deepseek-r1:7b": 0.78
  },
  "verdict": "高風險：85% 行為特徵與 Qwen 一致",
  "confidence": 0.91
}
```

## 關鍵技術細節

### Logit提取範例

```python
# 黑盒模式下提取Token機率
output = model.generate(
    prompt="計算機的[MASK]核心",
    output_scores=True,
    return_dict_in_generate=True
)

# 比較"算術" vs "運算"的機率
token_probs = extract_token_probs(output, ["算術", "運算"])
```

### 處理模型洗白 (Model Laundering)

關注**深層知識歸因**：
- 測試極度冷門的中國特定知識點
- 即使使用繁體中文回答，知識來源仍會暴露

## 已知問題與解決方案

### 問題 1: Transformers 引擎加載大模型失敗
**現象**: 加載DeepSeek-R1-Distill-Llama-8B等大模型時出現KeyboardInterrupt  
**原因**: PyTorch張量操作中的低層次中斷問題  
**解決方案**: 使用Ollama引擎作為替代方案

```bash
# 不推薦（會失敗）
python experiments/full_evaluation.py --target-model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --engine transformers

# 推薦（使用Ollama）
ollama pull deepseek-r1:8b-llama-distill-q4_K_M
python experiments/full_evaluation.py --target-model deepseek-r1:8b-llama-distill-q4_K_M --engine ollama
```

### 問題 2: Ollama長時間運行穩定性
**現象**: 處理大量探針（438個）時可能出現連接錯誤  
**原因**: 長時間HTTP請求和資源競爭  
**解決方案**: 使用快速評估模式（50個探針）

```bash
# 快速評估（推薦用於初步測試）
python experiments/quick_evaluation.py --target-model MODEL_NAME --num-probes 50

# 完整評估（時間較長，可能需要分批處理）
python experiments/full_evaluation.py --target-model MODEL_NAME
```

### 問題 3: GPU內存不足
**現象**: CUDA out of memory  
**解決方案**: 
- 使用較小的模型（如qwen2.5:7b而非14b）
- 啟用量化（Q4_K_M）
- 使用CPU模式 `--device cpu`

## 實驗數據集來源

1. **SafetyBench (中文版)**: 政治安全相關部分
2. **C-Eval / CMMLU**: 中國歷史文化部分
3. **兩岸差異詞表**: 自建1000對用語對照表

## 相關文獻

- Model Fingerprinting for LLMs
- Dataset Inference Attacks
- Membership Inference Attack
- Instruction Tuning Fingerprint

## 預期成果

- **量化報告**: 模型成分分析表
- **合規證明**: 技術性驗證手段
- **自動化工具**: 可部署的檢測腳本

## 測試報告

查看詳細測試報告：
- [GPU支持報告](GPU_SUPPORT_REPORT.md)
- [DeepSeek-R1測試報告](DEEPSEEK_R1_TEST_REPORT.md)
- [最終測試報告](FINAL_TEST_REPORT_20260204.md)
- [工作總結](FINAL_SUMMARY_20260204.md)

## 授權

本研究計畫用於學術研究與台灣資安合規需求。

## 引用

如使用本研究，請引用：
```
@misc{llm_provenance_2026,
  title={Black-box LLM Provenance Verification via Data Attribution and Behavioral Fingerprinting},
  author={Your Name},
  year={2026}
}
```
