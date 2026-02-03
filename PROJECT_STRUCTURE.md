# 項目文件清單

## 核心模組

### src/probes/ - 探針構建模組
- `political_probes.py`: 政治敏感性探針（台海關係、歷史事件、新疆西藏等）
- `linguistic_probes.py`: 語言習慣探針（兩岸用語差異）
- `memorization_probes.py`: 記憶化探針（中國特定語料庫測試）
- `__init__.py`: 統一接口

### src/fingerprint/ - 指紋提取模組
- `logit_extractor.py`: Logit 分佈指紋提取器
- `refusal_detector.py`: 拒絕響應檢測器
- `__init__.py`: 統一接口

### src/attribution/ - 歸因分析模組
- `similarity.py`: 相似度計算（Cosine, Euclidean, Pearson, KL）
- `anchor_models.py`: 錨點模型數據庫管理
- `__init__.py`: 溯源分析主函數

### src/utils/ - 工具模組
- `model_loader.py`: 模型加載器（支持 Ollama, Transformers）
- `metrics.py`: 評估指標計算
- `__init__.py`: 工具函數導出

## 實驗腳本

### experiments/
- `pilot_study.py`: 初步對比實驗（Qwen vs Llama）
- `full_evaluation.py`: 完整溯源評估流程
- `visualization.py`: 結果視覺化（t-SNE, UMAP, 熱圖）

## 配置與文檔

### 根目錄
- `README.md`: 完整的研究計畫說明
- `QUICKSTART.md`: 快速開始指南
- `requirements.txt`: Python 依賴清單
- `test_system.py`: 系統測試腳本
- `.gitignore`: Git 忽略規則

### configs/
- `default_config.yaml`: 默認配置文件

## 數據目錄

### data/
- `probes/`: 探針數據集（自動生成）
- `fingerprints/`: 提取的模型指紋
- `anchor_models/`: 錨點模型數據庫

### results/
- 實驗結果輸出目錄（JSON, HTML, PNG）

### logs/
- 日誌文件目錄

## 關鍵功能

### 探針類型（共3類）
1. **政治敏感性探針**：測試模型對兩岸政治議題的回應
2. **語言習慣探針**：比較兩岸用語差異（晶片/芯片、軟體/軟件等）
3. **記憶化探針**：檢測模型是否記憶中國特定訓練資料

### 指紋特徵（2種）
1. **Logit 指紋**：輸出機率分佈特徵向量
2. **拒絕指紋**：拒絕響應的模式和頻率

### 相似度度量（4種）
1. Cosine Similarity（餘弦相似度）
2. Euclidean Distance（歐幾里得距離）
3. Pearson Correlation（皮爾森相關係數）
4. KL Divergence（KL 散度）

### 視覺化方法（4種）
1. t-SNE（t-分佈隨機鄰域嵌入）
2. UMAP（均勻流形逼近與投影）
3. Similarity Matrix（相似度矩陣熱圖）
4. Hierarchical Clustering（層次聚類樹狀圖）

## 使用流程

### 標準工作流程
```
1. 安裝依賴 → pip install -r requirements.txt
2. 安裝 Ollama 並下載模型
3. 運行系統測試 → python test_system.py
4. 構建探針 → 自動生成到 data/probes/
5. 運行 pilot study → python experiments/pilot_study.py
6. 完整評估 → python experiments/full_evaluation.py --target-model <name>
7. 生成視覺化 → python experiments/visualization.py
```

### 輸出文件
- JSON 報告：包含完整的相似度分數和元數據
- HTML 報告：美觀的網頁格式報告
- PNG 圖表：t-SNE、UMAP 等視覺化圖表

## 技術棧

- **Python 3.8+**
- **PyTorch**: 深度學習框架
- **Transformers**: HuggingFace 模型接口
- **Ollama**: 本地 LLM 推理引擎
- **scikit-learn**: 機器學習工具
- **matplotlib/seaborn**: 數據視覺化
- **loguru**: 日誌管理

## 研究貢獻

這個系統實現了：
1. ✅ 黑盒模型溯源（無需訪問權重）
2. ✅ 多維度指紋提取（Logit + 拒絕模式）
3. ✅ 多種相似度度量（綜合評分）
4. ✅ 自動化評估流程
5. ✅ 視覺化分析工具
6. ✅ 適用於台灣資安合規需求

## 未來擴展

- [ ] 支持 vLLM 引擎
- [ ] 增加更多探針（語義理解、知識圖譜）
- [ ] 實現 API 服務模式
- [ ] 添加 Web UI
- [ ] 支持更多語言（日文、韓文）
- [ ] 對抗性測試（模型洗白檢測）

## 授權與引用

本項目用於學術研究目的。如使用請引用相關論文（待發表）。
