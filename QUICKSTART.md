# LLM 溯源技術研究 - 快速開始指南

## 安裝依賴

```powershell
# 創建虛擬環境（可選）
python -m venv venv
.\venv\Scripts\Activate.ps1

# 安裝依賴
pip install -r requirements.txt
```

## 安裝 Ollama 並下載模型

```powershell
# 下載並安裝 Ollama (https://ollama.ai)

# 下載錨點模型
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
ollama pull deepseek-r1:7b
```

## 使用範例

### 1. 構建探針數據集

```powershell
python -c "from src.probes import build_all_probes; build_all_probes()"
```

### 2. 運行 Pilot Study（快速驗證）

```powershell
# 比較 Qwen 和 Llama
python experiments/pilot_study.py

# 或指定不同的模型
python experiments/pilot_study.py --models qwen2.5:7b deepseek-r1:7b --num-probes 30
```

### 3. 完整評估

```powershell
# 對待測模型進行完整溯源分析
python experiments/full_evaluation.py --target-model <your_model_name>

# 範例
python experiments/full_evaluation.py --target-model qwen2.5:7b
```

### 4. 生成視覺化

```powershell
# 確保已有多個模型的指紋數據
python experiments/visualization.py --fingerprints-dir data/fingerprints
```

## 項目結構

```
thesis/
├── src/
│   ├── probes/              # 探針構建
│   ├── fingerprint/         # 指紋提取
│   ├── attribution/         # 歸因分析
│   └── utils/               # 工具函數
├── experiments/             # 實驗腳本
├── data/                    # 數據目錄
│   ├── probes/              # 探針數據
│   ├── fingerprints/        # 指紋數據
│   └── anchor_models/       # 錨點模型數據庫
├── results/                 # 實驗結果
└── configs/                 # 配置文件
```

## 常見問題

### Q: Ollama 連接失敗？
A: 確保 Ollama 服務正在運行，可以運行 `ollama list` 檢查。

### Q: 內存不足？
A: 減少 `--num-probes` 參數，或使用更小的模型。

### Q: 指紋提取很慢？
A: 這是正常的。完整的指紋提取可能需要數小時，可以先用 pilot study 快速驗證。

## 下一步

1. 查閱 [README.md](README.md) 了解詳細的研究方法
2. 運行 pilot study 驗證系統可行性
3. 提取更多錨點模型的指紋
4. 對實際待測模型進行完整評估
5. 分析結果並撰寫論文

## 聯絡

如有問題，請參考項目文檔或提交 Issue。
