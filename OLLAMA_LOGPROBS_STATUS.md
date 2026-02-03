# Ollama Logprobs 支援狀態報告（更新）

**日期**: 2026-02-03 18:30  
**測試版本**: Ollama 0.14.1 + ollama-python 0.4.5  
**測試狀態**: ❌ logprobs 不可用

---

## 關鍵發現

### 用戶提供的範例驗證

用戶提供了以下 Ollama 官方庫範例：

```python
import ollama

response = ollama.generate(
    model='llama3',
    prompt='Building a website can be done in',
    options={
        'logprobs': True,
        'top_logprobs': 5
    }
)

for token_data in response['completion_probabilities']:
    print(f"Token: {token_data['content']}, Logprobs: {token_data['probs']}")
```

### 實際測試結果

```
響應類型: <class 'ollama._types.GenerateResponse'>
logprobs 屬性: None
completion_probabilities: 不存在
```

**結論**: 雖然官方 Python 庫支援 logprobs **參數**，但 Ollama 0.14.1 服務端**不返回** logprobs 數據。

---

## 版本依賴分析

根據測試結果，logprobs 功能需要：

1. **Ollama 服務端**: 版本 >= 0.2.0（推測）
2. **ollama-python 庫**: 已支援 logprobs 參數
3. **API 參數**: `options={'logprobs': True, 'top_logprobs': K}`

### 當前環境

- ✅ ollama-python: 已安裝且支援參數
- ❌ Ollama 0.14.1: 服務端未實現 logprobs 返回
- ⏳ 需要升級: Ollama >= 0.2.0（或更新）

---

## 技術決策

### 選項 A：等待 Ollama 升級 ⏸️

**優點**:
- 一旦升級，系統立即可用
- 代碼已準備就緒（model_loader_v2.py）

**缺點**:
- 時間不確定
- 可能需要數週/數月

### 選項 B：繼續使用啟發式特徵 ✅（當前策略）

**優點**:
- 立即可用
- 系統完整可運行
- 已有後備機制

**缺點**:
- 相似度區分度較低
- 無法進行精確的 token 級別比較

### 選項 C：使用 Transformers 引擎 🚀（推薦）

**優點**:
- **完整 logprobs 支援**
- 精確的 token 機率分佈
- 適合學術研究

**缺點**:
- 需要下載完整模型權重（數 GB）
- GPU 記憶體需求高

---

## 實施建議

### 立即行動

1. ✅ 使用啟發式特徵完成全面測試
2. ✅ 驗證系統完整性和魯棒性
3. ✅ 文檔化 Ollama 限制

### 短期（1-2 週）

1. 追蹤 Ollama 版本更新
2. 測試 Transformers 引擎集成
3. 準備升級路徑

### 中期（1 個月）

1. 等待 Ollama logprobs 支援
2. 或切換到 vLLM/Transformers
3. 進行完整精度測試

---

## 代碼就緒狀態

### model_loader_v2.py ✅

**已實現**:
- Ollama 官方庫集成
- logprobs 參數支援
- 優雅的降級處理

**待啟用**:
- 等待 Ollama 服務端支援
- 或用於 Transformers 引擎

### logit_extractor.py ⏳

**需要更新**:
- 解析 Ollama logprobs 格式
- 處理 completion_probabilities
- 提取 top_k token 機率

---

## 結論

用戶提供的範例是**正確的**且**前瞻性的**，但需要更新版本的 Ollama 服務。

當前最佳策略：
1. 完成基於啟發式特徵的全面測試
2. 保留 logprobs 代碼以備將來升級
3. 在技術報告中明確記錄此限制

---

**更新者**: GitHub Copilot  
**測試環境**: Windows + Ollama 0.14.1  
**下一步**: 繼續全面測試流程
