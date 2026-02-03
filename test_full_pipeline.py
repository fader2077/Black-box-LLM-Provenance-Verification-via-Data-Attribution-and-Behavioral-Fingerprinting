"""
完整流程測試（接受 Ollama 無 logprobs 限制）
使用啟發式特徵進行指紋提取和溯源分析
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from loguru import logger
from src.utils import load_model
from src.fingerprint import extract_fingerprint
from src.attribution import trace_provenance

logger.info("=" * 80)
logger.info("完整流程驗證測試（Ollama 無 Logprobs 模式）")
logger.info("=" * 80)

# 測試配置
TEST_MODEL = "llama3.2:latest"
NUM_TEST_PROBES = 10

# 1. 載入探針
logger.info(f"\n[1/5] 載入 {NUM_TEST_PROBES} 個測試探針...")
all_probes_path = Path("data/probes/all_probes.json")
with open(all_probes_path, encoding='utf-8') as f:
    all_probes = json.load(f)

flat_probes = []
for probe_type, probes in all_probes.items():
    flat_probes.extend(probes)
test_probes = flat_probes[:NUM_TEST_PROBES]
logger.info(f"✓ 已載入 {len(test_probes)} 個探針")

# 2. 載入模型
logger.info(f"\n[2/5] 載入模型: {TEST_MODEL}...")
model = load_model(TEST_MODEL, engine="ollama")
logger.info("✓ 模型載入成功")

# 3. 提取指紋
logger.info("\n[3/5] 提取模型指紋...")
logger.info("注意：Ollama 0.14.1 不支援 logprobs，將使用啟發式特徵")
try:
    fingerprint = extract_fingerprint(model, test_probes)
    logger.info(f"✓ 指紋提取成功")
    logger.info(f"  特徵維度: {fingerprint}")
    
    # 檢查指紋內容
    if "logit_distribution" in fingerprint:
        logger.info(f"  Logit 分佈形狀: {fingerprint['logit_distribution'].shape if hasattr(fingerprint['logit_distribution'], 'shape') else 'N/A'}")
    if "refusal_patterns" in fingerprint:
        logger.info(f"  拒絕模式: {fingerprint['refusal_patterns']}")
except Exception as e:
    logger.error(f"✗ 指紋提取失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 執行溯源分析
logger.info("\n[4/5] 執行溯源分析...")
try:
    result = trace_provenance(fingerprint, anchor_db_path="data/anchor_models")
    logger.info("✓ 溯源分析完成")
    logger.info(f"  目標模型: {result.get('target_model', 'unknown')}")
    logger.info(f"  風險等級: {result['risk_assessment']['risk_level']}")
    logger.info(f"  判定: {result['risk_assessment']['verdict']}")
    
    if result['similarity_scores']:
        sorted_scores = sorted(result['similarity_scores'].items(), key=lambda x: x[1], reverse=True)
        logger.info("\n  相似度排名:")
        for i, (model_name, score) in enumerate(sorted_scores[:5], 1):
            logger.info(f"    {i}. {model_name}: {score:.4f}")
    else:
        logger.warning("  未計算相似度分數")
        
except Exception as e:
    logger.error(f"✗ 溯源分析失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 總結
logger.info("\n[5/5] 測試總結")
logger.info("=" * 80)
logger.info("✅ 完整流程測試通過！")
logger.info("")
logger.info("⚠️  重要說明:")
logger.info("   1. Ollama 0.14.1 不支援 logprobs 輸出")
logger.info("   2. 系統已自動回退到啟發式特徵")
logger.info("   3. 相似度分數可能較低（因缺乏精確的機率分佈）")
logger.info("   4. 對於生產環境，建議：")
logger.info("      - 升級到支援 logprobs 的 Ollama 版本")
logger.info("      - 或使用 vLLM/Transformers 引擎")
logger.info("      - 或收集更多啟發式特徵以提高區分度")
logger.info("=" * 80)
