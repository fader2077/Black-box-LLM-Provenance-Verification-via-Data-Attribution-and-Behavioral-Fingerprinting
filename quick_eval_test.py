"""
快速評估測試
使用少量探針進行測試
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.probes import build_all_probes
from src.fingerprint import extract_fingerprint
from src.attribution import trace_provenance, AnchorModelsDatabase
from src.utils import load_model
from loguru import logger

# 載入所有探針，但只使用前 10 個
logger.info("載入探針數據...")
all_probes_path = Path("data/probes/all_probes.json")
with open(all_probes_path, encoding='utf-8') as f:
    all_probes = json.load(f)

# 扁平化並取前 10 個
flat_probes = []
for probe_type, probes in all_probes.items():
    flat_probes.extend(probes)
test_probes = flat_probes[:10]
logger.info(f"使用 {len(test_probes)} 個探針進行測試")

# 載入模型
logger.info("載入模型: llama3.2:latest")
model = load_model("llama3.2:latest", engine="ollama")

# 提取指紋
logger.info("提取指紋...")
try:
    fingerprint = extract_fingerprint(model, test_probes)
    logger.info(f"指紋提取成功，特徵數: {len(fingerprint)}")
except Exception as e:
    logger.error(f"指紋提取失敗: {e}")
    sys.exit(1)

# 執行溯源分析
logger.info("執行溯源分析...")
try:
    result = trace_provenance(fingerprint, anchor_db_path="data/anchor_models")
    logger.info(f"溯源分析完成")
    logger.info(f"  風險等級: {result['risk_assessment']['risk_level']}")
    logger.info(f"  判定: {result['risk_assessment']['verdict']}")
    if result['similarity_scores']:
        best_match = max(result['similarity_scores'].items(), key=lambda x: x[1])
        logger.info(f"  最相似錨點: {best_match[0]} ({best_match[1]:.2%})")
except Exception as e:
    logger.error(f"溯源分析失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

logger.info("✅ 快速評估測試完成！")
