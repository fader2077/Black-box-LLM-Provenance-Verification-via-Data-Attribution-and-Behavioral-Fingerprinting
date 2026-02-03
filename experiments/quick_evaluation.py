"""
快速评估脚本 - 使用50个探针进行快速测试
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fingerprint import extract_fingerprint
from src.attribution import AnchorModelsDatabase
from src.attribution.similarity import SimilarityCalculator
from src.utils.unified_loader import load_model


def quick_evaluation(
    target_model: str,
    engine: str = "ollama",
    num_probes: int = 50
):
    """
    快速评估 - 使用较少探针
    
    Args:
        target_model: 待测模型
        engine: 推理引擎
        num_probes: 使用的探针数量
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info("LLM 溯源技術 - 快速評估")
    logger.info("=" * 80)
    logger.info(f"待測模型: {target_model}")
    logger.info(f"推理引擎: {engine}")
    logger.info(f"探針數量: {num_probes}")
    logger.info("=" * 80)
    
    # Step 1: 载入探针
    logger.info("\n[1/5] 準備探針數據集...")
    
    probes_path = Path("data/probes/all_probes.json")
    
    if not probes_path.exists():
        logger.error("探針文件不存在")
        return
    
    with open(probes_path, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
    
    # 合併所有探針
    all_probes = []
    for probe_type, probes in probes_data.items():
        all_probes.extend(probes)
    
    # 选取部分探针
    selected_probes = all_probes[:num_probes]
    logger.info(f"已載入 {len(selected_probes)} 個探針 (快速测试子集)")
    
    # Step 2: 检查锚点数据库
    logger.info("\n[2/5] 檢查錨點模型數據庫...")
    db = AnchorModelsDatabase()
    
    # 加载所有锚点指纹
    anchors_with_fp = []
    for model_name, data in db.anchor_models.items():
        if data.get("has_fingerprint"):
            fp = db.load_fingerprint(model_name)
            if fp:
                anchors_with_fp.append({
                    "model_id": model_name,
                    "fingerprint": fp,
                    "source": data["metadata"].get("source"),
                    "category": data["metadata"].get("category")
                })
    
    logger.info(f"  總錨點數: {len(db.anchor_models)}")
    logger.info(f"  已有指紋: {len(anchors_with_fp)}")
    
    if len(anchors_with_fp) == 0:
        logger.error("没有可用的锚点指纹")
        return
    
    # Step 3: 载入待测模型
    logger.info(f"\n[3/5] 載入待測模型: {target_model}...")
    model = load_model(
        model_name=target_model,
        engine=engine,
        device="cuda"
    )
    logger.info("✓ 模型載入成功")
    
    # Step 4: 提取模型指纹
    logger.info("\n[4/5] 提取模型指紋...")
    logger.info(f"此過程需要约 {num_probes * 3 // 60} 分钟...")
    
    fingerprint = extract_fingerprint(model, selected_probes)
    
    if not fingerprint:
        logger.error("指纹提取失败")
        return
    
    logger.info("✓ 指紋提取完成")
    
    # Step 5: 计算相似度
    logger.info("\n[5/5] 計算與錨點模型的相似度...")
    
    sim_calc = SimilarityCalculator()
    
    results = []
    for anchor in anchors_with_fp:
        anchor_id = anchor["model_id"]
        anchor_fp = anchor["fingerprint"]
        
        sim_result = sim_calc.calculate_fingerprint_similarity(fingerprint, anchor_fp)
        score = sim_result["overall_similarity"]
        
        results.append({
            "anchor": anchor_id,
            "source": anchor["source"],
            "category": anchor["category"],
            "similarity": score
        })
        
        logger.info(f"  vs {anchor_id:30s} {score:.4f}")
    
    # 排序并显示结果
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("評估結果（按相似度排序）")
    logger.info("=" * 80)
    
    for idx, result in enumerate(results, 1):
        logger.info(
            f"{idx}. {result['anchor']:30s} "
            f"[{result['category']:10s}] "
            f"{result['similarity']:.4f}"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ 評估完成")
    logger.info(f"最相似錨點: {results[0]['anchor']} ({results[0]['similarity']:.4f})")
    logger.info("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="快速模型评估")
    parser.add_argument("--target-model", required=True, help="待測模型名稱")
    parser.add_argument("--engine", default="ollama", help="推理引擎 (ollama/transformers)")
    parser.add_argument("--num-probes", type=int, default=50, help="使用的探針數量")
    
    args = parser.parse_args()
    
    quick_evaluation(
        target_model=args.target_model,
        engine=args.engine,
        num_probes=args.num_probes
    )


if __name__ == "__main__":
    main()
