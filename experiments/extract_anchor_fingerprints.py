"""
提取錨點模型指紋
為數據庫中的所有錨點模型提取指紋
"""

import sys
import json
import argparse
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes import build_all_probes
from src.fingerprint import extract_fingerprint
from src.attribution import AnchorModelsDatabase
from src.utils import load_model


def extract_anchor_fingerprints(
    models: list = None,
    num_probes: int = 50,
    force: bool = False
):
    """
    提取錨點模型的指紋
    
    Args:
        models: 要提取的模型列表（None 表示所有）
        num_probes: 使用的探針數量
        force: 是否強制重新提取
    """
    logger.info("=" * 80)
    logger.info("錨點模型指紋提取")
    logger.info("=" * 80)
    
    # 載入錨點數據庫
    db = AnchorModelsDatabase()
    
    # 確定要處理的模型
    if models is None:
        models = list(db.anchor_models.keys())
    
    logger.info(f"將處理 {len(models)} 個錨點模型")
    
    # 載入探針
    logger.info("載入探針數據...")
    probes_path = Path("data/probes/all_probes.json")
    
    if not probes_path.exists():
        logger.info("構建探針數據...")
        probes_data = build_all_probes()
    else:
        with open(probes_path, 'r', encoding='utf-8') as f:
            probes_data = json.load(f)
    
    # 選取部分探針
    all_probes = []
    for probe_type, probes in probes_data.items():
        all_probes.extend(probes)
    
    # 限制探針數量以加快測試
    selected_probes = all_probes[:num_probes]
    logger.info(f"使用 {len(selected_probes)} 個探針")
    
    # 逐個處理模型
    for idx, model_name in enumerate(models, 1):
        logger.info(f"\n[{idx}/{len(models)}] 處理 {model_name}...")
        
        # 檢查是否已有指紋
        if not force and db.anchor_models[model_name].get("has_fingerprint"):
            logger.info(f"  跳過（已有指紋）")
            continue
        
        try:
            # 載入模型
            logger.info(f"  載入模型...")
            model = load_model(model_name, engine="ollama")
            
            # 提取指紋（使用較少探針以加快速度）
            logger.info(f"  提取指紋（使用 {len(selected_probes)} 個探針）...")
            fingerprint = extract_fingerprint(
                model,
                selected_probes,
                include_logit=False,  # 暫時關閉 logit 以加快速度
                include_refusal=True
            )
            
            # 保存到數據庫
            db.store_fingerprint(model_name, fingerprint)
            logger.info(f"  ✓ {model_name} 指紋提取完成")
        
        except Exception as e:
            logger.error(f"  ✗ {model_name} 提取失敗: {e}")
            continue
    
    # 顯示摘要
    summary = db.export_database_summary()
    logger.info("\n" + "=" * 80)
    logger.info("提取完成摘要:")
    logger.info(f"  已有指紋: {summary['with_fingerprint']}")
    logger.info(f"  缺少指紋: {summary['without_fingerprint']}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="提取錨點模型指紋")
    parser.add_argument(
        "--models",
        nargs='+',
        help="要提取的模型名稱（默認為所有）"
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=50,
        help="使用的探針數量"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重新提取已有指紋的模型"
    )
    
    args = parser.parse_args()
    
    extract_anchor_fingerprints(
        models=args.models,
        num_probes=args.num_probes,
        force=args.force
    )


if __name__ == "__main__":
    main()
