"""
完整評估流程
對待測模型進行完整的溯源分析
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes import build_all_probes
from src.fingerprint import extract_fingerprint
from src.attribution import trace_provenance, generate_html_report, AnchorModelsDatabase
from src.utils import load_model


def setup_anchor_database(db_path: str = "data/anchor_models"):
    """
    設置錨點模型數據庫
    如果尚未提取錨點模型指紋，則進行提取
    """
    logger.info("檢查錨點模型數據庫...")
    
    db = AnchorModelsDatabase(db_path)
    summary = db.export_database_summary()
    
    logger.info(f"  總錨點數: {summary['total_anchors']}")
    logger.info(f"  已有指紋: {summary['with_fingerprint']}")
    logger.info(f"  缺少指紋: {summary['without_fingerprint']}")
    
    if summary['without_fingerprint'] > 0:
        logger.warning(f"發現 {summary['without_fingerprint']} 個錨點模型缺少指紋")
        logger.info("提示: 運行以下命令提取錨點模型指紋:")
        logger.info("  python experiments/extract_anchor_fingerprints.py")
    
    return db


def run_full_evaluation(
    target_model: str,
    output_file: str = None,
    engine: str = "ollama",
    use_cache: bool = True
):
    """
    執行完整的溯源評估
    
    Args:
        target_model: 待測模型名稱
        output_file: 輸出文件路徑
        engine: 推理引擎
        use_cache: 是否使用緩存的探針數據
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_file is None:
        output_file = f"results/evaluation_{target_model.replace(':', '_')}_{timestamp}.json"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("LLM 溯源技術 - 完整評估")
    logger.info("=" * 80)
    logger.info(f"待測模型: {target_model}")
    logger.info(f"推理引擎: {engine}")
    logger.info(f"時間戳: {timestamp}")
    logger.info("=" * 80)
    
    # Step 1: 載入或構建探針
    logger.info("\n[1/5] 準備探針數據集...")
    
    probes_path = Path("data/probes/all_probes.json")
    
    if use_cache and probes_path.exists():
        logger.info("使用緩存的探針數據")
        with open(probes_path, 'r', encoding='utf-8') as f:
            probes_data = json.load(f)
    else:
        logger.info("構建新的探針數據集")
        probes_data = build_all_probes()
    
    # 合併所有探針
    all_probes = []
    for probe_type, probes in probes_data.items():
        all_probes.extend(probes)
    
    logger.info(f"已載入 {len(all_probes)} 個探針")
    
    # Step 2: 檢查錨點數據庫
    logger.info("\n[2/5] 檢查錨點模型數據庫...")
    db = setup_anchor_database()
    
    # Step 3: 載入待測模型
    logger.info(f"\n[3/5] 載入待測模型: {target_model}...")
    
    try:
        model = load_model(target_model, engine=engine)
        logger.info("✓ 模型載入成功")
    except Exception as e:
        logger.error(f"✗ 模型載入失敗: {e}")
        return None
    
    # Step 4: 提取指紋
    logger.info(f"\n[4/5] 提取模型指紋...")
    logger.info("此過程可能需要較長時間，請耐心等待...")
    
    try:
        fingerprint = extract_fingerprint(
            model,
            all_probes,
            include_logit=True,
            include_refusal=True
        )
        
        # 保存指紋
        fingerprint_path = output_path.parent / f"{target_model.replace(':', '_')}_fingerprint.json"
        with open(fingerprint_path, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f, indent=2)
        
        logger.info(f"✓ 指紋已保存: {fingerprint_path}")
    
    except Exception as e:
        logger.error(f"✗ 指紋提取失敗: {e}")
        return None
    
    # Step 5: 執行溯源分析
    logger.info(f"\n[5/5] 執行溯源分析...")
    
    try:
        report = trace_provenance(
            fingerprint,
            anchor_db_path="data/anchor_models"
        )
        
        # 保存 JSON 報告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ JSON 報告已保存: {output_path}")
        
        # 生成 HTML 報告
        html_path = output_path.with_suffix('.html')
        generate_html_report(report, str(html_path))
        
        logger.info(f"✓ HTML 報告已保存: {html_path}")
    
    except Exception as e:
        logger.error(f"✗ 溯源分析失敗: {e}")
        return None
    
    # 打印摘要
    print_evaluation_summary(report)
    
    return report


def print_evaluation_summary(report: dict):
    """打印評估摘要"""
    logger.info("\n" + "=" * 80)
    logger.info("評估結果摘要")
    logger.info("=" * 80)
    
    # 基本信息
    logger.info(f"\n待測模型: {report['target_model']}")
    logger.info(f"分析時間: {report['analysis_timestamp']}")
    
    # 風險評估
    risk = report['risk_assessment']
    logger.info(f"\n風險等級: {risk['risk_level']}")
    logger.info(f"判定結果: {risk['verdict']}")
    logger.info(f"置信度: {risk['confidence']:.2%}")
    
    # 最佳匹配
    best = report['best_match']
    logger.info(f"\n最佳匹配:")
    logger.info(f"  模型: {best['model_name']}")
    logger.info(f"  相似度: {best['similarity_score']:.2%}")
    logger.info(f"  來源: {best['source']}")
    logger.info(f"  類別: {best['category']}")
    
    # 所有相似度
    logger.info(f"\n所有錨點模型相似度:")
    for model, score in sorted(
        report['similarity_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        logger.info(f"  {model:20s}: {score:.2%}")
    
    # 按來源統計
    logger.info(f"\n按來源平均相似度:")
    for source, score in sorted(
        report['source_analysis'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        logger.info(f"  {source:15s}: {score:.2%}")
    
    # 警告信息
    if 'warning' in report:
        logger.warning(f"\n⚠️  {report['warning']}")
    
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="LLM 溯源技術完整評估")
    parser.add_argument(
        "--target-model",
        required=True,
        help="待測模型名稱"
    )
    parser.add_argument(
        "--output",
        help="輸出文件路徑（默認自動生成）"
    )
    parser.add_argument(
        "--engine",
        default="ollama",
        choices=["ollama", "transformers", "vllm"],
        help="推理引擎"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="不使用緩存的探針數據"
    )
    
    args = parser.parse_args()
    
    run_full_evaluation(
        target_model=args.target_model,
        output_file=args.output,
        engine=args.engine,
        use_cache=not args.no_cache
    )


if __name__ == "__main__":
    main()
