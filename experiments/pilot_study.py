"""
Pilot Study: Qwen vs Llama 對比實驗
使用少量探針測試兩個模型的行為差異
"""

import sys
import json
from pathlib import Path
import argparse
from loguru import logger

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes import build_all_probes
from src.fingerprint import extract_fingerprint
from src.attribution import trace_provenance, generate_html_report
from src.utils import load_model


def run_pilot_study(
    model1_name: str = "qwen2.5:7b",
    model2_name: str = "llama3.2:3b",
    num_probes: int = 50,
    output_dir: str = "results/pilot_study"
):
    """
    執行初步對比實驗
    
    Args:
        model1_name: 第一個模型名稱
        model2_name: 第二個模型名稱
        num_probes: 使用的探針數量
        output_dir: 輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Pilot Study: LLM 溯源技術初步驗證")
    logger.info("=" * 80)
    
    # Step 1: 構建探針
    logger.info("\n[1/5] 構建測試探針...")
    probes_data = build_all_probes(output_dir="data/probes")
    
    # 選取部分探針進行測試
    political_probes = probes_data["political_sensitivity"][:num_probes//3]
    linguistic_probes = probes_data["linguistic_shibboleths"][:num_probes//3]
    memorization_probes = probes_data["memorization"][:num_probes//3]
    
    selected_probes = political_probes + linguistic_probes + memorization_probes
    
    logger.info(f"已選取 {len(selected_probes)} 個探針用於測試")
    
    # Step 2: 載入模型
    logger.info(f"\n[2/5] 載入模型...")
    logger.info(f"  模型1: {model1_name}")
    logger.info(f"  模型2: {model2_name}")
    
    try:
        model1 = load_model(model1_name, engine="ollama")
        model2 = load_model(model2_name, engine="ollama")
    except Exception as e:
        logger.error(f"模型載入失敗: {e}")
        logger.error("請確保 Ollama 已安裝並運行，且模型已下載")
        return
    
    # Step 3: 提取指紋
    logger.info(f"\n[3/5] 提取模型指紋...")
    
    logger.info(f"  正在處理 {model1_name}...")
    fp1 = extract_fingerprint(
        model1,
        selected_probes,
        include_logit=True,
        include_refusal=True
    )
    fp1_path = output_path / f"{model1_name.replace(':', '_')}_fingerprint.json"
    with open(fp1_path, 'w', encoding='utf-8') as f:
        json.dump(fp1, f, indent=2)
    logger.info(f"  ✓ 指紋已保存: {fp1_path}")
    
    logger.info(f"  正在處理 {model2_name}...")
    fp2 = extract_fingerprint(
        model2,
        selected_probes,
        include_logit=True,
        include_refusal=True
    )
    fp2_path = output_path / f"{model2_name.replace(':', '_')}_fingerprint.json"
    with open(fp2_path, 'w', encoding='utf-8') as f:
        json.dump(fp2, f, indent=2)
    logger.info(f"  ✓ 指紋已保存: {fp2_path}")
    
    # Step 4: 計算相似度
    logger.info(f"\n[4/5] 計算指紋相似度...")
    from src.attribution.similarity import SimilarityCalculator
    
    calc = SimilarityCalculator()
    
    # 提取向量
    vec1 = fp1["logit_fingerprint"]["vector"]
    vec2 = fp2["logit_fingerprint"]["vector"]
    
    similarity_metrics = calc.calculate_all_metrics(vec1, vec2)
    
    logger.info(f"  Cosine Similarity: {similarity_metrics['cosine_similarity']:.4f}")
    logger.info(f"  Euclidean Similarity: {similarity_metrics['euclidean_similarity']:.4f}")
    logger.info(f"  Pearson Correlation: {similarity_metrics['pearson_correlation']:.4f}")
    logger.info(f"  KL Similarity: {similarity_metrics['kl_similarity']:.4f}")
    logger.info(f"  Ensemble Score: {similarity_metrics['ensemble_score']:.4f}")
    
    # 比較拒絕模式
    if fp1.get("refusal_fingerprint") and fp2.get("refusal_fingerprint"):
        refusal_sim = calc.compare_refusal_patterns(
            fp1["refusal_fingerprint"],
            fp2["refusal_fingerprint"]
        )
        logger.info(f"\n  拒絕模式相似度:")
        logger.info(f"    拒絕率相似度: {refusal_sim['refusal_rate_similarity']:.4f}")
        logger.info(f"    風格分佈相似度: {refusal_sim['style_distribution_similarity']:.4f}")
        logger.info(f"    模式重疊度: {refusal_sim['pattern_overlap']:.4f}")
    
    # Step 5: 生成報告
    logger.info(f"\n[5/5] 生成對比報告...")
    
    report = {
        "experiment": "pilot_study",
        "models": {
            "model1": {
                "name": model1_name,
                "fingerprint_path": str(fp1_path),
            },
            "model2": {
                "name": model2_name,
                "fingerprint_path": str(fp2_path),
            }
        },
        "num_probes": len(selected_probes),
        "similarity_metrics": similarity_metrics,
        "conclusion": generate_conclusion(similarity_metrics, model1_name, model2_name),
    }
    
    report_path = output_path / "pilot_study_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 報告已保存: {report_path}")
    
    # 打印結論
    logger.info("\n" + "=" * 80)
    logger.info("實驗結論:")
    logger.info("=" * 80)
    logger.info(report["conclusion"])
    logger.info("=" * 80)
    
    return report


def generate_conclusion(metrics: dict, model1: str, model2: str) -> str:
    """根據指標生成結論"""
    ensemble_score = metrics["ensemble_score"]
    
    if ensemble_score > 0.7:
        conclusion = (
            f"{model1} 和 {model2} 顯示出高度相似性（{ensemble_score:.2%}）。\n"
            f"這可能表示它們共享相同的訓練數據或架構基礎。"
        )
    elif ensemble_score > 0.4:
        conclusion = (
            f"{model1} 和 {model2} 顯示出中等程度的相似性（{ensemble_score:.2%}）。\n"
            f"它們可能有部分共同的訓練數據或相似的訓練方法。"
        )
    else:
        conclusion = (
            f"{model1} 和 {model2} 顯示出較低的相似性（{ensemble_score:.2%}）。\n"
            f"這表示它們具有不同的訓練數據來源和特徵。\n"
            f"這驗證了我們的指紋技術能夠有效區分不同來源的模型。"
        )
    
    return conclusion


def main():
    parser = argparse.ArgumentParser(description="LLM 溯源技術 Pilot Study")
    parser.add_argument(
        "--models",
        nargs=2,
        default=["qwen2.5:7b", "llama3.2:3b"],
        help="要比較的兩個模型名稱"
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=50,
        help="使用的探針數量"
    )
    parser.add_argument(
        "--output",
        default="results/pilot_study",
        help="輸出目錄"
    )
    
    args = parser.parse_args()
    
    run_pilot_study(
        model1_name=args.models[0],
        model2_name=args.models[1],
        num_probes=args.num_probes,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
