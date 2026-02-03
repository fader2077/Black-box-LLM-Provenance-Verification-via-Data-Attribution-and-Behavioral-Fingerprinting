"""
基于现有锚点的DeepSeek-R1快速评估
使用已提取的锚点指纹进行相似度分析
"""

import sys
import json
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.attribution.similarity import SimilarityCalculator


def quick_analysis():
    """快速分析现有指纹"""
    
    logger.info("=" * 80)
    logger.info("DeepSeek-R1 快速相似度分析")
    logger.info("基于现有锚点指纹")
    logger.info("=" * 80)
    
    # 加载锚点指纹
    anchors = [
        {
            "name": "gpt2",
            "path": "data/anchor_models/gpt2_fingerprint.json",
            "category": "gpt",
            "source": "openai"
        },
        {
            "name": "gpt2-medium",
            "path": "data/anchor_models/gpt2_medium_fingerprint.json",
            "category": "gpt",
            "source": "openai"
        },
        {
            "name": "deepseek-r1:7b",
            "path": "data/anchor_models/deepseek_r1_7b_fingerprint.json",
            "category": "deepseek",
            "source": "china"
        }
    ]
    
    logger.info(f"\n加载 {len(anchors)} 个锚点指纹...")
    
    loaded_anchors = []
    for anchor in anchors:
        path = Path(anchor["path"])
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                fp = json.load(f)
                loaded_anchors.append({
                    **anchor,
                    "fingerprint": fp
                })
                logger.info(f"  ✓ {anchor['name']}")
        else:
            logger.warning(f"  ✗ {anchor['name']} - 文件不存在")
    
    if len(loaded_anchors) < 2:
        logger.error("锚点指纹太少，无法进行比较")
        return
    
    # 检查是否有部分提取的DeepSeek-R1指纹
    deepseek_fps = list(Path("checkpoints").glob("deepseek*checkpoint.json"))
    deepseek_fps.extend(list(Path("results").glob("deepseek*fingerprint.json")))
    
    if not deepseek_fps:
        logger.error("未找到DeepSeek-R1指纹文件")
        logger.info("\n建议: 先运行以下命令提取指纹:")
        logger.info("  python experiments/robust_fingerprint_extraction.py --model deepseek-r1:8b --engine ollama --num-probes 50 --batch-size 5 --device cuda")
        return
    
    logger.info(f"\n找到 {len(deepseek_fps)} 个DeepSeek相关文件")
    for fp in deepseek_fps:
        logger.info(f"  - {fp}")
    
    # 使用最新的文件
    latest_fp = max(deepseek_fps, key=lambda p: p.stat().st_mtime)
    logger.info(f"\n使用最新文件: {latest_fp.name}")
    
    with open(latest_fp, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    
    # 检查是否是检查点文件
    if "partial_results" in target_data:
        logger.info("这是检查点文件，需要转换为完整指纹")
        # 简单转换
        import numpy as np
        partial_results = target_data["partial_results"]
        
        # 提取特征
        feature_vectors = []
        for result in partial_results:
            if 'error' not in result:
                feature_vectors.append(result['features'])
        
        all_features = np.concatenate(feature_vectors)
        
        target_fp = {
            "model_name": "deepseek-r1:8b (部分)",
            "logit_fingerprint": {
                "vector": all_features.tolist(),
                "dimension": len(all_features)
            }
        }
        logger.info(f"  ✓ 转换成功: {len(partial_results)} 个探针")
    else:
        target_fp = target_data
        logger.info(f"  ✓ 完整指纹文件")
    
    # 计算相似度
    logger.info("\n计算相似度...")
    sim_calc = SimilarityCalculator()
    
    results = []
    for anchor in loaded_anchors:
        anchor_fp = anchor["fingerprint"]
        
        sim_result = sim_calc.calculate_fingerprint_similarity(target_fp, anchor_fp)
        score = sim_result["overall_similarity"]
        
        results.append({
            "anchor": anchor["name"],
            "category": anchor["category"],
            "source": anchor["source"],
            "similarity": score
        })
        
        logger.info(f"  vs {anchor['name']:20s} [{anchor['category']:10s}] {score:.4f}")
    
    # 排序并输出结论
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("相似度排名")
    logger.info("=" * 80)
    
    for idx, result in enumerate(results, 1):
        logger.info(f"{idx}. {result['anchor']:20s} [{result['category']:10s}] {result['similarity']:.4f}")
    
    top_match = results[0]
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 最相似: {top_match['anchor']} ({top_match['similarity']:.4f})")
    logger.info(f"   类别: {top_match['category']}")
    logger.info(f"   来源: {top_match['source']}")
    
    # 类别统计
    category_scores = {}
    for result in results:
        cat = result['category']
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(result['similarity'])
    
    category_avg = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}
    
    logger.info("\n📊 类别平均相似度:")
    for cat, avg_score in sorted(category_avg.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"   {cat:15s} {avg_score:.4f}")
    
    logger.info("=" * 80)
    
    # 保存结果
    report = {
        "target_model": target_fp.get("model_name", "deepseek-r1:8b"),
        "source_file": str(latest_fp),
        "anchors_used": [a["name"] for a in loaded_anchors],
        "similarities": results,
        "category_averages": category_avg,
        "top_match": top_match
    }
    
    report_path = Path("results/quick_analysis_result.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ 报告保存到: {report_path}")


if __name__ == "__main__":
    quick_analysis()
