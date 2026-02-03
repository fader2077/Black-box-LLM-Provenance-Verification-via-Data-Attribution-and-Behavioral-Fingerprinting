"""
全面测试流程 - DeepSeek-R1 谱系判定
使用GPU，带完整错误处理
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.unified_loader import load_model
from src.attribution.anchor_models import AnchorModelsDatabase
from src.attribution.similarity import SimilarityCalculator
from experiments.robust_fingerprint_extraction import RobustFingerprintExtractor


def comprehensive_test():
    """执行全面测试流程"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info("DeepSeek-R1 谱系判定 - 全面测试")
    logger.info("=" * 80)
    logger.info(f"时间戳: {timestamp}")
    logger.info(f"使用设备: GPU (CUDA)")
    logger.info("=" * 80)
    
    # 测试配置
    test_models = [
        {
            "name": "deepseek-r1:8b",
            "engine": "ollama",
            "description": "DeepSeek-R1 8B基础版本"
        },
        {
            "name": "deepseek-r1:8b-llama-distill-q4_K_M",
            "engine": "ollama",
            "description": "DeepSeek-R1-Distill-Llama-8B (量化版本)"
        }
    ]
    
    num_probes = 100  # 使用100个探针进行快速但可靠的测试
    batch_size = 5
    
    # 加载探针
    logger.info("\n[步骤 1/4] 加载探针...")
    probes_path = Path("data/probes/all_probes.json")
    
    if not probes_path.exists():
        logger.error("探针文件不存在")
        return
    
    with open(probes_path, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
    
    all_probes = []
    for probe_type, probes in probes_data.items():
        all_probes.extend(probes)
    
    selected_probes = all_probes[:num_probes]
    logger.info(f"✓ 已加载 {len(selected_probes)} 个探针")
    
    # 加载锚点数据库
    logger.info("\n[步骤 2/4] 加载锚点数据库...")
    db = AnchorModelsDatabase()
    
    # 验证锚点
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
                logger.info(f"  ✓ {model_name} ({data['metadata'].get('category')})")
    
    logger.info(f"总计: {len(anchors_with_fp)} 个锚点模型可用")
    
    if len(anchors_with_fp) < 3:
        logger.warning("警告: 锚点模型数量较少，可能影响判定准确性")
    
    # 检查是否有llama锚点
    has_llama = any(a['category'] == 'llama' for a in anchors_with_fp)
    has_deepseek = any(a['category'] == 'deepseek' for a in anchors_with_fp)
    
    if not has_llama:
        logger.warning("⚠️ 警告: 没有Llama锚点，无法判定与Llama的相似度")
    if not has_deepseek:
        logger.warning("⚠️ 警告: 没有DeepSeek锚点，无法判定与DeepSeek的相似度")
    
    # 相似度计算器
    sim_calc = SimilarityCalculator()
    
    # 测试结果存储
    all_results = []
    
    # 测试每个模型
    logger.info(f"\n[步骤 3/4] 测试 {len(test_models)} 个模型...")
    
    for idx, test_model_config in enumerate(test_models, 1):
        model_name = test_model_config["name"]
        engine = test_model_config["engine"]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"测试 {idx}/{len(test_models)}: {model_name}")
        logger.info(f"描述: {test_model_config['description']}")
        logger.info(f"{'='*80}")
        
        try:
            # 加载模型
            logger.info(f"\n  [3.{idx}.1] 加载模型...")
            model = load_model(
                model_name=model_name,
                engine=engine,
                device="cuda"
            )
            logger.info(f"  ✓ 模型加载成功")
            
            # 提取指纹
            logger.info(f"\n  [3.{idx}.2] 提取指纹...")
            logger.info(f"  探针数量: {num_probes}")
            logger.info(f"  批处理大小: {batch_size}")
            logger.info(f"  预计时间: {num_probes * 3.5 / 60:.1f} 分钟")
            
            extractor = RobustFingerprintExtractor(
                model=model,
                batch_size=batch_size,
                max_retries=3
            )
            
            fingerprint = extractor.extract_with_retry(
                probes=selected_probes,
                model_id=model_name,
                resume_from_checkpoint=True
            )
            
            if not fingerprint:
                logger.error(f"  ✗ 指纹提取失败")
                continue
            
            logger.info(f"  ✓ 指纹提取成功")
            logger.info(f"  特征维度: {fingerprint['logit_fingerprint']['dimension']}")
            
            # 保存指纹
            fp_path = Path(f"results/{model_name.replace(':', '_')}_fingerprint.json")
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fp_path, 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, indent=2, ensure_ascii=False)
            logger.info(f"  保存到: {fp_path}")
            
            # 计算相似度
            logger.info(f"\n  [3.{idx}.3] 计算与锚点的相似度...")
            
            similarities = []
            for anchor in anchors_with_fp:
                anchor_id = anchor["model_id"]
                anchor_fp = anchor["fingerprint"]
                
                sim_result = sim_calc.calculate_fingerprint_similarity(fingerprint, anchor_fp)
                score = sim_result["overall_similarity"]
                
                similarities.append({
                    "anchor": anchor_id,
                    "category": anchor["category"],
                    "source": anchor["source"],
                    "similarity": score
                })
                
                logger.info(f"    vs {anchor_id:30s} [{anchor['category']:10s}] {score:.4f}")
            
            # 排序相似度
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 判定
            top_match = similarities[0]
            logger.info(f"\n  🎯 最相似: {top_match['anchor']} ({top_match['similarity']:.4f})")
            logger.info(f"  📊 类别: {top_match['category']}")
            logger.info(f"  📍 来源: {top_match['source']}")
            
            # 分类统计
            category_scores = {}
            for sim in similarities:
                cat = sim['category']
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(sim['similarity'])
            
            category_avg = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}
            
            logger.info(f"\n  📈 类别平均相似度:")
            for cat, avg_score in sorted(category_avg.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {cat:15s} {avg_score:.4f}")
            
            # 判定结论
            if 'llama' in category_avg and 'deepseek' in category_avg:
                llama_score = category_avg['llama']
                deepseek_score = category_avg['deepseek']
                diff = abs(llama_score - deepseek_score)
                
                if llama_score > deepseek_score:
                    verdict = f"更接近 Llama 家族 (差异: {diff:.4f})"
                else:
                    verdict = f"更接近 DeepSeek 家族 (差异: {diff:.4f})"
                
                logger.info(f"\n  ⚖️  判定: {verdict}")
            else:
                verdict = "无法判定 (缺少Llama或DeepSeek锚点)"
                logger.info(f"\n  ⚠️  {verdict}")
            
            # 保存结果
            result = {
                "model": model_name,
                "description": test_model_config["description"],
                "timestamp": timestamp,
                "num_probes": num_probes,
                "similarities": similarities,
                "category_averages": category_avg,
                "top_match": top_match,
                "verdict": verdict
            }
            
            all_results.append(result)
            
        except KeyboardInterrupt:
            logger.warning(f"检测到中断，保存已有结果...")
            break
        except Exception as e:
            logger.error(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成最终报告
    logger.info(f"\n[步骤 4/4] 生成测试报告...")
    
    report_path = Path(f"results/comprehensive_test_report_{timestamp}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": timestamp,
        "test_config": {
            "num_probes": num_probes,
            "batch_size": batch_size,
            "device": "cuda",
            "anchors": [a["model_id"] for a in anchors_with_fp]
        },
        "results": all_results,
        "summary": {
            "total_models_tested": len(all_results),
            "successful_tests": len(all_results)
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 报告保存到: {report_path}")
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    
    for result in all_results:
        logger.info(f"\n{result['model']}:")
        logger.info(f"  最相似: {result['top_match']['anchor']} ({result['top_match']['similarity']:.4f})")
        logger.info(f"  判定: {result['verdict']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 全面测试完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    comprehensive_test()
