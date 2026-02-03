"""
快速测试 DeepSeek-R1 模型与锚点的相似度
测试目标：确定 DeepSeek-R1 更接近 Llama 还是 DeepSeek 家族
"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.utils.unified_loader import load_model
from src.fingerprint import extract_fingerprint
from src.attribution.anchor_models import AnchorModelsDatabase
from src.attribution.similarity import SimilarityCalculator
from src.probes import build_all_probes

def test_deepseek_similarity():
    """测试 DeepSeek-R1 相似度"""
    
    logger.info("=" * 80)
    logger.info("DeepSeek-R1 相似度测试")
    logger.info("=" * 80)
    
    # 1. 加载探针（使用较小子集以加快测试）
    logger.info("\n[1/4] 加载探针...")
    
    probes_path = Path("data/probes/all_probes.json")
    if probes_path.exists():
        with open(probes_path, 'r', encoding='utf-8') as f:
            probes_data = json.load(f)
        all_probes = []
        for probe_type, probes in probes_data.items():
            all_probes.extend(probes)
    else:
        logger.error("探针文件不存在")
        return
    
    # 使用前50个探针进行快速测试
    probes = all_probes[:50]
    logger.info(f"✓ 已加载 {len(probes)} 个探针（快速测试子集）")
    
    # 2. 加载锚点数据库
    logger.info("\n[2/4] 加载锚点数据库...")
    anchor_db = AnchorModelsDatabase()
    
    # 3. 初始化相似度计算器
    sim_calc = SimilarityCalculator()
    logger.info(f"✓ 已加载 {len(anchor_db.anchors)} 个锚点模型:")
    for anchor in anchor_db.anchors:
        logger.info(f"  - {anchor['model_id']} ({anchor['engine']})")
    
    # 3. 测试目标模型
    target_models = [
        "deepseek-r1:7b",
        "deepseek-r1:8b",
    ]
    
    results = {}
    
    for target_model in target_models:
        logger.info(f"\n[3/4] 测试模型: {target_model}")
        
        try:
            # 加载模型
            logger.info(f"  加载模型...")
            model = load_model(
                model_name=target_model,
                engine="ollama",
                device="cuda"
            )
            logger.info(f"  ✓ 模型加载成功")
            
            # 提取指纹
            logger.info(f"  提取指纹...")
            fingerprint = extract_fingerprint(model, probes, model_id=target_model)
            
            if not fingerprint:
                logger.error(f"  ✗ 指纹提取失败")
                continue
                
            logger.info(f"  ✓ 指纹提取成功")
            
            # 4. 计算与各锚点的相似度
            logger.info(f"\n[4/4] 计算相似度...")
            similarities = {}
            
            for anchor in anchor_db.anchors:
                anchor_id = anchor['model_id']
                anchor_fp = anchor['fingerprint']
                
                # 计算相似度
                sim_result = sim_calc.calculate_fingerprint_similarity(fingerprint, anchor_fp)
                score = sim_result['overall_similarity']
                similarities[anchor_id] = score
                logger.info(f"  vs {anchor_id}: {score:.4f}")
            
            results[target_model] = similarities
            
        except Exception as e:
            logger.error(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. 输出总结
    logger.info("\n" + "=" * 80)
    logger.info("测试结果总结")
    logger.info("=" * 80)
    
    for model, sims in results.items():
        logger.info(f"\n{model}:")
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        for anchor_id, score in sorted_sims:
            logger.info(f"  {anchor_id:30s} {score:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ 测试完成")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_deepseek_similarity()
