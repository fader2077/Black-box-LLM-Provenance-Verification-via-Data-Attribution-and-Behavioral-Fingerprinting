"""
簡化的端到端測試
測試核心功能鏈路
"""

import sys
from pathlib import Path
import json
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.probes import build_all_probes
from src.fingerprint import RefusalDetector
from src.attribution import SimilarityCalculator, AnchorModelsDatabase
from src.utils import load_model


def test_model_loading():
    """測試模型加載"""
    logger.info("=" * 60)
    logger.info("測試 1: 模型加載")
    logger.info("=" * 60)
    
    try:
        # 使用可用的模型進行測試
        model = load_model("llama3.2:latest", engine="ollama")
        
        # 測試生成
        response = model.generate("Hello, how are you?", max_new_tokens=20)
        logger.info(f"模型回應: {response[:100]}")
        
        logger.info("✓ 模型加載和生成測試通過")
        return True
    
    except Exception as e:
        logger.error(f"✗ 模型加載測試失敗: {e}")
        return False


def test_refusal_detection():
    """測試拒絕檢測功能"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 2: 拒絕檢測")
    logger.info("=" * 60)
    
    try:
        detector = RefusalDetector()
        
        # 載入模型
        model = load_model("llama3.2:latest", engine="ollama")
        
        # 測試幾個政治敏感問題
        test_prompts = [
            "請問台灣是一個國家嗎？",
            "What happened at Tiananmen Square in 1989?",
            "What is 2+2?",  # 正常問題
        ]
        
        results = []
        for prompt in test_prompts:
            logger.info(f"\n測試提示: {prompt}")
            response = model.generate(prompt, max_new_tokens=100)
            logger.info(f"回應: {response[:150]}...")
            
            detection = detector.detect_refusal(response)
            logger.info(f"是否拒絕: {detection['is_refusal']}")
            if detection['is_refusal']:
                logger.info(f"拒絕類型: {detection['refusal_type']}")
            
            results.append(detection)
        
        logger.info("\n✓ 拒絕檢測測試完成")
        return True
    
    except Exception as e:
        logger.error(f"✗ 拒絕檢測測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_probe_system():
    """測試探針系統"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 3: 探針系統")
    logger.info("=" * 60)
    
    try:
        # 檢查探針文件是否存在
        probes_path = Path("data/probes/all_probes.json")
        
        if probes_path.exists():
            with open(probes_path, 'r', encoding='utf-8') as f:
                probes_data = json.load(f)
            
            total = sum(len(probes) for probes in probes_data.values())
            logger.info(f"已載入探針: {total} 個")
            
            for probe_type, probes in probes_data.items():
                logger.info(f"  {probe_type}: {len(probes)} 個")
        else:
            logger.info("探針文件不存在，將構建...")
            probes_data = build_all_probes()
        
        logger.info("✓ 探針系統測試通過")
        return True
    
    except Exception as e:
        logger.error(f"✗ 探針系統測試失敗: {e}")
        return False


def test_similarity_calculation():
    """測試相似度計算"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 4: 相似度計算")
    logger.info("=" * 60)
    
    try:
        import numpy as np
        
        calc = SimilarityCalculator()
        
        # 創建測試向量
        vec1 = np.random.randn(100)
        vec2 = vec1 + np.random.randn(100) * 0.1  # 相似
        vec3 = np.random.randn(100)  # 不相似
        
        # 測試相似向量
        sim_similar = calc.calculate_all_metrics(vec1, vec2)
        logger.info(f"相似向量 ensemble_score: {sim_similar['ensemble_score']:.4f}")
        
        # 測試不相似向量
        sim_different = calc.calculate_all_metrics(vec1, vec3)
        logger.info(f"不相似向量 ensemble_score: {sim_different['ensemble_score']:.4f}")
        
        # 驗證邏輯
        assert sim_similar['ensemble_score'] > sim_different['ensemble_score'], \
            "相似向量的分數應該高於不相似向量"
        
        logger.info("✓ 相似度計算測試通過")
        return True
    
    except Exception as e:
        logger.error(f"✗ 相似度計算測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anchor_database():
    """測試錨點數據庫"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 5: 錨點數據庫")
    logger.info("=" * 60)
    
    try:
        db = AnchorModelsDatabase()
        
        # 檢查數據庫
        summary = db.export_database_summary()
        logger.info(f"總錨點數: {summary['total_anchors']}")
        logger.info(f"已有指紋: {summary['with_fingerprint']}")
        logger.info(f"缺少指紋: {summary['without_fingerprint']}")
        
        # 驗證完整性
        integrity = db.verify_database_integrity()
        if integrity['is_valid']:
            logger.info("數據庫完整性: ✓")
        else:
            logger.warning(f"數據庫有 {len(integrity['issues'])} 個問題")
        
        logger.info("✓ 錨點數據庫測試通過")
        return True
    
    except Exception as e:
        logger.error(f"✗ 錨點數據庫測試失敗: {e}")
        return False


def main():
    """運行所有端到端測試"""
    logger.info("=" * 80)
    logger.info("LLM 溯源技術 - 端到端功能測試")
    logger.info("=" * 80)
    
    tests = [
        ("探針系統", test_probe_system),
        ("相似度計算", test_similarity_calculation),
        ("錨點數據庫", test_anchor_database),
        ("模型加載", test_model_loading),
        ("拒絕檢測", test_refusal_detection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"測試 {test_name} 異常: {e}")
            results[test_name] = False
    
    # 打印摘要
    logger.info("\n" + "=" * 80)
    logger.info("測試摘要")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = "✓ 通過" if result else "✗ 失敗"
        logger.info(f"{test_name:20s}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    logger.info("=" * 80)
    logger.info(f"總結: {passed}/{total} 測試通過")
    
    if passed == total:
        logger.info("🎉 所有端到端測試通過！")
    else:
        logger.warning("⚠️  部分測試失敗")
    
    logger.info("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
