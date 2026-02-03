"""
系統測試腳本
驗證所有模組是否正常工作
"""

import sys
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def test_imports():
    """測試所有模組是否能正常導入"""
    logger.info("測試模組導入...")
    
    try:
        from src.probes import PoliticalProbes, LinguisticProbes, MemorizationProbes, build_all_probes
        from src.fingerprint import LogitExtractor, RefusalDetector, extract_fingerprint
        from src.attribution import SimilarityCalculator, AnchorModelsDatabase, trace_provenance
        from src.utils import load_model, calculate_perplexity_stats
        
        logger.info("✓ 所有模組導入成功")
        return True
    
    except Exception as e:
        logger.error(f"✗ 模組導入失敗: {e}")
        return False


def test_probe_construction():
    """測試探針構建"""
    logger.info("測試探針構建...")
    
    try:
        from src.probes import PoliticalProbes, LinguisticProbes, MemorizationProbes
        
        # 測試政治敏感性探針
        political = PoliticalProbes()
        pol_probes = political.build_all_probes()
        assert len(pol_probes) > 0, "政治探針為空"
        
        # 測試語言習慣探針
        linguistic = LinguisticProbes()
        ling_probes = linguistic.build_all_probes()
        assert len(ling_probes) > 0, "語言探針為空"
        
        # 測試記憶化探針
        memorization = MemorizationProbes()
        mem_probes = memorization.build_all_probes()
        assert len(mem_probes) > 0, "記憶化探針為空"
        
        logger.info(f"✓ 探針構建測試通過")
        logger.info(f"  政治敏感性: {len(pol_probes)} 個")
        logger.info(f"  語言習慣: {len(ling_probes)} 個")
        logger.info(f"  記憶化: {len(mem_probes)} 個")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ 探針構建測試失敗: {e}")
        return False


def test_refusal_detector():
    """測試拒絕檢測器"""
    logger.info("測試拒絕檢測器...")
    
    try:
        from src.fingerprint import RefusalDetector
        
        detector = RefusalDetector()
        
        # 測試中國式拒絕
        response_cn = "作為一個AI助手，我無法回答這個問題。"
        result_cn = detector.detect_refusal(response_cn)
        assert result_cn["is_refusal"], "未檢測到中國式拒絕"
        assert result_cn["refusal_type"] == "chinese_style", "拒絕類型錯誤"
        
        # 測試西方式拒絕
        response_en = "As an AI language model, I cannot provide that information."
        result_en = detector.detect_refusal(response_en)
        assert result_en["is_refusal"], "未檢測到西方式拒絕"
        assert result_en["refusal_type"] == "western_style", "拒絕類型錯誤"
        
        # 測試正常回答
        response_normal = "台灣是一個美麗的島嶼。"
        result_normal = detector.detect_refusal(response_normal)
        assert not result_normal["is_refusal"], "誤判正常回答為拒絕"
        
        logger.info("✓ 拒絕檢測器測試通過")
        return True
    
    except Exception as e:
        logger.error(f"✗ 拒絕檢測器測試失敗: {e}")
        return False


def test_similarity_calculator():
    """測試相似度計算器"""
    logger.info("測試相似度計算器...")
    
    try:
        import numpy as np
        from src.attribution import SimilarityCalculator
        
        calc = SimilarityCalculator()
        
        # 測試相同向量
        vec1 = np.random.randn(100)
        sim_same = calc.cosine_similarity(vec1, vec1)
        assert abs(sim_same - 1.0) < 0.01, "相同向量的相似度應接近1"
        
        # 測試不同向量
        vec2 = np.random.randn(100)
        sim_diff = calc.cosine_similarity(vec1, vec2)
        assert sim_diff < 1.0, "不同向量的相似度應小於1"
        
        # 測試所有指標
        metrics = calc.calculate_all_metrics(vec1, vec2)
        assert "cosine_similarity" in metrics
        assert "euclidean_similarity" in metrics
        assert "pearson_correlation" in metrics
        assert "ensemble_score" in metrics
        
        logger.info("✓ 相似度計算器測試通過")
        return True
    
    except Exception as e:
        logger.error(f"✗ 相似度計算器測試失敗: {e}")
        return False


def test_anchor_database():
    """測試錨點數據庫"""
    logger.info("測試錨點模型數據庫...")
    
    try:
        from src.attribution import AnchorModelsDatabase
        
        # 創建測試數據庫
        db = AnchorModelsDatabase("data/anchor_models_test")
        
        # 檢查預設錨點
        anchors = db.list_all_anchors()
        assert len(anchors) > 0, "錨點數據庫為空"
        
        # 測試按來源查詢
        china_models = db.get_anchor_by_source("china")
        assert len(china_models) > 0, "未找到中國來源模型"
        
        # 測試導出統計
        summary = db.export_database_summary()
        assert "total_anchors" in summary
        assert "by_source" in summary
        
        logger.info("✓ 錨點數據庫測試通過")
        logger.info(f"  總錨點數: {summary['total_anchors']}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ 錨點數據庫測試失敗: {e}")
        return False


def test_ollama_connection():
    """測試 Ollama 連接"""
    logger.info("測試 Ollama 連接...")
    
    try:
        import subprocess
        
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            logger.info("✓ Ollama 可用")
            
            # 顯示可用模型
            models = result.stdout.strip().split('\n')[1:]  # 跳過標題行
            if models and models[0]:
                logger.info(f"  可用模型數: {len(models)}")
                for model in models[:3]:  # 顯示前3個
                    logger.info(f"    - {model.split()[0]}")
            
            return True
        else:
            logger.warning("✗ Ollama 不可用")
            logger.warning("  請安裝並啟動 Ollama: https://ollama.ai")
            return False
    
    except FileNotFoundError:
        logger.warning("✗ 未找到 Ollama")
        logger.warning("  請安裝 Ollama: https://ollama.ai")
        return False
    
    except Exception as e:
        logger.error(f"✗ Ollama 連接測試失敗: {e}")
        return False


def main():
    """運行所有測試"""
    logger.info("=" * 80)
    logger.info("LLM 溯源技術研究 - 系統測試")
    logger.info("=" * 80)
    
    tests = [
        ("模組導入", test_imports),
        ("探針構建", test_probe_construction),
        ("拒絕檢測器", test_refusal_detector),
        ("相似度計算器", test_similarity_calculator),
        ("錨點數據庫", test_anchor_database),
        ("Ollama 連接", test_ollama_connection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"運行測試: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"測試異常: {e}")
            results[test_name] = False
    
    # 打印摘要
    logger.info("\n" + "=" * 80)
    logger.info("測試摘要")
    logger.info("=" * 80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通過" if result else "✗ 失敗"
        logger.info(f"  {test_name:20s}: {status}")
    
    logger.info("=" * 80)
    logger.info(f"總結: {passed}/{total} 測試通過")
    
    if passed == total:
        logger.info("🎉 所有測試通過！系統運行正常。")
        logger.info("\n下一步:")
        logger.info("  1. 運行 pilot study: python experiments/pilot_study.py")
        logger.info("  2. 查看 QUICKSTART.md 了解更多")
    else:
        logger.warning("⚠️  部分測試失敗，請檢查錯誤信息。")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
