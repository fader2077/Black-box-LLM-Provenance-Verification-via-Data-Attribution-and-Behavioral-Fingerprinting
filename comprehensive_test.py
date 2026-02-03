"""
全面測試腳本 - Transformers 引擎
測試所有核心功能模組
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
import json

# 配置日誌
logger.add("test_results.log", rotation="10 MB")

def test_1_module_imports():
    """測試 1: 核心模組導入"""
    logger.info("=" * 80)
    logger.info("測試 1: 核心模組導入")
    logger.info("=" * 80)
    
    try:
        from src.utils.unified_loader import load_model
        from src.utils.model_loader_transformers import TransformersModelLoader
        from src.probes import build_all_probes
        from src.fingerprint import extract_fingerprint
        from src.fingerprint.logit_extractor import LogitExtractor
        from src.attribution import trace_provenance
        from src.attribution.anchor_models import AnchorModelsDatabase
        
        logger.success("✅ 所有核心模組成功導入")
        return True
    except Exception as e:
        logger.error(f"❌ 模組導入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_model_loading():
    """測試 2: Transformers 模型載入"""
    logger.info("\n" + "=" * 80)
    logger.info("測試 2: Transformers 模型載入")
    logger.info("=" * 80)
    
    try:
        from src.utils.unified_loader import load_model
        
        logger.info("載入 GPT-2 模型...")
        model = load_model("gpt2", engine="transformers")
        
        # 測試基本生成
        logger.info("測試基本文本生成...")
        result = model.generate("Hello, my name is", max_tokens=5)
        logger.info(f"生成結果: {result}")
        
        # 測試 logprobs 生成
        logger.info("測試 logprobs 提取...")
        result_with_logprobs = model.generate_with_logprobs(
            "The capital of France is",
            max_tokens=3,
            top_k_logprobs=3
        )
        
        if 'logprobs' in result_with_logprobs and result_with_logprobs['logprobs']:
            logger.success(f"✅ Logprobs 提取成功，數量: {len(result_with_logprobs['logprobs'])}")
            return True
        else:
            logger.error("❌ Logprobs 提取失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ 模型載入測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_probe_system():
    """測試 3: 探針系統"""
    logger.info("\n" + "=" * 80)
    logger.info("測試 3: 探針系統")
    logger.info("=" * 80)
    
    try:
        from src.probes import build_all_probes
        
        logger.info("構建探針集...")
        probes = build_all_probes()
        
        total_probes = sum(len(p) for p in probes.values())
        logger.info(f"探針類型數: {len(probes)}")
        logger.info(f"總探針數: {total_probes}")
        
        for probe_type, probe_list in probes.items():
            logger.info(f"  {probe_type}: {len(probe_list)} 個")
        
        if total_probes > 0:
            logger.success("✅ 探針系統正常")
            return True
        else:
            logger.error("❌ 探針數量為 0")
            return False
            
    except Exception as e:
        logger.error(f"❌ 探針系統測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_fingerprint_extraction():
    """測試 4: 指紋提取"""
    logger.info("\n" + "=" * 80)
    logger.info("測試 4: 指紋提取（使用少量探針）")
    logger.info("=" * 80)
    
    try:
        from src.utils.unified_loader import load_model
        from src.fingerprint import extract_fingerprint
        import json
        
        # 載入少量探針用於測試
        logger.info("載入測試探針...")
        with open("data/probes/all_probes.json", encoding='utf-8') as f:
            all_probes = json.load(f)
        
        # 只用前 20 個探針測試
        flat_probes = []
        for probe_type, probes in all_probes.items():
            flat_probes.extend(probes)
        test_probes = flat_probes[:20]
        logger.info(f"使用 {len(test_probes)} 個探針進行測試")
        
        # 載入模型
        logger.info("載入 GPT-2 模型...")
        model = load_model("gpt2", engine="transformers")
        
        # 提取指紋
        logger.info("提取指紋...")
        fingerprint = extract_fingerprint(
            model,
            test_probes,
            include_logit=True,
            include_refusal=True
        )
        
        logger.info(f"指紋結構: {fingerprint.keys()}")
        
        if 'logit_fingerprint' in fingerprint and fingerprint['logit_fingerprint']:
            fp_dim = fingerprint['logit_fingerprint']['dimension']
            logger.info(f"Logit 分佈維度: {fp_dim}")
            logger.success("✅ 指紋提取成功")
            return True
        else:
            logger.error("❌ 指紋格式不正確")
            return False
            
    except Exception as e:
        logger.error(f"❌ 指紋提取測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_provenance_tracing():
    """測試 5: 溯源分析"""
    logger.info("\n" + "=" * 80)
    logger.info("測試 5: 溯源分析")
    logger.info("=" * 80)
    
    try:
        from src.attribution import trace_provenance
        from src.attribution.anchor_models import AnchorModelsDatabase
        import numpy as np
        
        # 檢查錨點數據庫
        logger.info("檢查錨點數據庫...")
        db = AnchorModelsDatabase()
        anchors = db.list_all_anchors()
        logger.info(f"可用錨點數: {len(anchors)}")
        
        # 創建測試指紋
        logger.info("創建測試指紋...")
        test_fingerprint = {
            'logit_distribution': np.random.randn(100),
            'refusal_patterns': {},
            'metadata': {
                'model_name': 'test-model',
                'num_probes': 20
            }
        }
        
        # 執行溯源
        logger.info("執行溯源分析...")
        result = trace_provenance(test_fingerprint)
        
        logger.info(f"風險等級: {result['risk_assessment']['risk_level']}")
        logger.info(f"相似度分數數量: {len(result['similarity_scores'])}")
        
        if result and 'risk_assessment' in result:
            logger.success("✅ 溯源分析成功")
            return True
        else:
            logger.error("❌ 溯源分析失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ 溯源分析測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_report_generation():
    """測試 6: 報告生成"""
    logger.info("\n" + "=" * 80)
    logger.info("測試 6: 報告生成")
    logger.info("=" * 80)
    
    try:
        from src.attribution import generate_html_report
        from datetime import datetime
        
        # 創建測試報告數據
        test_report = {
            'target_model': 'test-gpt2',
            'analysis_timestamp': datetime.now().isoformat(),
            'risk_assessment': {
                'risk_level': '低風險 (Low Risk)',
                'verdict': '模型可能來自已知的開源項目',
                'confidence': 0.75
            },
            'best_match': {
                'model_name': 'qwen2.5:7b',
                'similarity_score': 0.05,
                'source': 'china',
                'category': 'qwen'
            },
            'similarity_scores': {
                'qwen2.5:7b': 0.05,
                'llama3.2:3b': 0.03
            },
            'source_analysis': {
                'china': 0.05,
                'meta': 0.03
            }
        }
        
        # 生成 HTML 報告
        logger.info("生成 HTML 報告...")
        output_path = "test_report.html"
        generate_html_report(test_report, output_path)
        
        if Path(output_path).exists():
            logger.success(f"✅ HTML 報告生成成功: {output_path}")
            return True
        else:
            logger.error("❌ HTML 報告生成失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ 報告生成測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_end_to_end():
    """測試 7: 端到端完整流程"""
    logger.info("\n" + "=" * 80)
    logger.info("測試 7: 端到端完整流程")
    logger.info("=" * 80)
    
    try:
        from src.utils.unified_loader import load_model
        from src.fingerprint import extract_fingerprint
        from src.attribution import trace_provenance, generate_html_report
        import json
        
        # 載入探針
        logger.info("載入探針...")
        with open("data/probes/all_probes.json", encoding='utf-8') as f:
            all_probes = json.load(f)
        
        flat_probes = []
        for probe_type, probes in all_probes.items():
            flat_probes.extend(probes)
        test_probes = flat_probes[:30]  # 使用 30 個探針
        logger.info(f"使用 {len(test_probes)} 個探針")
        
        # 載入模型
        logger.info("載入模型...")
        model = load_model("gpt2", engine="transformers")
        
        # 提取指紋
        logger.info("提取指紋...")
        fingerprint = extract_fingerprint(model, test_probes)
        
        # 溯源分析
        logger.info("溯源分析...")
        result = trace_provenance(fingerprint)
        
        # 生成報告
        logger.info("生成報告...")
        generate_html_report(result, "e2e_test_report.html")
        
        # 保存 JSON
        with open("e2e_test_report.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.success("✅ 端到端測試成功")
        logger.info(f"  風險等級: {result['risk_assessment']['risk_level']}")
        logger.info(f"  最佳匹配: {result['best_match']['model_name']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 端到端測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """執行所有測試"""
    logger.info("=" * 80)
    logger.info("LLM 溯源技術 - Transformers 引擎全面測試")
    logger.info("=" * 80)
    
    tests = [
        ("模組導入", test_1_module_imports),
        ("模型載入", test_2_model_loading),
        ("探針系統", test_3_probe_system),
        ("指紋提取", test_4_fingerprint_extraction),
        ("溯源分析", test_5_provenance_tracing),
        ("報告生成", test_6_report_generation),
        ("端到端流程", test_7_end_to_end),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"測試 {test_name} 執行異常: {e}")
            results[test_name] = False
    
    # 總結
    logger.info("\n" + "=" * 80)
    logger.info("測試總結")
    logger.info("=" * 80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        logger.info(f"{test_name:20s}: {status}")
    
    logger.info(f"\n總計: {passed}/{total} 個測試通過")
    
    if passed == total:
        logger.success("\n🎉 所有測試通過！系統運行正常。")
        return 0
    else:
        logger.error(f"\n⚠️ {total - passed} 個測試失敗，需要修復。")
        return 1


if __name__ == "__main__":
    exit(main())
