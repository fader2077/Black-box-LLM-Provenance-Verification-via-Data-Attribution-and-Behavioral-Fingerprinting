"""
歸因分析模組初始化
提供完整的溯源分析功能
"""

from .similarity import SimilarityCalculator
from .anchor_models import AnchorModelsDatabase
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger


def trace_provenance(
    target_fingerprint: Dict,
    anchor_db_path: str = "data/anchor_models",
    threshold_high: float = 0.75,
    threshold_medium: float = 0.50,
    threshold_low: float = 0.25
) -> Dict:
    """
    執行完整的溯源分析
    
    Args:
        target_fingerprint: 待測模型的指紋
        anchor_db_path: 錨點模型數據庫路徑
        threshold_high: 高風險閾值
        threshold_medium: 中風險閾值
        threshold_low: 低風險閾值
    
    Returns:
        溯源分析結果
    """
    logger.info("開始執行溯源分析...")
    
    # 載入錨點模型數據庫
    db = AnchorModelsDatabase(anchor_db_path)
    calc = SimilarityCalculator()
    
    # 結果容器
    similarities = {}
    detailed_results = []
    
    # 與每個錨點模型進行比較
    for model_name in db.anchor_models.keys():
        if not db.anchor_models[model_name].get("has_fingerprint"):
            logger.warning(f"跳過 {model_name}（無指紋數據）")
            continue
        
        # 載入錨點指紋
        anchor_fp = db.load_fingerprint(model_name)
        if not anchor_fp:
            continue
        
        # 計算相似度
        similarity_result = calc.calculate_fingerprint_similarity(
            target_fingerprint,
            anchor_fp
        )
        
        overall_score = similarity_result["overall_similarity"]
        similarities[model_name] = overall_score
        
        # 儲存詳細結果
        detailed_results.append({
            "model_name": model_name,
            "source": db.anchor_models[model_name]["metadata"].get("source"),
            "category": db.anchor_models[model_name]["metadata"].get("category"),
            "overall_similarity": overall_score,
            "logit_similarity": similarity_result.get("logit_similarity", {}),
            "refusal_similarity": similarity_result.get("refusal_similarity", {}),
        })
        
        logger.info(f"  {model_name}: {overall_score:.4f}")
    
    if not similarities:
        logger.error("未能與任何錨點模型進行比較")
        return {
            "error": "No anchor models with fingerprints available",
            "verdict": "無法判定",
        }
    
    # 找出最相似的模型
    most_similar_model = max(similarities.items(), key=lambda x: x[1])
    best_match_name = most_similar_model[0]
    best_match_score = most_similar_model[1]
    
    # 獲取最佳匹配的元數據
    best_match_metadata = db.anchor_models[best_match_name]["metadata"]
    
    # 判定風險等級
    if best_match_score >= threshold_high:
        risk_level = "高風險 (High Risk)"
        verdict = f"{int(best_match_score * 100)}% 行為特徵與 {best_match_name} 一致"
    elif best_match_score >= threshold_medium:
        risk_level = "中風險 (Medium Risk)"
        verdict = f"與 {best_match_name} 有 {int(best_match_score * 100)}% 相似度"
    elif best_match_score >= threshold_low:
        risk_level = "低風險 (Low Risk)"
        verdict = f"與 {best_match_name} 有輕微相似 ({int(best_match_score * 100)}%)"
    else:
        risk_level = "未知 (Unknown)"
        verdict = "無法識別模型來源"
    
    # 按來源聚類統計
    source_scores = {}
    for model_name, score in similarities.items():
        source = db.anchor_models[model_name]["metadata"].get("source", "unknown")
        if source not in source_scores:
            source_scores[source] = []
        source_scores[source].append(score)
    
    # 計算各來源的平均相似度
    source_avg_scores = {
        source: np.mean(scores) 
        for source, scores in source_scores.items()
    }
    
    # 構建報告
    report = {
        "target_model": target_fingerprint.get("model_name", "unknown"),
        "analysis_timestamp": target_fingerprint.get("timestamp"),
        
        # 最佳匹配
        "best_match": {
            "model_name": best_match_name,
            "similarity_score": best_match_score,
            "source": best_match_metadata.get("source"),
            "category": best_match_metadata.get("category"),
            "vendor": best_match_metadata.get("vendor"),
        },
        
        # 風險評估
        "risk_assessment": {
            "risk_level": risk_level,
            "verdict": verdict,
            "confidence": best_match_score,
        },
        
        # 所有相似度分數
        "similarity_scores": similarities,
        
        # 按來源統計
        "source_analysis": source_avg_scores,
        
        # 詳細結果
        "detailed_results": detailed_results,
        
        # 閾值設定
        "thresholds": {
            "high_risk": threshold_high,
            "medium_risk": threshold_medium,
            "low_risk": threshold_low,
        },
    }
    
    # 如果是中國來源的高風險匹配，添加警告
    if best_match_metadata.get("source") == "china" and best_match_score >= threshold_high:
        report["warning"] = (
            "警告：檢測到高度相似的中國大陸來源模型特徵。"
            "建議進一步人工審核，確認是否符合資安合規要求。"
        )
    
    logger.info("✓ 溯源分析完成")
    logger.info(f"  最佳匹配: {best_match_name} ({best_match_score:.2%})")
    logger.info(f"  風險等級: {risk_level}")
    
    return report


def generate_html_report(report: Dict, output_path: str):
    """
    生成 HTML 格式的溯源報告
    
    Args:
        report: 溯源分析結果
        output_path: 輸出文件路徑
    """
    from pathlib import Path
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <title>LLM 溯源分析報告</title>
        <style>
            body {{
                font-family: 'Microsoft JhengHei', Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .risk-high {{ color: #dc3545; font-weight: bold; }}
            .risk-medium {{ color: #ffc107; font-weight: bold; }}
            .risk-low {{ color: #28a745; font-weight: bold; }}
            .score-bar {{
                height: 30px;
                background-color: #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
                margin: 10px 0;
            }}
            .score-fill {{
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                transition: width 0.3s ease;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 黑盒 LLM 溯源分析報告</h1>
            <p>Black-box LLM Provenance Verification</p>
        </div>
        
        <div class="section">
            <h2>📊 風險評估</h2>
            <p><strong>待測模型：</strong>{report['target_model']}</p>
            <p><strong>風險等級：</strong><span class="risk-{report['risk_assessment']['risk_level'].split()[0].lower()}">{report['risk_assessment']['risk_level']}</span></p>
            <p><strong>判定結果：</strong>{report['risk_assessment']['verdict']}</p>
            <p><strong>置信度：</strong>{report['risk_assessment']['confidence']:.2%}</p>
            
            <div class="score-bar">
                <div class="score-fill" style="width: {report['risk_assessment']['confidence'] * 100}%"></div>
            </div>
            
            {f'<p style="color: #dc3545; font-weight: bold;">⚠️ {report.get("warning", "")}</p>' if report.get("warning") else ""}
        </div>
        
        <div class="section">
            <h2>🎯 最佳匹配</h2>
            <table>
                <tr>
                    <th>屬性</th>
                    <th>值</th>
                </tr>
                <tr>
                    <td>模型名稱</td>
                    <td>{report['best_match']['model_name']}</td>
                </tr>
                <tr>
                    <td>相似度</td>
                    <td>{report['best_match']['similarity_score']:.2%}</td>
                </tr>
                <tr>
                    <td>來源</td>
                    <td>{report['best_match']['source']}</td>
                </tr>
                <tr>
                    <td>類別</td>
                    <td>{report['best_match']['category']}</td>
                </tr>
                <tr>
                    <td>廠商</td>
                    <td>{report['best_match'].get('vendor', 'N/A')}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 所有錨點模型相似度</h2>
            <table>
                <tr>
                    <th>模型名稱</th>
                    <th>相似度</th>
                    <th>視覺化</th>
                </tr>
                {''.join([f'''
                <tr>
                    <td>{name}</td>
                    <td>{score:.2%}</td>
                    <td>
                        <div class="score-bar" style="height: 20px;">
                            <div class="score-fill" style="width: {score * 100}%; height: 100%;"></div>
                        </div>
                    </td>
                </tr>
                ''' for name, score in sorted(report['similarity_scores'].items(), key=lambda x: x[1], reverse=True)])}
            </table>
        </div>
        
        <div class="section">
            <h2>🌍 按來源分析</h2>
            <table>
                <tr>
                    <th>來源</th>
                    <th>平均相似度</th>
                </tr>
                {''.join([f'''
                <tr>
                    <td>{source}</td>
                    <td>{score:.2%}</td>
                </tr>
                ''' for source, score in sorted(report['source_analysis'].items(), key=lambda x: x[1], reverse=True)])}
            </table>
        </div>
        
        <footer style="text-align: center; margin-top: 40px; color: #666;">
            <p>生成時間: {report['analysis_timestamp']}</p>
            <p>© 2026 LLM Provenance Verification System</p>
        </footer>
    </body>
    </html>
    """
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    logger.info(f"✓ HTML 報告已生成: {output_path}")


__all__ = [
    "SimilarityCalculator",
    "AnchorModelsDatabase",
    "trace_provenance",
    "generate_html_report",
]
