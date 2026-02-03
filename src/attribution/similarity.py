"""
相似度計算模組
提供多種相似度度量方法
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from loguru import logger


class SimilarityCalculator:
    """
    計算指紋之間的相似度
    支持多種相似度度量
    """
    
    def __init__(self):
        self.metrics = [
            "cosine",
            "euclidean",
            "pearson",
            "kl_divergence",
        ]
    
    def cosine_similarity(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> float:
        """
        計算餘弦相似度
        值域 [-1, 1]，越接近 1 表示越相似
        
        Args:
            vec1, vec2: 特徵向量
        
        Returns:
            相似度分數
        """
        if len(vec1) != len(vec2):
            logger.warning(f"向量維度不匹配: {len(vec1)} vs {len(vec2)}")
            # 補齊較短的向量
            max_len = max(len(vec1), len(vec2))
            vec1 = np.pad(vec1, (0, max_len - len(vec1)), mode='constant')
            vec2 = np.pad(vec2, (0, max_len - len(vec2)), mode='constant')
        
        # 使用 1 - cosine_distance 得到相似度
        similarity = 1 - cosine(vec1, vec2)
        
        # 處理 NaN
        if np.isnan(similarity):
            return 0.0
        
        return float(similarity)
    
    def euclidean_distance(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> float:
        """
        計算歐幾里得距離
        距離越小表示越相似
        返回歸一化的相似度分數 [0, 1]
        """
        if len(vec1) != len(vec2):
            max_len = max(len(vec1), len(vec2))
            vec1 = np.pad(vec1, (0, max_len - len(vec1)), mode='constant')
            vec2 = np.pad(vec2, (0, max_len - len(vec2)), mode='constant')
        
        dist = euclidean(vec1, vec2)
        
        # 轉換為相似度（使用 RBF kernel 概念）
        # similarity = exp(-distance / sigma)
        sigma = np.std(vec1) + np.std(vec2) + 1e-10
        similarity = np.exp(-dist / sigma)
        
        return float(similarity)
    
    def pearson_correlation(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> float:
        """
        計算 Pearson 相關係數
        值域 [-1, 1]，越接近 1 表示正相關越強
        """
        if len(vec1) != len(vec2):
            max_len = max(len(vec1), len(vec2))
            vec1 = np.pad(vec1, (0, max_len - len(vec1)), mode='constant')
            vec2 = np.pad(vec2, (0, max_len - len(vec2)), mode='constant')
        
        if len(vec1) < 2:
            return 0.0
        
        try:
            corr, _ = pearsonr(vec1, vec2)
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def kl_divergence(
        self, 
        p: np.ndarray, 
        q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        計算 KL 散度 (Kullback-Leibler Divergence)
        用於比較兩個機率分佈
        值越小表示分佈越相似
        
        注意：KL 散度不對稱，這裡計算對稱版本 (Jensen-Shannon Divergence)
        """
        if len(p) != len(q):
            max_len = max(len(p), len(q))
            p = np.pad(p, (0, max_len - len(p)), mode='constant')
            q = np.pad(q, (0, max_len - len(q)), mode='constant')
        
        # 確保是機率分佈（歸一化且非負）
        p = np.abs(p) + epsilon
        q = np.abs(q) + epsilon
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # 計算對稱 KL 散度（Jensen-Shannon Divergence）
        m = 0.5 * (p + q)
        
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        
        js_div = 0.5 * (kl_pm + kl_qm)
        
        # 轉換為相似度分數 [0, 1]
        # JS divergence 範圍是 [0, ln(2)]，所以除以 ln(2) 歸一化
        similarity = 1 - (js_div / np.log(2))
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def calculate_all_metrics(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> Dict[str, float]:
        """
        計算所有相似度指標
        
        Returns:
            包含所有指標的字典
        """
        results = {
            "cosine_similarity": self.cosine_similarity(vec1, vec2),
            "euclidean_similarity": self.euclidean_distance(vec1, vec2),
            "pearson_correlation": self.pearson_correlation(vec1, vec2),
            "kl_similarity": self.kl_divergence(vec1, vec2),
        }
        
        # 計算綜合分數（加權平均）
        weights = {
            "cosine_similarity": 0.4,
            "euclidean_similarity": 0.2,
            "pearson_correlation": 0.2,
            "kl_similarity": 0.2,
        }
        
        results["ensemble_score"] = sum(
            results[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return results
    
    def compare_refusal_patterns(
        self,
        fp1_refusal: Dict,
        fp2_refusal: Dict
    ) -> Dict[str, float]:
        """
        比較兩個模型的拒絕模式相似度
        
        Args:
            fp1_refusal, fp2_refusal: 拒絕指紋字典
        
        Returns:
            相似度指標
        """
        metrics = {}
        
        # 比較拒絕率
        rate1 = fp1_refusal.get("refusal_rate", 0.0)
        rate2 = fp2_refusal.get("refusal_rate", 0.0)
        metrics["refusal_rate_similarity"] = 1 - abs(rate1 - rate2)
        
        # 比較拒絕風格分佈
        style_vec1 = np.array([
            fp1_refusal.get("chinese_style_count", 0),
            fp1_refusal.get("western_style_count", 0),
            fp1_refusal.get("mixed_style_count", 0),
        ], dtype=float)
        
        style_vec2 = np.array([
            fp2_refusal.get("chinese_style_count", 0),
            fp2_refusal.get("western_style_count", 0),
            fp2_refusal.get("mixed_style_count", 0),
        ], dtype=float)
        
        if np.sum(style_vec1) > 0 and np.sum(style_vec2) > 0:
            # 歸一化
            style_vec1 = style_vec1 / np.sum(style_vec1)
            style_vec2 = style_vec2 / np.sum(style_vec2)
            
            metrics["style_distribution_similarity"] = self.cosine_similarity(
                style_vec1, style_vec2
            )
        else:
            metrics["style_distribution_similarity"] = 0.0
        
        # 比較拒絕模式重疊度
        patterns1 = set(fp1_refusal.get("pattern_distribution", {}).keys())
        patterns2 = set(fp2_refusal.get("pattern_distribution", {}).keys())
        
        if patterns1 or patterns2:
            overlap = len(patterns1 & patterns2)
            union = len(patterns1 | patterns2)
            metrics["pattern_overlap"] = overlap / union if union > 0 else 0.0
        else:
            metrics["pattern_overlap"] = 0.0
        
        # 綜合相似度
        metrics["refusal_ensemble_score"] = np.mean([
            metrics["refusal_rate_similarity"],
            metrics["style_distribution_similarity"],
            metrics["pattern_overlap"],
        ])
        
        return metrics
    
    def calculate_fingerprint_similarity(
        self,
        fp1: Dict,
        fp2: Dict
    ) -> Dict:
        """
        計算兩個完整指紋的相似度
        
        Args:
            fp1, fp2: 完整的指紋字典（包含 logit 和 refusal）
        
        Returns:
            完整的相似度分析
        """
        result = {
            "logit_similarity": {},
            "refusal_similarity": {},
            "overall_similarity": 0.0,
        }
        
        # 比較 Logit 指紋
        if fp1.get("logit_fingerprint") and fp2.get("logit_fingerprint"):
            vec1 = np.array(fp1["logit_fingerprint"]["vector"])
            vec2 = np.array(fp2["logit_fingerprint"]["vector"])
            
            result["logit_similarity"] = self.calculate_all_metrics(vec1, vec2)
        
        # 比較拒絕指紋
        if fp1.get("refusal_fingerprint") and fp2.get("refusal_fingerprint"):
            result["refusal_similarity"] = self.compare_refusal_patterns(
                fp1["refusal_fingerprint"],
                fp2["refusal_fingerprint"]
            )
        
        # 計算整體相似度（加權平均）
        logit_score = result["logit_similarity"].get("ensemble_score", 0.0)
        refusal_score = result["refusal_similarity"].get("refusal_ensemble_score", 0.0)
        
        # 🔧 修复：如果拒绝指纹不存在，只使用 logit 分数
        if fp1.get("refusal_fingerprint") and fp2.get("refusal_fingerprint"):
            # 两个指纹都有拒绝数据，使用加权平均
            # Logit 指紋權重較高（0.7），拒絕模式權重較低（0.3）
            result["overall_similarity"] = 0.7 * logit_score + 0.3 * refusal_score
        else:
            # 拒绝指纹缺失，只使用 logit 分数
            result["overall_similarity"] = logit_score
        
        return result


def main():
    """測試相似度計算器"""
    calc = SimilarityCalculator()
    
    # 測試向量
    vec1 = np.random.randn(100)
    vec2 = vec1 + np.random.randn(100) * 0.1  # 相似向量
    vec3 = np.random.randn(100)  # 不相似向量
    
    print("相似度計算器測試:")
    print("=" * 60)
    
    print("\n測試1: 相似向量 (vec1 vs vec2)")
    metrics = calc.calculate_all_metrics(vec1, vec2)
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n測試2: 不相似向量 (vec1 vs vec3)")
    metrics = calc.calculate_all_metrics(vec1, vec3)
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()
