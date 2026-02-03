"""
評估指標計算
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_classification_metrics(
    y_true: List[int],
    y_pred: List[int]
) -> Dict[str, float]:
    """
    計算分類指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
    
    Returns:
        指標字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    return metrics


def calculate_rouge_scores(
    predictions: List[str],
    references: List[str]
) -> Dict:
    """
    計算 ROUGE 分數（用於文本生成評估）
    
    Args:
        predictions: 預測文本列表
        references: 參考文本列表
    
    Returns:
        ROUGE 分數
    """
    try:
        from rouge import Rouge
        
        rouge = Rouge()
        scores = rouge.get_scores(predictions, references, avg=True)
        
        return scores
    
    except ImportError:
        return {"error": "Please install rouge: pip install rouge"}


def calculate_perplexity_stats(
    perplexities: List[float]
) -> Dict[str, float]:
    """
    計算困惑度統計資訊
    
    Args:
        perplexities: 困惑度列表
    
    Returns:
        統計指標
    """
    perplexities = np.array(perplexities)
    
    # 過濾掉無效值
    valid_ppls = perplexities[np.isfinite(perplexities)]
    
    if len(valid_ppls) == 0:
        return {
            "mean": float('inf'),
            "median": float('inf'),
            "std": 0.0,
            "min": float('inf'),
            "max": float('inf'),
        }
    
    return {
        "mean": float(np.mean(valid_ppls)),
        "median": float(np.median(valid_ppls)),
        "std": float(np.std(valid_ppls)),
        "min": float(np.min(valid_ppls)),
        "max": float(np.max(valid_ppls)),
        "q25": float(np.percentile(valid_ppls, 25)),
        "q75": float(np.percentile(valid_ppls, 75)),
    }


def calculate_attribution_accuracy(
    predictions: List[str],
    ground_truth: List[str]
) -> Dict:
    """
    計算歸因準確度
    
    Args:
        predictions: 預測的模型來源
        ground_truth: 真實的模型來源
    
    Returns:
        準確度統計
    """
    correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
    total = len(predictions)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "error_rate": 1 - accuracy,
    }


__all__ = [
    "calculate_classification_metrics",
    "calculate_rouge_scores",
    "calculate_perplexity_stats",
    "calculate_attribution_accuracy",
]
