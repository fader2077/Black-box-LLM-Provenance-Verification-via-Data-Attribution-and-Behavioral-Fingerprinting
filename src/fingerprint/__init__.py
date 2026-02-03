"""
指紋提取模組初始化
"""

from .logit_extractor import LogitExtractor
from .refusal_detector import RefusalDetector
import numpy as np
from typing import Dict, List
from loguru import logger


def extract_fingerprint(
    model_interface,
    probes: List[Dict],
    include_logit: bool = True,
    include_refusal: bool = True,
    top_k: int = 20
) -> Dict:
    """
    統一的指紋提取接口
    
    Args:
        model_interface: 模型推理接口
        probes: 探針列表
        include_logit: 是否包含 Logit 指紋
        include_refusal: 是否包含拒絕響應指紋
        top_k: Logit 提取的 top-k 參數
    
    Returns:
        完整的指紋字典
    """
    fingerprint = {
        "model_name": getattr(model_interface, 'model_name', 'unknown'),
        "timestamp": None,
        "logit_fingerprint": None,
        "refusal_fingerprint": None,
    }
    
    # 提取 Logit 指紋
    if include_logit:
        logger.info("提取 Logit 分佈指紋...")
        logit_extractor = LogitExtractor(model_interface, top_k=top_k)
        
        logit_fp = logit_extractor.extract_fingerprint_from_probes(probes)
        fingerprint["logit_fingerprint"] = {
            "vector": logit_fp.tolist(),
            "dimension": len(logit_fp),
            "stats": {
                "mean": float(np.mean(logit_fp)),
                "std": float(np.std(logit_fp)),
                "min": float(np.min(logit_fp)),
                "max": float(np.max(logit_fp)),
            }
        }
    
    # 提取拒絕響應指紋
    if include_refusal:
        logger.info("提取拒絕響應指紋...")
        refusal_detector = RefusalDetector()
        
        # 只對政治敏感性探針進行拒絕檢測
        political_probes = [p for p in probes if p.get("probe_type") == "political_sensitivity"]
        
        if political_probes:
            refusal_fp = refusal_detector.extract_refusal_fingerprint(
                political_probes, 
                model_interface
            )
            fingerprint["refusal_fingerprint"] = refusal_fp
        else:
            logger.warning("未找到政治敏感性探針，跳過拒絕檢測")
    
    # 添加時間戳
    from datetime import datetime
    fingerprint["timestamp"] = datetime.now().isoformat()
    
    logger.info("✓ 完整指紋提取完成")
    
    return fingerprint


__all__ = [
    "LogitExtractor",
    "RefusalDetector",
    "extract_fingerprint",
]
