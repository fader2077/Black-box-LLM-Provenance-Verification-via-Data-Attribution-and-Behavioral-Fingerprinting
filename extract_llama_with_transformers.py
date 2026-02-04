"""使用 Transformers 引擎重新提取 Llama-3.1-8B 锚点指纹"""
import sys
sys.path.insert(0, '.')

from src.utils.model_loader_v2 import TransformersModelLoader
from src.fingerprint.logit_extractor import LogitExtractor
from src.utils.probe_loader import ProbeLoader
import json
import torch
from datetime import datetime
from pathlib import Path
from loguru import logger

# 配置
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # HuggingFace model
OUTPUT_FILE = "data/anchor_models/llama3_1_8b_fingerprint_full.json"
NUM_PROBES = 30  # 使用30个探针
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"设备: {DEVICE}")
logger.info(f"模型: {MODEL_NAME}")
logger.info(f"探针数量: {NUM_PROBES}")

# 加载探针
logger.info("加载探针...")
probe_loader = ProbeLoader("data/probes/probe_dataset_v3.json")
probes = probe_loader.load_probes()[:NUM_PROBES]
logger.info(f"✓ 已加载 {len(probes)} 个探针")

# 加载模型
logger.info("加载模型（使用 Transformers 引擎）...")
try:
    model_loader = TransformersModelLoader(
        model_path=MODEL_NAME,
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    logger.info("✓ 模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    logger.info("可能需要先从 HuggingFace 下载模型")
    logger.info(f"或使用其他 Llama 模型，如:")
    logger.info("  - meta-llama/Meta-Llama-3-8B-Instruct")
    logger.info("  - meta-llama/Llama-2-7b-chat-hf")
    sys.exit(1)

# 创建提取器
extractor = LogitExtractor(
    model=model_loader,
    top_k=20
)

# 提取指纹
all_fingerprints = []
successful = 0
failed = 0

logger.info(f"开始提取指纹...")
for i, probe in enumerate(probes):
    try:
        logger.info(f"[{i+1}/{len(probes)}] 处理探针: {probe['question'][:50]}...")
        
        # 提取logits
        logits = extractor.extract_logits(
            prompt=probe['question'],
            target_tokens=probe['options']
        )
        
        # 保存指纹向量
        if "top_k_probs" in logits:
            all_fingerprints.extend(logits["top_k_probs"])
            successful += 1
            logger.success(f"✓ 探针 {i} 成功")
        else:
            logger.warning(f"⚠ 探针 {i} 无 logprobs")
            failed += 1
            
    except Exception as e:
        logger.error(f"✗ 探针 {i} 失败: {e}")
        failed += 1

# 保存指纹
fingerprint_data = {
    "model_name": "llama3.1:8b",
    "hf_model": MODEL_NAME,
    "timestamp": datetime.now().isoformat(),
    "logit_fingerprint": {
        "vector": all_fingerprints,
        "dimension": len(all_fingerprints),
        "stats": {
            "mean": sum(all_fingerprints) / len(all_fingerprints) if all_fingerprints else 0,
            "std": 0,  # 简化版本
            "min": min(all_fingerprints) if all_fingerprints else 0,
            "max": max(all_fingerprints) if all_fingerprints else 0
        }
    },
    "extraction_stats": {
        "total_probes": len(probes),
        "successful_probes": successful,
        "failed_probes": failed,
        "success_rate": successful / len(probes) if probes else 0
    },
    "engine": "transformers",
    "device": DEVICE
}

# 创建输出目录
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

# 保存文件
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(fingerprint_data, f, ensure_ascii=False, indent=2)

logger.success(f"✓ 指纹已保存: {OUTPUT_FILE}")
logger.info(f"  成功率: {successful}/{len(probes)} ({100*successful/len(probes):.1f}%)")
logger.info(f"  维度: {len(all_fingerprints)}")
logger.info(f"  范围: [{min(all_fingerprints):.3f}, {max(all_fingerprints):.3f}]")
