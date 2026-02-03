#!/usr/bin/env python3
"""
完全重建所有锚点指纹，使用统一的探针数量
确保数据质量和维度一致性
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.probes import build_all_probes
from src.fingerprint import extract_fingerprint
from src.utils.unified_loader import load_model

# 锚点模型映射
ANCHOR_MODEL_MAPPING = {
    "qwen2.5:0.5b": {
        "hf_model": "Qwen/Qwen2.5-0.5B",
        "ollama_model": "qwen2.5:7b",
        "engine": "transformers",
        "metadata": {
            "name": "qwen2.5:0.5b",
            "source": "china",
            "category": "qwen",
            "vendor": "Alibaba",
            "base_model": "Qwen2.5-0.5B",
            "description": "阿里巴巴 Qwen 系列模型（0.5B参数）"
        }
    },
    "gpt2": {
        "hf_model": "gpt2",
        "ollama_model": None,
        "engine": "transformers",
        "metadata": {
            "name": "gpt2",
            "source": "openai",
            "category": "gpt",
            "vendor": "OpenAI",
            "base_model": "GPT-2",
            "description": "OpenAI GPT-2 模型（124M参数）"
        }
    },
    "gpt2-medium": {
        "hf_model": "gpt2-medium",
        "ollama_model": None,
        "engine": "transformers",
        "metadata": {
            "name": "gpt2-medium",
            "source": "openai",
            "category": "gpt",
            "vendor": "OpenAI",
            "base_model": "GPT-2-Medium",
            "description": "OpenAI GPT-2 Medium 模型（355M参数）"
        }
    },
    "yi:6b": {
        "hf_model": "01-ai/Yi-6B",
        "ollama_model": "yi:6b",
        "engine": "transformers",
        "metadata": {
            "name": "yi:6b",
            "source": "china",
            "category": "yi",
            "vendor": "01.AI",
            "base_model": "Yi-6B",
            "description": "零一万物 Yi 系列模型（6B参数）"
        }
    },
    "deepseek-r1:7b": {
        "hf_model": None,  # 需要授权
        "ollama_model": "deepseek-r1:7b",
        "engine": "ollama",
        "metadata": {
            "name": "deepseek-r1:7b",
            "source": "china",
            "category": "deepseek",
            "vendor": "DeepSeek",
            "base_model": "DeepSeek-R1-7B",
            "description": "DeepSeek-R1 系列（7B参数）"
        }
    },
    "llama3.2:1b": {
        "hf_model": "meta-llama/Llama-3.2-1B",
        "ollama_model": "llama3.2:1b",
        "engine": "transformers",  # 优先尝试 transformers
        "metadata": {
            "name": "llama3.2:1b",
            "source": "meta",
            "category": "llama",
            "vendor": "Meta",
            "base_model": "Llama-3.2-1B",
            "description": "Meta Llama 3.2 系列（1B参数）"
        }
    },
    "gemma2:2b": {
        "hf_model": "google/gemma-2b",
        "ollama_model": "gemma2:2b",
        "engine": "transformers",
        "metadata": {
            "name": "gemma2:2b",
            "source": "google",
            "category": "gemma",
            "vendor": "Google",
            "base_model": "Gemma-2B",
            "description": "Google Gemma 系列（2B参数）"
        }
    },
}


def extract_single_anchor(
    anchor_name: str,
    config: Dict,
    probes: List,
    output_dir: Path,
    force: bool = False
) -> Optional[Dict]:
    """提取单个锚点模型的指纹"""
    
    output_file = output_dir / f"{anchor_name.replace(':', '_').replace('-', '_')}_fingerprint.json"
    
    # 检查是否已存在
    if output_file.exists() and not force:
        logger.info(f"⏭️  跳过 {anchor_name}（已存在，使用 --force 强制重新提取）")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📍 提取锚点: {anchor_name}")
    logger.info(f"{'='*60}")
    
    engine = config["engine"]
    model_id = config["hf_model"] if engine == "transformers" else config["ollama_model"]
    
    if model_id is None:
        logger.warning(f"⚠️  {anchor_name} 无可用模型，跳过")
        return None
    
    logger.info(f"  引擎: {engine}")
    logger.info(f"  模型: {model_id}")
    
    try:
        # 加载模型
        logger.info(f"⏳ 加载模型...")
        model = load_model(model_id, engine=engine)
        
        # 提取指纹
        logger.info(f"⏳ 提取指纹（{len(probes)} 个探针）...")
        fingerprint = extract_fingerprint(
            model_interface=model,
            probes=probes,
            include_logit=True,
            include_refusal=False  # 暂时禁用拒绝检测，加快速度
        )
        
        # 验证指纹质量
        fp_vector = fingerprint["logit_fingerprint"]["vector"]
        fp_dim = len(fp_vector)
        
        # 检查是否全是 0
        if all(v == 0.0 for v in fp_vector):
            logger.error(f"❌ {anchor_name} 指纹全是 0，提取失败！")
            return None
        
        # 检查维度
        if fp_dim < 10:
            logger.warning(f"⚠️  {anchor_name} 指纹维度过小: {fp_dim}")
        
        logger.success(f"✅ {anchor_name} 指纹提取成功")
        logger.info(f"  维度: {fp_dim}")
        logger.info(f"  均值: {fingerprint['logit_fingerprint']['stats']['mean']:.4f}")
        logger.info(f"  标准差: {fingerprint['logit_fingerprint']['stats']['std']:.4f}")
        
        # 保存指纹
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 已保存: {output_file}")
        
        return {
            "anchor_name": anchor_name,
            "config": config,
            "fingerprint_file": str(output_file.relative_to(project_root)),
            "dimension": fp_dim,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ {anchor_name} 提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重新提取所有锚点指纹")
    parser.add_argument("--num-probes", type=int, default=150,
                        help="每个类别的探针数量（总数=3*num_probes）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新提取（即使文件已存在）")
    parser.add_argument("--anchors", nargs="+",
                        help="只提取指定的锚点（默认全部）")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🔄 重新提取所有锚点指纹")
    logger.info("="*80)
    
    # 准备输出目录
    output_dir = project_root / "data" / "anchor_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载探针
    logger.info(f"\n📋 加载探针...")
    
    probes_path = project_root / "data" / "probes" / "all_probes.json"
    if probes_path.exists():
        logger.info(f"使用缓存的探针数据: {probes_path}")
        with open(probes_path, 'r', encoding='utf-8') as f:
            probes_data = json.load(f)
    else:
        logger.info("构建新的探针数据集")
        probes_data = build_all_probes()
    
    # 合并所有探针
    all_probes = []
    for probe_type, probes in probes_data.items():
        all_probes.extend(probes)
    
    # 如果指定了探针数量，随机抽样
    if args.num_probes > 0 and len(all_probes) > args.num_probes * 3:
        import random
        random.seed(42)
        all_probes = random.sample(all_probes, args.num_probes * 3)
        logger.info(f"已抽样 {len(all_probes)} 个探针")
    else:
        logger.info(f"使用全部 {len(all_probes)} 个探针")
    
    logger.success(f"✅ 已加载 {len(all_probes)} 个探针")
    
    # 选择要提取的锚点
    if args.anchors:
        anchors_to_extract = {k: v for k, v in ANCHOR_MODEL_MAPPING.items() if k in args.anchors}
        if not anchors_to_extract:
            logger.error(f"❌ 未找到指定的锚点: {args.anchors}")
            sys.exit(1)
    else:
        anchors_to_extract = ANCHOR_MODEL_MAPPING
    
    logger.info(f"\n将提取 {len(anchors_to_extract)} 个锚点:")
    for name in anchors_to_extract:
        logger.info(f"  • {name}")
    
    # 提取所有锚点
    results = []
    for anchor_name, config in anchors_to_extract.items():
        result = extract_single_anchor(
            anchor_name=anchor_name,
            config=config,
            probes=all_probes,
            output_dir=output_dir,
            force=args.force
        )
        if result:
            results.append(result)
    
    # 更新 metadata
    logger.info(f"\n📝 更新 metadata.json...")
    metadata = {}
    
    for result in results:
        anchor_name = result["anchor_name"]
        config = result["config"]
        
        metadata[anchor_name] = {
            "metadata": config["metadata"],
            "fingerprint_file": result["fingerprint_file"],
            "has_fingerprint": True,
            "hf_model": config["hf_model"],
            "ollama_model": config["ollama_model"],
            "engine": config["engine"],
            "dimension": result["dimension"]
        }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.success(f"✅ metadata 已保存: {metadata_file}")
    
    # 总结
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 提取结果总结")
    logger.info(f"{'='*80}")
    logger.info(f"成功: {len(results)}/{len(anchors_to_extract)}")
    
    if results:
        logger.info(f"\n成功提取的锚点:")
        for result in results:
            logger.info(f"  ✅ {result['anchor_name']:20s} (维度: {result['dimension']})")
    
    failed = len(anchors_to_extract) - len(results)
    if failed > 0:
        logger.warning(f"\n⚠️  失败: {failed} 个锚点")
    
    logger.success(f"\n🎉 完成！")


if __name__ == "__main__":
    main()
