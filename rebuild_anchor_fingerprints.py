"""
使用 Transformers 引擎重新提取锚点模型指纹

由于 Ollama 0.14.1 不支持 logprobs，我们使用 HuggingFace 上的对应模型重新提取
"""

import json
from pathlib import Path
from loguru import logger

from src.utils.unified_loader import load_model
from src.fingerprint import extract_fingerprint
from src.probes import build_all_probes


# HuggingFace 模型映射（对应原有的 Ollama 模型）
ANCHOR_MODEL_MAPPING = {
    # 原 Ollama 模型名 -> (HuggingFace 模型, 描述, 元数据)
    "qwen2.5:7b": {
        "hf_model": "Qwen/Qwen2.5-0.5B",  # 使用较小的版本以节省时间/资源
        "description": "阿里巴巴 Qwen 系列模型",
        "metadata": {
            "name": "qwen2.5:0.5b",
            "source": "china",
            "category": "qwen",
            "vendor": "Alibaba",
            "base_model": "Qwen2.5-0.5B",
            "description": "阿里巴巴 Qwen 系列模型（0.5B参数）"
        }
    },
    "deepseek-r1:7b": {
        "hf_model": "gpt2",  # DeepSeek R1 需要授权，使用 GPT-2 作为替代示例
        "description": "OpenAI GPT-2（替代 DeepSeek R1）",
        "metadata": {
            "name": "gpt2",
            "source": "openai",
            "category": "gpt",
            "vendor": "OpenAI",
            "base_model": "GPT-2",
            "description": "OpenAI GPT-2 模型"
        }
    },
    "yi:6b": {
        "hf_model": "01-ai/Yi-6B",  # Yi 模型可能需要授权
        "description": "零一万物 Yi 系列",
        "metadata": {
            "name": "yi:6b",
            "source": "china",
            "category": "yi",
            "vendor": "01.AI",
            "base_model": "Yi-6B",
            "description": "零一万物 Yi 系列模型"
        }
    },
    "llama3.2:3b": {
        "hf_model": "meta-llama/Llama-3.2-1B",  # Llama 3.2 需要授权，使用1B版本
        "description": "Meta Llama 3.2 系列",
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
        "hf_model": "google/gemma-2b",  # Gemma 2B
        "description": "Google Gemma 2B",
        "metadata": {
            "name": "gemma:2b",
            "source": "google",
            "category": "gemma",
            "vendor": "Google",
            "base_model": "Gemma-2B",
            "description": "Google Gemma 系列（2B参数）"
        }
    }
}


def extract_anchor_fingerprints(num_probes: int = 50):
    """
    使用 Transformers 引擎提取锚点模型指纹
    
    Args:
        num_probes: 每类探针使用的数量
    """
    logger.info("=" * 80)
    logger.info("使用 Transformers 引擎重新提取锚点模型指纹")
    logger.info("=" * 80)
    
    # 1. 加载探针
    probes_file = Path("data/probes/all_probes.json")
    if not probes_file.exists():
        logger.info("构建探针...")
        build_all_probes()
    
    with open(probes_file, 'r', encoding='utf-8') as f:
        all_probes = json.load(f)
    
    # 使用所有探针（不限制数量）
    test_probes = []
    for probe_type in all_probes.keys():
        test_probes.extend(all_probes[probe_type])  # 🔧 移除数量限制，使用全部探针
    
    logger.info(f"使用 {len(test_probes)} 个探针进行指纹提取 (完整数据集)")
    
    # 2. 创建输出目录
    anchor_dir = Path("data/anchor_models")
    anchor_dir.mkdir(exist_ok=True, parents=True)
    
    # 3. 提取每个锚点模型的指纹
    metadata_dict = {}
    success_count = 0
    
    for ollama_name, config in ANCHOR_MODEL_MAPPING.items():
        hf_model = config["hf_model"]
        description = config["description"]
        metadata = config["metadata"]
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"处理锚点: {description}")
        logger.info(f"  原模型: {ollama_name}")
        logger.info(f"  HF模型: {hf_model}")
        logger.info(f"{'=' * 80}")
        
        try:
            # 加载模型
            logger.info("  [1/3] 加载模型...")
            model = load_model(hf_model, engine="transformers")
            
            # 提取指纹
            logger.info("  [2/3] 提取指纹...")
            fingerprint = extract_fingerprint(model, test_probes)
            
            # 保存指纹
            logger.info("  [3/3] 保存指纹...")
            safe_name = ollama_name.replace(":", "_").replace("/", "_")
            fp_file = anchor_dir / f"{safe_name}_fingerprint.json"
            
            with open(fp_file, 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, indent=2, ensure_ascii=False)
            
            # 更新元数据
            metadata_dict[metadata["name"]] = {
                "metadata": metadata,
                "fingerprint_file": str(fp_file),
                "has_fingerprint": True,
                "hf_model": hf_model
            }
            
            logger.success(f"  ✓ 指纹已保存: {fp_file.name}")
            logger.info(f"  维度: {fingerprint['logit_fingerprint']['dimension']}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"  ✗ 失败: {e}")
            logger.warning(f"  跳过模型: {hf_model}")
            
            # 即使失败也要记录元数据
            metadata_dict[metadata["name"]] = {
                "metadata": metadata,
                "fingerprint_file": f"data/anchor_models/{safe_name}_fingerprint.json",
                "has_fingerprint": False,
                "hf_model": hf_model,
                "error": str(e)
            }
    
    # 4. 保存元数据
    logger.info(f"\n{'=' * 80}")
    logger.info("保存锚点数据库元数据...")
    metadata_file = anchor_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    logger.success(f"✓ 元数据已保存: {metadata_file}")
    
    # 5. 总结
    logger.info(f"\n{'=' * 80}")
    logger.info("提取总结")
    logger.info(f"{'=' * 80}")
    logger.info(f"成功: {success_count}/{len(ANCHOR_MODEL_MAPPING)}")
    logger.info(f"失败: {len(ANCHOR_MODEL_MAPPING) - success_count}/{len(ANCHOR_MODEL_MAPPING)}")
    
    if success_count == len(ANCHOR_MODEL_MAPPING):
        logger.success("\n🎉 所有锚点指纹提取成功！")
    elif success_count > 0:
        logger.warning(f"\n⚠️ 部分锚点提取成功 ({success_count}/{len(ANCHOR_MODEL_MAPPING)})")
    else:
        logger.error("\n❌ 所有锚点提取失败")
    
    return success_count > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 Transformers 引擎重新提取锚点指纹")
    parser.add_argument("--num-probes", type=int, default=50,
                        help="每类探针使用的数量 (默认: 50)")
    
    args = parser.parse_args()
    
    success = extract_anchor_fingerprints(args.num_probes)
    
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("下一步: 重新运行完整评估")
        logger.info("=" * 80)
        logger.info("python experiments/full_evaluation.py --target-model gpt2 --engine transformers")
    else:
        logger.error("\n锚点指纹提取失败，请检查错误信息")
