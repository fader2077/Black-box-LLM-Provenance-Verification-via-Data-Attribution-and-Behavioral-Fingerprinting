"""
添加llama3.2:3b锚点模型到数据库
"""

import sys
import json
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.attribution.anchor_models import AnchorModelsDatabase
from src.fingerprint import extract_fingerprint
from src.utils.unified_loader import load_model

def add_llama_anchor():
    """添加llama3.2:3b作为锚点"""
    
    logger.info("=" * 80)
    logger.info("添加 Llama3.2:3b 锚点模型")
    logger.info("=" * 80)
    
    # 1. 加载锚点数据库
    logger.info("\n[1/4] 加载锚点数据库...")
    db = AnchorModelsDatabase()
    
    # 2. 添加锚点模型（如果不存在）
    model_name = "llama3.2:3b"
    logger.info(f"\n[2/4] 添加锚点模型: {model_name}...")
    
    if model_name not in db.anchor_models:
        db.add_anchor_model(
            name=model_name,
            source="meta",
            category="llama",
            metadata={
                "vendor": "Meta",
                "base_model": "Llama-3.2-3B",
                "description": "Meta Llama 3.2 系列（3B参数）"
            }
        )
        logger.info(f"✓ 已添加锚点模型: {model_name}")
    else:
        logger.info(f"✓ 锚点模型已存在: {model_name}")
    
    # 3. 加载探针
    logger.info("\n[3/4] 加载探针...")
    probes_path = Path("data/probes/all_probes.json")
    if not probes_path.exists():
        logger.error("探针文件不存在，请先运行 build_all_probes")
        return
    
    with open(probes_path, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
    
    all_probes = []
    for probe_type, probes in probes_data.items():
        all_probes.extend(probes)
    
    logger.info(f"✓ 已加载 {len(all_probes)} 个探针")
    
    # 4. 提取指纹
    logger.info(f"\n[4/4] 提取模型指纹...")
    logger.info("此过程可能需要较长时间，请耐心等待...")
    
    try:
        # 加载模型
        logger.info(f"  加载模型...")
        model = load_model(
            model_name=model_name,
            engine="ollama",
            device="cuda"
        )
        logger.info(f"  ✓ 模型加载成功")
        
        # 提取指纹
        logger.info(f"  提取指纹...")
        fingerprint = extract_fingerprint(model, all_probes)
        
        if not fingerprint:
            logger.error(f"  ✗ 指纹提取失败")
            return
        
        logger.info(f"  ✓ 指纹提取成功")
        
        # 保存指纹
        logger.info(f"  保存指纹到数据库...")
        db.store_fingerprint(model_name, fingerprint)
        logger.info(f"  ✓ 指纹已保存")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✓ 成功添加 {model_name} 锚点模型")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    add_llama_anchor()
