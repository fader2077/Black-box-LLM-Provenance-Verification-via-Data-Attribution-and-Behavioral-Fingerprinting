"""
探針模組初始化
提供統一的探針構建接口
"""

from .political_probes import PoliticalProbes
from .linguistic_probes import LinguisticProbes
from .memorization_probes import MemorizationProbes
import json
from pathlib import Path


def build_all_probes(output_dir: str = "data/probes") -> dict:
    """
    構建所有類型的探針資料集
    
    Args:
        output_dir: 輸出目錄路徑
    
    Returns:
        包含所有探針的字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("開始構建歸因探針資料集...")
    print("="*60)
    
    # 1. 構建政治敏感性探針
    print("\n[1/3] 構建政治敏感性探針...")
    political_builder = PoliticalProbes()
    political_probes = political_builder.build_all_probes()
    political_builder.save_to_file(str(output_path / "political_sensitivity_probes.json"))
    
    # 2. 構建語言習慣探針
    print("\n[2/3] 構建語言習慣探針...")
    linguistic_builder = LinguisticProbes()
    linguistic_probes = linguistic_builder.build_all_probes()
    linguistic_builder.save_to_file(str(output_path / "linguistic_shibboleths_probes.json"))
    linguistic_builder.save_word_pairs(str(output_path / "word_pairs.csv"))
    
    # 3. 構建記憶化探針
    print("\n[3/3] 構建記憶化探針...")
    memorization_builder = MemorizationProbes()
    memorization_probes = memorization_builder.build_all_probes()
    memorization_builder.save_to_file(str(output_path / "memorization_probes.json"))
    
    # 合併所有探針
    all_probes = {
        "political_sensitivity": political_probes,
        "linguistic_shibboleths": linguistic_probes,
        "memorization": memorization_probes,
    }
    
    # 保存合併的探針集
    combined_path = output_path / "all_probes.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_probes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 所有探針已合併保存: {combined_path}")
    
    # 總體統計
    total_count = len(political_probes) + len(linguistic_probes) + len(memorization_probes)
    
    print("\n" + "="*60)
    print("探針構建完成！")
    print("="*60)
    print(f"總探針數: {total_count}")
    print(f"  - 政治敏感性: {len(political_probes)}")
    print(f"  - 語言習慣: {len(linguistic_probes)}")
    print(f"  - 記憶化: {len(memorization_probes)}")
    print("="*60)
    
    return all_probes


__all__ = [
    "PoliticalProbes",
    "LinguisticProbes",
    "MemorizationProbes",
    "build_all_probes",
]
