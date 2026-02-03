"""
錨點模型管理
管理已知來源的模型指紋數據庫
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from loguru import logger


class AnchorModelsDatabase:
    """
    錨點模型數據庫管理器
    儲存和管理已知來源模型的指紋
    """
    
    def __init__(self, db_path: str = "data/anchor_models"):
        """
        Args:
            db_path: 數據庫存儲路徑
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.anchor_models = {}
        self.metadata_file = self.db_path / "metadata.json"
        
        # 載入現有數據庫
        self._load_database()
    
    def _load_database(self):
        """載入錨點模型數據庫"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.anchor_models = json.load(f)
            logger.info(f"已載入 {len(self.anchor_models)} 個錨點模型")
        else:
            logger.info("未找到現有數據庫，將創建新數據庫")
            self._initialize_default_anchors()
    
    def _initialize_default_anchors(self):
        """初始化預設的錨點模型列表"""
        default_anchors = [
            {
                "name": "qwen2.5:7b",
                "source": "china",
                "category": "qwen",
                "vendor": "Alibaba",
                "base_model": "Qwen2.5-7B",
                "description": "阿里巴巴 Qwen 系列模型",
            },
            {
                "name": "deepseek-r1:7b",
                "source": "china",
                "category": "deepseek",
                "vendor": "DeepSeek",
                "base_model": "DeepSeek-R1-7B",
                "description": "DeepSeek 推理優化模型",
            },
            {
                "name": "yi:6b",
                "source": "china",
                "category": "yi",
                "vendor": "01.AI",
                "base_model": "Yi-6B",
                "description": "零一萬物 Yi 系列模型",
            },
            {
                "name": "llama3.2:3b",
                "source": "meta",
                "category": "llama",
                "vendor": "Meta",
                "base_model": "Llama-3.2-3B",
                "description": "Meta Llama 3.2 系列",
            },
            {
                "name": "gemma2:2b",
                "source": "google",
                "category": "gemma",
                "vendor": "Google",
                "base_model": "Gemma-2-2B",
                "description": "Google Gemma 2 系列",
            },
        ]
        
        for anchor in default_anchors:
            self.anchor_models[anchor["name"]] = {
                "metadata": anchor,
                "fingerprint_file": None,
                "has_fingerprint": False,
            }
        
        self._save_metadata()
        logger.info(f"已初始化 {len(default_anchors)} 個預設錨點模型")
    
    def add_anchor_model(
        self,
        name: str,
        source: str,
        category: str,
        metadata: Optional[Dict] = None
    ):
        """
        添加新的錨點模型
        
        Args:
            name: 模型名稱
            source: 來源（china, meta, google, etc.）
            category: 類別（qwen, llama, gemma, etc.）
            metadata: 其他元數據
        """
        if name in self.anchor_models:
            logger.warning(f"錨點模型 {name} 已存在，將更新元數據")
        
        anchor_data = {
            "metadata": {
                "name": name,
                "source": source,
                "category": category,
                **(metadata or {})
            },
            "fingerprint_file": None,
            "has_fingerprint": False,
        }
        
        self.anchor_models[name] = anchor_data
        self._save_metadata()
        
        logger.info(f"✓ 已添加錨點模型: {name}")
    
    def store_fingerprint(
        self,
        model_name: str,
        fingerprint: Dict
    ):
        """
        儲存錨點模型的指紋
        
        Args:
            model_name: 模型名稱
            fingerprint: 指紋數據
        """
        if model_name not in self.anchor_models:
            logger.error(f"模型 {model_name} 不在錨點列表中")
            return
        
        # 保存指紋到獨立文件
        fingerprint_file = self.db_path / f"{model_name.replace(':', '_')}_fingerprint.json"
        
        with open(fingerprint_file, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f, indent=2)
        
        # 更新元數據
        self.anchor_models[model_name]["fingerprint_file"] = str(fingerprint_file)
        self.anchor_models[model_name]["has_fingerprint"] = True
        
        self._save_metadata()
        
        logger.info(f"✓ 已儲存 {model_name} 的指紋")
    
    def load_fingerprint(self, model_name: str) -> Optional[Dict]:
        """
        載入錨點模型的指紋
        
        Args:
            model_name: 模型名稱
        
        Returns:
            指紋數據，如果不存在則返回 None
        """
        if model_name not in self.anchor_models:
            logger.error(f"模型 {model_name} 不在錨點列表中")
            return None
        
        fingerprint_file = self.anchor_models[model_name].get("fingerprint_file")
        
        if not fingerprint_file or not Path(fingerprint_file).exists():
            logger.warning(f"模型 {model_name} 的指紋文件不存在")
            return None
        
        with open(fingerprint_file, 'r', encoding='utf-8') as f:
            fingerprint = json.load(f)
        
        return fingerprint
    
    def get_anchor_by_source(self, source: str) -> List[str]:
        """
        獲取特定來源的所有錨點模型
        
        Args:
            source: 來源標籤（china, meta, google, etc.）
        
        Returns:
            模型名稱列表
        """
        return [
            name for name, data in self.anchor_models.items()
            if data["metadata"].get("source") == source
        ]
    
    def get_anchor_by_category(self, category: str) -> List[str]:
        """
        獲取特定類別的所有錨點模型
        
        Args:
            category: 類別標籤（qwen, llama, deepseek, etc.）
        
        Returns:
            模型名稱列表
        """
        return [
            name for name, data in self.anchor_models.items()
            if data["metadata"].get("category") == category
        ]
    
    def list_all_anchors(self) -> Dict:
        """
        列出所有錨點模型及其狀態
        
        Returns:
            錨點模型字典
        """
        summary = {}
        
        for name, data in self.anchor_models.items():
            summary[name] = {
                "source": data["metadata"].get("source"),
                "category": data["metadata"].get("category"),
                "has_fingerprint": data.get("has_fingerprint", False),
            }
        
        return summary
    
    def _save_metadata(self):
        """保存元數據到文件"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.anchor_models, f, indent=2, ensure_ascii=False)
    
    def export_database_summary(self) -> Dict:
        """
        導出數據庫摘要統計
        
        Returns:
            統計信息
        """
        total = len(self.anchor_models)
        with_fingerprint = sum(
            1 for data in self.anchor_models.values() 
            if data.get("has_fingerprint")
        )
        
        sources = {}
        categories = {}
        
        for data in self.anchor_models.values():
            source = data["metadata"].get("source", "unknown")
            category = data["metadata"].get("category", "unknown")
            
            sources[source] = sources.get(source, 0) + 1
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_anchors": total,
            "with_fingerprint": with_fingerprint,
            "without_fingerprint": total - with_fingerprint,
            "by_source": sources,
            "by_category": categories,
        }
    
    def verify_database_integrity(self) -> Dict:
        """
        驗證數據庫完整性
        
        Returns:
            驗證結果
        """
        issues = []
        
        for name, data in self.anchor_models.items():
            # 檢查指紋文件是否存在
            if data.get("has_fingerprint"):
                fp_file = data.get("fingerprint_file")
                if not fp_file or not Path(fp_file).exists():
                    issues.append({
                        "model": name,
                        "issue": "fingerprint_file_missing",
                        "severity": "warning",
                    })
            
            # 檢查元數據完整性
            required_fields = ["name", "source", "category"]
            for field in required_fields:
                if field not in data["metadata"]:
                    issues.append({
                        "model": name,
                        "issue": f"missing_metadata_{field}",
                        "severity": "error",
                    })
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "checked_models": len(self.anchor_models),
        }


def main():
    """測試錨點模型數據庫"""
    db = AnchorModelsDatabase()
    
    print("錨點模型數據庫測試:")
    print("=" * 60)
    
    # 列出所有錨點
    print("\n所有錨點模型:")
    anchors = db.list_all_anchors()
    for name, info in anchors.items():
        status = "✓" if info["has_fingerprint"] else "✗"
        print(f"  {status} {name} ({info['source']}/{info['category']})")
    
    # 統計摘要
    print("\n數據庫統計:")
    summary = db.export_database_summary()
    print(f"  總錨點數: {summary['total_anchors']}")
    print(f"  已有指紋: {summary['with_fingerprint']}")
    print(f"  缺少指紋: {summary['without_fingerprint']}")
    print(f"  按來源: {summary['by_source']}")
    print(f"  按類別: {summary['by_category']}")
    
    # 驗證完整性
    print("\n數據庫完整性驗證:")
    integrity = db.verify_database_integrity()
    if integrity["is_valid"]:
        print("  ✓ 數據庫完整")
    else:
        print(f"  ✗ 發現 {len(integrity['issues'])} 個問題")
        for issue in integrity["issues"][:5]:
            print(f"    - {issue}")


if __name__ == "__main__":
    main()
