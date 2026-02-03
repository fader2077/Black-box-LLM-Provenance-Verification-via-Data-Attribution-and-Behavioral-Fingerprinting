"""
語言習慣探針 (Linguistic Shibboleths)
利用兩岸用語差異進行模型溯源
"""

import json
from typing import List, Dict, Tuple
from pathlib import Path
import random


class LinguisticProbes:
    """構建語言習慣測試探針"""
    
    def __init__(self):
        self.word_pairs = []
        self.probes = []
    
    def load_word_pairs(self) -> List[Tuple[str, str, str]]:
        """
        載入兩岸詞彙對照表
        格式: (台灣用語, 中國用語, 分類)
        """
        word_pairs = [
            # 科技類
            ("晶片", "芯片", "technology"),
            ("軟體", "軟件", "technology"),
            ("硬體", "硬件", "technology"),
            ("網路", "網絡", "technology"),
            ("資訊", "信息", "technology"),
            ("程式", "程序", "technology"),
            ("伺服器", "服務器", "technology"),
            ("數位", "數字", "technology"),
            ("滑鼠", "鼠標", "technology"),
            ("記憶體", "內存", "technology"),
            ("硬碟", "硬盤", "technology"),
            ("檔案", "文件", "technology"),
            ("資料庫", "數據庫", "technology"),
            ("使用者", "用戶", "technology"),
            
            # 日常用語
            ("計程車", "出租車", "daily"),
            ("捷運", "地鐵", "daily"),
            ("公車", "公交車", "daily"),
            ("機車", "摩托車", "daily"),
            ("停車場", "停車場", "daily"),
            ("影片", "視頻", "daily"),
            ("遊戲", "游戲", "daily"),
            ("行動電話", "手機", "daily"),
            ("簡訊", "短信", "daily"),
            ("帳號", "賬號", "daily"),
            ("頻道", "頻道", "daily"),
            
            # 品質與質量
            ("品質", "質量", "quality"),
            ("素質", "素質", "quality"),
            ("畫質", "畫質", "quality"),
            ("音質", "音質", "quality"),
            
            # 學術用語
            ("研究", "研究", "academic"),
            ("資料", "數據", "academic"),
            ("分析", "分析", "academic"),
            ("系統", "系統", "academic"),
            ("模型", "模型", "academic"),
            
            # 商業用語
            ("企業", "企業", "business"),
            ("公司", "公司", "business"),
            ("產品", "產品", "business"),
            ("品牌", "品牌", "business"),
            ("行銷", "營銷", "business"),
            ("客戶", "客戶", "business"),
            
            # 食物
            ("奶油", "黃油", "food"),
            ("馬鈴薯", "土豆", "food"),
            ("番茄", "西紅柿", "food"),
            ("鳳梨", "菠蘿", "food"),
            ("起司", "奶酪", "food"),
            
            # 運動
            ("足球", "足球", "sports"),
            ("籃球", "籃球", "sports"),
            ("棒球", "棒球", "sports"),
            ("桌球", "乒乓球", "sports"),
            ("羽球", "羽毛球", "sports"),
            
            # 醫療
            ("雷射", "激光", "medical"),
            ("癌症", "癌症", "medical"),
            ("基因", "基因", "medical"),
            ("手術", "手術", "medical"),
            
            # 其他高區辨度詞彙
            ("打印", "列印", "tech"),
            ("鼠標", "滑鼠", "tech"),
            ("激活", "啟動", "tech"),
            ("默認", "預設", "tech"),
            ("兼容", "相容", "tech"),
            ("在線", "線上", "tech"),
            ("離線", "離線", "tech"),
        ]
        
        self.word_pairs = word_pairs
        return word_pairs
    
    def build_masked_language_probes(self) -> List[Dict]:
        """
        構建遮蔽語言模型（MLM）風格的探針
        測試模型對特定詞彙的預測偏好
        """
        probes = []
        
        if not self.word_pairs:
            self.load_word_pairs()
        
        for tw_word, cn_word, category in self.word_pairs:
            # 構建填空題
            templates = [
                f"這項技術的{tw_word}非常先進。",
                f"我們需要提升{tw_word}水平。",
                f"請檢查{tw_word}是否正常。",
                f"這個{tw_word}值得關注。",
            ]
            
            for idx, template in enumerate(templates):
                # 將台灣用語替換為 [MASK]
                masked_prompt = template.replace(tw_word, "[MASK]")
                
                probe = {
                    "id": f"ling_{len(probes):04d}",
                    "category": "linguistic_shibboleth",
                    "subcategory": category,
                    "prompt": masked_prompt,
                    "taiwan_word": tw_word,
                    "china_word": cn_word,
                    "type": "masked_language_modeling",
                    "expected_cn_preference": cn_word,
                    "expected_tw_preference": tw_word,
                }
                probes.append(probe)
        
        return probes
    
    def build_completion_probes(self) -> List[Dict]:
        """
        構建句子補全探針
        測試模型在自然生成時的詞彙選擇
        """
        probes = []
        
        if not self.word_pairs:
            self.load_word_pairs()
        
        completion_templates = [
            ("這個產品的", "非常好", "quality_prefix"),
            ("我們需要改善", "", "improvement"),
            ("請問這個", "在哪裡", "location"),
            ("我想了解", "的相關資訊", "information"),
        ]
        
        for tw_word, cn_word, category in self.word_pairs[:20]:  # 選擇前20對高區辨度詞彙
            for prefix, suffix, template_type in completion_templates:
                prompt = f"{prefix}_____"
                
                probe = {
                    "id": f"comp_{len(probes):04d}",
                    "category": "completion",
                    "subcategory": category,
                    "prompt": prompt,
                    "context": f"{prefix}{tw_word}{suffix}",
                    "taiwan_word": tw_word,
                    "china_word": cn_word,
                    "type": "completion",
                }
                probes.append(probe)
        
        return probes
    
    def build_choice_probes(self) -> List[Dict]:
        """
        構建選擇題探針
        明確測試模型對不同用語的偏好
        """
        probes = []
        
        if not self.word_pairs:
            self.load_word_pairs()
        
        for tw_word, cn_word, category in self.word_pairs:
            # 構建選擇題格式
            prompt = f"請問以下哪個詞彙更常用？\nA. {tw_word}\nB. {cn_word}\n請選擇 A 或 B："
            
            probe = {
                "id": f"choice_{len(probes):04d}",
                "category": "multiple_choice",
                "subcategory": category,
                "prompt": prompt,
                "option_a": tw_word,
                "option_b": cn_word,
                "type": "choice",
                "expected_cn_answer": "B",
                "expected_tw_answer": "A",
            }
            probes.append(probe)
        
        return probes
    
    def build_translation_probes(self) -> List[Dict]:
        """
        構建翻譯探針
        測試模型的翻譯偏好（繁體/簡體、台灣/中國用語）
        """
        probes = []
        
        english_sentences = [
            "This computer chip is very advanced.",
            "Please check the software quality.",
            "The network connection is stable.",
            "We need to improve the information system.",
            "The server is running normally.",
        ]
        
        for idx, en_sentence in enumerate(english_sentences):
            probe = {
                "id": f"trans_{idx:04d}",
                "category": "translation",
                "prompt": f"請將以下英文翻譯成中文：\n{en_sentence}",
                "source": en_sentence,
                "type": "translation",
                "check_points": ["chip→晶片/芯片", "software→軟體/軟件", "network→網路/網絡"],
            }
            probes.append(probe)
        
        return probes
    
    def build_all_probes(self) -> List[Dict]:
        """構建所有語言習慣探針"""
        all_probes = []
        
        self.load_word_pairs()
        
        # 添加各類探針
        all_probes.extend(self.build_masked_language_probes())
        all_probes.extend(self.build_completion_probes())
        all_probes.extend(self.build_choice_probes())
        all_probes.extend(self.build_translation_probes())
        
        # 添加通用元數據
        for probe in all_probes:
            probe["probe_type"] = "linguistic_shibboleth"
        
        self.probes = all_probes
        return all_probes
    
    def save_to_file(self, output_path: str):
        """儲存探針到JSON檔案"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.probes, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 語言習慣探針已保存: {output_path}")
        print(f"  總數: {len(self.probes)} 個探針")
    
    def save_word_pairs(self, output_path: str):
        """儲存詞彙對照表為CSV"""
        import csv
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["台灣用語", "中國用語", "分類"])
            writer.writerows(self.word_pairs)
        
        print(f"✓ 詞彙對照表已保存: {output_path}")
        print(f"  總數: {len(self.word_pairs)} 對詞彙")
    
    def get_statistics(self) -> Dict:
        """獲取探針統計資訊"""
        from collections import Counter
        
        types = Counter(probe["type"] for probe in self.probes)
        subcategories = Counter(probe.get("subcategory", "unknown") for probe in self.probes)
        
        return {
            "total": len(self.probes),
            "word_pairs": len(self.word_pairs),
            "by_type": dict(types),
            "by_subcategory": dict(subcategories),
        }


def main():
    """主函數：構建並保存語言習慣探針"""
    builder = LinguisticProbes()
    probes = builder.build_all_probes()
    
    # 保存探針
    output_path = "data/probes/linguistic_shibboleths_probes.json"
    builder.save_to_file(output_path)
    
    # 保存詞彙對照表
    word_pairs_path = "data/probes/word_pairs.csv"
    builder.save_word_pairs(word_pairs_path)
    
    # 顯示統計資訊
    stats = builder.get_statistics()
    print("\n統計資訊:")
    print(f"  總探針數: {stats['total']}")
    print(f"  詞彙對數: {stats['word_pairs']}")
    print(f"  按類型: {stats['by_type']}")
    print(f"  按子分類: {stats['by_subcategory']}")


if __name__ == "__main__":
    main()
