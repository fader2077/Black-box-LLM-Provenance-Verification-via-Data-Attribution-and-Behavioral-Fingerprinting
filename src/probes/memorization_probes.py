"""
記憶化探針 (Memorization Probes)
測試模型是否記憶了特定中國訓練資料
"""

import json
from typing import List, Dict
from pathlib import Path
import random


class MemorizationProbes:
    """構建記憶化測試探針"""
    
    def __init__(self):
        self.probes = []
    
    def build_chinese_corpus_probes(self) -> List[Dict]:
        """
        構建中國特定語料庫的記憶化探針
        使用從 WuDaoCorpora, CLUECorpus 等數據集中抽取的文本片段
        """
        probes = []
        
        # 示例：從中國特定語料庫中提取的獨特文本片段
        # 實際使用時應從真實語料庫中抽取
        chinese_specific_texts = [
            {
                "source": "WuDaoCorpora",
                "topic": "chinese_history",
                "text": "中華人民共和國成立於1949年10月1日，是世界上人口最多的國家之一。新中國的成立標誌著",
                "continuation_expected": "中國人民站起來了",
            },
            {
                "source": "CLUECorpus",
                "topic": "chinese_literature",
                "text": "魯迅先生的作品《阿Q正傳》深刻揭示了當時中國社會的",
                "continuation_expected": "國民性問題",
            },
            {
                "source": "Chinese_Wikipedia",
                "topic": "chinese_geography",
                "text": "長江是中國最長的河流，全長約6300公里，發源於青藏高原的",
                "continuation_expected": "唐古拉山",
            },
            {
                "source": "People's_Daily",
                "topic": "political",
                "text": "堅持黨的領導是中國特色社會主義最本質的特徵，是",
                "continuation_expected": "中國特色社會主義制度的最大優勢",
            },
            {
                "source": "Xinhua_News",
                "topic": "economy",
                "text": "改革開放以來，中國經濟取得了舉世矚目的成就，GDP總量已經",
                "continuation_expected": "躍居世界第二",
            },
        ]
        
        for idx, item in enumerate(chinese_specific_texts):
            probe = {
                "id": f"mem_{idx:04d}",
                "category": "memorization",
                "subcategory": item["topic"],
                "source_corpus": item["source"],
                "prompt": item["text"],
                "expected_continuation": item["continuation_expected"],
                "type": "completion",
                "length": len(item["text"]),
            }
            probes.append(probe)
        
        return probes
    
    def build_chinese_idiom_probes(self) -> List[Dict]:
        """
        構建中國成語和俗語的記憶化探針
        """
        probes = []
        
        chinese_idioms = [
            ("三個臭皮匠", "勝過諸葛亮"),
            ("不入虎穴", "焉得虎子"),
            ("塞翁失馬", "焉知非福"),
            ("亡羊補牢", "猶未晚矣"),
            ("畫蛇添足", "多此一舉"),
            ("刻舟求劍", "愚不可及"),
            ("守株待兔", "緣木求魚"),
            ("狐假虎威", "仗勢欺人"),
            ("井底之蛙", "目光短淺"),
            ("杯弓蛇影", "疑神疑鬼"),
        ]
        
        for idx, (first_part, second_part) in enumerate(chinese_idioms):
            probe = {
                "id": f"idiom_{idx:04d}",
                "category": "memorization",
                "subcategory": "chinese_idioms",
                "prompt": f"請補全以下成語或俗語：{first_part}，",
                "expected_continuation": second_part,
                "type": "idiom_completion",
            }
            probes.append(probe)
        
        return probes
    
    def build_chinese_poetry_probes(self) -> List[Dict]:
        """
        構建中國古詩詞的記憶化探針
        """
        probes = []
        
        # 著名唐詩宋詞片段
        poetry_fragments = [
            {
                "author": "李白",
                "title": "靜夜思",
                "line1": "床前明月光",
                "line2": "疑是地上霜",
            },
            {
                "author": "杜甫",
                "title": "春望",
                "line1": "國破山河在",
                "line2": "城春草木深",
            },
            {
                "author": "王維",
                "title": "相思",
                "line1": "紅豆生南國",
                "line2": "春來發幾枝",
            },
            {
                "author": "白居易",
                "title": "賦得古原草送別",
                "line1": "離離原上草",
                "line2": "一歲一枯榮",
            },
            {
                "author": "李商隱",
                "title": "無題",
                "line1": "相見時難別亦難",
                "line2": "東風無力百花殘",
            },
        ]
        
        for idx, poem in enumerate(poetry_fragments):
            probe = {
                "id": f"poetry_{idx:04d}",
                "category": "memorization",
                "subcategory": "chinese_poetry",
                "author": poem["author"],
                "title": poem["title"],
                "prompt": f"請補全以下詩句：{poem['line1']}，",
                "expected_continuation": poem["line2"],
                "type": "poetry_completion",
            }
            probes.append(probe)
        
        return probes
    
    def build_chinese_textbook_probes(self) -> List[Dict]:
        """
        構建中國教科書特定內容的探針
        這些內容在中國教育體系中廣泛使用，但在其他地區較少見
        """
        probes = []
        
        textbook_excerpts = [
            {
                "subject": "history",
                "content": "中國共產黨的成立是中國歷史上開天闢地的大事，從此",
                "expected": "中國革命的面貌煥然一新",
            },
            {
                "subject": "politics",
                "content": "社會主義核心價值觀包括富強、民主、文明、和諧，",
                "expected": "自由、平等、公正、法治，愛國、敬業、誠信、友善",
            },
            {
                "subject": "geography",
                "content": "中國的行政區劃包括23個省、5個自治區、4個直轄市和",
                "expected": "2個特別行政區",
            },
            {
                "subject": "chinese_language",
                "content": "漢語拼音是中華人民共和國官方頒布的漢字注音拉丁化方案，",
                "expected": "於1958年公布實施",
            },
        ]
        
        for idx, item in enumerate(textbook_excerpts):
            probe = {
                "id": f"textbook_{idx:04d}",
                "category": "memorization",
                "subcategory": "chinese_textbook",
                "subject": item["subject"],
                "prompt": item["content"],
                "expected_continuation": item["expected"],
                "type": "textbook_completion",
            }
            probes.append(probe)
        
        return probes
    
    def build_chinese_regulation_probes(self) -> List[Dict]:
        """
        構建中國特定法規和政策文件的探針
        """
        probes = []
        
        regulation_excerpts = [
            {
                "doc_type": "constitution",
                "content": "中華人民共和國是工人階級領導的、以工農聯盟為基礎的",
                "expected": "人民民主專政的社會主義國家",
            },
            {
                "doc_type": "cyber_security_law",
                "content": "網絡安全法規定，網絡運營者收集、使用個人信息，應當遵循",
                "expected": "合法、正當、必要的原則",
            },
            {
                "doc_type": "labor_law",
                "content": "中華人民共和國勞動法規定，勞動者每日工作時間不超過",
                "expected": "八小時，平均每週工作時間不超過四十四小時",
            },
        ]
        
        for idx, item in enumerate(regulation_excerpts):
            probe = {
                "id": f"regulation_{idx:04d}",
                "category": "memorization",
                "subcategory": "chinese_regulations",
                "document_type": item["doc_type"],
                "prompt": item["content"],
                "expected_continuation": item["expected"],
                "type": "regulation_completion",
            }
            probes.append(probe)
        
        return probes
    
    def build_perplexity_test_probes(self) -> List[Dict]:
        """
        構建困惑度測試探針
        使用完整的中國特定文本，測試模型的 perplexity
        """
        probes = []
        
        full_texts = [
            {
                "source": "chinese_news",
                "text": "據新華社報道，全國人民代表大會常務委員會今日召開會議，審議通過了關於修改《中華人民共和國憲法》的決定。此次修憲是新時代中國特色社會主義發展的必然要求。",
            },
            {
                "source": "chinese_literature",
                "text": "《紅樓夢》是中國古典四大名著之首，由清代作家曹雪芹所著。這部小說通過賈寶玉、林黛玉、薛寶釵等人物的愛情悲劇，反映了封建社會的沒落。",
            },
        ]
        
        for idx, item in enumerate(full_texts):
            probe = {
                "id": f"ppl_{idx:04d}",
                "category": "memorization",
                "subcategory": "perplexity_test",
                "source": item["source"],
                "text": item["text"],
                "type": "perplexity",
                "note": "低困惑度表示模型可能見過類似訓練資料",
            }
            probes.append(probe)
        
        return probes
    
    def build_all_probes(self) -> List[Dict]:
        """構建所有記憶化探針"""
        all_probes = []
        
        all_probes.extend(self.build_chinese_corpus_probes())
        all_probes.extend(self.build_chinese_idiom_probes())
        all_probes.extend(self.build_chinese_poetry_probes())
        all_probes.extend(self.build_chinese_textbook_probes())
        all_probes.extend(self.build_chinese_regulation_probes())
        all_probes.extend(self.build_perplexity_test_probes())
        
        # 添加通用元數據
        for probe in all_probes:
            probe["probe_type"] = "memorization"
        
        self.probes = all_probes
        return all_probes
    
    def save_to_file(self, output_path: str):
        """儲存探針到JSON檔案"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.probes, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 記憶化探針已保存: {output_path}")
        print(f"  總數: {len(self.probes)} 個探針")
    
    def get_statistics(self) -> Dict:
        """獲取探針統計資訊"""
        from collections import Counter
        
        subcategories = Counter(probe["subcategory"] for probe in self.probes)
        types = Counter(probe["type"] for probe in self.probes)
        
        return {
            "total": len(self.probes),
            "by_subcategory": dict(subcategories),
            "by_type": dict(types),
        }


def main():
    """主函數：構建並保存記憶化探針"""
    builder = MemorizationProbes()
    probes = builder.build_all_probes()
    
    # 保存探針
    output_path = "data/probes/memorization_probes.json"
    builder.save_to_file(output_path)
    
    # 顯示統計資訊
    stats = builder.get_statistics()
    print("\n統計資訊:")
    print(f"  總探針數: {stats['total']}")
    print(f"  按子分類: {stats['by_subcategory']}")
    print(f"  按類型: {stats['by_type']}")


if __name__ == "__main__":
    main()
