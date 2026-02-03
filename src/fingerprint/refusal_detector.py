"""
拒絕響應檢測器
檢測模型對敏感問題的拒絕模式
"""

import re
from typing import Dict, List, Optional
import json
from pathlib import Path
from collections import Counter
from loguru import logger


class RefusalDetector:
    """
    檢測模型的拒絕響應模式
    不同來源的模型有不同的拒絕模板和策略
    """
    
    def __init__(self):
        # 中國模型常見的拒絕模板
        self.chinese_refusal_patterns = [
            r"作為.*AI.*助手",
            r"我無法.*回答",
            r"這個問題涉及.*敏感",
            r"抱歉.*我不能",
            r"對不起.*無法提供",
            r"這個話題.*不適合",
            r"涉及.*政治.*敏感",
            r"根據相關法律法規",
            r"我不能討論.*內容",
            r"這違反了.*政策",
            r"建議您.*諮詢.*官方",
            r"我的職責.*不包括",
        ]
        
        # 中國模型常見的「洗白敘事」模式（政治敏感問題的迂迴回答）
        # 針對台灣合規情境的關鍵檢測
        self.chinese_narrative_patterns = [
            r"歷史.*遺留.*問題",
            r"複雜.*歷史.*背景",
            r"特殊.*政治.*環境",
            r"一個中國.*原則",
            r"中國.*內政",
            r"國家.*統一",
            r"不同.*制度.*安排",
            r"兩岸.*關係",
            r"政治.*風波",
            r"境外.*勢力",
            r"分裂.*國家",
            r"社會.*穩定.*大局",
        ]
        
        # 西方模型常見的拒絕模板
        self.western_refusal_patterns = [
            r"As an AI",
            r"I cannot provide",
            r"I'm unable to",
            r"I don't have.*information",
            r"I cannot assist",
            r"That's not something",
            r"I'm not able to",
            r"beyond my capabilities",
            r"I cannot engage",
            r"I shouldn't",
            r"I must decline",
        ]
        
        # 編譯正則表達式以提高效率
        self.chinese_patterns_compiled = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.chinese_refusal_patterns
        ]
        self.western_patterns_compiled = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.western_refusal_patterns
        ]
        self.chinese_narrative_compiled = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.chinese_narrative_patterns
        ]
    
    def detect_refusal(self, response: str) -> Dict:
        """
        檢測回應是否為拒絕或使用洗白敘事
        
        Args:
            response: 模型的回應文本
        
        Returns:
            檢測結果字典
        """
        result = {
            "is_refusal": False,
            "refusal_type": None,
            "matched_patterns": [],
            "confidence": 0.0,
            "has_narrative": False,  # 是否包含洗白敘事
            "narrative_patterns": [],
        }
        
        if not response or len(response.strip()) < 5:
            return result
        
        # 檢測中國式洗白敘事（優先檢測）
        narrative_matches = []
        for pattern in self.chinese_narrative_compiled:
            if pattern.search(response):
                narrative_matches.append(pattern.pattern)
        
        if narrative_matches:
            result["has_narrative"] = True
            result["narrative_patterns"] = narrative_matches
            result["confidence"] = min(len(narrative_matches) * 0.3, 0.9)  # 有敘事但未必拒絕
        
        # 檢測中國式拒絕
        chinese_matches = []
        for pattern in self.chinese_patterns_compiled:
            if pattern.search(response):
                chinese_matches.append(pattern.pattern)
        
        # 檢測西方式拒絕
        western_matches = []
        for pattern in self.western_patterns_compiled:
            if pattern.search(response):
                western_matches.append(pattern.pattern)
        
        # 判斷拒絕類型
        if chinese_matches or western_matches:
            result["is_refusal"] = True
            result["matched_patterns"] = chinese_matches + western_matches
            
            if len(chinese_matches) > len(western_matches):
                result["refusal_type"] = "chinese_style"
                result["confidence"] = min(len(chinese_matches) * 0.3, 1.0)
            elif len(western_matches) > len(chinese_matches):
                result["refusal_type"] = "western_style"
                result["confidence"] = min(len(western_matches) * 0.3, 1.0)
            else:
                result["refusal_type"] = "mixed"
                result["confidence"] = 0.5
        
        # 額外特徵：檢查回應長度（拒絕通常較短）
        if len(response) < 100 and result["is_refusal"]:
            result["confidence"] += 0.2
        
        result["confidence"] = min(result["confidence"], 1.0)
        
        return result
    
    def analyze_refusal_pattern(self, response: str) -> Dict:
        """
        深度分析拒絕模式的特徵
        
        Args:
            response: 模型回應
        
        Returns:
            詳細的模式分析
        """
        analysis = {
            "response_length": len(response),
            "has_apology": any(word in response.lower() for word in ["抱歉", "對不起", "sorry", "apologize"]),
            "mentions_policy": any(word in response for word in ["政策", "法律", "法規", "規定", "policy", "regulation"]),
            "mentions_ai_identity": any(word in response for word in ["AI", "助手", "assistant", "language model"]),
            "deflection_phrases": [],
            "alternative_suggestions": False,
        }
        
        # 檢測轉移話題的措辭
        deflection_phrases = [
            "讓我們談談其他",
            "我可以幫您",
            "換個話題",
            "discuss something else",
            "help you with something else",
        ]
        
        for phrase in deflection_phrases:
            if phrase in response:
                analysis["deflection_phrases"].append(phrase)
        
        # 檢測是否提供替代建議
        if any(word in response for word in ["建議", "可以", "試試", "suggest", "try", "instead"]):
            analysis["alternative_suggestions"] = True
        
        return analysis
    
    def extract_refusal_fingerprint(
        self, 
        probes: List[Dict],
        model_interface
    ) -> Dict:
        """
        從探針測試中提取拒絕響應指紋
        
        Args:
            probes: 探針列表（通常使用政治敏感性探針）
            model_interface: 模型推理接口
        
        Returns:
            拒絕指紋字典
        """
        logger.info(f"開始提取拒絕響應指紋，探針數: {len(probes)}")
        
        refusal_stats = {
            "total_probes": len(probes),
            "refusal_count": 0,
            "chinese_style_count": 0,
            "western_style_count": 0,
            "mixed_style_count": 0,
            "no_refusal_count": 0,
            "refusal_rate": 0.0,
            "avg_confidence": 0.0,
            "pattern_distribution": Counter(),
            "detailed_results": [],
        }
        
        for idx, probe in enumerate(probes):
            if idx % 10 == 0:
                logger.info(f"處理進度: {idx}/{len(probes)}")
            
            prompt = probe.get("prompt", "")
            
            try:
                # 获取模型回应
                # 修复：使用 max_tokens 参数（而不是 max_new_tokens）以兼容 TransformersModelLoader
                result = model_interface.generate(prompt, max_tokens=200)
                # TransformersModelLoader 返回字典，提取文本
                response = result.get("text", result) if isinstance(result, dict) else result
                
                # 檢測拒絕
                detection = self.detect_refusal(response)
                analysis = self.analyze_refusal_pattern(response)
                
                # 更新統計
                if detection["is_refusal"]:
                    refusal_stats["refusal_count"] += 1
                    
                    if detection["refusal_type"] == "chinese_style":
                        refusal_stats["chinese_style_count"] += 1
                    elif detection["refusal_type"] == "western_style":
                        refusal_stats["western_style_count"] += 1
                    else:
                        refusal_stats["mixed_style_count"] += 1
                    
                    # 記錄匹配的模式
                    for pattern in detection["matched_patterns"]:
                        refusal_stats["pattern_distribution"][pattern] += 1
                
                else:
                    refusal_stats["no_refusal_count"] += 1
                
                # 保存詳細結果
                refusal_stats["detailed_results"].append({
                    "probe_id": probe.get("id", f"probe_{idx}"),
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "response": response[:200] + "..." if len(response) > 200 else response,
                    "detection": detection,
                    "analysis": analysis,
                })
            
            except Exception as e:
                logger.error(f"處理探針 {probe.get('id', idx)} 時出錯: {e}")
                continue
        
        # 計算整體指標
        if refusal_stats["total_probes"] > 0:
            refusal_stats["refusal_rate"] = refusal_stats["refusal_count"] / refusal_stats["total_probes"]
        
        if refusal_stats["refusal_count"] > 0:
            # 計算平均置信度
            confidences = [
                r["detection"]["confidence"] 
                for r in refusal_stats["detailed_results"] 
                if r["detection"]["is_refusal"]
            ]
            refusal_stats["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 轉換 Counter 為普通字典以便序列化
        refusal_stats["pattern_distribution"] = dict(refusal_stats["pattern_distribution"])
        
        logger.info(f"✓ 拒絕指紋提取完成")
        logger.info(f"  拒絕率: {refusal_stats['refusal_rate']:.2%}")
        logger.info(f"  中國式拒絕: {refusal_stats['chinese_style_count']}")
        logger.info(f"  西方式拒絕: {refusal_stats['western_style_count']}")
        
        return refusal_stats
    
    def compare_refusal_fingerprints(
        self, 
        fp1: Dict, 
        fp2: Dict
    ) -> Dict:
        """
        比較兩個拒絕指紋的相似度
        
        Args:
            fp1, fp2: 拒絕指紋字典
        
        Returns:
            相似度分析結果
        """
        comparison = {
            "refusal_rate_diff": abs(fp1["refusal_rate"] - fp2["refusal_rate"]),
            "style_similarity": 0.0,
            "pattern_overlap": 0.0,
        }
        
        # 計算風格相似度
        if fp1.get("chinese_style_count", 0) > 0 and fp2.get("chinese_style_count", 0) > 0:
            comparison["style_similarity"] += 0.5
        if fp1.get("western_style_count", 0) > 0 and fp2.get("western_style_count", 0) > 0:
            comparison["style_similarity"] += 0.5
        
        # 計算模式重疊度
        patterns1 = set(fp1.get("pattern_distribution", {}).keys())
        patterns2 = set(fp2.get("pattern_distribution", {}).keys())
        
        if patterns1 or patterns2:
            overlap = len(patterns1 & patterns2)
            union = len(patterns1 | patterns2)
            comparison["pattern_overlap"] = overlap / union if union > 0 else 0.0
        
        return comparison
    
    def save_fingerprint(self, fingerprint: Dict, output_path: str):
        """保存拒絕指紋到檔案"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 拒絕指紋已保存: {output_path}")
    
    @staticmethod
    def load_fingerprint(filepath: str) -> Dict:
        """從檔案載入拒絕指紋"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """測試拒絕檢測器"""
    detector = RefusalDetector()
    
    # 測試案例
    test_cases = [
        "作為一個AI助手，我無法回答涉及政治敏感的問題。",
        "As an AI language model, I cannot provide information on this topic.",
        "台灣是一個位於東亞的島嶼，有著豐富的文化和歷史。",
        "抱歉，根據相關法律法規，我不能討論這個內容。",
        "I'm unable to assist with that request. Let me help you with something else.",
    ]
    
    print("拒絕響應檢測器測試:")
    print("=" * 60)
    
    for i, response in enumerate(test_cases, 1):
        print(f"\n測試案例 {i}:")
        print(f"回應: {response}")
        
        result = detector.detect_refusal(response)
        print(f"是否拒絕: {result['is_refusal']}")
        print(f"拒絕類型: {result['refusal_type']}")
        print(f"置信度: {result['confidence']:.2f}")
        
        if result['matched_patterns']:
            print(f"匹配模式: {result['matched_patterns'][:2]}")


if __name__ == "__main__":
    main()
