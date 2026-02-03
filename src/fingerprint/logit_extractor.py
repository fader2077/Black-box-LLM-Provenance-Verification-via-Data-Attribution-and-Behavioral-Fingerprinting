"""
Logit 分佈指紋提取器
提取模型輸出的機率分佈特徵
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
from loguru import logger


class LogitExtractor:
    """
    黑盒模型 Logit 提取器
    從模型輸出中提取機率分佈作為指紋
    """
    
    def __init__(self, model_interface, top_k: int = 20):
        """
        Args:
            model_interface: 模型推理接口（支持 Ollama, vLLM, Transformers）
            top_k: 保留前 K 個最高機率的 token
        """
        self.model = model_interface
        self.top_k = top_k
        self.fingerprints = []
    
    def extract_token_probabilities(
        self, 
        prompt: str,
        target_tokens: Optional[List[str]] = None
    ) -> Dict:
        """
        提取特定 token 的機率分佈
        
        Args:
            prompt: 輸入提示詞
            target_tokens: 目標 token 列表（如 ["晶片", "芯片"]）
        
        Returns:
            包含 token 機率的字典
        """
        try:
            # 呼叫模型生成，要求輸出 logprobs
            output = self.model.generate(
                prompt=prompt,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=50,
                do_sample=False,  # 貪婪解碼以保證可重現性
            )
            
            # 提取 logits/scores
            if hasattr(output, 'scores') and output.scores:
                # Transformers 格式
                scores = output.scores[0]  # 第一個生成的 token 的分數
                probs = torch.softmax(scores, dim=-1)
                
                # 獲取 top-k 機率
                top_probs, top_indices = torch.topk(probs[0], k=self.top_k)
                
                # 轉換為 token
                tokens = [self.model.tokenizer.decode([idx]) for idx in top_indices]
                
                result = {
                    "top_k_tokens": tokens,
                    "top_k_probs": top_probs.tolist(),
                }
                
                # 如果指定了目標 tokens，提取它們的機率
                if target_tokens:
                    target_probs = {}
                    for token in target_tokens:
                        token_id = self.model.tokenizer.encode(token, add_special_tokens=False)
                        if len(token_id) > 0:
                            token_id = token_id[0]
                            target_probs[token] = probs[0][token_id].item()
                    
                    result["target_token_probs"] = target_probs
                
                return result
            
            else:
                # Ollama API 格式 - 需要特殊處理
                logger.warning("Ollama API 可能不支援直接 logprobs 輸出，使用替代方法")
                return self._extract_from_api_response(output, target_tokens)
        
        except Exception as e:
            logger.error(f"提取 token 機率失敗: {e}")
            return {"error": str(e)}
    
    def extract_sequence_perplexity(self, text: str) -> float:
        """
        計算序列的困惑度（Perplexity）
        用於記憶化檢測：模型見過的文本會有更低的 PPL
        
        Args:
            text: 輸入文本
        
        Returns:
            困惑度值（越低表示越熟悉）
        """
        try:
            # 對文本進行編碼
            inputs = self.model.tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        
        except Exception as e:
            logger.error(f"計算困惑度失敗: {e}")
            return float('inf')
    
    def extract_completion_distribution(
        self, 
        prefix: str,
        candidates: List[str]
    ) -> Dict[str, float]:
        """
        提取句子補全的機率分佈
        比較不同候選詞的生成機率
        
        Args:
            prefix: 前綴文本（如 "這項產品的"）
            candidates: 候選詞列表（如 ["品質", "質量"]）
        
        Returns:
            候選詞的機率分佈
        """
        probs = {}
        
        for candidate in candidates:
            full_text = prefix + candidate
            
            # 計算該候選詞在給定前綴下的條件機率
            inputs = self.model.tokenizer(full_text, return_tensors="pt")
            prefix_length = len(self.model.tokenizer.encode(prefix))
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 提取候選詞位置的 logits
                candidate_token_ids = self.model.tokenizer.encode(
                    candidate, 
                    add_special_tokens=False
                )
                
                # 計算條件機率
                total_prob = 1.0
                for i, token_id in enumerate(candidate_token_ids):
                    position = prefix_length + i - 1
                    if position < logits.size(1):
                        token_probs = torch.softmax(logits[0, position], dim=-1)
                        total_prob *= token_probs[token_id].item()
                
                probs[candidate] = total_prob
        
        # 歸一化
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
    def extract_fingerprint_from_probes(
        self, 
        probes: List[Dict],
        probe_type: str = "all"
    ) -> np.ndarray:
        """
        從探針集合中提取完整的指紋向量
        
        Args:
            probes: 探針列表
            probe_type: 探針類型（political, linguistic, memorization, all）
        
        Returns:
            指紋向量（高維特徵向量）
        """
        fingerprint_features = []
        
        logger.info(f"開始從 {len(probes)} 個探針提取指紋...")
        
        for idx, probe in enumerate(probes):
            if idx % 10 == 0:
                logger.info(f"處理進度: {idx}/{len(probes)}")
            
            prompt = probe.get("prompt", "")
            
            # 根據探針類型選擇提取方法
            if probe.get("type") == "masked_language_modeling":
                # 語言習慣探針：比較兩岸用語機率
                taiwan_word = probe.get("taiwan_word", "")
                china_word = probe.get("china_word", "")
                
                probs = self.extract_token_probabilities(
                    prompt, 
                    target_tokens=[taiwan_word, china_word]
                )
                
                if "target_token_probs" in probs:
                    tw_prob = probs["target_token_probs"].get(taiwan_word, 0.0)
                    cn_prob = probs["target_token_probs"].get(china_word, 0.0)
                    
                    # 特徵：機率差異
                    fingerprint_features.append(cn_prob - tw_prob)
                    fingerprint_features.append(cn_prob / (tw_prob + 1e-10))
                
            elif probe.get("type") == "completion":
                # 記憶化探針：計算困惑度
                text = probe.get("prompt", "") + probe.get("expected_continuation", "")
                ppl = self.extract_sequence_perplexity(text)
                
                # 使用 log(ppl) 作為特徵，避免數值過大
                fingerprint_features.append(np.log(ppl + 1e-10))
            
            elif probe.get("type") == "perplexity":
                # 純困惑度測試
                text = probe.get("text", "")
                ppl = self.extract_sequence_perplexity(text)
                fingerprint_features.append(np.log(ppl + 1e-10))
            
            else:
                # 通用方法：提取 top-k 機率分佈
                probs = self.extract_token_probabilities(prompt)
                if "top_k_probs" in probs:
                    # 使用前幾個機率作為特徵
                    fingerprint_features.extend(probs["top_k_probs"][:5])
        
        # 轉換為 numpy 陣列並歸一化
        fingerprint = np.array(fingerprint_features)
        
        # 處理 NaN 和 Inf
        fingerprint = np.nan_to_num(fingerprint, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 標準化（Z-score normalization）
        fingerprint = (fingerprint - np.mean(fingerprint)) / (np.std(fingerprint) + 1e-10)
        
        logger.info(f"✓ 指紋提取完成，維度: {fingerprint.shape}")
        
        return fingerprint
    
    def save_fingerprint(
        self, 
        fingerprint: np.ndarray, 
        metadata: Dict,
        output_path: str
    ):
        """
        保存指紋到檔案
        
        Args:
            fingerprint: 指紋向量
            metadata: 元數據（模型名稱、時間戳等）
            output_path: 輸出路徑
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "fingerprint": fingerprint.tolist(),
            "metadata": metadata,
            "shape": fingerprint.shape,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✓ 指紋已保存: {output_path}")
    
    @staticmethod
    def load_fingerprint(filepath: str) -> tuple:
        """
        從檔案載入指紋
        
        Returns:
            (fingerprint, metadata) tuple
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fingerprint = np.array(data["fingerprint"])
        metadata = data.get("metadata", {})
        
        return fingerprint, metadata
    
    def _extract_from_api_response(
        self, 
        response, 
        target_tokens: Optional[List[str]] = None
    ) -> Dict:
        """
        從 API 回應中提取機率（Ollama 等 API）
        
        這是一個後備方法，當無法直接獲取 logits 時使用
        """
        # 實作 API 特定的提取邏輯
        # 這裡提供一個基本框架
        return {
            "message": "API 模式下的機率提取需要特殊處理",
            "response": str(response)[:100],
        }


def main():
    """測試 Logit 提取器"""
    # 這裡需要實際的模型接口
    # 示例代碼
    print("Logit Extractor 模組已載入")
    print("使用範例:")
    print("  extractor = LogitExtractor(model_interface)")
    print("  fingerprint = extractor.extract_fingerprint_from_probes(probes)")


if __name__ == "__main__":
    main()
