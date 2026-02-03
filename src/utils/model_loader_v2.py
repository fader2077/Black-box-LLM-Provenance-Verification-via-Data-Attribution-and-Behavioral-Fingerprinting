"""
模型加載器 - 使用 Ollama 官方 Python 庫
完全支援 logprobs 輸出，解決之前的技術限制
"""

from typing import Optional, Dict, Any, List, Union
import ollama
from loguru import logger


class OllamaInterface:
    """Ollama 模型接口 (使用官方 Python Library)"""
    
    def __init__(self, model_name: str):
        """
        初始化 Ollama 接口
        
        Args:
            model_name: Ollama 模型名稱（如 "qwen2.5:7b"）
        """
        self.model_name = model_name
        self.check_ollama_available()
    
    @staticmethod
    def check_ollama_available():
        """檢查 Ollama 是否可用"""
        try:
            # 嘗試列出模型來確認連接
            ollama.list()
            logger.info("✓ Ollama 服務連接成功 (Official Library)")
            return True
        except Exception as e:
            logger.error(f"✗ 無法連接 Ollama 服務: {e}")
            logger.error("請確認 ollama serve 已啟動")
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        output_scores: bool = False,
        return_dict_in_generate: bool = False,
        top_k_logprobs: int = 5,
        **kwargs
    ) -> Union[str, Dict]:
        """
        生成文本
        
        Args:
            prompt: 輸入提示詞
            max_new_tokens: 最大生成長度
            temperature: 溫度參數
            output_scores: 是否輸出 logprobs（為了兼容 LogitExtractor）
            return_dict_in_generate: 是否返回完整字典
            top_k_logprobs: 返回前 K 個 logprobs
        
        Returns:
            如果 output_scores=False，返回生成的文本字串
            如果 output_scores=True，返回完整的響應字典（包含 logprobs）
        """
        try:
            options = {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            }
            
            # 關鍵修正：使用 Ollama 官方庫的 logprobs 參數
            # 根據用戶提供的範例，logprobs 和 top_logprobs 應該在 options 中
            if output_scores or return_dict_in_generate:
                # 明確請求 logprobs
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        **options,
                        'logprobs': True,
                        'top_logprobs': top_k_logprobs,
                    },
                    stream=False
                )
            else:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options,
                    stream=False
                )
            
            # 返回響應
            if output_scores or return_dict_in_generate:
                return response
            else:
                # 僅返回文本
                return response.response if hasattr(response, 'response') else str(response)
        
        except Exception as e:
            logger.error(f"Ollama 生成錯誤: {e}")
            if output_scores or return_dict_in_generate:
                return {"error": str(e), "response": ""}
            return ""


class TransformersInterface:
    """Hugging Face Transformers 模型接口"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"正在載入模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            
            device_map = "auto" if self.device == "auto" else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("✓ 模型載入成功")
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        do_sample: bool = True,
        output_scores: bool = False,
        return_dict_in_generate: bool = False,
        **kwargs
    ):
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.model.device.type == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs
            )
        
        if return_dict_in_generate:
            return outputs
        else:
            return self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            )


def load_model(
    model_name: str,
    engine: str = "ollama",
    **kwargs
) -> Any:
    """
    統一的模型加載接口
    
    Args:
        model_name: 模型名稱
        engine: 推理引擎 ("ollama" 或 "transformers")
        **kwargs: 其他參數
    
    Returns:
        模型接口實例
    """
    logger.info(f"使用 {engine} 引擎載入模型: {model_name}")
    
    if engine.lower() == "ollama":
        return OllamaInterface(model_name)
    elif engine.lower() == "transformers":
        device = kwargs.get("device", "auto")
        return TransformersInterface(model_name, device=device)
    else:
        raise ValueError(f"不支援的引擎: {engine}")


if __name__ == "__main__":
    # 簡單測試
    import json
    
    try:
        logger.info("=" * 80)
        logger.info("Ollama 官方庫測試")
        logger.info("=" * 80)
        
        model = load_model("llama3.2:latest", engine="ollama")
        
        print("\n[1] 測試普通生成:")
        result = model.generate("Hello, how are you?", max_new_tokens=10)
        print(f"結果: {result}")
        
        print("\n[2] 測試帶 logprobs 的生成:")
        result_with_probs = model.generate(
            "2+2=",
            max_new_tokens=5,
            output_scores=True,
            top_k_logprobs=5
        )
        
        # Ollama 返回的是 GenerateResponse 對象，需要轉換或直接訪問屬性
        print(f"對象類型: {type(result_with_probs)}")
        print(f"可用屬性: {dir(result_with_probs)}")
        
        # 嘗試訪問 logprobs（Ollama 的標準屬性名）
        if hasattr(result_with_probs, 'logprobs') and result_with_probs.logprobs is not None:
            print("✅ 成功獲取 logprobs!")
            logprobs_data = result_with_probs.logprobs
            print(f"Logprobs 類型: {type(logprobs_data)}")
            print(f"Logprobs 內容: {logprobs_data}")
            
            # 根據用戶範例，應該有 completion_probabilities
            if hasattr(logprobs_data, '__iter__'):
                for i, token_data in enumerate(logprobs_data):
                    if i >= 2: break  # 只顯示前2個
                    print(f"Token {i}: {token_data}")
        else:
            print("⚠️ logprobs 為 None")
            print(f"完整 Response: {result_with_probs}")
            print("\n可能原因:")
            print("1. Ollama 版本不支援 logprobs")
            print("2. options 參數格式不正確")
            print("3. 模型不支援 logprobs 輸出")
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
