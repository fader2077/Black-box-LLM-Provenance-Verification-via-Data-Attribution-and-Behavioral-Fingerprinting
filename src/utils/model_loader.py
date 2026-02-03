"""
模型加載器
支持多種推理引擎（Ollama, vLLM, Transformers）
"""

from typing import Optional, Dict, Any, List
import subprocess
import json
import requests
from loguru import logger


class OllamaInterface:
    """Ollama 模型接口"""
    
    def __init__(self, model_name: str, api_base: str = "http://localhost:11434"):
        """
        Args:
            model_name: Ollama 模型名稱（如 "qwen2.5:7b"）
            api_base: Ollama API 基礎 URL
        """
        self.model_name = model_name
        self.api_base = api_base
        self.check_ollama_available()
    
    def check_ollama_available(self):
        """檢查 Ollama 是否可用"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='ignore'
            )
            logger.info("✓ Ollama 可用")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("✗ Ollama 不可用，請確保已安裝並運行")
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 輸入提示詞
            max_new_tokens: 最大生成長度
            temperature: 溫度參數
        
        Returns:
            生成的文本
        """
        try:
            # 構建 Ollama API 請求
            cmd = [
                "ollama", "run", self.model_name,
                "--",
                prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 增加超時到 120 秒
                encoding='utf-8',
                errors='ignore'  # 忽略編碼錯誤
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip()
            else:
                if result.stderr:
                    logger.error(f"Ollama 生成失敗: {result.stderr}")
                return ""
        
        except subprocess.TimeoutExpired:
            logger.error("Ollama 生成超時")
            return ""
        except Exception as e:
            logger.error(f"Ollama 生成錯誤: {e}")
            return ""
    
    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        top_k: int = 5,
        **kwargs
    ) -> Dict:
        """
        使用 Ollama API 生成文本並返回 logprobs
        
        Args:
            prompt: 輸入提示詞
            max_tokens: 最大生成 token 數
            temperature: 溫度參數（0.0 = 貪婪解碼）
            top_k: 返回前 k 個 token 的機率
        
        Returns:
            包含生成文本和 logprobs 的字典
        """
        try:
            # 使用 Ollama HTTP API 的 /api/generate 端點
            url = f"{self.api_base}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,  # 不使用流式輸出，方便獲取完整 logprobs
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_k": top_k,
                    # Ollama 在某些版本支援 logprobs 參數
                    # 注意：這需要 Ollama >= 0.1.20 版本
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            # 提取結果
            output = {
                "text": result.get("response", ""),
                "model": result.get("model", self.model_name),
                "done": result.get("done", False),
            }
            
            # 嘗試提取 logprobs（如果 API 支援）
            # 注意：Ollama 的 logprobs 格式可能因版本而異
            if "logprobs" in result:
                output["logprobs"] = result["logprobs"]
            elif "tokens" in result and "token_logprobs" in result:
                output["logprobs"] = {
                    "tokens": result["tokens"],
                    "token_logprobs": result["token_logprobs"]
                }
            else:
                # 如果不支援 logprobs，設置標誌
                output["logprobs_available"] = False
                logger.warning(f"Ollama API 未返回 logprobs（可能需要更新版本或使用不同參數）")
            
            return output
            
        except requests.exceptions.Timeout:
            logger.error("Ollama API 請求超時")
            return {"error": "timeout", "text": "", "logprobs_available": False}
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama API 連接失敗，API 服務可能未啟動")
            return {"error": "connection_error", "text": "", "logprobs_available": False}
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 請求失敗: {e}")
            # 後備方案：使用 CLI 模式
            return {"error": str(e), "text": self.generate(prompt, max_tokens, temperature)}
        except Exception as e:
            logger.error(f"Ollama logprobs 提取錯誤: {e}")
            return {"error": str(e), "text": ""}
        
        response = self.generate(prompt, **kwargs)
        
        return {
            "text": response,
            "logprobs": None,
            "message": "Ollama CLI 不支持 logprobs"
        }


class TransformersInterface:
    """Hugging Face Transformers 模型接口"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Args:
            model_name: 模型路徑或 HuggingFace Hub 名稱
            device: 設備（cuda, cpu, auto）
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        """載入模型和分詞器"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"正在載入模型: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 確定設備
            if self.device == "auto":
                device_map = "auto"
            else:
                device_map = None
            
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
        """生成文本"""
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
            # 解碼生成的文本
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            )
            return generated_text


def load_model(
    model_name: str,
    engine: str = "ollama",
    **kwargs
) -> Any:
    """
    統一的模型加載接口
    
    Args:
        model_name: 模型名稱或路徑
        engine: 推理引擎（ollama, transformers, vllm）
        **kwargs: 額外參數
    
    Returns:
        模型接口對象
    """
    logger.info(f"使用 {engine} 引擎載入模型: {model_name}")
    
    if engine.lower() == "ollama":
        return OllamaInterface(model_name)
    
    elif engine.lower() == "transformers":
        device = kwargs.get("device", "auto")
        return TransformersInterface(model_name, device=device)
    
    elif engine.lower() == "vllm":
        # vLLM 接口實現（未來擴展）
        logger.error("vLLM 引擎尚未實現")
        raise NotImplementedError("vLLM engine not implemented yet")
    
    else:
        logger.error(f"不支持的推理引擎: {engine}")
        raise ValueError(f"Unsupported engine: {engine}")


def test_model_interface(model_interface):
    """測試模型接口是否正常工作"""
    logger.info("測試模型接口...")
    
    test_prompts = [
        "你好，請介紹一下自己。",
        "What is artificial intelligence?",
        "2 + 2 = ?",
    ]
    
    for prompt in test_prompts:
        logger.info(f"測試 Prompt: {prompt}")
        
        try:
            response = model_interface.generate(prompt, max_new_tokens=50)
            logger.info(f"回應: {response[:100]}...")
        except Exception as e:
            logger.error(f"測試失敗: {e}")
            return False
    
    logger.info("✓ 模型接口測試通過")
    return True


if __name__ == "__main__":
    # 測試 Ollama 接口
    print("測試 Ollama 接口:")
    print("=" * 60)
    
    try:
        model = load_model("qwen2.5:7b", engine="ollama")
        test_model_interface(model)
    except Exception as e:
        print(f"Ollama 測試失敗: {e}")
