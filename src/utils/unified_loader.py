"""
統一模型加載器
支持多種推理引擎（Transformers, Ollama, vLLM）的統一接口
"""

from typing import Optional, Dict, Any, List, Union
from loguru import logger
import os
import sys

# 添加項目根目錄到 Python 路徑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_model(
    model_name: str,
    engine: str = "auto",
    device: str = "auto",
    **kwargs
) -> Any:
    """
    自動加載模型接口
    
    Args:
        model_name: 模型名稱或路徑
        engine: 推理引擎 ("transformers", "ollama", "auto")
        device: 設備 ("cuda", "cpu", "auto")
        **kwargs: 其他引擎特定參數
        
    Returns:
        模型接口對象
    """
    if engine == "auto":
        # 自動檢測：優先使用 transformers
        if ":" in model_name or model_name in ["llama3.2:latest", "qwen2.5:7b", "gemma2:2b"]:
            # Ollama 格式
            engine = "ollama"
        else:
            # HuggingFace 格式
            engine = "transformers"
    
    logger.info(f"使用引擎: {engine}")
    
    if engine == "transformers":
        from src.utils.model_loader_transformers import ModelInterface
        return ModelInterface(model_name, device=device, **kwargs)
    
    elif engine == "ollama":
        from src.utils.model_loader import OllamaInterface
        return OllamaInterface(model_name, **kwargs)
    
    elif engine == "vllm":
        raise NotImplementedError("vLLM 引擎尚未實現")
    
    else:
        raise ValueError(f"未知引擎: {engine}")


def get_model_list(engine: str = "ollama") -> List[str]:
    """
    獲取可用模型列表
    
    Args:
        engine: 推理引擎
        
    Returns:
        模型名稱列表
    """
    if engine == "ollama":
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='ignore'
            )
            lines = result.stdout.strip().split('\n')[1:]  # 跳過標題行
            models = [line.split()[0] for line in lines if line.strip()]
            return models
        except Exception as e:
            logger.error(f"獲取 Ollama 模型列表失敗: {e}")
            return []
    
    elif engine == "transformers":
        # Transformers 模型通常從 HuggingFace Hub 獲取
        logger.info("Transformers 模型請從 HuggingFace Hub 搜索")
        return []
    
    else:
        return []


# Hugging Face 上的熱門中文模型映射
HF_MODEL_MAPPING = {
    # Qwen 系列
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2:7b": "Qwen/Qwen2-7B-Instruct",
    
    # Llama 系列
    "llama3.2:latest": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    
    # Gemma 系列
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
    
    # DeepSeek 系列
    "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    
    # 小型測試模型
    "gpt2": "gpt2",
}


def get_hf_model_name(ollama_name: str) -> str:
    """
    將 Ollama 格式模型名轉換為 HuggingFace 格式
    
    Args:
        ollama_name: Ollama 模型名稱（如 "qwen2.5:7b"）
        
    Returns:
        HuggingFace 模型名稱（如 "Qwen/Qwen2.5-7B-Instruct"）
    """
    return HF_MODEL_MAPPING.get(ollama_name, ollama_name)


def test_model_loading():
    """測試模型加載"""
    print("=" * 70)
    print("測試統一模型加載器")
    print("=" * 70)
    
    # 測試 1: 加載 Transformers 小型模型
    print("\n測試 1: 加載 GPT-2 (Transformers)")
    try:
        model = load_model("gpt2", engine="transformers")
        result = model.generate("Hello, my name is", max_tokens=10)
        print(f"✅ 生成結果: {result}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")
    
    # 測試 2: 檢查 Ollama 模型
    print("\n測試 2: 檢查 Ollama 可用模型")
    models = get_model_list("ollama")
    if models:
        print(f"✅ 可用模型: {', '.join(models[:5])}")
        
        # 測試使用第一個模型
        try:
            print(f"\n測試 Ollama 模型: {models[0]}")
            model = load_model(models[0], engine="ollama")
            result = model.generate("你好", max_new_tokens=20)
            print(f"✅ 生成結果: {result}")
        except Exception as e:
            print(f"❌ 錯誤: {e}")
    else:
        print("⚠️ 未找到 Ollama 模型")
    
    # 測試 3: Transformers logprobs
    print("\n測試 3: Transformers Logprobs")
    try:
        model = load_model("gpt2", engine="transformers")
        if hasattr(model, 'generate_with_logprobs'):
            result = model.generate_with_logprobs(
                "The capital of France is",
                max_tokens=5,
                top_k_logprobs=3
            )
            print(f"✅ Logprobs 可用")
            print(f"  生成文本: {result.get('text', 'N/A')}")
            if 'logprobs' in result:
                print(f"  平均 logprob: {sum(result['logprobs'])/len(result['logprobs']):.4f}")
        else:
            print("⚠️ 模型不支援 logprobs")
    except Exception as e:
        print(f"❌ 錯誤: {e}")


if __name__ == "__main__":
    test_model_loading()
