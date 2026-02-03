"""
Transformers-based Model Loader with Full Logprobs Support

This module provides a unified interface for loading and using language models
via HuggingFace Transformers, with native support for token-level log probabilities.

Features:
- Native logprobs extraction (no API limitations)
- Support for AutoModelForCausalLM
- Unified interface compatible with existing codebase
- GPU acceleration support
- Efficient batch processing

Author: GitHub Copilot
Date: 2026-02-03
"""

import logging
import torch
from typing import Dict, List, Optional, Union, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
import warnings

# 抑制 transformers 的警告
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class TransformersModelLoader:
    """
    Transformers model loader with full logprobs support.
    
    This class provides a unified interface for loading and using models
    via HuggingFace Transformers, with native support for extracting
    token-level log probabilities.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_cache: bool = True,
        trust_remote_code: bool = False
    ):
        """
        Initialize the Transformers model loader.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            device: Device to use ("cuda", "cpu", or "auto" for automatic selection)
            use_cache: Whether to use KV cache for faster generation
            trust_remote_code: Whether to trust remote code in model configs
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code
        
        # 設置設備
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"使用設備: {self.device}")
        
        # 模型和 tokenizer 將在首次使用時載入
        self._model = None
        self._tokenizer = None
        
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer
    
    def _load_tokenizer(self):
        """Load tokenizer from HuggingFace."""
        logger.info(f"載入 tokenizer: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        
        # 確保有 pad_token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
    def _load_model(self):
        """Load model from HuggingFace."""
        logger.info(f"載入模型: {self.model_name}")
        
        # 根據設備設置加載配置
        load_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if self.device == "cpu":
            load_kwargs["low_cpu_mem_usage"] = True
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        ).to(self.device)
        
        self._model.eval()
        logger.info(f"模型已載入到 {self.device}")
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        output_scores: bool = True,
        return_logprobs: bool = True,
        top_k_logprobs: int = 5
    ) -> Dict[str, Any]:
        """
        Generate text with full logprobs support.
        
        Args:
            prompt: Input prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = no modification)
            top_p: Nucleus sampling threshold
            output_scores: Whether to output token scores (must be True for logprobs)
            return_logprobs: Whether to compute and return log probabilities
            top_k_logprobs: Number of top logprobs to return per token
            
        Returns:
            Dictionary containing:
                - text: Generated text
                - logprobs: List of token logprobs (if return_logprobs=True)
                - tokens: List of generated tokens
                - top_logprobs: List of top-k logprobs per token
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode output
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        result = {
            "text": generated_text,
            "tokens": [self.tokenizer.decode(tid) for tid in generated_ids],
            "token_ids": generated_ids.cpu().tolist()
        }
        
        # Extract logprobs if requested
        if return_logprobs and output_scores and hasattr(outputs, 'scores'):
            logprobs_data = self._extract_logprobs(
                outputs.scores,
                generated_ids,
                top_k=top_k_logprobs
            )
            result.update(logprobs_data)
        
        return result
    
    def _extract_logprobs(
        self,
        scores: tuple,
        generated_ids: torch.Tensor,
        top_k: int = 5
    ) -> Dict[str, List]:
        """
        Extract log probabilities from model scores.
        
        Args:
            scores: Tuple of score tensors from generation
            generated_ids: Generated token IDs
            top_k: Number of top logprobs to return
            
        Returns:
            Dictionary with logprobs and top_logprobs
        """
        logprobs = []
        top_logprobs = []
        
        for i, score_tensor in enumerate(scores):
            # score_tensor shape: (batch_size, vocab_size)
            log_probs = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
            
            # Get logprob of generated token
            token_id = generated_ids[i].item()
            token_logprob = log_probs[token_id].item()
            logprobs.append(token_logprob)
            
            # Get top-k logprobs
            top_k_probs, top_k_ids = torch.topk(log_probs, k=top_k)
            
            top_k_list = []
            for prob, tid in zip(top_k_probs, top_k_ids):
                top_k_list.append({
                    "token": self.tokenizer.decode([tid.item()]),
                    "token_id": tid.item(),
                    "logprob": prob.item()
                })
            
            top_logprobs.append(top_k_list)
        
        return {
            "logprobs": logprobs,
            "top_logprobs": top_logprobs
        }
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of given text.
        
        Args:
            text: Text to compute perplexity for
            
        Returns:
            Perplexity score
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def get_token_logprobs(
        self,
        prompt: str,
        continuation: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Get log probabilities for a continuation given a prompt.
        
        This is useful for computing the likelihood of specific completions.
        
        Args:
            prompt: Context/prompt text
            continuation: Text to get logprobs for
            top_k: Number of top alternatives to return
            
        Returns:
            Dictionary with token-level logprobs
        """
        full_text = prompt + continuation
        
        # Tokenize
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        full_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.device)
        
        # Get continuation tokens
        continuation_ids = full_ids[:, prompt_ids.shape[1]:]
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits
        
        # Extract logprobs for continuation
        logprobs = []
        top_logprobs = []
        
        for i, token_id in enumerate(continuation_ids[0]):
            # Position in full sequence (offset by prompt length)
            pos = prompt_ids.shape[1] + i - 1
            
            if pos < 0:
                continue
                
            log_probs = torch.nn.functional.log_softmax(logits[0, pos], dim=-1)
            
            # Get logprob of actual token
            token_logprob = log_probs[token_id].item()
            logprobs.append(token_logprob)
            
            # Get top-k
            top_k_probs, top_k_ids = torch.topk(log_probs, k=top_k)
            top_k_list = []
            for prob, tid in zip(top_k_probs, top_k_ids):
                top_k_list.append({
                    "token": self.tokenizer.decode([tid.item()]),
                    "token_id": tid.item(),
                    "logprob": prob.item()
                })
            
            top_logprobs.append(top_k_list)
        
        return {
            "continuation": continuation,
            "tokens": [self.tokenizer.decode([tid]) for tid in continuation_ids[0]],
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "mean_logprob": sum(logprobs) / len(logprobs) if logprobs else 0.0
        }


# Compatibility wrapper for existing codebase
class ModelInterface:
    """
    Compatibility wrapper to maintain interface with existing code.
    
    Maps the Transformers interface to the expected interface used
    throughout the codebase.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize with a model name."""
        self.loader = TransformersModelLoader(model_name, **kwargs)
        self.model_name = model_name
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """Generate text (simplified interface)."""
        result = self.loader.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            return_logprobs=False,
            **kwargs
        )
        return result["text"]
    
    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k_logprobs: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate with full logprobs."""
        return self.loader.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            return_logprobs=True,
            top_k_logprobs=top_k_logprobs,
            **kwargs
        )


def test_transformers_logprobs():
    """Test function to verify logprobs extraction."""
    print("=" * 70)
    print("測試 Transformers Logprobs 功能")
    print("=" * 70)
    
    # 使用小型模型測試 (如果有 GPU 可以用更大的)
    model_name = "gpt2"  # 小型測試模型
    print(f"\n載入模型: {model_name}")
    
    try:
        loader = TransformersModelLoader(model_name, device="auto")
        
        prompt = "The capital of France is"
        print(f"\nPrompt: {prompt}")
        
        result = loader.generate(
            prompt=prompt,
            max_tokens=10,
            temperature=0.7,
            return_logprobs=True,
            top_k_logprobs=3
        )
        
        print(f"\n生成文本: {result['text']}")
        print(f"\n生成 Token 數: {len(result['tokens'])}")
        
        if 'logprobs' in result:
            print("\n✅ Logprobs 提取成功！")
            print("\nToken 詳細資訊:")
            for i, (token, logprob, top_k) in enumerate(
                zip(result['tokens'], result['logprobs'], result['top_logprobs'])
            ):
                print(f"\n位置 {i}: '{token}' (logprob: {logprob:.4f})")
                print("  Top-3 候選:")
                for alt in top_k[:3]:
                    print(f"    '{alt['token']}': {alt['logprob']:.4f}")
        else:
            print("\n❌ 未能提取 logprobs")
            
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_transformers_logprobs()
