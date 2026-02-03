"""
Ollama Logprobs API 煙霧測試
驗證 /v1/completions 端點是否正確返回 logprobs
"""
import requests
import json
from loguru import logger

def test_ollama_logprobs_api():
    """測試 Ollama OpenAI 兼容接口的 logprobs 功能"""
    
    logger.info("=" * 80)
    logger.info("Ollama Logprobs API 煙霧測試")
    logger.info("=" * 80)
    
    # 測試參數
    api_base = "http://localhost:11434"
    model_name = "llama3.2:latest"
    test_prompt = "測試"
    
    # 測試 /v1/completions 端點
    url = f"{api_base}/v1/completions"
    
    payload = {
        "model": model_name,
        "prompt": test_prompt,
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 1.0,
        "logprobs": 5,  # 要求返回 top 5 logprobs
        "stream": False
    }
    
    try:
        logger.info(f"請求 URL: {url}")
        logger.info(f"請求 Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        logger.info("\n✅ API 請求成功！")
        logger.info(f"回應狀態碼: {response.status_code}")
        logger.info(f"\n完整回應:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        
        # 驗證 logprobs 數據
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            text = choice.get("text", "")
            logprobs_data = choice.get("logprobs", {})
            
            logger.info(f"\n生成文本: {text}")
            
            if logprobs_data:
                logger.info("\n✅ 成功獲取 logprobs！")
                logger.info(f"Logprobs 鍵: {list(logprobs_data.keys())}")
                
                if "tokens" in logprobs_data:
                    logger.info(f"Token 數量: {len(logprobs_data['tokens'])}")
                    logger.info(f"前 3 個 Tokens: {logprobs_data['tokens'][:3]}")
                
                if "token_logprobs" in logprobs_data:
                    logger.info(f"Token Logprobs: {logprobs_data['token_logprobs'][:3]}")
                
                if "top_logprobs" in logprobs_data:
                    logger.info(f"Top Logprobs 範例: {logprobs_data['top_logprobs'][:2]}")
                
                logger.info("\n🎉 驗收通過！Ollama API 正確返回 logprobs 數據")
                return True
            else:
                logger.error("\n❌ 錯誤：未獲取到 logprobs 數據")
                logger.error("這可能意味著：")
                logger.error("1. Ollama 版本不支援 logprobs（需要 >= 0.1.20）")
                logger.error("2. 模型不支援 logprobs 輸出")
                logger.error("3. API 配置問題")
                return False
        else:
            logger.error("\n❌ 錯誤：API 未返回 choices")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("\n❌ 連接錯誤：無法連接到 Ollama API")
        logger.error("請確認：")
        logger.error("1. Ollama 服務已啟動（執行 'ollama serve'）")
        logger.error("2. API 端口為 11434（默認端口）")
        return False
    except requests.exceptions.Timeout:
        logger.error("\n❌ 請求超時")
        return False
    except Exception as e:
        logger.error(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ollama_logprobs_api()
    
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✅ 煙霧測試通過！可以繼續進行完整測試")
        logger.info("=" * 80)
    else:
        logger.error("\n" + "=" * 80)
        logger.error("❌ 煙霧測試失敗！請先修正 API 問題")
        logger.error("=" * 80)
