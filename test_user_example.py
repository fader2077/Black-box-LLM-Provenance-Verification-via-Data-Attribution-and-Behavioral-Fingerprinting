"""
直接測試用戶提供的 Ollama logprobs 範例
"""
import ollama

print("=" * 80)
print("測試用戶提供的 Ollama logprobs 範例")
print("=" * 80)

try:
    print("\n執行用戶提供的範例代碼...")
    response = ollama.generate(
        model='llama3.2:latest',
        prompt='Building a website can be done in',
        options={
            'logprobs': True,
            'top_logprobs': 5
        }
    )
    
    print(f"\n響應類型: {type(response)}")
    print(f"\nlogprobs 屬性: {response.logprobs}")
    
    # 檢查是否有 completion_probabilities
    if hasattr(response, 'completion_probabilities'):
        print("\n✅ 找到 completion_probabilities!")
        for token_data in response.completion_probabilities:
            print(f"Token: {token_data.get('content', 'N/A')}, Logprobs: {token_data.get('probs', 'N/A')}")
            break  # 只顯示第一個
    else:
        print("\n❌ 沒有 completion_probabilities 屬性")
        
    # 檢查所有可用屬性
    print("\n所有非私有屬性:")
    for attr in dir(response):
        if not attr.startswith('_'):
            value = getattr(response, attr)
            if not callable(value):
                print(f"  {attr}: {type(value).__name__}")
                
except Exception as e:
    print(f"\n❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("結論:")
print("如果 logprobs 為 None，說明:")
print("1. Ollama 0.14.1 可能不支援 logprobs 參數")
print("2. 需要升級 Ollama 到支援 logprobs 的版本")
print("3. 或者該範例適用於更新的 Ollama 版本")
print("=" * 80)
