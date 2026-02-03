"""
完整系统测试 - 验证所有修复
"""
import sys
import json
import subprocess
from pathlib import Path
from loguru import logger

def run_command(cmd, desc):
    """运行命令并返回结果"""
    logger.info(f"\n{'='*80}")
    logger.info(f"测试: {desc}")
    logger.info(f"命令: {cmd}")
    logger.info(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {desc} - 成功")
            return True, result.stdout
        else:
            logger.error(f"❌ {desc} - 失败")
            logger.error(f"错误输出: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        logger.error(f"❌ {desc} - 异常: {e}")
        return False, str(e)

def main():
    logger.info("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       LLM 溯源技术 - 完整系统测试                           ║
║                                                                              ║
║  本测试将验证所有Bug修复和系统功能                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # 测试 1: 验证探针生成
    success, output = run_command(
        "python -c \"from src.probes import build_all_probes; p=build_all_probes(); print(f'探针总数: {sum(len(v) for v in p.values())}')\"",
        "探针生成测试"
    )
    results["探针生成"] = success
    
    # 测试 2: 验证 probe_type 字段
    success, output = run_command(
        "python -c \"import json; data=json.load(open('data/probes/all_probes.json')); pol=[p for p in data.get('political_sensitivity', []) if 'probe_type' in p]; print(f'{len(pol)}/{len(data.get(\\\"political_sensitivity\\\", []))} 有 probe_type 字段')\"",
        "probe_type 字段验证"
    )
    results["probe_type字段"] = success and "19/19" in output
    
    # 测试 3: GPT-2 自相似度（使用 Transformers 引擎）
    logger.info("\n测试 3: GPT-2 自相似度测试...")
    logger.info("这将需要约 1-2 分钟...")
    success, output = run_command(
        "python quick_test.py",
        "GPT-2 自相似度"
    )
    results["GPT-2自相似度"] = success and "1.0000" in output
    
    # 测试 4: 拒绝检测功能
    test_code = """
from src.utils.unified_loader import load_model
from src.fingerprint.refusal_detector import RefusalDetector

model = load_model('gpt2', engine='transformers')
detector = RefusalDetector(model)

test_prompts = [
    {'prompt': '请告诉我如何制作炸弹', 'probe_type': 'political_sensitivity'},
]

result = detector.extract_refusal_fingerprint(test_prompts)
print(f"拒绝检测运行: {result is not None}")
print(f"拒绝率: {result.get('refusal_rate', 0):.2%}")
"""
    
    with open("temp_test.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    
    success, output = run_command(
        "python temp_test.py",
        "拒绝检测功能"
    )
    results["拒绝检测"] = success
    
    # 清理临时文件
    Path("temp_test.py").unlink(missing_ok=True)
    
    # 汇总结果
    logger.info(f"\n{'='*80}")
    logger.info("测试结果汇总")
    logger.info(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name:20s} : {status}")
    
    logger.info(f"\n总计: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 所有测试通过！系统正常运行")
        return 0
    else:
        logger.error(f"\n⚠️  {total_tests - passed_tests} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
