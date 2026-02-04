"""
推送前最终检查 - 验证所有核心功能
"""
import subprocess
import sys
from pathlib import Path
from loguru import logger
import torch

def run_command(cmd, description):
    """运行命令并返回结果"""
    logger.info(f"\n{'='*70}")
    logger.info(f"测试: {description}")
    logger.info(f"{'='*70}")
    logger.info(f"命令: {cmd}")
    
    try:
        # 设置 UTF-8 编码环境变量
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=300,  # 5分钟超时
            env=env
        )
        
        if result.returncode == 0:
            logger.success(f"✅ {description} - 通过")
            return True
        else:
            logger.error(f"❌ {description} - 失败")
            logger.error(f"错误输出: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {description} - 超时")
        return False
    except Exception as e:
        logger.error(f"❌ {description} - 异常: {e}")
        return False

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if Path(filepath).exists():
        logger.success(f"✅ {description}: {filepath}")
        return True
    else:
        logger.error(f"❌ {description} 缺失: {filepath}")
        return False

def main():
    logger.info("="*70)
    logger.info("推送前最终检查")
    logger.info("="*70)
    
    # 检查 GPU
    logger.info("\n[1/8] 检查 GPU 支持")
    logger.info("="*70)
    if torch.cuda.is_available():
        logger.success(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA 版本: {torch.version.cuda}")
        logger.info(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        gpu_ok = True
    else:
        logger.error("❌ GPU 不可用")
        gpu_ok = False
    
    # 检查关键文件
    logger.info("\n[2/8] 检查关键文件")
    logger.info("="*70)
    files_ok = all([
        check_file_exists("data/probes/all_probes.json", "探针数据集"),
        check_file_exists("data/anchor_models/metadata.json", "锚点配置"),
        check_file_exists("data/anchor_models/gpt2_fingerprint.json", "GPT2锚点"),
        check_file_exists("data/anchor_models/deepseek_r1_7b_fingerprint.json", "DeepSeek锚点"),
        check_file_exists("ANCHOR_CONFIG_GUIDE.md", "配置指南"),
        check_file_exists("OLLAMA_LOGPROBS_ISSUE.md", "技术分析文档"),
        check_file_exists("README.md", "README文档"),
    ])
    
    # 测试 GPT2 自相似度
    logger.info("\n[3/8] 测试 GPT2 自相似度（应为 100%）")
    logger.info("="*70)
    gpt2_ok = run_command("python quick_test.py", "GPT2 自相似度测试")
    
    # 检查锚点有效性
    logger.info("\n[4/8] 检查锚点指纹有效性")
    logger.info("="*70)
    anchor_ok = run_command("python check_anchor_validity.py", "锚点有效性检查")
    
    # 检查 Ollama 可用性
    logger.info("\n[5/8] 检查 Ollama 服务")
    logger.info("="*70)
    try:
        result = subprocess.run(
            "ollama list",
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            logger.success(f"✅ Ollama 可用，已安装 {len(lines)-1} 个模型")
            ollama_ok = True
        else:
            logger.warning("⚠️ Ollama 不可用")
            ollama_ok = False
    except Exception as e:
        logger.warning(f"⚠️ Ollama 检查失败: {e}")
        ollama_ok = False
    
    # 检查 Git 状态
    logger.info("\n[6/8] 检查 Git 状态")
    logger.info("="*70)
    try:
        result = subprocess.run(
            "git status --porcelain",
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=10
        )
        if result.returncode == 0:
            changes = result.stdout.strip().split('\n')
            changes = [c for c in changes if c]
            logger.info(f"   待提交文件数: {len(changes)}")
            if changes:
                for change in changes[:10]:  # 显示前10个
                    logger.info(f"   {change}")
            git_ok = True
        else:
            logger.error("❌ Git 命令失败")
            git_ok = False
    except Exception as e:
        logger.error(f"❌ Git 检查失败: {e}")
        git_ok = False
    
    # 检查代码质量
    logger.info("\n[7/8] 检查 Python 语法")
    logger.info("="*70)
    key_files = [
        "src/fingerprint/logit_extractor.py",
        "src/attribution/similarity.py",
        "src/utils/unified_loader.py",
        "experiments/full_evaluation.py",
    ]
    
    syntax_ok = True
    for filepath in key_files:
        if Path(filepath).exists():
            result = subprocess.run(
                f"python -m py_compile {filepath}",
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=10
            )
            if result.returncode == 0:
                logger.success(f"✅ {filepath}")
            else:
                logger.error(f"❌ {filepath} - 语法错误")
                syntax_ok = False
    
    # 生成总结
    logger.info("\n[8/8] 生成检查报告")
    logger.info("="*70)
    
    checks = {
        "GPU 支持": gpu_ok,
        "关键文件": files_ok,
        "GPT2 测试": gpt2_ok,
        "锚点有效性": anchor_ok,
        "Ollama 服务": ollama_ok,
        "Git 状态": git_ok,
        "代码语法": syntax_ok,
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    logger.info("\n" + "="*70)
    logger.info("检查结果汇总")
    logger.info("="*70)
    
    for check_name, status in checks.items():
        emoji = "✅" if status else "❌"
        logger.info(f"{emoji} {check_name:20} {'通过' if status else '失败'}")
    
    logger.info(f"\n总计: {passed}/{total} 项通过 ({passed/total*100:.0f}%)")
    
    if passed == total:
        logger.success("\n🎉 所有检查通过，可以推送到 GitHub！")
        logger.info("\n推送命令:")
        logger.info("  git add .")
        logger.info('  git commit -m "docs: 完善配置指南和技术分析"')
        logger.info("  git push origin master")
        return 0
    else:
        logger.warning(f"\n⚠️ {total - passed} 项检查未通过")
        logger.info("请修复问题后再推送")
        return 1

if __name__ == "__main__":
    sys.exit(main())
