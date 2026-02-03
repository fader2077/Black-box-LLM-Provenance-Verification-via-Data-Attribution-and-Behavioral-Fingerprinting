# Git Commit and Push Script
# 修复相似度计算Bug并推送到GitHub

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  LLM 溯源技术 - Bug修复提交" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查Git状态
Write-Host "[1/5] 检查Git状态..." -ForegroundColor Yellow
git status

Write-Host ""
Read-Host "按 Enter 继续查看改动..."

# 2. 查看具体改动
Write-Host "`n[2/5] 查看文件改动..." -ForegroundColor Yellow
git diff src/attribution/similarity.py
git diff src/probes/political_probes.py

Write-Host ""
Read-Host "按 Enter 继续添加文件..."

# 3. 添加修改的文件
Write-Host "`n[3/5] 添加修改的文件..." -ForegroundColor Yellow
git add src/attribution/similarity.py
git add src/probes/political_probes.py
git add data/probes/all_probes.json
git add quick_test.py
git add BUGFIX_20260204.md
git add rebuild_anchor_fingerprints.py
git add run_complete_test.py

Write-Host "✓ 文件已添加到暂存区" -ForegroundColor Green

# 4. 创建提交
Write-Host "`n[4/5] 创建Git提交..." -ForegroundColor Yellow
$commitMessage = @"
fix: 修复模型相似度计算错误 (70% -> 100%)

问题:
- GPT-2 自相似度只有 70%，而不是预期的 100%

根因:
1. 主要: 默认引擎选择错误 (ollama vs transformers)
   - full_evaluation.py 默认使用 ollama 引擎
   - HuggingFace模型应使用 transformers 引擎
   - Ollama失败后使用伪概率，导致指纹错误

2. 次要: 相似度计算在拒绝指纹缺失时仍用加权平均
   - 即使 logit_score=1.0，结果也只有 0.7

3. 次要: probe_type 字段缺失
   - 政治探针无法被识别，拒绝检测被跳过

修复:
- similarity.py: 拒绝指纹缺失时直接使用 logit 分数
- political_probes.py: 添加 probe_type 字段
- 重新生成探针缓存

测试结果:
✅ GPT-2 vs GPT-2: 100.00% (Cosine: 1.0, Pearson: 1.0, KL: 1.0)
✅ 所有核心功能验证通过

使用提示:
- HuggingFace模型: --engine transformers
- Ollama模型: --engine ollama (默认)

Closes #相似度计算错误
"@

git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ 提交成功" -ForegroundColor Green
} else {
    Write-Host "✗ 提交失败" -ForegroundColor Red
    exit 1
}

# 5. 推送到远程仓库
Write-Host "`n[5/5] 推送到GitHub..." -ForegroundColor Yellow
Write-Host "当前分支:" -ForegroundColor Cyan
git branch --show-current

Write-Host ""
$confirm = Read-Host "确认推送到远程仓库？(y/n)"

if ($confirm -eq 'y' -or $confirm -eq 'Y') {
    git push
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ 推送成功！" -ForegroundColor Green
        Write-Host "`n================================================" -ForegroundColor Cyan
        Write-Host "  修复已成功提交并推送到GitHub" -ForegroundColor Green
        Write-Host "================================================" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ 推送失败" -ForegroundColor Red
        Write-Host "请检查网络连接和远程仓库权限" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n已取消推送" -ForegroundColor Yellow
    Write-Host "可以稍后手动推送: git push" -ForegroundColor Cyan
}
