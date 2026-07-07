$python = "D:\software\Anaconda\envs\ai4city-mas\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "未找到 Python 环境: $python"
    exit 1
}

& $python "D:\course\ai4city\agent_rag_memory_demo_v2.py"
