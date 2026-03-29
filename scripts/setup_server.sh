#!/bin/bash
# =============================================================================
# setup_server.sh — 租用服务器一键部署 aira-cogito 实验环境
# =============================================================================
#
# 前提：服务器已有 CUDA 驱动（nvidia-smi 能跑）
#
# 用法：
#   bash setup_server.sh
#
# 需要手动做的事（脚本会提醒）：
#   1. 修改 .env 中的 API key
#   2. rsync AIRS-Bench 数据集（~11GB）
#   3. 设置 cron 清理任务
# =============================================================================

set -euo pipefail

WORK_DIR="${WORK_DIR:-/root}"
REPO_DIR="$WORK_DIR/Research_Society"
CONDA_ENV="aira-dojo"
PYTHON_VER="3.12.9"

echo "=========================================="
echo " aira-cogito 环境部署脚本"
echo " 工作目录: $WORK_DIR"
echo "=========================================="

# ─── Step 1: 检查 CUDA ──────────────────────────────────────────────────
echo ""
echo "[Step 1/8] 检查 CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi 不可用，请确认 CUDA 驱动已安装"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "驱动版本: $CUDA_VER"

# ─── Step 2: 安装 Miniconda ─────────────────────────────────────────────
echo ""
echo "[Step 2/8] 安装 Miniconda..."
if [ -d "$HOME/miniconda3" ]; then
    echo "Miniconda 已存在，跳过安装"
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    echo "Miniconda 安装完成"
fi

# 初始化 conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash 2>/dev/null || true

# ─── Step 3: 创建 conda 环境 ────────────────────────────────────────────
echo ""
echo "[Step 3/8] 创建 conda 环境 ($CONDA_ENV, Python $PYTHON_VER)..."
if conda env list | grep -q "$CONDA_ENV"; then
    echo "环境 $CONDA_ENV 已存在，跳过创建"
else
    conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y
fi
conda activate "$CONDA_ENV"
echo "Python: $(python --version)"

# ─── Step 4: 安装 PyTorch ────────────────────────────────────────────────
echo ""
echo "[Step 4/8] 安装 PyTorch..."
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch 已安装: $(python -c 'import torch; print(torch.__version__)')"
else
    # 检查 CUDA 版本决定用哪个 wheel
    if nvcc --version 2>/dev/null | grep -q "12\." ; then
        echo "检测到 CUDA 12.x，安装 torch+cu124"
        pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu124
    else
        echo "未检测到 CUDA 12.x，安装 CPU 版本（请手动重装 GPU 版本）"
        pip install torch==2.8.0 torchvision
    fi
fi

# 验证
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ─── Step 5: 安装 torch-geometric + torch-cluster ───────────────────────
echo ""
echo "[Step 5/8] 安装 torch-geometric..."
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; print(f'cu{torch.version.cuda.replace(\".\",\"\")[:3]}' if torch.cuda.is_available() else 'cpu')")
echo "PyTorch: $TORCH_VER, CUDA tag: $CUDA_TAG"

pip install torch-geometric==2.7.0
pip install torchmetrics==1.9.0

# torch-cluster: 尝试从 pyg 源安装预编译包
echo "安装 torch-cluster（可能需要编译，耐心等待）..."
pip install torch-cluster -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" 2>/dev/null \
    || pip install torch-cluster -f "https://data.pyg.org/whl/torch-2.6.0+${CUDA_TAG}.html" 2>/dev/null \
    || echo "[WARN] torch-cluster 预编译包未找到，尝试从源码编译..." && pip install torch-cluster

# ─── Step 6: 安装其他依赖 ────────────────────────────────────────────────
echo ""
echo "[Step 6/8] 安装 Python 依赖包..."
if [ -f "$REPO_DIR/scripts/requirements.txt" ]; then
    pip install -r "$REPO_DIR/scripts/requirements.txt"
else
    echo "[WARN] requirements.txt 未找到，请先 git clone 仓库"
    echo "  git clone https://github.com/WangRentu/Research_Society.git $REPO_DIR"
    exit 1
fi

# 安装 spacy 英文模型
python -m spacy download en_core_web_sm 2>/dev/null || echo "[WARN] spacy 模型下载失败，可稍后重试"

# ─── Step 7: 安装项目（editable mode）────────────────────────────────────
echo ""
echo "[Step 7/8] 安装 aira-cogito & aira-dojo..."
pip install -e "$REPO_DIR/forks/aira-dojo/"
pip install -e "$REPO_DIR/forks/aira-cogito/"

# ─── Step 8: 配置 .env ──────────────────────────────────────────────────
echo ""
echo "[Step 8/8] 生成 .env 模板..."
ENV_FILE="$REPO_DIR/forks/aira-cogito/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'ENVEOF'
# === DashScope API ===
PRIMARY_KEY=<替换为你的DashScope API Key>
PRIMARY_KEY_O1_PREVIEW=<替换为你的DashScope API Key>
HOST_O1_PREVIEW=<替换为你的DashScope endpoint>

# === 路径配置（已自动设置，按需修改）===
LOGGING_DIR="/root/Research_Society/forks/aira-cogito/outputs"
MLE_BENCH_DATA_DIR="/root/Research_Society/forks/MLGym"
AIRS_BENCH_TASKS_DIR="/root/Research_Society/forks/airs-bench/airsbench/tasks/rad"
AIRS_BENCH_DATA_DIR="/root/Research_Society/forks/airs-bench/datasets/airs_raw_datasets"
SUPERIMAGE_DIR=""
DEFAULT_SLURM_ACCOUNT=""
DEFAULT_SLURM_PARTITION=""
DEFAULT_SLURM_QOS=""

# === HuggingFace 镜像（国内服务器必须）===
HF_ENDPOINT="https://hf-mirror.com"
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0
ENVEOF
    echo ".env 模板已生成: $ENV_FILE"
    echo "[TODO] 请编辑此文件填入你的 API Key"
else
    echo ".env 已存在，跳过"
fi

# ─── 验证 ────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " 环境验证"
echo "=========================================="
cd "$REPO_DIR/forks/aira-cogito"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

python -c "
import torch
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'GPU count:   {torch.cuda.device_count()}')

import hydra, omegaconf, litellm
print(f'Hydra:       {hydra.__version__}')
print(f'OmegaConf:   {omegaconf.__version__}')
print(f'LiteLLM:     {litellm.__version__}')

import transformers, torch_geometric
print(f'Transformers:{transformers.__version__}')
print(f'PyG:         {torch_geometric.__version__}')

from dojo.solvers.greedy.greedy import GreedyCognitoSolver
from dojo.core.solvers.utils.cognitive_state import CognitiveState
print(f'aira-cogito: import OK')
print()
print('环境部署完成!')
"

echo ""
echo "=========================================="
echo " 剩余手动步骤"
echo "=========================================="
echo "1. 编辑 .env 填入 API Key:"
echo "   vim $ENV_FILE"
echo ""
echo "2. 传输 AIRS-Bench 数据集（~11GB）:"
echo "   # 在本机执行:"
echo "   rsync -avz --progress /data1/intern/research_society/forks/airs-bench/ root@<IP>:$REPO_DIR/forks/airs-bench/"
echo ""
echo "3. 设置清理 cron:"
echo "   chmod +x $REPO_DIR/scripts/cleanup_workspace.sh"
echo "   (crontab -l 2>/dev/null; echo \"*/10 * * * * $REPO_DIR/scripts/cleanup_workspace.sh $REPO_DIR/forks/aira-cogito/outputs 30 >> /tmp/cleanup.log 2>&1\") | crontab -"
echo ""
echo "4. 测试 API 连接:"
echo "   cd $REPO_DIR/forks/aira-cogito && set -a && source .env && set +a"
echo "   python -c \"import litellm; r=litellm.completion(model='qwen3.5-397b-a17b',messages=[{'role':'user','content':'hi'}],max_tokens=5); print(r.choices[0].message.content)\""
echo ""
