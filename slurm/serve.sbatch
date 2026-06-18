#!/bin/bash
# =============================================================================
# DiffusionGemma demo — SLURM serve template for 1x NVIDIA H100 (80 GB).
#
# What this does:
#   1. Requests one H100 on a compute node.
#   2. Sets up the Hugging Face cache and HF auth.
#   3. Launches the demo web server with `uv run dlm`.
#   4. Prints the EXACT ssh tunnel command so you can open the UI on your laptop.
#
# Edit the obvious PLACEHOLDERS (LOGIN_NODE, ACCOUNT, PARTITION, ...) before use.
# Submit with:   sbatch slurm/serve.sbatch
# Watch output:  tail -f slurm-<jobid>.out
# =============================================================================

#SBATCH --job-name=dgemma-demo
#SBATCH --partition=PARTITION        # <-- your GPU partition (e.g. gpu, h100, accel)
#SBATCH --account=ACCOUNT            # <-- your allocation / project account
#SBATCH --gres=gpu:h100:1            # <-- one H100 (adjust the gres name to your site)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # <-- CPUs for dataloading / tokenization
#SBATCH --mem=128G                   # <-- host RAM (weights stream to GPU; this is headroom)
#SBATCH --time=04:00:00              # <-- walltime HH:MM:SS
#SBATCH --output=slurm-%j.out        # stdout+stderr -> slurm-<jobid>.out

set -euo pipefail

# --- Login node you SSH through (the node you `sbatch` from). EDIT THIS. -------
LOGIN_NODE=login.cluster.example.edu   # <-- e.g. login1.hpc.myuni.edu
SSH_USER=${USER}                       # <-- your cluster username (defaults to $USER)

# --- Port the server listens on (override: PORT=9000 sbatch slurm/serve.sbatch)
PORT=${PORT:-8000}

# --- Optional environment modules (uncomment/adjust to match your site) -------
# module purge
# module load cuda/12.4
# module load python/3.11
# module load uv

# --- Hugging Face cache + auth ------------------------------------------------
# Point HF_HOME at fast/large shared scratch so the ~52 GB (bf16) download is
# cached and reused across jobs. EDIT the path.
export HF_HOME=${HF_HOME:-/scratch/${SSH_USER}/hf_home}
mkdir -p "${HF_HOME}"
# Provide your token (or `huggingface-cli login` once into the cache above).
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
export MODEL_ID=${MODEL_ID:-google/diffusiongemma-26B-A4B-it}

# --- Tunnel instructions ------------------------------------------------------
COMPUTE_NODE=$(hostname -s)
echo "=================================================================="
echo " DiffusionGemma demo starting"
echo "   compute node : ${COMPUTE_NODE}"
echo "   port         : ${PORT}"
echo "   model        : ${MODEL_ID}"
echo "   HF_HOME      : ${HF_HOME}"
echo "------------------------------------------------------------------"
echo " From your LAPTOP, open an SSH tunnel (copy/paste this line):"
echo ""
echo "   ssh -N -L ${PORT}:${COMPUTE_NODE}:${PORT} ${SSH_USER}@${LOGIN_NODE}"
echo ""
echo " Then browse to:   http://localhost:${PORT}"
echo "=================================================================="

# --- Launch the server --------------------------------------------------------
# Run from the repo root. `uv run` resolves/creates the env on the fly; ensure
# the GPU extra is installed first with:  uv sync --extra gpu
uv run dlm --model-id "${MODEL_ID}" --host 0.0.0.0 --port "${PORT}" --precision bf16
