#!/usr/bin/env bash
set -euo pipefail

ProjRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── Paths (edit before running) ──
VlmPretrainedPath="/path/to/Qwen3-VL-4B-Instruct"
DataRoot="/path/to/finetune_data"
OutputDir="${ProjRoot}/outputs"
DeepspeedConfig="${ProjRoot}/configs/deepspeed_zero2.json"
FastTokenizerPath="/path/to/fast"

PretrainedCkpt=""
if [ -z "${PretrainedCkpt}" ]; then
    echo "[ERROR] PretrainedCkpt is required. Set it inside the script before running." >&2
    exit 1
fi

# ── Data ──
RepoIds="LabUtopia/Level3_TransportBeaker"
DatasetSchema="labutopia_level3_transportbeaker_v3"
ExternalStatsPath="${DataRoot}/${RepoIds}/meta/stats_canonical_grip.json"

# ── Cluster ──
NumGpus=4
MainProcessPort=29652

# ── Model ──
Dtype="bfloat16"
DitNumLayers=18
DitNumHeads=8
DitHeadDim=128
ChunkSize=50
MaxStateDim=32
MaxActionDim=32
ImageHeight=224
ImageWidth=224

# ── Training ──
BatchSize=48
GradientAccumulationSteps=1
NumWorkers=8
TotalSteps=80000
SaveFreq=10000
LogFreq=50
Seed=42

Lr="5e-5"
VlmLr="5e-5"
DitLr="5e-5"
WeightDecay=0.01
GradClipNorm=1.0
WarmupSteps=2000
DecaySteps="${TotalSteps}"
# Cosine decay to 2.5e-6; flat full-LR (DecayLr==Lr) diverged ~step 56k on the KI base.
DecayLr="2.5e-6"

FreezeVisionEncoder=true
GradientCheckpointing=true
GcVisualEncoder=true
GcLanguageModel=false
GcDit=false
AttnImplementation="flash_attention_2"

ActionMode="delta"
NormalizeArmJoints=true
NormalizeGripper=true
GripperNormMode="q01_q99"
SnapGripperToBinary=false
GripperMaxWidth=0.04
GripperCanonicalDim=7

if [ "${ActionMode}" != "delta" ]; then
    echo "[ERROR] LabUtopia finetune requires ActionMode=delta." >&2
    exit 1
fi

TrainingPhase="posttrain"
KnowledgeInsulation=false
UseFastTokenizer=false
KiMseWeight=10.0

JobName="labvla_finetune_transportbeaker_$(date +%Y%m%d_%H%M%S)"

# ── Environment ──
export PYTHONPATH="src:."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ALGO=Ring
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export MALLOC_ARENA_MAX=2

# ── Launch ──
cd "${ProjRoot}"

DsArgs=()
[ -f "${DeepspeedConfig}" ] && DsArgs=(--use_deepspeed --deepspeed_config_file "${DeepspeedConfig}")

exec accelerate launch \
    --num_processes "${NumGpus}" \
    --num_machines 1 \
    --main_process_port "${MainProcessPort}" \
    --mixed_precision bf16 \
    "${DsArgs[@]}" \
    scripts/train.py \
    --vlm_pretrained_path "${VlmPretrainedPath}" \
    --dtype "${Dtype}" \
    --dit_num_layers "${DitNumLayers}" \
    --dit_num_heads "${DitNumHeads}" \
    --dit_head_dim "${DitHeadDim}" \
    --chunk_size "${ChunkSize}" \
    --max_state_dim "${MaxStateDim}" \
    --max_action_dim "${MaxActionDim}" \
    --batch_size "${BatchSize}" \
    --gradient_accumulation_steps "${GradientAccumulationSteps}" \
    --num_workers "${NumWorkers}" \
    --steps "${TotalSteps}" \
    --save_freq "${SaveFreq}" \
    --log_freq "${LogFreq}" \
    --seed "${Seed}" \
    --image_height "${ImageHeight}" \
    --image_width "${ImageWidth}" \
    --lr "${Lr}" \
    --vlm_lr "${VlmLr}" \
    --dit_lr "${DitLr}" \
    --weight_decay "${WeightDecay}" \
    --grad_clip_norm "${GradClipNorm}" \
    --warmup_steps "${WarmupSteps}" \
    --decay_steps "${DecaySteps}" \
    --decay_lr "${DecayLr}" \
    --freeze_vision_encoder "${FreezeVisionEncoder}" \
    --gradient_checkpointing "${GradientCheckpointing}" \
    --gc_visual_encoder "${GcVisualEncoder}" \
    --gc_language_model "${GcLanguageModel}" \
    --gc_dit "${GcDit}" \
    --attn_implementation "${AttnImplementation}" \
    --action_mode "${ActionMode}" \
    --normalize_arm_joints "${NormalizeArmJoints}" \
    --normalize_gripper "${NormalizeGripper}" \
    --gripper_norm_mode "${GripperNormMode}" \
    --snap_gripper_to_binary "${SnapGripperToBinary}" \
    --gripper_max_width "${GripperMaxWidth}" \
    --gripper_canonical_dim "${GripperCanonicalDim}" \
    --training_phase "${TrainingPhase}" \
    --knowledge_isolation "${KnowledgeInsulation}" \
    --use_fast_tokenizer "${UseFastTokenizer}" \
    --fast_tokenizer_path "${FastTokenizerPath}" \
    --ki_mse_weight "${KiMseWeight}" \
    --repo_ids "${RepoIds}" \
    --data_root "${DataRoot}" \
    --dataset_schema "${DatasetSchema}" \
    --external_stats_path "${ExternalStatsPath}" \
    --output_dir "${OutputDir}" \
    --job_name "${JobName}" \
    --resume "${PretrainedCkpt}" \
    --load_weights_only true
