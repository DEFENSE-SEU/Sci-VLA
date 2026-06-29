#!/usr/bin/env bash
set -euo pipefail

ProjRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── Paths (edit before running) ──
VlmPretrainedPath="/path/to/Qwen3-VL-4B-Instruct"
DataRoot="/path/to/pretrain_data"
OutputDir="${ProjRoot}/outputs"
DeepspeedConfig="${ProjRoot}/configs/deepspeed_zero2.json"
FastTokenizerPath="/path/to/fast"

# ── Cluster ──
NumGpus=8
NumMachines=2
MachineRank=0
MasterAddr="127.0.0.1"
MasterPort=29653
NumProcesses=$((NumGpus * NumMachines))

# ── Data (pick one or comma-separate for multi-dataset training) ──
# Available datasets:
#   oxe-auge_clean        — Open X-Embodiment augmented collection (v3 format)
#   robointer_droid_clean — RoboInter DROID subset (v2.1 format)
# To add your own dataset, create a DatasetSchema under schemas/ and append
# the repo name here (see schemas/README or schemas/oxe_auge.py for examples).
RepoIds="oxe-auge_clean"

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
BatchSize=64
GradientAccumulationSteps=1
NumWorkers=4
DataloaderPrefetchFactor=2
PrefetchBuffer=true
PrefetchBufferSize=2
DistTimeoutSeconds=3600
TrimTokenPaddingToBatch=true
SourceShapeConvergence=true
DataErrorSkip=true
DataErrorSkipMaxAttempts=64
DataErrorLogFirst=20
TotalSteps=80000
SaveFreq=10000
MaxKeepCkpts=8
LogFreq=1
Seed=42

Lr="1e-5"
VlmLr="1e-5"
DitLr="1e-4"
WeightDecay=0.01
GradClipNorm=1.0
WarmupSteps=8000
DecaySteps="${TotalSteps}"
DecayLr="5e-7"

FreezeVisionEncoder=false
GradientCheckpointing=true
GcVisualEncoder=false
GcLanguageModel=true
GcDit=false
AttnImplementation="flash_attention_2"
ActionMode="delta"
NormalizeArmJoints=true
NormalizeGripper=true

TrainingPhase="posttrain"
KnowledgeIsolation=true
UseFastTokenizer=true
KiMseWeight=10.0
DiscreteActionVocabSize=2048
DiscreteActionMaxLength=192

ResumeCheckpoint=""
JobName="labvla_ki_posttrain_$(date +%Y%m%d_%H%M%S)"

# ── Environment ──
export PYTHONPATH="src:."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MASTER_ADDR="${MasterAddr}"
export MASTER_PORT="${MasterPort}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_ALGO=Ring
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export MALLOC_ARENA_MAX=2

# ── Launch ──
cd "${ProjRoot}"

ResumeArgs=()
[ -n "${ResumeCheckpoint}" ] && ResumeArgs+=(--resume "${ResumeCheckpoint}" --load_weights_only true)

DsArgs=()
[ -f "${DeepspeedConfig}" ] && DsArgs=(--use_deepspeed --deepspeed_multinode_launcher standard --deepspeed_config_file "${DeepspeedConfig}")

exec accelerate launch \
    --num_processes "${NumProcesses}" \
    --num_machines "${NumMachines}" \
    --machine_rank "${MachineRank}" \
    --main_process_ip "${MasterAddr}" \
    --main_process_port "${MasterPort}" \
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
    --dataloader_prefetch_factor "${DataloaderPrefetchFactor}" \
    --prefetch_buffer "${PrefetchBuffer}" \
    --prefetch_buffer_size "${PrefetchBufferSize}" \
    --dist_timeout_seconds "${DistTimeoutSeconds}" \
    --trim_token_padding_to_batch "${TrimTokenPaddingToBatch}" \
    --source_shape_convergence "${SourceShapeConvergence}" \
    --data_error_skip "${DataErrorSkip}" \
    --data_error_skip_max_attempts "${DataErrorSkipMaxAttempts}" \
    --data_error_log_first "${DataErrorLogFirst}" \
    --steps "${TotalSteps}" \
    --save_freq "${SaveFreq}" \
    --max_keep_ckpts "${MaxKeepCkpts}" \
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
    --training_phase "${TrainingPhase}" \
    --knowledge_isolation "${KnowledgeIsolation}" \
    --use_fast_tokenizer "${UseFastTokenizer}" \
    --fast_tokenizer_path "${FastTokenizerPath}" \
    --ki_mse_weight "${KiMseWeight}" \
    --discrete_action_vocab_size "${DiscreteActionVocabSize}" \
    --discrete_action_max_length "${DiscreteActionMaxLength}" \
    --repo_ids "${RepoIds}" \
    --data_root "${DataRoot}" \
    --output_dir "${OutputDir}" \
    --job_name "${JobName}" \
    "${ResumeArgs[@]}"
