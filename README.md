# AtomBridge

<p align="center">
	<b>Agentic VLA Inference Plugin for Long-Horizon Scientific Manipulation</b>
</p>

<p align="center">
	<a href="https://arxiv.org/abs/2602.09430"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2602.09430-b31b1b.svg"></a>
</p>

This repository contains the AtomBridge pipeline and Mujoco simulation assets (including Autobio assets and newly added assets).

## Overview

AtomBridge focuses on agentic, long-horizon task execution in scientific lab simulation environments.

- Long-horizon task decomposition and execution
- Integration with OpenPI training and serving pipeline

## Quick Links

- [Paper](https://arxiv.org/abs/2602.09430)
- [Training Data Generation](#training-data-generation)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)

## Installation (editable local install)

```bash
conda create -n atombridge python=3.11
conda activate atombridge
conda install ffmpeg=7.1.1 -c conda-forge
cd third_party/openpi
pip install uv
uv pip install -e .
uv pip install 'mujoco==3.3.0' numpy scipy toppra trimesh shapely triangle manifold3d sympy zstandard tqdm networkx usd-core ffmpeg imageio[ffmpeg] matplotlib scikit-image openai 
```

For LABVLA comparison experiments, create the LABVLA environment separately:

```bash
conda create -n labvla python=3.10 -y
conda activate labvla
cd third_party/labvla
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install flash_attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
```

## Training Data Generation
Generate raw expert demonstrations for each scene separately:

```bash
python scripts/autobio_scripts/centrifuge5910_tasks.py --episodes 100 --log-root logs/centrifuge5910_tasks
python scripts/autobio_scripts/thermal_cycler_tasks.py --episodes 100 --log-root logs/thermal_cycler_tasks
```

Render camera views at 50 Hz:

```bash
bash scripts/autobio_scripts/render_all.bash logs/centrifuge5910_tasks
bash scripts/autobio_scripts/render_all.bash logs/thermal_cycler_tasks
```

Convert each scene to a separate LeRobot dataset:

```bash
python scripts/convert.py --data-dir logs/centrifuge5910_tasks --repo-id mani_centrifuge5910
python scripts/convert.py --data-dir logs/thermal_cycler_tasks --repo-id mani_thermalcycler
```

The converted datasets now include a frame-level task_is_complete label.
For the vision-language intervention-switch export, training, and annotated
video inference workflow, see
[scripts/intervention_switch/README.md](scripts/intervention_switch/README.md).

## Fine-tuning
When you need to finetune a new specific task, add config in `third_party/openpi/src/openpi/training/config.py`.

```bash
cd third_party/openpi
python scripts/compute_norm_stats.py --config-name mani_thermalcycler_pi05
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python scripts/train.py mani_thermalcycler_pi05 --exp-name mani_thermalcycler_pi05_finetune
```


<!-- ### LABVLA fine-tuning

AtomBridge UR5e demonstrations use the LABVLA schema `atombridge_ur5e_single_arm`.
The existing `scripts/convert.py` output can be reused directly.

```bash
cd third_party/labvla
python -m data_process stats \
  --dataset ~/.cache/huggingface/lerobot/mani_centrifuge5910 \
  --schema atombridge_ur5e_single_arm
```

Use LABVLA's training launcher or `scripts/train.py` with the AtomBridge repo id,
the LeRobot cache root, and `DatasetSchema=atombridge_ur5e_single_arm`. For example,
set the launcher variables to:

```bash
DataRoot=~/.cache/huggingface/lerobot
RepoIds=mani_centrifuge5910
DatasetSchema=atombridge_ur5e_single_arm
ExternalStatsPath=~/.cache/huggingface/lerobot/mani_centrifuge5910/meta/stats.json
```

Also set `PretrainedCkpt` and the model/tokenizer paths at the top of the
launcher. Then run the LABVLA fine-tuning launcher after editing those values:

```bash
bash launch/finetune/train_labutopia.sh
```

If you prefer a direct invocation, use LABVLA's `scripts/train.py` with the same
data root, repo id, schema, and stats settings from the launcher. -->



## Evaluation

### Run policy

To evaluate the policy model, open a shell and run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py policy:checkpoint --policy.config 'mani_thermalcycler_pi05' --policy.dir 'checkpoints/mani_thermalcycler_pi05/mani_thermalcycler_pi05_finetune/49999/'
```

### Start evaluate

Ready memory uses visual A/B comparisons over a demonstration trajectory to
retrieve a state immediately before the next atomic operation. It is used for
transitions between prompts, so run it with `--experiment-mode full` or
`--experiment-mode no-agent`. The `no-agent` mode skips transition planning and
directly restores the retrieved state.

First, export a reusable ready-memory index from a local LeRobot dataset. The
following command selects the longest demonstration for each atomic task and
exports every second frame:

```bash
python scripts/autobio_scripts/export_lerobot_ready_memory_index.py \
  --repo_id mani_thermalcycler \
  --output logs/ready_memory_index.json \
  --samples-per-task 1 \
  --selection longest \
  --frame-stride 2
```

Then configure a vision-language model and enable the index during evaluation:

```bash
export BASE_URL="https://openai.sufy.com/v1"
export MODEL_NAME="qwen3.5-397b-a17b"
export API_KEY=""

python ./scripts/autobio_scripts/evaluate.py \
  --task "thermal_cycler_long_task_1" \
  --completion-model-checkpoint checkpoints/completion_switch_v1_stride10/best.pt \
  --time_limit 30 \
  --experiment-mode full \
  --ready-memory-enabled \
  --ready-memory-db logs/ready_memory_index.json \
  --ready-memory-window-size 25 \
  --ready-memory-max-iterations 4 \
  --llm-backend-mode chat \
  --prompts "open lid of thermal cycler,place pcr Plate into thermal cycler,close lid of thermal cycler,screw tighten the knob,press the button,screw loosen the knob,open lid of thermal cycler,take pcr Plate from thermal cycler" 
```

To retrieve directly from a local LeRobot dataset without first exporting an
index, replace `--ready-memory-db ...` with:

```bash
--ready-memory-repo-id mani_thermalcycler
```

You can optionally add `--ready-memory-episode-index INDEX` to restrict direct
retrieval to one episode. Other useful controls are
`--ready-memory-min-frame-ratio` (default `0.05`, avoids selecting frame 0),
`--ready-memory-match-cutoff` (default `0.5`, fuzzy task matching for index
mode), and `--ready-memory-front-image-key` (default `observation/image`, direct
dataset mode). The selected state is written to
`logs/target_ready_state_selected.json`, and the transition-compatible result
is written to `logs/target_qpos_selected.json`.



### Experiment modes

`evaluate.py` uses `--experiment-mode` to select the transition strategy between prompts:

| Mode | Transition behavior |
| --- | --- |
| `no-transition` | Disable transitions between prompts. This is the default mode. |
| `baseline` | If the atomic task exactly matches a trajectory memory, extract that memory's initial state; otherwise, randomly use an indexed memory's initial state. It then uses RRT to restore to that state; if RRT fails, it prints `RRT_FAILED_SKIP_ACTION` and skips the failed arm motion. |
| `no-retrieval` | Do not provide a target pose. Run the planning/coding agents only, and do not append final target-pose restoration. |
| `no-agent` | Retrieve the target pose, skip planning/coding agents, and directly interpolate to the retrieved pose. |
| `full` | Full AtomBridge transition pipeline: retrieval, planning/coding agent execution. |


<!-- ### Evaluate LABVLA on simulations

Start a LABVLA websocket server from the LABVLA environment:

```bash
cd third_party/labvla
python deployment/serve_labvla.py \
  --pretrained_path /path/to/labvla/checkpoint \
  --port 8000 \
  --device cuda \
  --training_repo_id mani_thermalcycler
```

Then run AtomBridge evaluation with the LABVLA observation adapter:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --policy-backend labvla \
  --host 127.0.0.1 \
  --port 8000 \
  --task 'thermal_cycler_long_task_1' \
  --time_limit 30 \
  --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler" \
  --experiment-mode no-transition
``` -->

### Evaluate the model using local VLM model

```bash
python -m vllm.entrypoints.openai.api_server \
  --model xxx \
  --served-model-name qwen3.5-27b \
  --host 192.168.124.33 \
  --port 9000
```

then open another shell window and run:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --task "thermal_cycler_long_task_1" \
  --time_limit 30 \
  --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler" \
  --experiment-mode full \
  --llm-base-url http://192.168.124.33:9000/v1 \
  --llm-model-name qwen3.5-27b \
  --llm-api-key EMPTY \
  --llm-temperature 0.2 \
  --llm-top-p 0.9 \
  --llm-max-tokens 4096 \
  --llm-max-attempts 3 \
  --llm-timeout 120
```



<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DEFENSE-SEU/AtomBridge&type=Date)](https://star-history.com/#DEFENSE-SEU/AtomBridge&Date) -->


## Citation

If you find AtomBridge useful in your research, please cite the paper:

```bibtex
@article{pang2026sci,
  title={AtomBridge: Agentic VLA Inference Plugin for Long-Horizon Tasks in Scientific Experiments},
  author={Pang, Yiwen and Zhou, Bo and Li, Changjin and Wang, Xuanhao and Xu, Shengxiang and Wang, Deng-Bao and Zhang, Min-Ling and Di, Shimin},
  journal={arXiv preprint arXiv:2602.09430},
  year={2026}
}
```
