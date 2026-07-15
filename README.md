# Sci-VLA

<p align="center">
	<b>Agentic VLA Inference Plugin for Long-Horizon Scientific Manipulation</b>
</p>

<p align="center">
	<a href="https://arxiv.org/abs/2602.09430"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2602.09430-b31b1b.svg"></a>
</p>

This repository contains the Sci-VLA architecture and simulation assets (including Autobio assets and newly added assets).

## Overview

Sci-VLA focuses on agentic, long-horizon task execution in scientific lab simulation environments.

- Long-horizon task decomposition and execution
- Integration with OpenPI training and serving pipeline
- Rich simulation assets for robotics evaluation

## Quick Links

- [Paper](https://arxiv.org/abs/2602.09430)
- [Training Data Generation](#training-data-generation)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)

## Installation (editable local install)

```bash
conda create -n scivla python=3.11
conda activate scivla
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
<!-- sudo apt-get update
sudo apt-get install -y libegl1 libgles2 libgl1 libglvnd0 libosmesa6 libosmesa6-dev -->

## Training Data Generation
Generate raw expert demonstrations for each scene separately:

```bash
python scripts/autobio_scripts/centrifuge5910_tasks.py
python scripts/autobio_scripts/thermal_cycler_tasks.py
```

By default, these commands write episode folders directly to:

- `logs/centrifuge5910_tasks`
- `logs/thermal_cycler_tasks`

To change the number of episodes per sub-task or output directory:

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

## Fine-tuning
When you need to finetune a new specific task, add config in `third_party/openpi/src/openpi/training/config.py`.

```bash
cd third_party/openpi
python scripts/compute_norm_stats.py --config-name mani_thermalcycler_pi05
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python scripts/train.py mani_thermalcycler_pi05 --exp-name mani_thermalcycler_pi05_finetune
```

### LABVLA fine-tuning

Sci-VLA UR5e demonstrations use the LABVLA schema `scivla_ur5e_single_arm`.
The existing `scripts/convert.py` output can be reused directly.

```bash
cd third_party/labvla
python -m data_process stats \
  --dataset ~/.cache/huggingface/lerobot/mani_centrifuge5910 \
  --schema scivla_ur5e_single_arm
```

Use LABVLA's training launcher or `scripts/train.py` with the Sci-VLA repo id,
the LeRobot cache root, and `DatasetSchema=scivla_ur5e_single_arm`. For example,
set the launcher variables to:

```bash
DataRoot=~/.cache/huggingface/lerobot
RepoIds=mani_centrifuge5910
DatasetSchema=scivla_ur5e_single_arm
ExternalStatsPath=~/.cache/huggingface/lerobot/mani_centrifuge5910/meta/stats.json
```

Also set `PretrainedCkpt` and the model/tokenizer paths at the top of the
launcher. Then run the LABVLA fine-tuning launcher after editing those values:

```bash
bash launch/finetune/train_labutopia.sh
```

If you prefer a direct invocation, use LABVLA's `scripts/train.py` with the same
data root, repo id, schema, and stats settings from the launcher.

## Evaluation

### Extract initial qpos json file from lerobot dataset

```bash
python scripts/autobio_scripts/export_lerobot_initial_qpos.py --repo_id mani_thermalcycler 
```

### Convert jax model to pytorch model
If you want to use pytorch model to evaluate tasks, converting the jax checkpoint to pytorch is needed:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* ~/anaconda3/envs/scivla/lib/python3.11/site-packages/transformers
python scripts/convert_jax_model_to_pytorch.py --checkpoint_dir checkpoints/mani_thermalcycler_pi05/ --config_name mani_thermalcycler_pi05 --output_path checkpoints/mani_thermalcycler_pi05_pytorch
```

### Evaluate the policy model on simulations
To evaluate the policy model, open a shell and run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py policy:checkpoint --policy.config 'mani_thermalcycler_pi05' --policy.dir 'checkpoints/mani_thermalcycler_pi05/mani_thermalcycler_pi05_finetune/49999/'
```

then open another shell and run evaluation:

```bash
export BASE_URL="https://openai.sufy.com/v1"
export MODEL_NAME="qwen3.5-397b-a17b"
export API_KEY=""

python ./scripts/autobio_scripts/evaluate.py --task 'thermal_cycler_long_task_1' --time_limit 30 --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler" --experiment-mode full
```

Add `--no-render-video` to skip replay video capture and mp4 writing while still running evaluation and counting success.

### Problem-validation video demo

The policy server must already be running, and the local dataset must exist at
`~/.cache/huggingface/lerobot/mani_thermalcycler`. Then run the fixed demo with:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --problem-validation-demo \
  --seed 0 \
  --time_limit 30
```

The demo executes the fixed prompts `open the lid of the thermal cycler` and
`place pcrPlate into the thermal cycler`. The same `--seed` reproduces the same
placement trajectory and frame selection. The frame is sampled uniformly from
the first 30% of a trajectory matching `place pcrPlate into the thermal cycler`.
Front- and left-view MP4s are saved under `videos/` with the fixed
`problem_validation_open_lid_place_pcr_plate` filename prefix.

<!-- python ./scripts/autobio_scripts/evaluate.py --task 'centrifuge5910_long_task_1' --time_limit 30 --prompts "open the lid of the centrifuge5910,pick the experimental centrifuge tube from rack and place it into the centrifuge5910,pick the balance centrifuge tube from rack and place it into the centrifuge5910,close the lid of the centrifuge5910,press the screen button to start the centrifuge5910" --experiment-mode baseline --num_episodes 20 --no-render-video

python ./scripts/autobio_scripts/evaluate.py --task 'centrifuge5910_long_task_2' --time_limit 30 --prompts "press the screen button to start the centrifuge5910,open the lid of the centrifuge5910,pick the experimental centrifuge tube from the centrifuge5910 and place it on the rack,pick the balance centrifuge tube from the centrifuge5910 and place it on the rack,close the lid of the centrifuge5910" --experiment-mode baseline --num_episodes 20 --no-render-video

python ./scripts/autobio_scripts/evaluate.py --task 'thermal_cycler_long_task_2' --time_limit 30 --prompts "press the button to start the thermal cycler,screw loosen the knob of the thermal cycler,open the lid of the thermal cycler,take pcrPlate from the thermal cycler,close the lid of the thermal cycler" --experiment-mode baseline --num_episodes 20 --no-render-video

python ./scripts/autobio_scripts/evaluate.py --task 'thermal_cycler_long_task_1' --time_limit 30 --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler,screw loosen the knob of the thermal cycler,open the lid of the thermal cycler,take pcrPlate from the thermal cycler" --experiment-mode baseline --num_episodes 20 --no-render-video

python ./scripts/autobio_scripts/evaluate.py --task 'centrifuge5910_long_task_1' --time_limit 30 --prompts "open the lid of the centrifuge5910,pick the experimental centrifuge tube from rack and place it into the centrifuge5910,pick the balance centrifuge tube from rack and place it into the centrifuge5910,close the lid of the centrifuge5910,press the screen button to start the centrifuge5910,open the lid of the centrifuge5910,pick the experimental centrifuge tube from the centrifuge5910 and place it on the rack,close the lid of the centrifuge5910" --experiment-mode baseline --num_episodes 20 --no-render-video
-->

### Experiment modes

`evaluate.py` uses `--experiment-mode` to select the transition strategy between prompts:

| Mode | Transition behavior |
| --- | --- |
| `no-transition` | Disable transitions between prompts. This is the default mode. |
| `baseline` | Randomly sample an initial pose directly from the dataset without retrieval, then use RRT to restore to that pose; if RRT fails, it prints `FALLBACK` and falls back to interpolation. |
| `no-retrieval` | Do not provide a target pose. Run the planning/coding agents only, and do not append final target-pose restoration. |
| `no-agent` | Retrieve the target pose, skip planning/coding agents, and directly interpolate to the retrieved pose. |
| `full` | Full Sci-VLA transition pipeline: retrieval, planning agent, coding/primitive execution, and final target-pose restoration. |

For example:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --task 'thermal_cycler_long_task_1' \
  --time_limit 30 \
  --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler"  \
  --experiment-mode baseline
```

### Evaluate LABVLA on simulations

Start a LABVLA websocket server from the LABVLA environment:

```bash
cd third_party/labvla
python deployment/serve_labvla.py \
  --pretrained_path /path/to/labvla/checkpoint \
  --port 8000 \
  --device cuda \
  --training_repo_id mani_thermalcycler
```

Then run Sci-VLA evaluation with the LABVLA observation adapter:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --policy-backend labvla \
  --host 127.0.0.1 \
  --port 8000 \
  --task 'thermal_cycler_long_task_1' \
  --time_limit 30 \
  --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler" \
  --experiment-mode no-transition
```

### Evaluate the model using local VLM model (Qwen3.5)

Establish the policy model:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py policy:checkpoint --policy.config 'mani_thermalcycler_pi05' --policy.dir 'checkpoints/mani_thermalcycler_pi05/mani_thermalcycler_pi05_finetune/49999/'
```

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

[![Star History Chart](https://api.star-history.com/svg?repos=DEFENSE-SEU/Sci-VLA&type=Date)](https://star-history.com/#DEFENSE-SEU/Sci-VLA&Date) -->





## Citation

If you find Sci-VLA useful in your research, please cite the paper:

```bibtex
@article{pang2026sci,
  title={Sci-VLA: Agentic VLA Inference Plugin for Long-Horizon Tasks in Scientific Experiments},
  author={Pang, Yiwen and Zhou, Bo and Li, Changjin and Wang, Xuanhao and Xu, Shengxiang and Wang, Deng-Bao and Zhang, Min-Ling and Di, Shimin},
  journal={arXiv preprint arXiv:2602.09430},
  year={2026}
}
```
