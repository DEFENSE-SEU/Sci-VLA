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

If you already have both openpi and autobio environments, skip this section.

```bash
conda create -n scivla python=3.11
conda activate scivla
conda install ffmpeg=7.1.1 -c conda-forge
cd third_party/openpi
pip install uv
uv pip install -e .
uv pip install 'mujoco==3.3.0' numpy scipy toppra trimesh shapely triangle manifold3d sympy zstandard tqdm networkx usd-core ffmpeg imageio[ffmpeg] matplotlib scikit-image openai pytest chex
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
python scripts/compute_norm_stats.py --config-name mani_centrifuge5910_pi05
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python scripts/train.py mani_centrifuge5910_pi05 --exp-name mani_centrifuge5910_pi05_finetune
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
python scripts/autobio_scripts/export_lerobot_initial_qpos.py --repo_id mani_centrifuge5910 
```

### Convert jax model to pytorch model
If you want to use pytorch model to evaluate tasks, converting the jax checkpoint to pytorch is needed:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* ~/anaconda3/envs/scivla/lib/python3.11/site-packages/transformers
python scripts/convert_jax_model_to_pytorch.py --checkpoint_dir checkpoints/mani_centrifuge5910_pi05/ --config_name mani_centrifuge5910_pi05 --output_path checkpoints/mani_centrifuge5910_pi05_pytorch
```

### Evaluate the policy model on simulations
To evaluate the policy model, open a shell and run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py policy:checkpoint --policy.config 'mani_centrifuge5910_pi05' --policy.dir 'checkpoints/mani_centrifuge5910_pi05/mani_centrifuge5910_pi05_finetune/60000/'
```

then open another shell and run evaluation:

```bash
export BASE_URL="your_url"
export MODEL_NAME="your_model"
export API_KEY="your_api_key"

# example usage
python ./scripts/autobio_scripts/evaluate.py --task 'centrifuge5910_long_task_1' --time_limit 30 --prompts "open the lid of the centrifuge5910,pick the experimental centrifuge tube from rack and place it into the centrifuge5910,pick the balance centrifuge tube from rack and place it into the centrifuge5910,close the lid of the centrifuge5910,press the screen button to start the centrifuge5910"

python ./scripts/autobio_scripts/evaluate.py --task 'thermal_cycler_long_task_1' --time_limit 30 --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler"
```

### Evaluate LABVLA on simulations

Start a LABVLA websocket server from the LABVLA environment:

```bash
cd third_party/labvla
python deployment/serve_labvla.py \
  --pretrained_path /path/to/labvla/checkpoint \
  --port 8000 \
  --device cuda \
  --training_repo_id mani_centrifuge5910
```

Then run Sci-VLA evaluation with the LABVLA observation adapter:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --policy-backend labvla \
  --host 127.0.0.1 \
  --port 8000 \
  --task 'centrifuge5910_long_task_1' \
  --time_limit 30 \
  --prompts "open the lid of the centrifuge5910,pick the experimental centrifuge tube from rack and place it into the centrifuge5910,pick the balance centrifuge tube from rack and place it into the centrifuge5910,close the lid of the centrifuge5910,press the screen button to start the centrifuge5910"
```

### Evaluate the model using local VLM model (Qwen3.5)

Establish the policy model:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py policy:checkpoint --policy.config 'mani_centrifuge5910_pi05' --policy.dir 'checkpoints/mani_centrifuge5910_pi05/mani_centrifuge5910_pi05_finetune/100000/'
```

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 \
  --port 9000
```


then open another shell window and run:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --task "thermal_cycler_long_task_1" \
  --time_limit 30 \
  --prompts "open the lid of the centrifuge5910,pick the experimental centrifuge tube from rack and place it into the centrifuge5910,pick the balance centrifuge tube from rack and place it into the centrifuge5910,close the lid of the centrifuge5910,press the screen button to start the centrifuge5910" \
  --llm-base-url http://127.0.0.1:9000/v1 \
  --llm-model-name qwen3.5-9b \
  --llm-api-key EMPTY \
  --llm-temperature 0.2 \
  --llm-top-p 0.9 \
  --llm-max-tokens 4096 \
  --llm-max-attempts 3 \
  --llm-timeout 120 \
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
