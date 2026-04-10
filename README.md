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
<!-- sudo apt-get update
sudo apt-get install -y libegl1 libgles2 libgl1 libglvnd0 libosmesa6 libosmesa6-dev -->

## Training Data Generation
To generate long-horizon tasks, run:

```bash
python scripts/autobio_scripts/centrifuge5910_tasks.py
python scripts/autobio_scripts/thermal_cycler_tasks.py
```

Render camera view:

```bash
bash scripts/autobio_scripts/render_all.bash logs/
```

Then, move all data contained within the sub-task folders (folders named by timestamp) into the `long_tasks` folder.
Finally, convert `long_tasks` to lerobot data:

```bash
python scripts/convert.py --data-dir logs/long_tasks --repo-id long_tasks
```

## Fine-tuning
When you need to train specific task, add config in third_party/openpi/src/openpi/training/config.py. (Config `long_tasks` and `long_tasks_pi05` have already included.)

```bash
cd third_party/openpi
python scripts/compute_norm_stats.py --config-name long_tasks_pi05
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python scripts/train.py long_tasks_pi05 --exp-name long_tasks_pi05_finetune
```

## Evaluation

### Extract initial qpos json file from lerobot dataset

```bash
python scripts/autobio_scripts/export_lerobot_initial_qpos.py --repo_id long_tasks 
```

### Convert jax model to pytorch model
If you want to use pytorch model to evaluate tasks, converting the jax checkpoint to pytorch is needed:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* ~/anaconda3/envs/scivla/lib/python3.11/site-packages/transformers
python scripts/convert_jax_model_to_pytorch.py --checkpoint_dir checkpoints/long_tasks_pi05/ --config_name long_tasks_pi05 --output_path checkpoints/long_tasks_pi05_pytorch
```

### Evaluate the policy model on simulations
To evaluate the policy model, open a shell and run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py policy:checkpoint --policy.config 'long_tasks_pi05' --policy.dir 'checkpoints/long_tasks_pi05/long_tasks_pi05_finetune/100000/'
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

### Evaluate the model using local VLM model (Qwen3.5)

Establish the policy model:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py policy:checkpoint --policy.config 'long_tasks_pi05' --policy.dir 'checkpoints/long_tasks_pi05/long_tasks_pi05_finetune/100000/'
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
  --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler" \
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

