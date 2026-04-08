# Sci-VLA (Agentic VLA Inference Plugin)

This repository contains the Sci-VLA architecture and some simulation assets (including Autobio assets and newly added assets).

## Installation (editable local install):
If you already have both openpi and autobio environments, skip this section.

```bash
conda create -n scivla python=3.11
conda activate scivla
conda install ffmpeg=7.1.1 -c conda-forge
cd third_party/openpi
pip install uv
uv pip install -e .
uv pip install 'mujoco==3.3.0' numpy scipy toppra trimesh shapely triangle manifold3d sympy zstandard tqdm networkx usd-core ffmpeg imageio[ffmpeg] matplotlib scikit-image openai pytest chex
cp -r src/openpi/models_pytorch/transformers_replace/* ~/anaconda3/envs/scivla/lib/python3.11/site-packages/transformers
```


## Training Data Generation
To generate long-horizon tasks, run:

```bash
python scripts/autobio_scripts/centrifuge5910_tasks.py
python scripts/autobio_scripts/thermal_cycler_tasks.py
python scripts/autobio_scripts/cleaning_tasks.py
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
python scripts/compute_norm_stats.py --config-name long_tasks_pi05-lora
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python scripts/train.py long_tasks_pi05-lora --exp-name long_tasks_pi05-lora_finetune
```

## Evaluation

### Convert jax model to pytorch model
If you want to use pytorch model to evaluate tasks, converting the jax checkpoint to pytorch is needed:

```bash
python scripts/convert_jax_model_to_pytorch.py --checkpoint_dir checkpoints/long_tasks_pi05-lora/ --config_name long_tasks_pi05-lora --output_path checkpoints/long_tasks_pi05-lora_pytorch
```

### Evaluate the model on simulations
To evaluate the policy model, run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py policy:checkpoint --policy.config 'long_tasks_pi05-lora' --policy.dir 'checkpoints/long_tasks_pi05-lora/...'
```

then open another shell window and run:

```bash
export BASE_URL=your_url
export MODEL_NAME=your_model
export API_KEY=your_api_key

# cleaning table task
python ./scripts/autobio_scripts/evaluate.py --task 'place_centrifugeTube_into_basket' --num_episodes 20 --time_limit 30 --prompts "place pipette tip box into the basket,place pcr plate into the basket,place centrifuge tube into the basket"

# task A
python ./scripts/autobio_scripts/evaluate.py --task 'centrifuge5910_long_task_1' --num_episodes 20 --time_limit 30 --prompts "open the lid of the centrifuge5910,pick the experimental centrifuge tube from rack and place it into the centrifuge5910,pick the balance centrifuge tube from rack and place it into the centrifuge5910,close the lid of the centrifuge5910,press the screen button to start the centrifuge5910"

# task B
python ./scripts/autobio_scripts/evaluate.py --task 'centrifuge5910_long_task_2' --num_episodes 20 --time_limit 30 --prompts "press the screen button to stop the centrifuge5910,open the lid of the centrifuge5910,pick the experimental centrifuge tube from the centrifuge5910 and place it on the rack,pick the balance centrifuge tube from the centrifuge5910 and place it on the rack,close the lid of the centrifuge5910"

# task C
python ./scripts/autobio_scripts/evaluate.py --task 'thermal_cycler_long_task_1' --num_episodes 20 --time_limit 30 --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler"

# task D
python ./scripts/autobio_scripts/evaluate.py --task 'thermal_cycler_long_task_2' --num_episodes 20 --time_limit 30 --prompts "press the button of the thermal cycler,screw loosen the knob of the thermal cycler,open the lid of the thermal cycler,take pcrPlate from the thermal cycler,close the lid of the thermal cycler"

# task E
python ./scripts/autobio_scripts/evaluate.py --task 'centrifuge5910_long_task_1' --num_episodes 20 --time_limit 30 --prompts "open the lid of the centrifuge5910,pick the experimental centrifuge tube from rack and place it into the centrifuge5910,pick the balance centrifuge tube from rack and place it into the centrifuge5910,press the screen button to start the centrifuge5910,press the screen button to stop the centrifuge5910,pick the experimental centrifuge tube from the centrifuge5910 and place it on the rack,pick the balance centrifuge tube from the centrifuge5910 and place it on the rack,close the lid of the centrifuge5910"

# task F
python ./scripts/autobio_scripts/evaluate.py --task 'thermal_cycler_long_task_1' --num_episodes 20 --time_limit 30 --prompts "open the lid of the thermal cycler,place pcrPlate into the thermal cycler,close the lid of the thermal cycler,screw tighten the knob of the thermal cycler,press the button to start the thermal cycler,screw loosen the knob of the thermal cycler,open the lid of the thermal cycler,take pcrPlate from the thermal cycler"
```

