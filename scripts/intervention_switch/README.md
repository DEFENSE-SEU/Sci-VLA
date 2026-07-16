# Vision-language intervention switch

This first version predicts whether the task described by text is complete in
the current frame. It uses the first video or episode frame as the initial
visual state and a frozen CLIP image/text backbone with a small trainable binary
head.

## 1. Produce frame-level labels

New centrifuge5910_tasks and thermal_cycler_tasks demonstrations store
task_is_complete on every recorded MuJoCo state by calling the task's existing
physics-based check() function after each simulation step.

For raw logs collected before this change, backfill labels before conversion:

~~~bash
python scripts/autobio_scripts/backfill_completion_labels.py \
  logs/centrifuge5910_tasks logs/thermal_cycler_tasks
~~~

Then render and convert the raw logs. scripts/convert.py now writes
task_is_complete as a singleton float32 LeRobot feature and fails loudly when
a raw state is missing its label.

~~~bash
bash scripts/autobio_scripts/render_all.bash logs/centrifuge5910_tasks
bash scripts/autobio_scripts/render_all.bash logs/thermal_cycler_tasks

python scripts/convert.py --data-dir logs/centrifuge5910_tasks \
  --repo-id mani_centrifuge5910
python scripts/convert.py --data-dir logs/thermal_cycler_tasks \
  --repo-id mani_thermalcycler
~~~

## 2. Export the training dataset

Install the model-side dependencies in the same environment as LeRobot:

~~~bash
pip install -r scripts/intervention_switch/requirements.txt
~~~

Export frames, task descriptions, initial-frame references and labels from both
LeRobot datasets:

~~~bash
python scripts/intervention_switch/export_completion_dataset.py \
  --repo-id mani_centrifuge5910 mani_thermalcycler \
  --output-dir data/completion_switch_v1 \
  --image-key image
~~~

The output contains:

- manifest.jsonl: one image-text-label pair per line;
- images/: extracted RGB frames;
- summary.json: episode, split and class counts.

Splits are assigned by complete trajectory, never by individual frame. The
default task_text_config.json adds variable descriptions and physically valid
contradictory negatives. For example, a frame labeled complete for closing a
lid is paired with the open-lid goal as False. Contradictory pairs are added
only to positive terminal-state frames; arbitrary mismatched tasks are not
assumed to be negative.

Use --stride 2 or a larger value if 50 Hz trajectories contain too many nearly
identical frames. Edit or replace --text-config when adding tasks.

## 3. Train

~~~bash
python scripts/intervention_switch/train.py \
  --manifest data/completion_switch_v1/manifest.jsonl \
  --output-dir checkpoints/completion_switch_v1 \
  --epochs 20 \
  --batch-size 32
~~~

The CLIP backbone is frozen. Only the shared fusion/classification head is
trained. The script compensates for class imbalance, selects a validation
threshold subject to --min-precision (default 0.95), and saves best.pt,
history.jsonl, and test_metrics.json.

The first model download requires access to the Hugging Face model repository.
After it is cached, training and inference can run locally.

## 4. Annotate a video

~~~bash
python scripts/intervention_switch/infer_video.py \
  --checkpoint checkpoints/completion_switch_v1/best.pt \
  --video input.mp4 \
  --text "the thermal cycler lid is fully closed" \
  --output output_annotated.mp4
~~~

The first video frame is the initial-state reference. Each output frame shows
the raw completion probability and the debounced TASK COMPLETE: TRUE/FALSE
decision. By default, at least four of the latest five raw predictions must be
positive. For raw per-frame decisions without debounce, use:

~~~bash
--window-size 1 --required-positive 1
~~~

## Important limitation

This is a frame/initial-frame model, not a true video model. Persistent goals
such as lid position and object placement are appropriate. Momentary events
such as pressing a spring-return button may require a short temporal encoder or
robot-state input if the event is not visually persistent.
