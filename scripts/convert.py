#!/usr/bin/env python

import json
import os
import numpy as np
from pathlib import Path
from io import BytesIO
import shutil

import zstandard as zstd
import cv2
import tyro

# LeRobot renamed this cache variable. Normalize the legacy spelling before
# importing LeRobot, whose recent versions reject LEROBOT_HOME outright.
legacy_lerobot_home = os.environ.pop("LEROBOT_HOME", None)
if legacy_lerobot_home and "HF_LEROBOT_HOME" not in os.environ:
    os.environ["HF_LEROBOT_HOME"] = legacy_lerobot_home

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
LEROBOT_HOME = Path(
    os.getenv("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
).expanduser()

def take_state_split(arr, split):
    start = split['start']
    end = split['end']
    shape = tuple(split['shape'])
    dtype = split['dtype']
    return arr[..., start:end].reshape(arr.shape[:-1] + shape).astype(dtype)

def load_log(log_dir: Path):
    log_dir = Path(log_dir)
    with open(log_dir / "states.npy.zst", "rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as zstd_f:
            states_io = BytesIO(zstd_f.read())
    states = np.load(states_io)
    with open(log_dir / "info.json", "r") as f:
        info = json.load(f)
    with open(log_dir / "downsample.json", "r") as f:
        downsample = json.load(f)
    return states, info, downsample

def confirm(message: str):
    """Prompt the user for confirmation."""
    import sys
    if not sys.stdin.isatty():
        return True  # Assume yes if not in a terminal
    while True:
        response = input(f"{message} (y/n): ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def probe_log(log_dir: Path):
    with open(log_dir / "info.json", "r") as f:
        info = json.load(f)
    task = info["task"]
    state_dim = len(task["state_indices"])
    action_dim = len(task["action_indices"])
    with open(log_dir / "downsample.json", "r") as f:
        downsample = json.load(f)
    fps = downsample["fps"]
    height = downsample["height"]
    width = downsample["width"]
    camera_mapping = task["camera_mapping"]
    return fps, height, width, state_dim, action_dim, list(camera_mapping.keys())

def configure_lerobot_video_encoder(video_codec: str, video_crf: int, video_gop: int | None):
    valid_codecs = {"h264", "hevc", "libsvtav1"}
    if video_codec not in valid_codecs:
        raise ValueError(f"Unsupported video codec {video_codec!r}; expected one of {sorted(valid_codecs)}")

    original_encode_video_frames = lerobot_dataset_module.encode_video_frames

    def encode_video_frames_compat(imgs_dir, video_path, fps, **kwargs):
        kwargs["vcodec"] = video_codec
        kwargs["pix_fmt"] = "yuv420p"
        kwargs["crf"] = video_crf
        kwargs["g"] = video_gop
        return original_encode_video_frames(imgs_dir, video_path, fps, **kwargs)

    lerobot_dataset_module.encode_video_frames = encode_video_frames_compat


def validate_video_file(video_path: Path, *, min_mean: float = 1.0, min_std: float = 1.0):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open converted video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Converted video has no frames: {video_path}")

    sample_indices = sorted({0, frame_count // 2, frame_count - 1})
    means = []
    stds = []
    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Failed to decode frame {frame_index} from converted video: {video_path}")
        means.append(float(frame.mean()))
        stds.append(float(frame.std()))
    capture.release()

    if max(means) <= min_mean and max(stds) <= min_std:
        raise RuntimeError(
            f"Converted video looks black: {video_path} "
            f"means={means} stds={stds}"
        )


def validate_episode_videos(dataset: LeRobotDataset, episode_index: int, log_dir: Path):
    for video_key in dataset.meta.video_keys:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, video_key)
        try:
            validate_video_file(video_path)
        except Exception as exc:
            raise RuntimeError(
                f"Video validation failed for episode={episode_index} key={video_key!r} "
                f"source_log={log_dir}"
            ) from exc


def main(
    data_dir: str,
    repo_id: str,
    video_codec: str = "h264",
    video_crf: int = 23,
    video_gop: int | None = 2,
    validate_videos: bool = True,
):
    configure_lerobot_video_encoder(video_codec, video_crf, video_gop)
    parent_path = Path(data_dir)
    output_path = LEROBOT_HOME / repo_id

    if output_path.exists():
        if confirm(f"Output path {output_path} already exists. Delete it?"):
            shutil.rmtree(output_path)
        else:
            print("Exiting without changes.")
            return

    log_folders = sorted(d for d in parent_path.iterdir() if d.is_dir())
    assert len(log_folders) > 0, "No log folders found in the specified directory."

    fps, height, width, state_dim, action_dim, camera_keys = probe_log(log_folders[0])
    image_shape = (height, width, 3)
    print(f"FPS: {fps}")
    print(f"Image shape: {image_shape}")
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    print(f"Camera keys: {camera_keys}")

    features ={
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
        "task_is_complete": {
            "dtype": "float32",
            # LeRobot 0.4 maps shape (1,) to an Arrow scalar while its frame
            # validator still requires a NumPy array. A singleton 2-D feature
            # avoids that incompatible code path and remains scalar-valued.
            "shape": (1, 1),
            "names": ["task_is_complete"],
        },
    }
    for camera_key in camera_keys:
        features[camera_key] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=0,
    )

    for episode_index, log_dir in enumerate(log_folders):
        states, info, downsample = load_log(log_dir)
        
        state_splits = info["split"]
        task = info["task"]
        indices = downsample["indices"]
        prompt = task["prefix"]
        state_indices = task["state_indices"]
        action_indices = task["action_indices"]
        assert len(state_indices) == state_dim
        assert len(action_indices) == action_dim
        assert downsample["fps"] == fps

        camera_mapping = task["camera_mapping"]
        camera_files = downsample["cameras"]
        frame_infos = info.get("info", [])
        if len(frame_infos) != len(states):
            raise ValueError(
                f"Frame info/state count mismatch in {log_dir}: "
                f"{len(frame_infos)} != {len(states)}"
            )
        def get_camera_file(camera_name):
            camera_file = camera_files[camera_name]
            camera_stream = cv2.VideoCapture(str(log_dir / camera_file))
            if not camera_stream.isOpened():
                raise RuntimeError(f"Failed to open video stream for {log_dir}")
            return camera_stream
        camera_streams = {camera_key: get_camera_file(camera) for camera_key, camera in camera_mapping.items()}

        for i in indices:
            state_record = states[i]
            qpos = take_state_split(state_record, state_splits["qpos"])
            ctrl = take_state_split(state_record, state_splits["ctrl"])
            if "task_is_complete" not in frame_infos[i]:
                raise ValueError(
                    f"Missing task_is_complete at state index {i} in {log_dir}. "
                    "Re-collect the trajectory or run "
                    "scripts/autobio_scripts/backfill_completion_labels.py on the raw logs."
                )

            frame = {
                "state": qpos[state_indices].astype(np.float32),
                "actions": ctrl[action_indices].astype(np.float32),
                "task_is_complete": np.asarray(
                    [[bool(frame_infos[i]["task_is_complete"])]],
                    dtype=np.float32,
                ),
                # "task": prompt,
            }

            for camera_key, camera_stream in camera_streams.items():
                ret, image = camera_stream.read()
                assert ret, f"Failed to read image at index {i} from {log_dir}"
                assert image.shape == image_shape, f"Image shape mismatch at index {i}: {image.shape} != {image_shape}"
                frame[camera_key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            dataset.add_frame(frame, task=prompt)

        for camera_stream in camera_streams.values():
            assert not camera_stream.grab(), f"Not all frames were read from {log_dir}"
            camera_stream.release()

        dataset.save_episode()
        if validate_videos:
            validate_episode_videos(dataset, episode_index, log_dir)

if __name__ == "__main__":
    tyro.cli(main)
