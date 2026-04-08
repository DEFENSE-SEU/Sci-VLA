import os
os.environ["MUJOCO_GL"] = "egl"
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import sys

from tqdm import tqdm
import numpy as np

if TYPE_CHECKING:
    from evaluator import Evaluator, Policy

def make_policy(host: str, port: int) -> "Policy":
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    ws_policy = WebsocketClientPolicy(host, port)
    def policy_fn(obs: dict) -> np.ndarray:
        return ws_policy.infer(obs)['actions']
    return policy_fn

def evaluate_task(
    evaluator: "Evaluator",
    policy: "Policy",
    seed: int,
    time_limit: float,
    prompts: list[str] | None = None,
    use_transition_generation: bool = True,
):
    evaluator.task.reset(seed=seed)
    # evaluator.task.set_serializer(log_root="logs/xxxx", log_name=str(seed))
    return evaluator.evaluate(
        policy,
        time_limit,
        prompts=prompts,
        use_transition_generation=use_transition_generation,
    )

_evaluator: "Evaluator"
_policy: "Policy"
_prompts: list[str] | None = None
_time_limit: float
_use_transition_generation: bool
_log_file_handle = None


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_output_logging(log_path: str | None) -> Path:
    global _log_file_handle

    if log_path is None:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        final_log_path = log_dir / f"{timestamp}.log"
    else:
        final_log_path = Path(log_path)
        final_log_path.parent.mkdir(parents=True, exist_ok=True)

    _log_file_handle = open(final_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, _log_file_handle)
    sys.stderr = TeeStream(sys.stderr, _log_file_handle)
    print(f"[Log] Evaluate output is being saved to: {final_log_path}")
    return final_log_path

def init_worker(
    host: str,
    port: int,
    task_name: str,
    image_history: int,
    time_limit: float,
    video_fps: int,
    use_transition_generation: bool,
    queue,
    prompts: list[str] | None = None,
):
    import os
    render_device_id = queue.get()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_device_id)
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from task import create_task
    from evaluator import Evaluator
    global _evaluator, _policy, _prompts, _time_limit, _use_transition_generation
    task = create_task(task_name)
    _evaluator = Evaluator(task, image_history=image_history, video_fps=video_fps)
    _policy = make_policy(host, port)
    _prompts = prompts
    _time_limit = time_limit
    _use_transition_generation = use_transition_generation

def step_worker(seed: int):
    return evaluate_task(
        _evaluator,
        _policy,
        seed,
        _time_limit,
        _prompts,
        _use_transition_generation,
    )

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a policy using the WebSocket client.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket server port")
    parser.add_argument("--task", type=str, default="pickup", help="Task name")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to evaluate")
    parser.add_argument("--image_history", type=int, default=0, help="Image history for the policy")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for parallel evaluation, 0 for serial")
    parser.add_argument("--save", type=str, default=None, help="Output file for evaluation results")
    parser.add_argument("--seed", type=int, default=None, help="Master seed for evaluation")
    parser.add_argument("--render_device_id", type=str, default='0', help="Comma-separated list of GPU device IDs for rendering")
    parser.add_argument("--prompts", type=str, default=None, help="Comma-separated list of prompts to execute sequentially")
    parser.add_argument("--time_limit", type=float, default=100, help="per task time limit")
    parser.add_argument("--video_fps", type=int, default=20, help="Replay video FPS")
    parser.add_argument("--log_path", type=str, default=None, help="Path to save evaluate stdout/stderr log")
    parser.add_argument(
        "--use-transition-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run transition generation and execution between prompts",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_output_logging(args.log_path)

    master_rng = np.random.default_rng(args.seed)
    seeds = master_rng.integers(0, 2**32 - 1, size=args.num_episodes).tolist()
    prompts = args.prompts.split(',') if args.prompts else None
    time_limit = args.time_limit
    render_device_ids = args.render_device_id.split(',')
    assert len(render_device_ids) > 0

    if args.num_workers == 0:
        # Serial evaluation
        os.environ["MUJOCO_EGL_DEVICE_ID"] = render_device_ids[0]
        from task import create_task
        from evaluator import Evaluator
        policy = make_policy(args.host, args.port)
        task = create_task(args.task)
        evaluator = Evaluator(task, image_history=args.image_history, video_fps=args.video_fps)
        results = [
            evaluate_task(
                evaluator,
                policy,
                seed,
                time_limit,
                prompts,
                args.use_transition_generation,
            )
            for seed in tqdm(seeds)
        ]
    else:
        # Parallel evaluation
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        render_device_assignment = [
            i % len(render_device_ids) for i in range(args.num_workers)
        ]
        queue = multiprocessing.Queue()
            
        with ProcessPoolExecutor(
            max_workers=args.num_workers, initializer=init_worker,
            initargs=(
                args.host,
                args.port,
                args.task,
                args.image_history,
                time_limit,
                args.video_fps,
                args.use_transition_generation,
                queue,
                prompts,
            )
        ) as executor:
            for device_id in render_device_assignment:
                queue.put(device_id)
            results = list(tqdm(executor.map(step_worker, seeds), total=len(seeds), desc="Evaluating tasks"))

    results = [float(r) for r in results]

    if args.save:
        import json
        with open(args.save, 'w') as f:
            json.dump(results, f)
    else:
        print("Evaluation results:", results)
