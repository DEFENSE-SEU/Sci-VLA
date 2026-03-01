import os
os.environ["MUJOCO_GL"] = "egl"
from typing import TYPE_CHECKING

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

def evaluate_task(evaluator: "Evaluator", policy: "Policy", seed: int, time_limit: float, prompts: list[str] | None = None):
    evaluator.task.reset(seed=seed)
    # evaluator.task.set_serializer(log_root="logs/xxxx", log_name=str(seed))
    return evaluator.evaluate(policy, time_limit, prompts=prompts)

_evaluator: "Evaluator"
_policy: "Policy"
_prompts: list[str] | None = None

def init_worker(host: str, port: int, task_name: str, image_history: int, time_limit: float, queue, prompts: list[str] | None = None):
    import os
    render_device_id = queue.get()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_device_id)
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from task import create_task
    from evaluator import Evaluator
    global _evaluator, _policy, _prompts
    task = create_task(task_name)
    _evaluator = Evaluator(task, image_history=image_history)
    _policy = make_policy(host, port)
    _prompts = prompts
    _time_limit = time_limit

def step_worker(seed: int):
    return evaluate_task(_evaluator, _policy, seed, _time_limit, _prompts)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a policy using the WebSocket client.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket server port")
    parser.add_argument("--task", type=str, default="pickup", help="Task name")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--image_history", type=int, default=0, help="Image history for the policy")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for parallel evaluation, 0 for serial")
    parser.add_argument("--save", type=str, default=None, help="Output file for evaluation results")
    parser.add_argument("--seed", type=int, default=None, help="Master seed for evaluation")
    parser.add_argument("--render_device_id", type=str, default='0', help="Comma-separated list of GPU device IDs for rendering")
    parser.add_argument("--prompts", type=str, default=None, help="Comma-separated list of prompts to execute sequentially")
    parser.add_argument("--time_limit", type=float, default=100, help="per task time limit")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

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
        evaluator = Evaluator(task, image_history=args.image_history)
        results = [evaluate_task(evaluator, policy, seed, time_limit, prompts) for seed in tqdm(seeds)]
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
            initargs=(args.host, args.port, args.task, args.image_history, time_limit, queue, prompts)
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
