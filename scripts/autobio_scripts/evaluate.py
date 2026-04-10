import os
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import sys
import subprocess

from tqdm import tqdm
import numpy as np

if TYPE_CHECKING:
    from evaluator import Evaluator, Policy


def resolve_mujoco_gl(cli_gl: str | None = None) -> str:
    # Priority: CLI arg > existing MUJOCO_GL env > safe default.
    backend = cli_gl or os.environ.get("MUJOCO_GL") or "auto"
    backend = backend.lower()
    if backend not in {"auto", "egl", "osmesa", "glfw"}:
        raise ValueError(f"Unsupported MUJOCO_GL backend: {backend}")
    if backend == "auto":
        backend = probe_mujoco_gl_backend()
    return backend


def _probe_single_backend(backend: str) -> bool:
    env = os.environ.copy()
    env["MUJOCO_GL"] = backend
    if backend == "egl":
        env["PYOPENGL_PLATFORM"] = "egl"
    elif backend == "osmesa":
        env["PYOPENGL_PLATFORM"] = "osmesa"
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        env.pop("PYOPENGL_PLATFORM", None)
        env.pop("MUJOCO_EGL_DEVICE_ID", None)

    result = subprocess.run(
        [sys.executable, "-c", "import mujoco; print('ok')"],
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def probe_mujoco_gl_backend() -> str:
    # Prefer GPU EGL first on servers, then CPU OSMesa, then glfw.
    for backend in ["egl", "osmesa", "glfw"]:
        if _probe_single_backend(backend):
            print(f"[Render] Auto-selected MUJOCO_GL backend: {backend}")
            return backend

    raise RuntimeError(
        "No usable MUJOCO_GL backend found (tried egl/osmesa/glfw). "
        "Install server GL runtime libraries, then retry."
    )


def configure_mujoco_env(gl_backend: str, render_device_id: str | None = None):
    os.environ["MUJOCO_GL"] = gl_backend
    if gl_backend == "egl":
        if render_device_id is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_device_id)
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    elif gl_backend == "osmesa":
        # On headless servers, forcing OSMesa loader avoids inheriting an incompatible EGL platform.
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        # glfw path expects desktop GL/X11 stack; clear explicit PyOpenGL platform overrides.
        os.environ.pop("PYOPENGL_PLATFORM", None)
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

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
    no_planning: bool = False,
    no_interpolation: bool = False,
    llm_config: dict | None = None,
):
    evaluator.task.reset(seed=seed)
    # evaluator.task.set_serializer(log_root="logs/xxxx", log_name=str(seed))
    return evaluator.evaluate(
        policy,
        time_limit,
        prompts=prompts,
        use_transition_generation=use_transition_generation,
        no_planning=no_planning,
        no_interpolation=no_interpolation,
        llm_config=llm_config,
    )


def normalize_eval_result(raw_result):
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        success, timing = raw_result
        if isinstance(timing, dict):
            return bool(success), timing
        return bool(success), None

    if isinstance(raw_result, dict):
        success = bool(raw_result.get("success", False))
        timing = raw_result.get("timing")
        if isinstance(timing, dict):
            return success, timing
        return success, None

    return bool(raw_result), None


def print_running_average_timing(timings: list[dict]):
    if len(timings) == 0:
        return

    planning_avg = float(np.mean([t.get("transition_planning_avg_per_transition", 0.0) for t in timings]))
    transition_avg = float(np.mean([t.get("transition_avg_per_transition", 0.0) for t in timings]))
    episode_avg = float(np.mean([t.get("episode_total", 0.0) for t in timings]))
    ratio_avg = float(np.mean([t.get("transition_ratio", 0.0) for t in timings]))
    total_transitions = int(np.sum([t.get("transition_count", 0) for t in timings]))

    planning_total = float(np.sum([t.get("transition_planning_total", 0.0) for t in timings]))
    transition_total = float(np.sum([t.get("transition_total", 0.0) for t in timings]))
    global_planning_avg = (planning_total / total_transitions) if total_transitions > 0 else 0.0
    global_transition_avg = (transition_total / total_transitions) if total_transitions > 0 else 0.0

    print(
        f"[TimingAvg] episodes={len(timings)} | "
        f"transitions={total_transitions} | "
        f"avg planning/transition={planning_avg:.3f}s | "
        f"avg transition/transition={transition_avg:.3f}s | "
        f"global planning/transition={global_planning_avg:.3f}s | "
        f"global transition/transition={global_transition_avg:.3f}s | "
        f"avg episode total={episode_avg:.3f}s | "
        f"avg transition ratio={ratio_avg:.2f}%"
    )

_evaluator: "Evaluator"
_policy: "Policy"
_prompts: list[str] | None = None
_time_limit: float
_use_transition_generation: bool
_no_planning: bool
_no_interpolation: bool
_llm_config: dict | None = None
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
    no_planning: bool,
    no_interpolation: bool,
    llm_config: dict | None,
    mujoco_gl: str,
    queue,
    prompts: list[str] | None = None,
):
    import os
    render_device_id = queue.get()
    configure_mujoco_env(mujoco_gl, str(render_device_id))
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from task import create_task
    from evaluator import Evaluator
    global _evaluator, _policy, _prompts, _time_limit, _use_transition_generation, _no_planning, _no_interpolation, _llm_config
    task = create_task(task_name)
    _evaluator = Evaluator(task, image_history=image_history, video_fps=video_fps)
    _policy = make_policy(host, port)
    _prompts = prompts
    _time_limit = time_limit
    _use_transition_generation = use_transition_generation
    _no_planning = no_planning
    _no_interpolation = no_interpolation
    _llm_config = llm_config

def step_worker(seed: int):
    return evaluate_task(
        _evaluator,
        _policy,
        seed,
        _time_limit,
        _prompts,
        _use_transition_generation,
        _no_planning,
        _no_interpolation,
        _llm_config,
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
        "--mujoco-gl",
        type=str,
        default="auto",
        choices=["auto", "egl", "osmesa", "glfw"],
        help="MuJoCo rendering backend; auto probes egl -> osmesa -> glfw",
    )
    parser.add_argument(
        "--use-transition-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run transition generation and execution between prompts",
    )
    parser.add_argument(
        "--no_planning",
        action="store_true",
        help="Skip transition planning/code generation; keep qpos retrieval and target-qpos restore generation",
    )
    parser.add_argument(
        "--no_interpolation",
        action="store_true",
        help="Run retrieval/planning/codegen but skip final move_to_target_qpos in transition execute",
    )
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM base URL for transition generation")
    parser.add_argument("--llm-model-name", type=str, default=None, help="LLM model name for transition generation")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key for transition generation")
    parser.add_argument("--llm-temperature", type=float, default=None, help="LLM sampling temperature")
    parser.add_argument("--llm-top-p", type=float, default=None, help="LLM sampling top-p")
    parser.add_argument("--llm-max-tokens", type=int, default=None, help="LLM max output tokens")
    parser.add_argument("--llm-max-attempts", type=int, default=None, help="Max retry attempts per LLM stage")
    parser.add_argument("--llm-timeout", type=float, default=None, help="LLM request timeout in seconds")
    parser.add_argument(
        "--llm-thinking",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="LLM thinking mode for compatible models/backends",
    )
    parser.add_argument(
        "--llm-backend-mode",
        type=str,
        default="auto",
        choices=["auto", "responses", "chat"],
        help="LLM API mode: auto (local base_url->chat, remote->responses), or force responses/chat",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mujoco_gl = resolve_mujoco_gl(args.mujoco_gl)
    setup_output_logging(args.log_path)

    master_rng = np.random.default_rng(args.seed)
    seeds = master_rng.integers(0, 2**32 - 1, size=args.num_episodes).tolist()
    prompts = args.prompts.split(',') if args.prompts else None
    time_limit = args.time_limit
    render_device_ids = args.render_device_id.split(',')
    llm_config = {
        "base_url": args.llm_base_url,
        "model_name": args.llm_model_name,
        "api_key": args.llm_api_key,
        "temperature": args.llm_temperature,
        "top_p": args.llm_top_p,
        "max_tokens": args.llm_max_tokens,
        "max_attempts": args.llm_max_attempts,
        "timeout": args.llm_timeout,
        "thinking": args.llm_thinking,
        "backend_mode": args.llm_backend_mode,
    }
    assert len(render_device_ids) > 0
    success_results: list[float] = []
    episode_timings: list[dict] = []

    if args.num_workers == 0:
        # Serial evaluation
        configure_mujoco_env(mujoco_gl, render_device_ids[0])
        from task import create_task
        from evaluator import Evaluator
        policy = make_policy(args.host, args.port)
        task = create_task(args.task)
        evaluator = Evaluator(task, image_history=args.image_history, video_fps=args.video_fps)
        for seed in tqdm(seeds):
            raw_result = evaluate_task(
                evaluator,
                policy,
                seed,
                time_limit,
                prompts,
                args.use_transition_generation,
                args.no_planning,
                args.no_interpolation,
                llm_config,
            )
            success, timing = normalize_eval_result(raw_result)
            success_results.append(float(success))
            if timing is not None:
                episode_timings.append(timing)
            if args.num_episodes != 1 and len(episode_timings) > 0:
                print_running_average_timing(episode_timings)
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
                args.no_planning,
                args.no_interpolation,
                llm_config,
                mujoco_gl,
                queue,
                prompts,
            )
        ) as executor:
            for device_id in render_device_assignment:
                queue.put(device_id)
            for raw_result in tqdm(executor.map(step_worker, seeds), total=len(seeds), desc="Evaluating tasks"):
                success, timing = normalize_eval_result(raw_result)
                success_results.append(float(success))
                if timing is not None:
                    episode_timings.append(timing)
                if args.num_episodes != 1 and len(episode_timings) > 0:
                    print_running_average_timing(episode_timings)

    results = success_results
    if args.num_episodes != 1 and len(episode_timings) > 0:
        print("[TimingAvg] Final average over all episodes:")
        print_running_average_timing(episode_timings)
        total_transitions = int(np.sum([t.get("transition_count", 0) for t in episode_timings]))
        transition_total = float(np.sum([t.get("transition_total", 0.0) for t in episode_timings]))
        final_transition_avg = (transition_total / total_transitions) if total_transitions > 0 else 0.0
        print(
            f"[TimingAvg] Final global transition avg duration: "
            f"{final_transition_avg:.3f}s (total_transitions={total_transitions})"
        )

    if args.save:
        import json
        with open(args.save, 'w') as f:
            json.dump(results, f)
    else:
        print("Evaluation results:", results)
