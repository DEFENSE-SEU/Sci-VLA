import os
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import sys
import subprocess

from tqdm import tqdm
import numpy as np

from problem_validation_demo import ProblemValidationDemoConfig

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

POLICY_BACKENDS = ("openpi", "labvla")
EXPERIMENT_MODES = ("no-transition", "baseline", "no-retrieval", "no-agent", "full")


def apply_problem_validation_demo_profile(args) -> ProblemValidationDemoConfig | None:
    if not args.problem_validation_demo:
        return None

    prefix_percent = float(getattr(args, "problem_validation_prefix_percent", 30.0))
    if not np.isfinite(prefix_percent) or not 0.0 < prefix_percent <= 100.0:
        raise ValueError(
            "Problem-validation prefix percent must be in (0, 100], "
            f"got {prefix_percent}"
        )

    config = ProblemValidationDemoConfig(prefix_fraction=prefix_percent / 100.0)
    args.task = config.task_name
    args.prompts = ",".join(config.prompts)
    args.num_episodes = 1
    args.num_workers = 0
    args.render_video = True
    args.intervention_mode = "non_timeout"
    args.experiment_mode = "no-transition"
    args.use_transition_generation = False
    args.transition_mode = "none"
    args.no_planning = False
    args.no_interpolation = False
    args.no_retrieval = False
    return config


def prepare_policy_observation(obs: dict, policy_backend: str) -> dict:
    if policy_backend == "openpi":
        return obs

    if policy_backend != "labvla":
        raise ValueError(f"Unsupported policy backend: {policy_backend}")

    required_keys = [
        "observation/state",
        "observation/image",
        "observation/wrist_image",
        "observation/wrist_image_2",
    ]
    missing = [key for key in required_keys if key not in obs or obs[key] is None]
    if missing:
        raise ValueError(
            "LABVLA policy backend requires observation keys "
            f"{required_keys}; missing {missing}"
        )

    return {
        **obs,
        "state": obs["observation/state"],
        "camera_1_rgb": obs["observation/image"],
        "camera_2_rgb": obs["observation/wrist_image"],
        "camera_3_rgb": obs["observation/wrist_image_2"],
    }


def make_policy(host: str, port: int, policy_backend: str = "openpi") -> "Policy":
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    ws_policy = WebsocketClientPolicy(host, port)

    def policy_fn(obs: dict) -> np.ndarray:
        request_obs = prepare_policy_observation(obs, policy_backend)
        return ws_policy.infer(request_obs)["actions"]

    return policy_fn


def resolve_experiment_mode_config(
    *,
    experiment_mode: str,
    use_transition_generation: bool,
    transition_mode: str,
    no_planning: bool,
    no_interpolation: bool,
    no_retrieval: bool,
) -> dict:
    mode = (experiment_mode or "no-transition").strip().lower().replace("_", "-")
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment_mode: {experiment_mode}")

    if mode == "no-transition":
        return {
            "use_transition_generation": False,
            "transition_mode": "none",
            "no_planning": False,
            "no_interpolation": False,
            "no_retrieval": False,
        }

    if mode == "baseline":
        return {
            "use_transition_generation": True,
            "transition_mode": "random_dataset_task_pose_rrt",
            "no_planning": False,
            "no_interpolation": False,
            "no_retrieval": False,
        }

    if mode == "no-retrieval":
        return {
            "use_transition_generation": True,
            "transition_mode": "llm",
            "no_planning": False,
            "no_interpolation": True,
            "no_retrieval": True,
        }

    if mode == "no-agent":
        return {
            "use_transition_generation": True,
            "transition_mode": "llm",
            "no_planning": True,
            "no_interpolation": False,
            "no_retrieval": False,
        }

    return {
        "use_transition_generation": True,
        "transition_mode": "llm",
        "no_planning": False,
        "no_interpolation": False,
        "no_retrieval": False,
    }

def evaluate_task(
    evaluator: "Evaluator",
    policy: "Policy",
    seed: int,
    time_limit: float,
    prompts: list[str] | None = None,
    use_transition_generation: bool = True,
    transition_mode: str = "auto",
    no_planning: bool = False,
    no_interpolation: bool = False,
    no_retrieval: bool = False,
    control_fps: float = 50.0,
    llm_config: dict | None = None,
    use_task_judge: bool = False,
    max_prompt_retries: int = 1,
    judge_confidence_threshold: float = 0.6,
    judge_on_error: str = "fail",
    intervention_mode: str = "non_timeout",
    transition_seed: int | None = None,
    problem_validation_demo_config: ProblemValidationDemoConfig | None = None,
):
    evaluator.task.reset(seed=seed)
    # evaluator.task.set_serializer(log_root="logs/xxxx", log_name=str(seed))
    return evaluator.evaluate(
        policy,
        time_limit,
        prompts=prompts,
        use_transition_generation=use_transition_generation,
        transition_mode=transition_mode,
        no_planning=no_planning,
        no_interpolation=no_interpolation,
        no_retrieval=no_retrieval,
        control_fps=control_fps,
        llm_config=llm_config,
        use_task_judge=use_task_judge,
        max_prompt_retries=max_prompt_retries,
        judge_confidence_threshold=judge_confidence_threshold,
        judge_on_error=judge_on_error,
        intervention_mode=intervention_mode,
        transition_seed=transition_seed,
        **(
            {"problem_validation_demo_config": problem_validation_demo_config}
            if problem_validation_demo_config is not None
            else {}
        ),
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


def build_atomic_task_success_summary(
    episode_timings: list[dict],
    num_episodes: int,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}

    for timing in episode_timings:
        episode_results = timing.get("atomic_task_results", [])
        if not isinstance(episode_results, list):
            continue

        prompt_success_by_episode: dict[str, bool] = {}
        for result_position, result in enumerate(episode_results):
            if not isinstance(result, dict):
                continue
            prompt = result.get("prompt")
            if prompt is None:
                prompt = "__task__"
            prompt = str(prompt)
            try:
                prompt_index = int(result.get("prompt_index", result_position))
            except (TypeError, ValueError):
                prompt_index = result_position
            prompt_key = f"prompt_index={prompt_index} {prompt}"
            prompt_success_by_episode[prompt_key] = (
                prompt_success_by_episode.get(prompt_key, False)
                or bool(result.get("success", False))
            )

        for prompt_key, prompt_success in prompt_success_by_episode.items():
            if prompt_key not in summary:
                summary[prompt_key] = {
                    "success_count": 0,
                    "episode_count": 0,
                    "max_success": int(num_episodes),
                }
            summary[prompt_key]["episode_count"] += 1
            if prompt_success:
                summary[prompt_key]["success_count"] += 1

    return summary


def print_atomic_task_success_summary(summary: dict[str, dict[str, int]]):
    if len(summary) == 0:
        return

    print("[AtomicTaskSuccess] Final per-atomic-task success counts:")
    for prompt, stats in summary.items():
        print(
            f"[AtomicTaskSuccess] {prompt}: "
            f"{stats['success_count']}/{stats['max_success']} "
            f"(evaluated_episodes={stats['episode_count']})"
        )


def build_complete_episode_success_summary(success_results: list[float], num_episodes: int) -> dict[str, int]:
    return {
        "success_count": int(sum(1 for result in success_results if bool(result))),
        "max_success": int(num_episodes),
        "evaluated_episodes": int(len(success_results)),
    }


def print_complete_episode_success_summary(summary: dict[str, int]):
    print(
        "[EpisodeSuccess] Complete episode success count: "
        f"{summary['success_count']}/{summary['max_success']} "
        f"(evaluated_episodes={summary['evaluated_episodes']})"
    )


def _transition_collision_index(raw_key) -> int | None:
    text = str(raw_key)
    if text.startswith("transition_"):
        text = text[len("transition_"):]
    try:
        index = int(text)
    except (TypeError, ValueError):
        return None
    return index if index > 0 else None


def build_transition_collision_summary(
    episode_timings: list[dict],
    num_episodes: int,
    expected_transition_count: int | None = None,
) -> dict[str, float]:
    totals: dict[int, int] = {}
    max_index = max(0, int(expected_transition_count or 0))

    for timing in episode_timings:
        raw_counts = timing.get("transition_collision_counts", {})
        items = []
        if isinstance(raw_counts, dict):
            items = list(raw_counts.items())
        elif isinstance(raw_counts, list):
            items = list(enumerate(raw_counts, start=1))

        for raw_index, raw_count in items:
            index = _transition_collision_index(raw_index)
            if index is None:
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            totals[index] = totals.get(index, 0) + count
            max_index = max(max_index, index)

    denominator = int(num_episodes) if int(num_episodes) > 0 else max(1, len(episode_timings))
    return {
        f"transition_{index}": float(totals.get(index, 0)) / float(denominator)
        for index in range(1, max_index + 1)
    }


def print_transition_collision_summary(summary: dict[str, float]):
    if len(summary) == 0:
        return

    formatted = ", ".join(
        f"{transition_name}:{avg_count:.3f}"
        for transition_name, avg_count in summary.items()
    )
    print(f"[TransitionCollisionAvg] {formatted}")

_evaluator: "Evaluator"
_policy: "Policy"
_prompts: list[str] | None = None
_time_limit: float
_use_transition_generation: bool
_transition_mode: str
_no_planning: bool
_no_interpolation: bool
_no_retrieval: bool
_control_fps: float
_llm_config: dict | None = None
_use_task_judge: bool
_max_prompt_retries: int
_judge_confidence_threshold: float
_judge_on_error: str
_intervention_mode: str
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
    policy_backend: str,
    task_name: str,
    image_history: int,
    time_limit: float,
    video_fps: int,
    render_video: bool,
    use_transition_generation: bool,
    transition_mode: str,
    no_planning: bool,
    no_interpolation: bool,
    no_retrieval: bool,
    control_fps: float,
    llm_config: dict | None,
    use_task_judge: bool,
    max_prompt_retries: int,
    judge_confidence_threshold: float,
    judge_on_error: str,
    intervention_mode: str,
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
    global _evaluator, _policy, _prompts, _time_limit, _use_transition_generation, _transition_mode, _no_planning, _no_interpolation, _no_retrieval, _control_fps, _llm_config, _use_task_judge, _max_prompt_retries, _judge_confidence_threshold, _judge_on_error, _intervention_mode
    task = create_task(task_name)
    _evaluator = Evaluator(task, image_history=image_history, video_fps=video_fps, render_video=render_video)
    _policy = make_policy(host, port, policy_backend)
    _prompts = prompts
    _time_limit = time_limit
    _use_transition_generation = use_transition_generation
    _transition_mode = transition_mode
    _no_planning = no_planning
    _no_interpolation = no_interpolation
    _no_retrieval = no_retrieval
    _control_fps = control_fps
    _llm_config = llm_config
    _use_task_judge = use_task_judge
    _max_prompt_retries = max_prompt_retries
    _judge_confidence_threshold = judge_confidence_threshold
    _judge_on_error = judge_on_error
    _intervention_mode = intervention_mode

def step_worker(seed: int):
    return evaluate_task(
        _evaluator,
        _policy,
        seed,
        _time_limit,
        _prompts,
        _use_transition_generation,
        _transition_mode,
        _no_planning,
        _no_interpolation,
        _no_retrieval,
        _control_fps,
        _llm_config,
        _use_task_judge,
        _max_prompt_retries,
        _judge_confidence_threshold,
        _judge_on_error,
        _intervention_mode,
        transition_seed=seed,
    )

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a policy using the WebSocket client.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket server port")
    parser.add_argument(
        "--policy-backend",
        type=str,
        default="openpi",
        choices=POLICY_BACKENDS,
        help="Policy websocket payload adapter to use: openpi keeps current keys; labvla maps Sci-VLA keys to LABVLA camera/state keys.",
    )
    parser.add_argument(
        "--problem-validation-demo",
        action="store_true",
        help="Run the fixed thermal-cycler problem validation demo profile.",
    )
    parser.add_argument(
        "--problem-validation-prefix-percent",
        type=float,
        default=30.0,
        metavar="PERCENT",
        help=(
            "In problem-validation mode, sample the target state uniformly from "
            "the first PERCENT%% of the matching trajectory (default: 30)."
        ),
    )
    parser.add_argument("--task", type=str, default="pickup", help="Task name")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--image_history", type=int, default=0, help="Image history for the policy")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for parallel evaluation, 0 for serial")
    parser.add_argument("--save", type=str, default=None, help="Output file for evaluation results")
    parser.add_argument("--seed", type=int, default=None, help="Master seed for evaluation")
    parser.add_argument("--render_device_id", type=str, default='0', help="Comma-separated list of GPU device IDs for rendering")
    parser.add_argument("--prompts", type=str, default=None, help="Comma-separated list of prompts to execute sequentially")
    parser.add_argument("--time_limit", type=float, default=100, help="per task time limit")
    parser.add_argument(
        "--intervention-mode",
        type=str,
        default="timeout",
        choices=["non_timeout", "timeout"],
        help=(
            "Prompt intervention mode: non_timeout stops VLA as soon as local task success "
            "is detected; timeout runs each prompt until time_limit, then checks success."
        ),
    )
    parser.add_argument("--video_fps", type=int, default=20, help="Replay video FPS")
    parser.add_argument(
        "--render-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to capture and save replay videos; use --no-render-video to skip videos while still computing success.",
    )
    parser.add_argument(
        "--control_fps",
        type=float,
        default=50.0,
        help="Policy action execution rate in Hz; render_all.bash generates 50Hz datasets",
    )
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
        default=False,
        help="Whether to run transition generation and execution between prompts",
    )
    parser.add_argument(
        "--experiment-mode",
        type=str,
        default="no-transition",
        choices=EXPERIMENT_MODES,
        help=(
            "High-level experiment mode. no-transition: disable transitions; "
            "baseline: random dataset init pose + RRT transition, falling back to interpolation; "
            "no-retrieval: planning/coding agents only with no target restore; "
            "no-agent: retrieve target pose, then interpolate restore without planning/coding agents; "
            "full: retrieval + planning + coding."
        ),
    )
    parser.add_argument(
        "--transition-mode",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "llm",
            "retrieval_interp",
            "retrieval_collision_planner",
            "random_future_task_pose_collision_planner",
            "random_future_task_pose_rrt",
            "random_dataset_task_pose_rrt",
        ],
        help=(
            "Transition executor between prompts. auto preserves --use-transition-generation "
            "behavior; llm uses Sci transition generation; retrieval_* modes use non-LLM baselines."
        ),
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
    parser.add_argument(
        "--no_retrieval",
        action="store_true",
        help="Skip target-qpos retrieval for LLM transition generation; intended for no-retrieval ablation.",
    )
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM base URL for transition generation")
    parser.add_argument("--llm-model-name", type=str, default=None, help="LLM model name for transition generation")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key for transition generation")
    parser.add_argument("--llm-temperature", type=float, default=None, help="LLM sampling temperature")
    parser.add_argument("--llm-top-p", type=float, default=None, help="LLM sampling top-p")
    parser.add_argument("--llm-max-tokens", type=int, default=None, help="LLM max output tokens")
    parser.add_argument("--llm-max-attempts", type=int, default=None, help="Max retry attempts per LLM stage")
    parser.add_argument("--llm-timeout", type=float, default=None, help="LLM request timeout in seconds")
    parser.add_argument("--llm-image-max-side", type=int, default=None, help="Max image side before sending transition images to VLM; 0 disables compression")
    parser.add_argument("--llm-image-quality", type=int, default=None, help="JPEG quality for compressed transition images")
    parser.add_argument(
        "--llm-local-retrieval-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Try local fuzzy qpos prompt retrieval before calling the LLM",
    )
    parser.add_argument("--llm-local-retrieval-cutoff", type=float, default=None, help="Local fuzzy qpos prompt match cutoff")
    parser.add_argument(
        "--ready-memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use ReadyStateAgent visual A/B retrieval before transition planning.",
    )
    parser.add_argument("--ready-memory-db", type=str, default=None, help="Pre-extracted ready memory index JSON")
    parser.add_argument("--ready-memory-repo-id", type=str, default=None, help="LeRobot repo id fallback for ReadyStateAgent retrieval")
    parser.add_argument("--ready-memory-episode-index", type=int, default=None, help="Optional LeRobot episode index for ReadyStateAgent")
    parser.add_argument(
        "--ready-memory-window-size",
        type=float,
        default=None,
        help="Initial ReadyStateAgent A/B window length as a trajectory percentage N; 20 means 20%%",
    )
    parser.add_argument("--ready-memory-min-frame-ratio", type=float, default=None, help="Minimum selected frame ratio to avoid frame 0")
    parser.add_argument(
        "--ready-memory-max-iterations",
        type=int,
        default=None,
        help="Maximum ReadyStateAgent A/B judgements (default: 4)",
    )
    parser.add_argument("--ready-memory-match-cutoff", type=float, default=None, help="Fuzzy task prompt cutoff for ready memory index")
    parser.add_argument("--ready-memory-front-image-key", type=str, default=None, help="LeRobot image key for ReadyStateAgent repo mode")
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
    parser.add_argument(
        "--use-task-judge",
        action="store_true",
        help="Use a VLM judge after each prompt to decide whether to advance or retry the current prompt.",
    )
    parser.add_argument(
        "--max-prompt-retries",
        type=int,
        default=1,
        help="Maximum retry transitions for each prompt when the task judge reports failure.",
    )
    parser.add_argument(
        "--judge-confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum VLM judge confidence required to accept a successful prompt.",
    )
    parser.add_argument(
        "--judge-on-error",
        type=str,
        default="fail",
        choices=["fail", "pass"],
        help="How to handle judge request/parsing errors: fail retries conservatively or pass for debugging.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    problem_validation_demo_config = apply_problem_validation_demo_profile(args)
    if problem_validation_demo_config is not None:
        print(
            "[ProblemValidationDemo] "
            f"task={args.task} "
            f"prompts={args.prompts} "
            f"num_episodes={args.num_episodes} "
            f"num_workers={args.num_workers} "
            f"render_video={args.render_video} "
            f"intervention_mode={args.intervention_mode} "
            f"experiment_mode={args.experiment_mode} "
            f"use_transition_generation={args.use_transition_generation} "
            f"transition_mode={args.transition_mode} "
            f"prefix_percent={problem_validation_demo_config.prefix_fraction * 100:g} "
            f"no_planning={args.no_planning} "
            f"no_interpolation={args.no_interpolation} "
            f"no_retrieval={args.no_retrieval}"
        )
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
        "image_max_side": args.llm_image_max_side,
        "image_quality": args.llm_image_quality,
        "local_retrieval_first": args.llm_local_retrieval_first,
        "local_retrieval_cutoff": args.llm_local_retrieval_cutoff,
        "ready_memory_enabled": args.ready_memory_enabled,
        "ready_memory_db_path": args.ready_memory_db,
        "ready_memory_repo_id": args.ready_memory_repo_id,
        "ready_memory_episode_index": args.ready_memory_episode_index,
        "ready_memory_window_size": args.ready_memory_window_size,
        "ready_memory_min_frame_ratio": args.ready_memory_min_frame_ratio,
        "ready_memory_max_iterations": args.ready_memory_max_iterations,
        "ready_memory_match_cutoff": args.ready_memory_match_cutoff,
        "ready_memory_front_image_key": args.ready_memory_front_image_key,
        "thinking": args.llm_thinking,
        "backend_mode": args.llm_backend_mode,
    }
    experiment_config = resolve_experiment_mode_config(
        experiment_mode=args.experiment_mode,
        use_transition_generation=args.use_transition_generation,
        transition_mode=args.transition_mode,
        no_planning=args.no_planning,
        no_interpolation=args.no_interpolation,
        no_retrieval=args.no_retrieval,
    )
    print(
        "[ExperimentMode] "
        f"mode={args.experiment_mode} "
        f"use_transition_generation={experiment_config['use_transition_generation']} "
        f"transition_mode={experiment_config['transition_mode']} "
        f"no_planning={experiment_config['no_planning']} "
        f"no_interpolation={experiment_config['no_interpolation']} "
        f"no_retrieval={experiment_config['no_retrieval']}"
    )
    assert len(render_device_ids) > 0
    success_results: list[float] = []
    episode_timings: list[dict] = []

    if args.num_workers == 0:
        # Serial evaluation
        configure_mujoco_env(mujoco_gl, render_device_ids[0])
        from task import create_task
        from evaluator import Evaluator
        policy = make_policy(args.host, args.port, args.policy_backend)
        task = create_task(args.task)
        evaluator = Evaluator(
            task,
            image_history=args.image_history,
            video_fps=args.video_fps,
            render_video=args.render_video,
        )
        for seed in tqdm(seeds):
            raw_result = evaluate_task(
                evaluator,
                policy,
                seed,
                time_limit,
                prompts,
                experiment_config["use_transition_generation"],
                experiment_config["transition_mode"],
                experiment_config["no_planning"],
                experiment_config["no_interpolation"],
                experiment_config["no_retrieval"],
                args.control_fps,
                llm_config,
                args.use_task_judge,
                args.max_prompt_retries,
                args.judge_confidence_threshold,
                args.judge_on_error,
                args.intervention_mode,
                transition_seed=seed,
                problem_validation_demo_config=problem_validation_demo_config,
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
                args.policy_backend,
                args.task,
                args.image_history,
                time_limit,
                args.video_fps,
                args.render_video,
                experiment_config["use_transition_generation"],
                experiment_config["transition_mode"],
                experiment_config["no_planning"],
                experiment_config["no_interpolation"],
                experiment_config["no_retrieval"],
                args.control_fps,
                llm_config,
                args.use_task_judge,
                args.max_prompt_retries,
                args.judge_confidence_threshold,
                args.judge_on_error,
                args.intervention_mode,
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
    complete_episode_summary = build_complete_episode_success_summary(success_results, args.num_episodes)
    print_complete_episode_success_summary(complete_episode_summary)
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

    atomic_task_summary = build_atomic_task_success_summary(episode_timings, args.num_episodes)
    print_atomic_task_success_summary(atomic_task_summary)
    if experiment_config["transition_mode"] != "none":
        expected_transition_count = max(0, len(prompts) - 1) if prompts is not None else 0
        transition_collision_summary = build_transition_collision_summary(
            episode_timings,
            args.num_episodes,
            expected_transition_count=expected_transition_count,
        )
        print_transition_collision_summary(transition_collision_summary)

    if args.save:
        import json
        with open(args.save, 'w') as f:
            json.dump(results, f)
    else:
        print("Evaluation results:", results)
