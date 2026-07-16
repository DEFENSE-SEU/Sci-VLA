import importlib
import json
import sys
import types
from pathlib import Path


AUTOBIO_ROOT = Path(__file__).resolve().parent


def load_agent():
    if str(AUTOBIO_ROOT) not in sys.path:
        sys.path.insert(0, str(AUTOBIO_ROOT))
    return importlib.import_module("ready_memory_retrieval_agent")


def make_judgement(agent, *, a_exceeded, b_exceeded):
    return agent.normalize_ready_pair_judgement(
        {
            "A_exceeded_ready": a_exceeded,
            "B_exceeded_ready": b_exceeded,
            "confidence": 0.9,
            "reason": "test judgement",
        }
    )


def make_frames(length=100):
    return [
        {
            "frame_index": index,
            "image_path": f"frame_{index}.jpg",
            "state": [float(index)],
        }
        for index in range(length)
    ]


def run_search(monkeypatch, judgements, *, max_iterations=4):
    agent = load_agent()
    pending = iter(judgements)
    monkeypatch.setattr(agent, "judge_ready_pair", lambda **_kwargs: next(pending))
    return agent._run_window_search_on_frames(
        frames=make_frames(),
        task_prompt="pick up the pcr plate",
        window_size=20,
        min_frame_ratio=0.05,
        max_iterations=max_iterations,
        llm_config=None,
    )


def test_normalization_derives_target_only_from_b_exceeded():
    agent = load_agent()

    inconsistent = make_judgement(agent, a_exceeded=True, b_exceeded=False)
    accepted = make_judgement(agent, a_exceeded=False, b_exceeded=False)

    assert inconsistent.b_is_target_state is True
    assert accepted.b_is_target_state is True


def test_normalization_accepts_when_b_is_false_even_if_a_is_missing():
    agent = load_agent()

    judgement = agent.normalize_ready_pair_judgement(
        {"B_exceeded_ready": False, "confidence": 0.5}
    )

    assert judgement.a_exceeded_ready is True
    assert judgement.b_is_target_state is True


def test_analysis_treats_not_fully_open_and_too_close_as_exceeded():
    agent = load_agent()

    assert agent._infer_exceeded_from_analysis(
        {
            "gripper_state": "not_fully_open",
            "target_object_state": "stationary",
            "gripper_target_relation": "safe_pre_contact_gap",
        }
    ) is True
    assert agent._infer_exceeded_from_analysis(
        {
            "gripper_state": "fully_open",
            "target_object_state": "stationary",
            "gripper_target_relation": "too_close",
        }
    ) is True


def test_agent_prompt_only_requests_exceeded_booleans_and_defines_clearance():
    agent = load_agent()

    prompt = agent._build_ready_pair_prompt("pick up the pcr plate")

    assert '"A_exceeded_ready"' in prompt
    assert '"B_exceeded_ready"' in prompt
    assert '"B_is_target_state"' not in prompt
    assert "not fully released" in prompt
    assert "too close" in prompt


def test_false_true_keeps_left_half_of_window(monkeypatch):
    agent = load_agent()
    result = run_search(
        monkeypatch,
        [
            make_judgement(agent, a_exceeded=False, b_exceeded=True),
            make_judgement(agent, a_exceeded=False, b_exceeded=False),
        ],
    )

    assert [(item["A_index"], item["B_index"]) for item in result.history] == [
        (29, 49),
        (29, 39),
    ]
    assert result.selected_index == 39
    assert result.fallback_to_initial_frame is False


def test_true_true_shifts_left_by_current_window_length(monkeypatch):
    agent = load_agent()
    result = run_search(
        monkeypatch,
        [
            make_judgement(agent, a_exceeded=True, b_exceeded=True),
            make_judgement(agent, a_exceeded=False, b_exceeded=False),
        ],
    )

    assert [(item["A_index"], item["B_index"]) for item in result.history] == [
        (29, 49),
        (9, 29),
    ]
    assert result.selected_index == 29
    assert result.fallback_to_initial_frame is False


def test_false_false_accepts_b_immediately(monkeypatch):
    agent = load_agent()
    result = run_search(
        monkeypatch,
        [make_judgement(agent, a_exceeded=False, b_exceeded=False)],
    )

    assert result.selected_index == 49
    assert len(result.history) == 1
    assert result.fallback_to_initial_frame is False


def test_search_falls_back_to_initial_frame_after_four_iterations(monkeypatch, capsys):
    agent = load_agent()
    result = run_search(
        monkeypatch,
        [make_judgement(agent, a_exceeded=False, b_exceeded=True) for _ in range(4)],
    )

    assert len(result.history) == 4
    assert result.selected_index == 0
    assert result.fallback_to_initial_frame is True
    assert result.fallback_reason == "max_iterations_exhausted"
    assert "fallback to initial frame 0" in capsys.readouterr().out


def test_true_false_pair_accepts_b_immediately(monkeypatch):
    agent = load_agent()
    result = run_search(
        monkeypatch,
        [make_judgement(agent, a_exceeded=True, b_exceeded=False)],
    )

    assert [(item["A_index"], item["B_index"]) for item in result.history] == [
        (29, 49),
    ]
    assert result.selected_index == 49
    assert result.fallback_to_initial_frame is False


def test_index_retrieval_exports_initial_frame_state_and_image_on_fallback(
    monkeypatch,
    tmp_path,
):
    agent = load_agent()
    memory_path = tmp_path / "memory.json"
    output_path = tmp_path / "selected.json"
    memory_path.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "task": "pick up the pcr plate",
                        "memory_id": "pcr-plate-0",
                        "frames": [
                            {
                                "frame_index": index * 2,
                                "image_path": f"frame_{index}.jpg",
                                "state": [float(index), 1.0],
                            }
                            for index in range(20)
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    exceeded = make_judgement(agent, a_exceeded=False, b_exceeded=True)
    monkeypatch.setattr(agent, "judge_ready_pair", lambda **_kwargs: exceeded)

    result = agent.retrieve_ready_memory_from_index(
        memory_db_path=memory_path,
        task_prompt="pick up the pcr plate",
        window_size=20,
        output_path=output_path,
        min_frame_ratio=0.05,
        max_iterations=4,
    )

    assert result["fallback_to_initial_frame"] is True
    assert result["fallback_reason"] == "max_iterations_exhausted"
    assert result["ready_frame_local_index"] == 0
    assert result["ready_frame_index"] == 0
    assert result["target_state"] == [0.0, 1.0]
    assert result["target_front_image_path"] == str(tmp_path / "frame_0.jpg")
    assert json.loads(output_path.read_text(encoding="utf-8"))["target_state"] == [0.0, 1.0]


def test_cli_default_max_iterations_is_four(monkeypatch):
    agent = load_agent()
    monkeypatch.setattr(sys, "argv", ["ready_memory_retrieval_agent.py"])

    assert agent.parse_args().max_iterations == 4


def test_judge_ready_pair_preserves_planning_generation_defaults(monkeypatch, tmp_path):
    agent = load_agent()
    sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
    transition_generation = importlib.import_module("transition_generation")
    captured = {}

    def fake_request_json_object(**kwargs):
        captured.update(kwargs)
        return {
            "A_exceeded_ready": False,
            "B_exceeded_ready": False,
            "confidence": 1.0,
        }

    monkeypatch.setattr(
        transition_generation,
        "_request_json_object",
        fake_request_json_object,
    )
    frame_a = tmp_path / "a.jpg"
    frame_b = tmp_path / "b.jpg"
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")

    agent.judge_ready_pair(
        task_prompt="pick up the pcr plate",
        frame_a_path=frame_a,
        frame_b_path=frame_b,
        llm_config={
            "model_name": "test-model",
            "temperature": None,
            "max_tokens": None,
            "max_image_side": 0,
        },
        client=object(),
    )

    assert captured["temperature"] is None
    assert captured["max_tokens"] is None


def test_ready_agent_cli_generation_defaults_match_evaluate(monkeypatch):
    agent = load_agent()
    monkeypatch.setattr(sys, "argv", ["ready_memory_retrieval_agent.py"])

    args = agent.parse_args()

    assert args.llm_temperature is None
    assert args.llm_max_tokens is None
