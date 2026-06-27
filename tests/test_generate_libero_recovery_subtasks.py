import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_libero_recovery_subtasks.py"
SPEC = importlib.util.spec_from_file_location("generate_libero_recovery_subtasks", MODULE_PATH)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_perturb_robot_qpos_is_seeded_and_leaves_input_unchanged():
    base = np.array([1.0, 2.0, 3.0])
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)

    perturbed_a = generator.perturb_robot_qpos(base, noise_std=0.1, rng=rng_a)
    perturbed_b = generator.perturb_robot_qpos(base, noise_std=0.1, rng=rng_b)

    assert np.allclose(perturbed_a, perturbed_b)
    assert not np.allclose(perturbed_a, base)
    assert np.allclose(base, np.array([1.0, 2.0, 3.0]))


def test_assemble_recovery_prefixed_segment_drops_duplicate_target_state():
    recovery_states = np.array(
        [
            [100.0, 100.0],
            [200.0, 200.0],
            [1.0, 1.0],
        ]
    )
    recovery_actions = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.0],
        ]
    )
    original_states = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    original_actions = np.array(
        [
            [9.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )

    states, actions = generator.assemble_recovery_prefixed_segment(
        recovery_states=recovery_states,
        recovery_actions=recovery_actions,
        original_states=original_states,
        original_actions=original_actions,
        start_frame=1,
        end_frame=3,
    )

    assert states.tolist() == [
        [100.0, 100.0],
        [200.0, 200.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]
    assert actions.tolist() == [
        [0.1, 0.0],
        [0.2, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ]
    assert len(states) == len(actions)


def test_assemble_recovery_prefixed_segment_rejects_bad_recovery_lengths():
    recovery_states = np.zeros((2, 3))
    recovery_actions = np.zeros((2, 2))
    original_states = np.zeros((4, 3))
    original_actions = np.zeros((4, 2))

    try:
        generator.assemble_recovery_prefixed_segment(
            recovery_states=recovery_states,
            recovery_actions=recovery_actions,
            original_states=original_states,
            original_actions=original_actions,
            start_frame=0,
            end_frame=1,
        )
    except ValueError as exc:
        assert "one fewer" in str(exc)
    else:
        raise AssertionError("expected ValueError")
