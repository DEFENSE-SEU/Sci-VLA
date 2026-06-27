import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "split_libero_long_subtasks.py"
SPEC = importlib.util.spec_from_file_location("split_libero_long_subtasks", MODULE_PATH)
splitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splitter)


def test_find_first_stable_true_requires_consecutive_frames():
    assert splitter.find_first_stable_true([False, True, False, True, True], stable_window=2) == 3
    assert splitter.find_first_stable_true([False, True, False, True], stable_window=2) is None


def test_build_segments_groups_goals_that_complete_on_same_boundary():
    goal_states = [("On", "mug", "plate"), ("On", "book", "shelf"), ("Open", "drawer")]
    status_by_goal = [
        [False, False, True, True, True, True],
        [False, False, True, True, True, True],
        [False, False, False, False, True, True],
    ]

    segments = splitter.build_segments(
        goal_states=goal_states,
        status_by_goal=status_by_goal,
        total_frames=6,
        stable_window=2,
        min_segment_frames=2,
    )

    assert segments == [
        {
            "subtask_index": 0,
            "start_frame": 0,
            "end_frame": 2,
            "goal_indices": [0, 1],
            "goal_states": [["On", "mug", "plate"], ["On", "book", "shelf"]],
            "prompt": "On mug plate; On book shelf",
        },
        {
            "subtask_index": 1,
            "start_frame": 2,
            "end_frame": 4,
            "goal_indices": [2],
            "goal_states": [["Open", "drawer"]],
            "prompt": "Open drawer",
        },
        {
            "subtask_index": 2,
            "start_frame": 4,
            "end_frame": 5,
            "goal_indices": [],
            "goal_states": [],
            "prompt": "terminal stabilization",
        },
    ]


def test_build_segments_keeps_continuity_between_adjacent_segments():
    segments = splitter.build_segments(
        goal_states=[("A", "x"), ("B", "y")],
        status_by_goal=[
            [False, True, True, True],
            [False, False, True, True],
        ],
        total_frames=4,
        stable_window=1,
        min_segment_frames=1,
    )

    for prev, nxt in zip(segments, segments[1:]):
        assert prev["end_frame"] == nxt["start_frame"]
