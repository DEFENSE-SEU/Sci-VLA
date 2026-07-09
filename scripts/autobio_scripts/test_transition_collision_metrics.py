from types import SimpleNamespace


def test_transition_collision_summary_uses_num_episodes_as_denominator():
    from evaluate import build_transition_collision_summary

    summary = build_transition_collision_summary(
        [
            {"transition_collision_counts": {"1": 3, "2": 1}},
            {"transition_collision_counts": {"1": 5}},
        ],
        num_episodes=4,
        expected_transition_count=3,
    )

    assert summary == {
        "transition_1": 2.0,
        "transition_2": 0.25,
        "transition_3": 0.0,
    }


def test_robot_object_collision_counter_excludes_robot_robot_and_object_object_contacts():
    from evaluator import count_robot_object_collision_contacts

    model = SimpleNamespace(
        geom_names={
            0: "/ur:wrist_collision",
            1: "tube_collision",
            2: "/ur:2f85:left_finger",
            3: "rack_collision",
        }
    )
    data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(geom1=0, geom2=1),
            SimpleNamespace(geom1=0, geom2=2),
            SimpleNamespace(geom1=1, geom2=3),
            SimpleNamespace(geom1=2, geom2=3),
        ],
    )

    assert count_robot_object_collision_contacts(model, data) == 2
