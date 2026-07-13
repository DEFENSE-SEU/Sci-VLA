import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import build_atomic_task_success_summary


def test_atomic_task_success_summary_keeps_duplicate_prompt_indices_separate():
    summary = build_atomic_task_success_summary(
        [
            {
                "atomic_task_results": [
                    {
                        "prompt_index": 3,
                        "prompt": "close the lid of the centrifuge5910",
                        "success": True,
                        "attempt_index": 0,
                    },
                    {
                        "prompt_index": 7,
                        "prompt": "close the lid of the centrifuge5910",
                        "success": False,
                        "attempt_index": 0,
                    },
                ]
            }
        ],
        num_episodes=1,
    )

    assert list(summary) == [
        "prompt_index=3 close the lid of the centrifuge5910",
        "prompt_index=7 close the lid of the centrifuge5910",
    ]
    assert summary["prompt_index=3 close the lid of the centrifuge5910"]["success_count"] == 1
    assert summary["prompt_index=7 close the lid of the centrifuge5910"]["success_count"] == 0
