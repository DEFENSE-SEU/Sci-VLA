# VLM Task Judge Retry Design

Date: 2026-06-29

## Goal

Add a prompt-level VLM judgement plugin to Sci-VLA long-horizon evaluation. After each subtask prompt finishes, the evaluator should decide from end-state images whether that subtask succeeded. Successful subtasks advance to the next prompt. Failed subtasks transition back to the current prompt's initial qpos distribution and retry the same prompt.

This closes the current gap where the multi-prompt evaluator runs each prompt for `time_limit` and transitions forward as long as the simulation remains healthy, even if the visible task outcome failed.

## Chosen Approach

Use a VLM-only task judge for the first version.

The judge consumes the current prompt and two end-state camera views:

- front view from `table_cam_front` or the primary task image camera fallback
- side view from `table_cam_left` or the front-view fallback

It returns a strict JSON result:

```json
{
  "success": true,
  "confidence": 0.82,
  "reason": "the centrifuge lid appears open",
  "failure_mode": null
}
```

The first version will not use task-specific simulator rules or `Task.check()` for prompt-level judgement. Those can be added later as a second signal after VLM judgement logs reveal common false positives and false negatives.

## New Module

Create `scripts/autobio_scripts/transition_judgement.py`.

Responsibilities:

- Build the VLM judgement prompt.
- Convert current front/side images to data URLs.
- Call an OpenAI-compatible backend using the same LLM config style as transition generation.
- Parse and validate the strict JSON response.
- Normalize judge output to:
  - `success: bool`
  - `confidence: float`
  - `reason: str`
  - `failure_mode: str | None`

The module should reuse existing helper behavior from `transition_generation.py` where practical, especially JSON response parsing, backend mode selection, and image data URL encoding. It should avoid duplicating unrelated transition code.

## Evaluator Flow

Update the multi-prompt branch in `Evaluator.evaluate()`.

For each prompt `prompts[i]`:

1. Execute `run_prompt(prompt, time_limit)`.
2. If simulation is unhealthy, fail the episode.
3. Save end-state front and side images.
4. If `--use-task-judge` is disabled, preserve current behavior.
5. If `--use-task-judge` is enabled, call the VLM judge for `prompts[i]`.
6. Treat the prompt as successful only when:
   - `judge.success` is true; and
   - `judge.confidence >= judge_confidence_threshold`.
7. If successful:
   - for the final prompt, finish the episode;
   - otherwise transition to `prompts[i + 1]` using the existing Phase 1 validated transition path.
8. If failed:
   - if the retry count for `prompts[i]` is below `max_prompt_retries`, transition to `prompts[i]` and execute it again;
   - otherwise fail the episode.

Failure recovery should reuse the existing target-qpos retrieval and transition code generation path. The only difference is the target prompt:

- normal advance: target prompt is `prompts[i + 1]`
- retry recovery: target prompt is `prompts[i]`

This keeps recovery behavior inside the same validated transition mechanism added in Phase 1.

## CLI

Add these flags to `scripts/autobio_scripts/evaluate.py`:

- `--use-task-judge`
- `--max-prompt-retries`, default `1`
- `--judge-confidence-threshold`, default `0.6`
- `--judge-on-error`, choices `fail` and `pass`, default `fail`

`judge-on-error=fail` is the conservative default: if the VLM request fails, JSON parsing fails, or the result is malformed, the current prompt is treated as failed and will be retried if retries remain.

`judge-on-error=pass` is only for debugging unreliable local VLM endpoints. It allows the evaluator to preserve forward progress when the judge service is unavailable.

## Logging

Write one JSONL record per judgement to:

`logs/task_judgements.jsonl`

Each record should include:

- timestamp
- task name if available
- seed if available
- prompt index
- prompt text
- attempt index
- success
- confidence
- reason
- failure_mode
- front image path
- side image path
- action: `advance`, `retry`, or `fail_episode`
- raw judge result when available
- error text when judge execution fails

The image files should use unique names per prompt index and attempt, rather than repeatedly overwriting `logs/current_view.png`. Existing `logs/current_view.png` and `logs/current_side_view.png` can still be written for compatibility with transition generation.

## Error Handling

The judge should raise clear errors when:

- the model name is missing while `--use-task-judge` is enabled;
- required image files are missing;
- the backend returns no text;
- no JSON object can be parsed;
- `confidence` is not numeric;
- `success` is missing or not coercible to boolean.

The evaluator should catch judgement errors and convert them according to `judge-on-error`.

If a retry transition back to the current prompt cannot find a valid qpos candidate, the episode should fail. It should not blindly rerun the current prompt from an invalid state.

## Testing

Add tests that avoid live VLM calls, GPUs, and MuJoCo rendering:

- judgement normalization:
  - valid response with `success=true` and high confidence is accepted;
  - low confidence converts to an unsuccessful prompt-level decision;
  - malformed confidence raises a clear `ValueError`;

- evaluator retry policy as pure logic:
  - successful judgement returns `advance`;
  - failed judgement below retry limit returns `retry`;
  - failed judgement at retry limit returns `fail_episode`;
  - judge error follows `judge-on-error`;

- CLI parsing:
  - `--use-task-judge` enables the feature;
  - defaults are `max_prompt_retries=1`, `judge_confidence_threshold=0.6`, and `judge_on_error=fail`;

- transition target selection:
  - when a prompt fails, the recovery transition target is the current prompt, not the next prompt;
  - when a prompt succeeds, the normal transition target remains the next prompt.

## Non-Goals

This first version does not add rule-based simulator state checks.

This first version does not train a separate success classifier.

This first version does not generate custom recovery actions beyond reusing validated transition generation to return to the current prompt's initial qpos.

This first version does not change the VLA policy server protocol or policy action execution.

## Acceptance Criteria

- Existing evaluation behavior is unchanged unless `--use-task-judge` is passed.
- With `--use-task-judge`, every completed prompt produces a JSONL judgement record.
- A successful prompt advances to the next prompt.
- A failed prompt transitions back to the same prompt's initial qpos and retries until `max_prompt_retries` is exhausted.
- Exhausting retries fails the episode.
- Focused tests cover judgement parsing, retry decision logic, CLI defaults, and current-vs-next transition target selection without requiring a live VLM server.
