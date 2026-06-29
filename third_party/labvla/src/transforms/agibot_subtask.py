"""Per-frame subtask annotation builder for AgiBot World samples.

agibot_dual_arm has exactly one task description per task_NNN dir (1-row
tasks.parquet). The v30 adapter already resolves ``task_index`` → task string
into ``data["task"]``. Naively reusing that as the prediction target creates a
pathological loop: the prompt is a generic VLA
template that often substitutes ``task`` as the user-text instruction → the
model is told the answer in its input.

This transform:
  1. Copies ``data["task"]`` into ``data["annotation.subtask"]`` (the field
     declared by ``schemas/agibot_dual_arm.SCHEMA.annotation_losses``).
  2. Rewrites ``data["task"]`` to a generic descriptor so the prompt asks the
     model to predict the task instead of repeating it.

The transform is hydrated via ``hydrate_all`` based on schema id — for non-
agibot schemas it is a no-op.

Pipeline placement (set by ``LabVLADatasetConfig.__post_init__``):
  ... → unify_anno → BuildAgiBotSubtaskTransformFn → qwen_processor → ...

placing it AFTER unify_anno guarantees no collision with the 12-field RoboInter
unified-annotation builder, and BEFORE qwen_processor so ``data["task"]``
rewrite reaches the prompt builder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from src.transforms.core import DataDict, DataTransformFn


# Conservative generic prompt; short tokens to keep prefix budget tight.
DEFAULT_GENERIC_PROMPT = "Describe the high-level task being performed."


@DataTransformFn.register_subclass("build_agibot_subtask")
@dataclass
class BuildAgiBotSubtaskTransformFn(DataTransformFn):
    """Promote ``data['task']`` to ``data['annotation.subtask']`` for agibot.

    Inactive by default; ``hydrate_all`` flips ``enabled=True`` for schemas
    whose ``annotation_losses`` declare a spec with ``field='annotation.subtask'``.
    """

    enabled: bool = False
    generic_prompt: str = DEFAULT_GENERIC_PROMPT
    rewrite_task: bool = True   # set False to keep original task in prompt

    @staticmethod
    def _is_present(value) -> bool:
        """True if ``value`` is a real (non-empty, non-null-sentinel) string.

        Mirrors the null-stringification artifacts handled elsewhere in the
        annotation pipeline so a parquet "None"/"nan"/"[]" cell counts as
        MISSING rather than a real existing subtask.
        """
        if value is None:
            return False
        try:
            s = str(value).strip()
        except Exception:
            return False
        return bool(s) and s.lower() not in {"none", "nan", "null", "[]", "{}", "''", '""'}

    def __call__(self, data: DataDict) -> DataDict:
        if not self.enabled:
            return data
        # Only backfill when annotation.subtask is MISSING. A sample that
        # already carries a real subtask annotation is authoritative —
        # overwriting it with data["task"] and clobbering the prompt would
        # corrupt schemas/samples that legitimately provide subtask.
        if self._is_present(data.get("annotation.subtask")):
            return data
        raw = data.get("task", "")
        try:
            task_str = str(raw).strip() if raw is not None else ""
        except Exception:
            task_str = ""
        if not task_str:
            # Missing task — skip silently. AnnotationTokenizeTransformFn
            # treats absent annotation field as "no CE for this sample".
            return data
        data["annotation.subtask"] = task_str
        if self.rewrite_task:
            data["task"] = self.generic_prompt
        return data

    def hydrate(self, ctx) -> "BuildAgiBotSubtaskTransformFn":
        # Gate on an EXPLICIT schema identity, not just the presence of an
        # `annotation.subtask` loss field: gating on the field alone would fire
        # for ANY schema reusing that field name, clobbering its `task` with the
        # generic prompt and overwriting its `annotation.subtask`. The
        # task→subtask backfill + prompt rewrite is AgiBot-specific (one task
        # description per task dir), so require the AgiBot schema identity.
        # robot_type / schema_id carry the family tag.
        schema = ctx.schema
        _has_subtask_loss = any(
            getattr(spec, "field", None) == "annotation.subtask"
            for spec in (schema.annotation_losses or ())
        )
        _robot_type = str(getattr(schema, "robot_type", "") or "")
        _schema_id = str(getattr(schema, "schema_id", "") or "")
        _is_agibot_schema = (
            _robot_type.startswith("agibot") or _schema_id.startswith("agibot")
        )
        _agibot_enabled = bool(_has_subtask_loss and _is_agibot_schema)
        t = replace(self, enabled=_agibot_enabled)
        if _has_subtask_loss and not _is_agibot_schema:
            logging.info(
                f"{t.__class__.__name__} left disabled: schema "
                f"{_schema_id!r} declares annotation.subtask but is not an "
                f"AgiBot schema — skipping task→subtask backfill/prompt rewrite."
            )
        if _agibot_enabled:
            logging.info(
                f"Hydrated {t.__class__.__name__} enabled=True ({schema.schema_id})"
            )
        return t
