"""OXE-AugE merged schema (oxe-auge_clean).

Single-arm, 7 DoF + 1 gripper layout across all 180 sub-sources after merge.
Authored via `SingleArmBlueprint` — the blueprint compiler derives
`state_keys`, `action_keys`, `delta_mask`, `gripper_action_dims`, and
`arm_layout` from the one `SingleArm` spec below.

Merge-time handling of source heterogeneity (some sources are 6-DoF raw; we
promote them to 7-DoF canonical by zero-padding dim 6 and moving the raw
gripper to dim 7) is described declaratively by `MERGE_RECIPE`, which
`data_process/merge_v3/build_oxe_auge_clean.py` consumes.
"""
from src.schema.blueprint import SingleArmBlueprint, SingleArm
from src.schema.merge_recipe import MergeRecipe, SourceArmLayout
from src.schema.arm_layout import ArmLayoutSpec, ArmCount


SCHEMA = SingleArmBlueprint(
    schema_id="oxe_auge_v1",
    robot_type="oxe_auge_merged_single_arm",
    arm=SingleArm(
        dof=7,
        # State must read the canonical joints column, not `observation.state`:
        # the latter is the untouched per-source raw vector (mixed widths across
        # sources), while only `observation.joints` was canonicalized to 8-dim
        # (7 arm + 1 grip) by the merge. Since the state key is
        # DeltaActionTransformFn's t=0 reference, binding it to the raw vector
        # would train delta against an unrelated reference. State therefore reads
        # the virtual key below, which the adapter materializes per-frame as a
        # copy of `observation.joints` (declared in `virtual_state_sources`).
        # Joint/gripper state share one column, so the blueprint collapses to a
        # single 8-dim state key; same on the action side.
        joint_state_col="virtual.joints_state",
        joint_action_col="observation.joints",
        gripper_state_col="virtual.joints_state",
        gripper_action_col="observation.joints",
        arm_mode="delta",
        gripper_mode="abs",
    ),
    # Same-frame copy of the canonical joints column (see above).
    virtual_state_sources={"virtual.joints_state": "observation.joints"},
    cameras={
        "observation.images.image": "image0",
    },
    allow_extra_cameras=True,
    source_path=__file__,
    # oxe-auge_clean_v2 invariant: gripper dim 7 has been canonicalized to
    # open_fraction in [0, 1] (0=closed, 1=open) by data_process/merge_v3/
    # gripper_canonicalize.py during the v2 merge. The cross-dataset semantic
    # guard in scripts/train.py uses this annotation to gate any future
    # multi-repo mix that includes oxe-auge.
    gripper_semantic="open_fraction",
).build()


# Declarative source → canonical layout promotion rules. Consumed by the
# merge script at data prep time (not at training time).
MERGE_RECIPE = MergeRecipe(
    canonical=ArmLayoutSpec(
        arm_count=ArmCount.SINGLE,
        arm_dof=7,
        gripper_index_in_raw=7,
    ),
    source_layouts={
        # 6-DoF sources — zero-pad arm dim 6, move raw grip from 6 to 7.
        "berkeley_autolab_ur5": SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
        "bridge":               SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
        "jaco_play":            SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
    },
    # All other sources are 7-DoF with grip at raw index 7 → no rewrite.
    default_layout=SourceArmLayout(arm_dof=7, raw_gripper_idx=7),
)
