# Centrifuge 5910 Lid Open Button Design

## Goal

Update the Eppendorf 5910 centrifuge simulation so the existing `open_centrifuge5910_lid` task starts by pressing a distinct lid-release button. The button should be visually and semantically separate from the existing START button, and pressing it should make the closed lid pop up by about 30 degrees so the robot can continue opening it.

## Scope

- Add a new physical lid-release button to the centrifuge model.
- Keep `press_centrifuge5910_button` as the START/start-centrifuge operation.
- Do not add a new task name for pressing the lid-release button.
- Insert the lid-release button press at the start of the existing `open_centrifuge5910_lid` expert routine.
- Preserve the later open-lid sequence that uses the robot to grab and move the lid.

## Model Changes

In `model/instrument/centrifuge_eppendorf_5910_ri.xml`, add a distinct front-right hardware-style button:

- Body/site/geom names should use `lid_open_button` naming, not generic `open_button`, to avoid confusion with START.
- The button should have its own touch sensor.
- The button should be placed on the front-right side of the centrifuge body for clear visual distinction from the screen START button.
- The button should remain easy for the UR5e gripper to contact.

The lid hinge range is `0` to `1.94` radians, where the current task treats values near `1.94` as closed. A 30 degree pop-up corresponds to reducing the lid joint by about `0.5236` radians from the closed value, so the target is approximately:

```text
lid_pop_qpos = lid_qpos_max - 0.5236
```

## Behavior Changes

`Centrifuge_Eppendorf_5910` should detect the lid-release touch sensor during system updates. When pressed:

- deactivate `lid-lock`;
- command the lid toward the 30 degree pop-up target;
- keep the lid unlocked while it is popped open;
- avoid re-locking immediately after the button press.

The implementation should prefer existing system patterns:

- use `FlatButton` or equivalent sensor handling rather than ad hoc contact scans;
- store relevant IDs during `_reload`;
- initialize button/lid-release state during `_reset`;
- update `data.ctrl` only for the lid actuator added for this behavior.

## Task And Expert Changes

The existing `open_centrifuge5910_lid` task should start from a closed, locked lid as it does now. The expert execution should then:

1. Move to a pre-press pose near `lid_open_button_site`.
2. Press the lid-release button long enough for the touch sensor to trigger.
3. Wait briefly for the lid to pop to about 30 degrees.
4. Continue with the existing open-lid trajectory.

No new task should be added to the task list. The existing START button task keeps its current meaning.

## Verification

Verification should cover:

- the scene/model compiles with the new button, sensor, and actuator;
- reset for `open_centrifuge5910_lid` still starts with the lid closed and locked;
- pressing `lid_open_button` unlocks the lid and moves it near the 30 degree pop-up target;
- `press_centrifuge5910_button` still refers to START and is not repurposed;
- the `open_centrifuge5910_lid` expert runs without failing during the new initial press phase.

If the local JAX environment is still mismatched, use model-level compile checks first and report any blocked full-script verification separately.
