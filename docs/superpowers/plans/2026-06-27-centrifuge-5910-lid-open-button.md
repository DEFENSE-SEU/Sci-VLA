# Centrifuge 5910 Lid Open Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a distinct lid-release button to the Eppendorf 5910 model and make `open_centrifuge5910_lid` press it before the existing open-lid sequence.

**Architecture:** The MJCF model owns the physical button, touch sensor, and lid actuator. `Centrifuge_Eppendorf_5910` owns the button state machine: pressing the button unlocks the lid and commands a 30 degree pop-up target. The existing `open_centrifuge5910_lid` expert routine uses a helper pose to press the new button before continuing to the lid-grab trajectory.

**Tech Stack:** MuJoCo MJCF XML, Python 3.11, `mujoco`, existing `System`/`FlatButton` instrument framework, existing `Pose`/IK/TOPP expert utilities.

---

## File Structure

- Modify `model/instrument/centrifuge_eppendorf_5910_ri.xml`
  - Rename/replace the current generic `open_button` prototype with a `lid_open_button`.
  - Add a dedicated touch sensor named `lid_open_button`.
  - Add or adjust a lid actuator named `lid_opener` for the 30 degree pop-up behavior.
- Modify `scripts/autobio_scripts/instrument.py`
  - Add a `FlatButton('lid_open_button')` subsystem to `Centrifuge_Eppendorf_5910`.
  - Cache the lid actuator ID and 30 degree target.
  - Unlock and command the lid when the button is pressed.
- Modify `scripts/autobio_scripts/centrifuge5910_tasks.py`
  - Expose a reachable button press pose in `Centrifuge_5910`.
  - Insert the button press at the start of `open_centrifuge5910_lid`.
  - Do not add a new task name.

## Task 1: Model the Distinct Lid-Release Button

**Files:**
- Modify: `model/instrument/centrifuge_eppendorf_5910_ri.xml`

- [ ] **Step 1: Inspect current button and actuator names**

Run:

```bash
rg -n "open_button|button_sensor|lid_opener|lid-lock|<sensor>|<actuator>" model/instrument/centrifuge_eppendorf_5910_ri.xml
```

Expected: See the current generic `open_button`, `button_sensor`, and `lid_opener` definitions.

- [ ] **Step 2: Replace generic button names with lid-specific names**

In `model/instrument/centrifuge_eppendorf_5910_ri.xml`, replace the current block:

```xml
      <!-- 打开按钮 - 位置在控制面板上 -->
      <body name="open_button" pos="0.05 0.15 0.1">
        <joint name="open_button_joint" type="slide" axis="0 0 1" range="0 0.005" limited="true" damping="5" stiffness="1000" />
        <geom class="button" name="open_button_geom" />
        <site name="open_button_site" pos="0 0 0.005" size="0.012" type="cylinder" rgba="0.2 0.8 0.2 1" />
      </body>
```

with:

```xml
      <!-- Lid release button: distinct from the screen START button. -->
      <body name="lid_open_button" pos="0.25 -0.18 0.11">
        <joint name="lid_open_button_joint" type="slide" axis="0 0 1" range="-0.004 0" limited="true" damping="5" stiffness="1000" />
        <geom class="button" name="lid_open_button_geom" pos="0 0 0" />
        <site name="lid_open_button" pos="0 0 0.006" size="0.014" type="sphere" rgba="1 0.55 0.05 0.45" />
      </body>
```

Rationale: the C option places the new hardware-style button on the front-right body area. The exact position can be tuned after compilation/viewer inspection, but it should remain separate from the screen START region.

- [ ] **Step 3: Rename the touch sensor**

Replace:

```xml
  <sensor>
    <touch name="button_sensor" site="open_button_site" />
  </sensor>
```

with:

```xml
  <sensor>
    <touch name="lid_open_button" site="lid_open_button" />
  </sensor>
```

This lets `FlatButton('lid_open_button')` resolve both the subsystem and the sensor name through the existing namespace machinery.

- [ ] **Step 4: Adjust the lid actuator to be position-style**

Replace the current actuator:

```xml
  <actuator>
    <general name="lid_opener" joint="lid" gear="10" ctrlrange="0 1.94" forcerange="-50 50" />
  </actuator>
```

with:

```xml
  <actuator>
    <position name="lid_opener" joint="lid" kp="45" ctrlrange="0 1.94" forcerange="-80 80" />
  </actuator>
```

Rationale: the system will command an explicit lid joint target, so a position actuator matches the behavior better than a generic actuator with ambiguous force semantics.

- [ ] **Step 5: Compile the model scene**

Run:

```bash
python - <<'PY'
import mujoco
mujoco.MjSpec.from_file("model/scene/centrifuge_5910_tasks.xml").compile()
print("compiled")
PY
```

Expected: `compiled`.

- [ ] **Step 6: Commit the XML model change**

Run:

```bash
git add model/instrument/centrifuge_eppendorf_5910_ri.xml
git commit -m "feat: add centrifuge 5910 lid release button model"
```

Expected: commit succeeds with only the XML file staged.

## Task 2: Implement Lid-Release Button Behavior

**Files:**
- Modify: `scripts/autobio_scripts/instrument.py`

- [ ] **Step 1: Add the button subsystem**

In `scripts/autobio_scripts/instrument.py`, change:

```python
class Centrifuge_Eppendorf_5910(System):
    
    def _reload(self, model: mujoco.MjModel):
```

to:

```python
class Centrifuge_Eppendorf_5910(System):

    def _configure(self):
        self.lid_open_button = self.add_subsystem(FlatButton('lid_open_button'))

    def _reload(self, model: mujoco.MjModel):
```

- [ ] **Step 2: Cache actuator and pop-up target metadata**

Inside `Centrifuge_Eppendorf_5910._reload`, after:

```python
        self.lid_jntlimit = model.jnt_range[self.lid_joint]
```

add:

```python
        self.lid_opener = self.name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, 'lid_opener')
        self.lid_pop_angle = np.deg2rad(30.0)
        self.lid_pop_qpos = max(self.lid_jntlimit[0].item(), self.lid_qpos_max - self.lid_pop_angle)
```

- [ ] **Step 3: Initialize the release state on reset**

Replace:

```python
    def _reset(self, data: mujoco.MjData):
        self._bad_locking = False
        self._update(data)  # Delegate to _update
```

with:

```python
    def _reset(self, data: mujoco.MjData):
        self._bad_locking = False
        self._lid_release_active = False
        data.ctrl[self.lid_opener] = self.lid_qpos_max
        self._update(data)
```

- [ ] **Step 4: Update locking and pop-up behavior**

Replace the whole `Centrifuge_Eppendorf_5910._update` method:

```python
    def _update(self, data):
        lid_qpos = data.qpos[self.lid_qposadr]
        if lid_qpos < self.lid_qpos_max - 0.01:
        # try to lock while lid is open
            self._bad_locking = True
        else:
            self._bad_locking = False
        if not self._bad_locking:
        # lock when lid is closed
            data.eq_active[self.lid_lock] = 1
```

with:

```python
    def _update(self, data):
        lid_qpos = data.qpos[self.lid_qposadr]

        if self.lid_open_button.is_pressed:
            self._lid_release_active = True
            data.eq_active[self.lid_lock] = 0
            data.ctrl[self.lid_opener] = self.lid_pop_qpos

        if self._lid_release_active:
            data.eq_active[self.lid_lock] = 0
            data.ctrl[self.lid_opener] = self.lid_pop_qpos
            if lid_qpos <= self.lid_pop_qpos + 0.02:
                data.ctrl[self.lid_opener] = lid_qpos
            return

        if lid_qpos < self.lid_qpos_max - 0.01:
            self._bad_locking = True
        else:
            self._bad_locking = False

        if not self._bad_locking:
            data.eq_active[self.lid_lock] = 1
            data.ctrl[self.lid_opener] = self.lid_qpos_max
```

Rationale: pressing the new button unlocks the lid and commands the 30 degree pop-up target. The release state prevents the existing closed-lid auto-lock behavior from immediately re-locking the lid.

- [ ] **Step 5: Verify names resolve and reset works**

Run:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "scripts/autobio_scripts")
import mujoco
mujoco.mj_loadPluginLibrary("./libmjlab.so.3.3.0")
from task import SCENE_ROOT
from instrument import Centrifuge_Eppendorf_5910
from simulation import Manager

spec = mujoco.MjSpec.from_file(str(SCENE_ROOT / "centrifuge_5910_tasks.xml"))
instrument = Centrifuge_Eppendorf_5910("/centrifuge_eppendorf_5910:")
manager = Manager.from_spec(spec, [instrument])
manager.reload()
manager.reset(keyframe=0)
print("lid_lock", instrument.lid_lock)
print("lid_opener", instrument.lid_opener)
print("lid_pop_qpos", round(float(instrument.lid_pop_qpos), 4))
print("ok")
PY
```

Expected: IDs and `lid_pop_qpos` print, followed by `ok`.

- [ ] **Step 6: Commit behavior change**

Run:

```bash
git add scripts/autobio_scripts/instrument.py
git commit -m "feat: trigger centrifuge 5910 lid release button"
```

Expected: commit succeeds with only `instrument.py` staged.

## Task 3: Press the Button at the Start of `open_centrifuge5910_lid`

**Files:**
- Modify: `scripts/autobio_scripts/centrifuge5910_tasks.py`

- [ ] **Step 1: Cache the button site and add a pose helper**

In class `Centrifuge_5910`, update `_reset` from:

```python
    def _reset(self, data):
        super()._reset(data)  
        self.fk_lever = FK(1, self.model, data, f'{self.local_prefix}body', f'{self.local_prefix}lid')
```

to:

```python
    def _reset(self, data):
        super()._reset(data)
        self.fk_lever = FK(1, self.model, data, f'{self.local_prefix}body', f'{self.local_prefix}lid')
        self.lid_open_button_site = self.model.site(f'{self.local_prefix}lid_open_button').id
```

Then add this method after `get_eefpose_lever`:

```python
    def get_lid_open_button_pose(self, data: mujoco.MjData, mode: str = 'pre') -> Pose:
        site_pos = data.site_xpos[self.lid_open_button_site]
        site_mat = data.site_xmat[self.lid_open_button_site]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, site_mat)

        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, [0.0, 1.0, 0.0], np.pi)
        res_quat = np.zeros(4)
        mujoco.mju_mulQuat(res_quat, quat, rel_quat)

        match mode:
            case 'pre':
                rel_pos = np.array([0.0, 0.0, 0.08])
            case 'press':
                rel_pos = np.array([0.0, 0.0, 0.01])
            case _:
                raise ValueError(f"Unknown lid open button pose mode: {mode}")

        res_pos = site_pos + rel_pos
        return Pose(res_pos, res_quat)
```

Rationale: the helper exposes a single stable target for the expert routine without adding a new task.

- [ ] **Step 2: Add an expert helper to press the button**

In `Centrifuge5910ManipulateExpert`, after `gripper_control`, add:

```python
    def press_lid_open_button(self):
        pre_pose = self.instrument.get_lid_open_button_pose(self.data, mode='pre')
        press_pose = self.instrument.get_lid_open_button_pose(self.data, mode='press')
        self.gripper_control(250)
        self.move_to(pre_pose, num_steps=12)
        self.move_to(press_pose, num_steps=8)
        for _ in range(60):
            self.step_and_log({})
        self.move_to(pre_pose, num_steps=8)
        for _ in range(120):
            self.step_and_log({})
```

- [ ] **Step 3: Insert button press into existing open-lid task**

At the start of the `case 'open_centrifuge5910_lid':` block in `execute`, immediately after:

```python
            case 'open_centrifuge5910_lid':
```

insert:

```python
                self.press_lid_open_button()
```

Do not add a new entry to the `tasks` list at the bottom of the file.

- [ ] **Step 4: Remove obsolete manual open-button assumptions if present**

Run:

```bash
rg -n "open_button|button_sensor|lid_open_button|press_centrifuge5910_lid" scripts/autobio_scripts/centrifuge5910_tasks.py
```

Expected: only `lid_open_button` helper references appear. No new task name such as `press_centrifuge5910_lid_open_button` appears.

- [ ] **Step 5: Verify script-level import and reset**

Run:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "scripts/autobio_scripts")
import mujoco
mujoco.mj_loadPluginLibrary("./libmjlab.so.3.3.0")
from centrifuge5910_tasks import Centrifuge5910Manipulate

spec = Centrifuge5910Manipulate.load()
task = Centrifuge5910Manipulate(spec)
task.task = "open_centrifuge5910_lid"
info = task.reset(seed=0)
print(info["prefix"])
print("lid_qpos", round(float(task.data.qpos[task.instrument.lid_qposadr]), 4))
print("lid_lock", int(task.data.eq_active[task.instrument.lid_lock]))
print("button_site", task.instrument.lid_open_button_site)
PY
```

Expected: prefix is `open the lid of the centrifuge5910`, lid is near closed, lock is active or ready to be released by the button, and a button site ID prints.

- [ ] **Step 6: Run the open-lid expert if environment dependencies allow**

Run:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "scripts/autobio_scripts")
import mujoco
mujoco.mj_loadPluginLibrary("./libmjlab.so.3.3.0")
from centrifuge5910_tasks import Centrifuge5910Manipulate

spec = Centrifuge5910Manipulate.load()
expert = Centrifuge5910Manipulate.Expert(spec)
expert.task = "open_centrifuge5910_lid"
expert.reset(seed=0)
expert.press_lid_open_button()
lid_qpos = float(expert.data.qpos[expert.instrument.lid_qposadr])
target = float(expert.instrument.lid_pop_qpos)
print("lid_qpos", round(lid_qpos, 4))
print("target", round(target, 4))
print("released", lid_qpos <= expert.instrument.lid_qpos_max - 0.05)
PY
```

Expected: `released True`. If this is blocked by the known JAX package mismatch, report that full expert verification is blocked and keep the model-level checks as evidence.

- [ ] **Step 7: Commit expert change**

Run:

```bash
git add scripts/autobio_scripts/centrifuge5910_tasks.py
git commit -m "feat: press lid release during centrifuge 5910 open task"
```

Expected: commit succeeds with only `centrifuge5910_tasks.py` staged.

## Task 4: Final Verification

**Files:**
- Verify: `model/instrument/centrifuge_eppendorf_5910_ri.xml`
- Verify: `scripts/autobio_scripts/instrument.py`
- Verify: `scripts/autobio_scripts/centrifuge5910_tasks.py`

- [ ] **Step 1: Check git status excludes unrelated user files**

Run:

```bash
git status --short
```

Expected: no staged changes. Unrelated pre-existing untracked files such as `scripts/split_libero_long_subtasks.py`, `scripts/generate_libero_recovery_subtasks.py`, and `tests/` may remain untracked.

- [ ] **Step 2: Re-run model compile**

Run:

```bash
python - <<'PY'
import mujoco
mujoco.MjSpec.from_file("model/scene/centrifuge_5910_tasks.xml").compile()
print("compiled")
PY
```

Expected: `compiled`.

- [ ] **Step 3: Re-run reset/import verification**

Run the verification command from Task 3 Step 5.

Expected: import succeeds, reset succeeds, and `open_centrifuge5910_lid` still reports the existing prefix.

- [ ] **Step 4: Verify no new task name was introduced**

Run:

```bash
rg -n "press_centrifuge5910_lid|lid_open.*task|open_centrifuge5910_lid" scripts/autobio_scripts/centrifuge5910_tasks.py
```

Expected: `open_centrifuge5910_lid` exists; no `press_centrifuge5910_lid_open_button` or equivalent new task exists.

- [ ] **Step 5: Summarize verification result**

Record:

```text
Model compile: PASS or FAIL
Reset/import verification: PASS or FAIL
Button release verification: PASS, FAIL, or BLOCKED by environment
No new task added: PASS or FAIL
```

If any check fails, fix the implementation before final response.
