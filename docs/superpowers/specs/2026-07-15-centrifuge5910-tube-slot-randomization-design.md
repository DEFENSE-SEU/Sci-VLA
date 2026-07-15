# Centrifuge 5910 Tube Slot Randomization Design

## Goal

Correct the two Centrifuge 5910 take tasks so the experimental and balance tubes have stable identities and distinct centrifuge slots, while randomizing the non-target tube between its assigned centrifuge slot and its assigned rack position.

## Fixed identity mapping

- The experimental tube is `self.tube`, uses centrifuge slot `0`, and uses rack position `(row=1, col=4)`.
- The balance tube is `self.tube2`, uses centrifuge slot `1`, and uses rack position `(row=0, col=2)`.

These assignments match the existing `place_experimental_tube_into_centrifuge5910` and `place_balance_tube_into_centrifuge5910` tasks.

## Initial-state behavior

For `take_experimental_tube_from_centrifuge5910`:

- Place the experimental tube in slot `0` on every reset.
- Place the balance tube with equal probability either in slot `1` or at its rack position.
- Keep the experimental tube's destination at its rack position.

For `take_balance_tube_from_centrifuge5910`:

- Place the balance tube in slot `1` on every reset.
- Place the experimental tube with equal probability either in slot `0` or at its rack position.
- Keep the balance tube's destination at its rack position.

The random branch will use NumPy's random generator so the existing `reset(seed)` call makes the sampled state reproducible.

## Expert behavior

- The experimental take expert targets slot `0` and derives its end-effector pose from `self.tube`.
- The balance take expert targets slot `1` and derives its end-effector pose from `self.tube2`.

No motion-path changes beyond correcting the selected tube and slot are included.

## Implementation structure

Define named constants for the two tube slot IDs and rack coordinates. Add a focused helper on `Centrifuge5910Manipulate` that places a supplied tube with equal probability in its assigned centrifuge slot or rack position. Both take-task reset branches use the constants and helper, while the expert branches use the same slot constants.

## Tests

Add regression coverage that verifies:

1. The target tube always starts in its assigned centrifuge slot.
2. The non-target tube can start in its assigned centrifuge slot.
3. The non-target tube can start at its assigned rack position.
4. The two expert branches target different assigned slots and use the correct tube object.
5. The relevant Centrifuge 5910 test suite continues to pass.

## Scope and compatibility

The change is limited to the two take-task reset branches, their expert target selection, named mapping constants, the random-placement helper, and regression tests. Existing unrelated working-tree edits, including commented waits and lid-motion changes, remain untouched.
