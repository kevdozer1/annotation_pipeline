# BridgeEngine Annotation Rubric

Last updated: 2026-06-06

This is the frozen internal rubric for the 100-episode BridgeData experiment. The public fork can make rubrics configurable, but this internal result should not keep changing the scoring target.

## What The Score Means

The score is not pure task success. It is training usefulness for visible robot-manipulation learning.

Good episodes have clear, visible cause-effect manipulation boundaries: approach, contact or grasp, object motion, release or stable end state. Long or unusual multi-step structure is not a penalty if the boundaries are visible. That kind of unusual structure belongs in anomaly/value scoring, not in quality rejection.

## Score Definitions

| Score | Label | Meaning |
|---:|---|---|
| 1 | clear reject | Not usable for visible manipulation-boundary learning. The action is absent, wrong, too occluded, or the episode gives no reliable cause-effect signal. |
| 2 | reject | Weak or mostly wrong interaction. Some robot/object evidence may exist, but it is not enough for the main keep set. |
| 3 | near reject | Some useful evidence, but too occluded, incomplete, ambiguous, or unstable for reliable training without caution. |
| 4 | near keep | Usable demonstration with a visible manipulation arc, but with some imperfection such as partial occlusion, extra motion, or minor ambiguity. |
| 5 | clear keep | Clean visible cause-effect interaction with clear manipulation boundaries and useful training signal. Long structured episodes can be 5/5. |

## Key Corrections From Visual Review

- Do not penalize a long episode merely because it has multiple stacked subtasks.
- Do not downgrade a clear visible pan/object movement after placement if the task boundaries remain interpretable.
- Do downgrade episodes where the arm occludes most of the meaningful action.
- Do downgrade episodes where the task never finishes and the end state is not stable.
- Treat short unreliable end-state glimpses as weak evidence, not as clear success.
- Use anomaly/value score to surface interesting unusual structure, not to reject it.

## Internal Versus Public Release

For this repo, freeze the rubric above before running the final head-to-head experiment.

For the public fork, expose this as a configurable rubric template. The public claim should be: BridgeEngine supports rubric-calibrated pi0.7-style annotation pipelines, validated here on robot manipulation video. It should not claim that one universal score works across all video classes.
