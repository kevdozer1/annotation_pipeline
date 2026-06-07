# BridgeEngine Public Release Plan

Last updated: 2026-06-06

## Public Product Claim

BridgeEngine should be released as:

> A configurable pi0.7-style annotation and curation pipeline for video datasets, validated on robot manipulation video and architected to adapt to other video classes through dataset adapters, VLM providers, and rubric templates.

Do not release it as:

> A universal video annotation system proven to work on every data class.

The validated class is BridgeData-style robot manipulation video. Human-motion and general-video adaptation are credible extensions, not validated claims yet.

## Split The Work Into Two Repos

### Internal Research Repo

Purpose: finish the controlled scientific claim.

Keep frozen:

- 100-episode BridgeData subset
- calibrated score rubric
- held-out split
- LeWM frozen-adapter metric
- seed list
- head-to-head family definitions

Do not over-generalize this repo before the head-to-head is finished.

### Clean Public Fork

Purpose: make the annotation pipeline useful to other people.

Make configurable:

- dataset adapters
- VLM backends
- prompt templates
- scoring rubrics
- quality gates
- review GUI fields
- export formats
- optional benchmark adapters

## Public-Fork Module Shape

Recommended package layout:

```text
bridgeengine/
  adapters/
    base.py
    bridgedata.py
    folder_video.py
  backends/
    base.py
    gemini.py
    openai.py
    mock.py
  rubrics/
    robot_manipulation.yaml
    human_motion.yaml
    general_video.yaml
  labelers/
    subtask_segmenter.py
    episode_metadata.py
    subgoal_images.py
  review/
    app.py
  quality/
    gates.py
  export/
    parquet.py
    lerobot.py
    rlds.py
  eval/
    base.py
    lewm.py
```

## Minimum Useful Public Release

1. A folder-video dataset adapter.
2. Gemini/OpenAI/mock backend plugins.
3. A robot-manipulation rubric template.
4. The browser review GUI.
5. Quality report and reliability report commands.
6. Parquet snapshot export.
7. A tiny demo dataset or synthetic fixture.
8. Clear docs explaining how to bring your own rubric.

## What To Defer From Public V1

- Full LeWM benchmark setup.
- SAM/VDA/CoTracker perception baselines.
- Full BridgeData download tooling.
- Claims about human-motion or general-video performance.
- Distributed labeling.
- Production dataset versioning formats.

## Why The Score Disagreement Is A Feature, Not Just A Bug

Gemini and OpenAI did not match Kevin's scores exactly. That should be presented honestly:

- VLMs are useful cheap first-pass annotators.
- Raw VLM scores are not automatically calibrated to a user's training-data rubric.
- BridgeEngine's value is the inspect-calibrate-measure loop.
- The release should let users bring their own examples and rubric, then measure reliability.

That framing is stronger than pretending hosted VLM scores are universally correct.
