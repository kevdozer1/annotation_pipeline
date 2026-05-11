from __future__ import annotations

import pandas as pd

EPISODE_COLUMNS = [
    "episode_id",
    "source_path_video",
    "source_path_actions",
    "source_path_meta",
    "source_path_frames",
    "num_steps",
    "language_instruction",
    "snapshot_id",
]

STEP_COLUMNS = [
    "episode_id",
    "step_idx",
    "timestamp",
    "action",
    "state",
    "snapshot_id",
]

SENSOR_COLUMNS = [
    "episode_id",
    "sensor_name",
    "calibration_json",
    "snapshot_id",
]

LABEL_COLUMNS = [
    "episode_id",
    "step_idx",
    "labeler_name",
    "labeler_version",
    "label_payload_path",
    "confidence",
    "provenance_json",
    "snapshot_id",
]


def empty_labels(snapshot_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=LABEL_COLUMNS)
    frame["snapshot_id"] = frame.get("snapshot_id", pd.Series(dtype="str"))
    return frame

