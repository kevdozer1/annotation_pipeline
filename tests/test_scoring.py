from __future__ import annotations

from bridgeengine.scoring import score_metadata_for_curation


def test_curation_score_promotes_structured_ambiguous_task_result() -> None:
    metadata = {
        "quality": 3,
        "mistake": True,
        "reason": "The robot moves the knife to the pot area, but the final frame is ambiguous and the requested placement is only partially completed.",
    }
    segments = [
        {"segment_idx": 0, "start_step": 0, "end_step": 9, "subtask_text": "move gripper toward the yellow-handled knife near the metal pot"},
        {"segment_idx": 1, "start_step": 10, "end_step": 18, "subtask_text": "carry the yellow-handled knife into the metal pot"},
        {"segment_idx": 2, "start_step": 19, "end_step": 21, "subtask_text": "release the yellow-handled knife in the metal pot"},
    ]

    score = score_metadata_for_curation("episode_x", metadata, segments)

    assert score.task_success_quality == 3
    assert score.curation_quality == 4
    assert score.curation_keep is True
    assert score.boundary_clarity == "clear"


def test_curation_score_rejects_no_visible_grasp_release_cycle() -> None:
    metadata = {
        "quality": 1,
        "mistake": True,
        "reason": "The robot moves near the pot but does not visibly grasp or release the spoon into it; the pot appears empty in the final frame.",
    }
    segments = [
        {"segment_idx": 0, "start_step": 0, "end_step": 10, "subtask_text": "move gripper to the blue spoon beside the pot"},
        {"segment_idx": 1, "start_step": 11, "end_step": 22, "subtask_text": "grasp and carry the blue spoon into the metal pot"},
    ]

    score = score_metadata_for_curation("episode_y", metadata, segments)

    assert score.curation_quality == 1
    assert score.curation_keep is False
    assert score.boundary_clarity == "weak"


def test_curation_score_near_rejects_unclear_localized_end_state_attempt() -> None:
    metadata = {
        "quality": 1,
        "mistake": True,
        "reason": (
            "The robot approaches the cups but no cup is visibly grasped, lifted, transported, or released into the sink. "
            "By the final frame, the cup remains outside the sink near the gripper, so the task is unfinished."
        ),
    }
    segments = [
        {"segment_idx": 0, "start_step": 0, "end_step": 3, "subtask_text": "position gripper over the sink and dish rack area"},
        {"segment_idx": 1, "start_step": 4, "end_step": 19, "subtask_text": "move down toward the blue cup on the tray"},
        {"segment_idx": 2, "start_step": 20, "end_step": 22, "subtask_text": "grip and stabilize the blue cup on the tray"},
    ]

    score = score_metadata_for_curation("episode_near_reject", metadata, segments)

    assert score.curation_quality == 3
    assert score.curation_keep is False
    assert score.boundary_clarity == "partial"


def test_curation_score_clear_rejects_short_destination_miss() -> None:
    metadata = {
        "quality": 2,
        "mistake": True,
        "reason": "The robot grasps and lifts the pan from the sink area, but it never transports or places it on a stove; the task is unfinished and the destination is not reached.",
    }
    segments = [
        {"segment_idx": 0, "start_step": 0, "end_step": 10, "subtask_text": "move to and grasp the metal pan in the sink"},
        {"segment_idx": 1, "start_step": 11, "end_step": 22, "subtask_text": "lift and carry the metal pan out of the sink toward the stove"},
    ]

    score = score_metadata_for_curation("episode_clear_reject", metadata, segments)

    assert score.curation_quality == 1
    assert score.curation_keep is False
    assert score.boundary_clarity == "weak"


def test_curation_score_clear_keeps_long_stacked_boundaries() -> None:
    metadata = {
        "quality": 4,
        "mistake": False,
        "reason": "The robot appears to grasp the egg and place it into the pot/pan, and the pot/pan is on the stove at the end. The only limitation is that the pot/pan appears to have already been on the stove rather than being moved there during the episode.",
    }
    segments = [
        {"segment_idx": 0, "start_step": 0, "end_step": 24, "subtask_text": "move to the white egg and grasp it"},
        {"segment_idx": 1, "start_step": 25, "end_step": 37, "subtask_text": "carry the white egg toward the metal pot/pan"},
        {"segment_idx": 2, "start_step": 38, "end_step": 54, "subtask_text": "lower and release the white egg into the metal pot/pan"},
        {"segment_idx": 3, "start_step": 55, "end_step": 77, "subtask_text": "position the metal pot/pan with the egg on the stove burner area"},
    ]

    score = score_metadata_for_curation("episode_structured_keep", metadata, segments)

    assert score.curation_quality == 5
    assert score.curation_keep is True
    assert score.boundary_clarity == "clear"


def test_curation_score_clear_keeps_successful_pickup_without_release() -> None:
    metadata = {
        "quality": 5,
        "task_success_quality": 5,
        "mistake": False,
        "reason": "The robot successfully approached, grasped, and lifted the pan off the stove, fully completing the task.",
    }
    segments = [
        {"segment_idx": 0, "start_step": 0, "end_step": 4, "subtask_text": "grasp pan handle"},
        {"segment_idx": 1, "start_step": 5, "end_step": 14, "subtask_text": "lift pan from stove"},
    ]

    score = score_metadata_for_curation("episode_pickup", metadata, segments)

    assert score.curation_quality == 5
    assert score.curation_keep is True
    assert score.boundary_clarity == "clear"
