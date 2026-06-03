# Deviations

- Python 3.10 is used for the contained `.venv` because Python 3.11 is not installed on this workstation.
- A deterministic subtask/metadata fallback adapter remains for CI and downstream plumbing tests, but live Moondream labels are required for demo claims and benchmark runs.
- Raw VLM responses are stored under `bridgeengine_data/snapshots/<snapshot_id>/raw_vlm_outputs/<episode_id>/` so label provenance can be inspected before training.
- Live subtask labels currently use gripper/action transitions for temporal boundaries plus Moondream for semantic interval text because pure Moondream frame-index segmentation produced gappy, repetitive boundaries.
- The current live Moondream labels are intentionally blocked from benchmark use by quality gates until repeated subtask text and metadata judge contradictions are repaired.
- The subgoal-image labeler uses the actual end-of-segment frame instead of a generated future image, keeping the POC tractable while preserving the training interface.
- The mask, depth, and track labelers are preserved as `bridgeengine.labelers.perceptive` comparison modules but are no longer part of the main pi0.7-style benchmark.
- The benchmark grid uses a deterministic CPU proxy for latent MSE instead of launching 12 heavyweight LEWM GPU training runs; the adapter boundary is isolated in `bridgeengine.benchmark.train_lewm`.
- The snapshot and cut manifests use a deterministic `created_at_utc` timestamp so reproducibility tests can compare manifest bytes exactly.
- A Streamlit viewer is included even though the original Mode A text said no web interface, because Kevin explicitly asked for an inspection surface for demo readiness.
- The pre-pivot perception-era benchmark CSV is archived at `bench_results/pre_pivot/bench_results.csv`; fresh pivot benchmark output is intentionally ignored until live labels are green-lit.
