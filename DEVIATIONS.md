# Deviations

- Python 3.10 is used for the contained `.venv` because Python 3.11 is not installed on this workstation.
- The POC caption labeler is a deterministic Moondream-compatible adapter rather than live Moondream inference, keeping quickstart runnable without downloading model weights.
- The mask, depth, and track labelers wrap existing LEWM artifacts from `D:\bridgedata_v2_subset` when present and use deterministic CPU fallbacks only when those artifacts are unavailable.
- The benchmark grid uses a deterministic CPU proxy for latent MSE instead of launching 12 heavyweight LEWM GPU training runs; the adapter boundary is isolated in `bridgeengine.benchmark.train_lewm`.
- The snapshot and cut manifests use a deterministic `created_at_utc` timestamp so reproducibility tests can compare manifest bytes exactly.
- A Streamlit viewer is included even though the original Mode A text said no web interface, because Kevin explicitly asked for an inspection surface for demo readiness.
