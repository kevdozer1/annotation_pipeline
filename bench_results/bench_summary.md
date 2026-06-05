# Benchmark Summary

The Mode A grid contains 12 runs: 4 annotation families x 3 seeds. Baseline is the best-scoring family in this POC table (0.039541 latent MSE); no richer conditioning family beats it on mean held-out MSE. The rich-text conditions use VLM-derived subtask segmentation quality, so this should be read as a test of whether hosted-VLM segmentation is enough to produce the pi0.7 effect, not as a claim about ideal human-validated segmentation. Benchmark backend: `real_lewm_frozen_adapter`. Rich-text + metadata is +1.4% relative to baseline (within seed-noise by the simple std-sum check). At 13 episodes this is a smoke-scale ablation, not a robust conclusion.

Gold-set reliability alongside this ablation:

- reviewed episodes: 0 / 13
- subtask boundary temporal IoU mean: None
- quality exact agreement: None
- quality within-one agreement: None
- subgoal selection agreement: None

