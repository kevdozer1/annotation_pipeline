# Benchmark Summary

The Mode A proxy grid contains 12 runs: 4 annotation families x 1 scale x 3 seeds. Hybrid is the best-scoring family in this POC table (0.132443 latent MSE), and perceptive supervision beats baseline by 16.0% at 13 episodes. These numbers are deterministic CPU proxy results wired to the LEWM experiment contract; the project is ready for a real GPU LEWM sweep by replacing `bridgeengine.benchmark.train_lewm.run_family_seed` with the heavyweight training adapter.
