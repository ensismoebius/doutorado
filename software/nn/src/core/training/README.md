# Training

- `Trainer.hpp` owns generic autoencoder and supervised training loops with callback-driven reporting.
- Batch progress callbacks now emit fractional `TrainingState::batch_progress` updates while a batch is still executing, so UIs can render live intra-batch movement instead of only whole-batch jumps.
- `ITrainingCallback` remains backward compatible: existing callbacks can ignore `on_batch_progress`, while progress-aware callbacks can map fractional batch work onto their own bars.
- Regression coverage lives in `tests/trainer_genericity_gtest.cpp`, including a check that batch progress reaches `1.0F` by batch end.