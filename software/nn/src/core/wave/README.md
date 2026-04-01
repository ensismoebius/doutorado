# wave

Purpose
- Audio processing helpers and feature extraction utilities used by demos and preprocessing pipelines.

Usage
- Feature extraction functions and signal processing pipelines are implemented here; call them from preprocessing code before feeding tensors to models.

Tests
- See `src/core/wave/tests/` for examples and expected outputs.

Recent updates
- WAV write open-file failures now use `NN_LOG_ERROR` instead of direct stdout writes, keeping error reporting consistent with centralized logging.

Optimization techniques and references
- Centralized logging channel for failure paths: standardizes observability and reduces output-routing divergence across CLI, tests, and redirected streams (see [1], [2]).

Bibliographic references
- [1] Martin Kleppmann. Designing Data-Intensive Applications. O'Reilly, 2017.
- [2] Martin Fowler. Refactoring: Improving the Design of Existing Code (2nd ed.). Addison-Wesley, 2018.
