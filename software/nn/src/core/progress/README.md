# Progress

- Progress rendering in ProgressManager now writes padded/truncated cells directly to the output stream instead of creating intermediate fitted strings.
- This avoids GCC false-positive string overflow diagnostics seen during optimized builds while keeping terminal output formatting unchanged.
- Validation performed: rebuilt guayaquil target after the change and confirmed successful link without the prior ProgressManager string-overflow warnings.
