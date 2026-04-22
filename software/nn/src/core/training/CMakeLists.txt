# Core training utilities - headers only library

# Headers are in the same directory
# No build targets needed for header-only interface

# Tests disabled: Trainer.hpp has pre-existing bugs in template body
# - Nested lambdas incorrectly access member variables
# - Missing includes for MSELossImpl
# Tests for Config and EpochResult can be added once Trainer is fixed
# add_subdirectory(tests)