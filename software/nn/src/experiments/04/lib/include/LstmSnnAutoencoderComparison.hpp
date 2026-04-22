#pragma once

// Forward declaration for the Experiment04 runner used by experiment03/experiment04 mains.
// Implementation lives in src/experiments/04/experiment04.cpp.
class LstmAutoencoderExperiment
{
   public:
    auto run(int argc, char* argv[]) -> int;
};
