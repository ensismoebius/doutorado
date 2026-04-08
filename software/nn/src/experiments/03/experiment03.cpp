/**
 * @file src/experiments/03/experiment03.cpp
 * @brief Lightweight launcher for Experiment03 CLI.
 *
 * This translation unit implements the small main() wrapper used to parse CLI
 * parameters and invoke the `Experiment03` driver. It is intentionally thin;
 * the experiment implementation lives under `lib/src/` and the runtime
 * configuration is declared in `lib/include/experiment03.hpp`.
 */
#include "lib/include/experiment03.hpp"

#include <cstddef>
#include <iostream>
#include <memory>

#include "default_config.hpp"
#include "lib/include/cli.hpp"
#include "nn/logging/StreamRedirector.hpp"

using nn::logging::StreamRedirector;
using std::cerr;
using std::cout;
using std::exception;
using std::make_shared;
using std::size_t;

auto main(int argc, char* argv[]) -> int
{
    Config config = parseCliParams(argc, argv, default_config);
    StreamRedirector redirect(true, true);
    Experiment03 experiment(config);
    return experiment.run();
}
