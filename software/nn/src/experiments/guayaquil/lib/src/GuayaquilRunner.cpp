#include "../include/E04Runner.hpp"

#include <string>

#include "../include/E04Cli.hpp"

namespace e04
{

auto should_run_comparative_from_cli(int argc, char* argv[]) -> bool
{
    return e04::should_run_comparative_cli(argc, argv);
}

auto should_run_from_cli(int argc, char* argv[]) -> bool
{
    return should_run_comparative_from_cli(argc, argv);
}

} // namespace e04
