#include "../include/Meeting01Runner.hpp"

#include <string>

#include "../include/Meeting01Cli.hpp"

namespace meeting01
{

auto should_run_comparative_from_cli(int argc, char* argv[]) -> bool
{
    return meeting01::should_run_comparative_cli(argc, argv);
}

auto should_run_from_cli(int argc, char* argv[]) -> bool
{
    return should_run_comparative_from_cli(argc, argv);
}

} // namespace meeting01
