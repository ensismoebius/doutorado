#include "../include/GuayaquilRunner.hpp"

#include <string>

#include "../include/GuayaquilCli.hpp"

namespace guayaquil
{

auto should_run_comparative_from_cli(int argc, char* argv[]) -> bool
{
    return guayaquil::should_run_comparative_cli(argc, argv);
}

auto should_run_from_cli(int argc, char* argv[]) -> bool
{
    return should_run_comparative_from_cli(argc, argv);
}

} // namespace guayaquil
