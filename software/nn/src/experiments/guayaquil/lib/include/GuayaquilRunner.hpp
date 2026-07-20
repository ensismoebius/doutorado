#pragma once

namespace guayaquil
{
auto should_run_from_cli(int argc, char* argv[]) -> bool;
auto should_run_comparative_from_cli(int argc, char* argv[]) -> bool;
auto run_comparative_experiment(int argc, char* argv[]) -> int;
} // namespace guayaquil
