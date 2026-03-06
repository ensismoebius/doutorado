/**
 * @file loadingData.cpp
 * @brief Tiny utility demo: inspect a MATLAB `.mat` file via project dataLoaders helpers.
 */

#include <iostream>
#include <string>

#include "nn/dataLoaders/mat_file_utils.hpp"

auto main(int argc, char* argv[]) -> int
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <mat-file>\n";
        return 1;
    }

    try
    {
        const std::string mat_path = argv[1];
        auto variables = matioCpp::utils::list_variable_names(mat_path);
        if (variables.empty())
        {
            std::cerr << "Error: no variables found (or failed to read MAT file): " << mat_path
                      << '\n';
            return 1;
        }

        std::cout << "Variables in file: " << mat_path << '\n';
        for (const auto& name : variables)
        {
            auto dims_opt = matioCpp::utils::get_variable_dimensions(mat_path, name);
            if (!dims_opt)
            {
                std::cout << "  " << name << ": [unavailable dimensions]\n";
                continue;
            }

            const auto& dims = *dims_opt;
            std::cout << "  " << name << ": [";
            for (size_t i = 0; i < dims.size(); ++i)
            {
                if (i != 0)
                {
                    std::cout << "x";
                }
                std::cout << dims[i];
            }
            std::cout << "]\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
