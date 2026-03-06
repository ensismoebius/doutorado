/**
 * @file loadingData.cpp
 * @brief Tiny utility demo: inspect a MATLAB `.mat` file via project dataLoaders helpers.
 */

#include <CLI/CLI.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "nn/dataLoaders/mat_file_utils.hpp"

using CLI::App;
using matioCpp::utils::get_variable_dimensions;
using matioCpp::utils::list_variable_names;
using std::cerr;
using std::cout;
using std::exception;
using std::string;

auto main(int argc, char* argv[]) -> int
{
    const std::vector<string> default_mat_paths = {
        "/home/ensismoebius/Documentos/UNESP/doutorado/databases/BaseDeDatosHablaImaginada/S01/S01_Audio.mat",
        "/home/ensismoebius/Documentos/UNESP/doutorado/databases/BaseDeDatosHablaImaginada/S01/S01_EEG.mat"};

    string mat_path;

    App app{"Inspect variables and dimensions from a MATLAB .mat file."};

    app.add_option(           //
           "mat_file",        //
           mat_path,          //
           "Path to MAT file" //
           )
          ->expected(0, 1)
        ->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    try
    {
        const std::vector<string> files_to_inspect =
            mat_path.empty() ? default_mat_paths : std::vector<string>{mat_path};

        bool had_errors = false;

        for (const auto& file_path : files_to_inspect)
        {
            auto variables = list_variable_names(file_path);

            if (variables.empty())
            {
                cerr << "Error: no variables found: " << file_path << '\n';
                had_errors = true;
                continue;
            }

            cout << "Variables in file: " << file_path << '\n';

            for (const auto& name : variables)
            {
                auto dims_opt = get_variable_dimensions(file_path, name);
                if (!dims_opt)
                {
                    cout << "  " << name << ": [unavailable dimensions]\n";
                    continue;
                }

                const auto& dims = *dims_opt;

                cout << "  " << name << ": [";
                for (size_t i = 0; i < dims.size(); ++i)
                {
                    if (i != 0)
                    {
                        cout << "x";
                    }
                    cout << dims[i];
                }
                cout << "]\n";
            }
        }

        if (had_errors)
        {
            return 1;
        }
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
