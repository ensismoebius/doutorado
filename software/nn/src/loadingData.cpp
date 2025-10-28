#include <iostream>

#include "dataLoaders/MatFile.h"

using std::cerr;
using std::cout;
using std::decay_t;
using std::string;
using std::visit;

using namespace matio;

auto main() -> int {
    // Load an existing MAT file (S02)
    const string mat_path =
        "/home/ensismoebius/Documentos/UNESP/doutorado/databases/Base de "
        "Datos Habla Imaginada/S02/S02_Audio.mat";

    MatFile mat_file;

    if (!mat_file.open(mat_path)) {
        cerr << "Failed to open MAT file: " << mat_path << '\n';
        return 1;
    }

    auto variables = mat_file.read_all_variables();

    cout << "Variables in file:\n";
    for (const auto& [name, var] : variables) {
        cout << "  " << name << ": " << var.type_name() << " [";
        for (size_t i = 0; i < var.dimensions.size(); ++i) {
            if (i != 0) {
                cout << "x";
            }
            cout << var.dimensions[i];
        }
        cout << "]\n";

        // Visit the variant to print all data for each supported type
        visit(
            [&](auto&& data) {
                using VARIABLE_TYPE = decay_t<decltype(data)>;
                if constexpr (std::is_same_v<VARIABLE_TYPE, string>) {
                    // It's a string; print it
                    cout << "Data (string): " << data << "\n";
                } else {
                    // It's a numeric vector; print all elements
                    cout << "    Data (" << var.type_name() << "): ";
                    for (size_t i = 0; i < data.size(); ++i) {
                        cout << +data[i];
                        if (i + 1 < data.size()) {
                            cout << " ";
                        }
                    }
                    cout << "\n";
                }
            },
            var.data);
    }

    return 0;
}