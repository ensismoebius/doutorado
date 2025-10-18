#include <iostream>
#include <vector>

#include "dataLoaders/MatFile.h"

using namespace matio;

auto main() -> int {
  // Load an existing MAT file (S02)
  const std::string mat_path =
      "/home/ensismoebius/Documentos/UNESP/doutorado/databases/Base de "
      "Datos Habla Imaginada/S02/S02_Audio.mat";

  MatFile mat_file;
  if (!mat_file.open(mat_path)) {
    std::cerr << "Failed to open MAT file: " << mat_path << '\n';
    return 1;
  }

  auto variables = mat_file.read_all_variables();

  std::cout << "Variables in file:\n";
  for (const auto& [name, var] : variables) {
    std::cout << "  " << name << ": " << var.type_name() << " [";
    for (size_t i = 0; i < var.dimensions.size(); ++i) {
      if (i != 0) {
        std::cout << "x";
      }
      std::cout << var.dimensions[i];
    }
    std::cout << "]\n";

    // Visit the variant to print all data for each supported type
    std::visit(
        [&](auto&& data) {
          using T = std::decay_t<decltype(data)>;
          if constexpr (std::is_same_v<T, std::string>) {
            std::cout << "    Data (string): " << data << "\n";
          } else {
            // It's a numeric vector; print all elements
            std::cout << "    Data (" << var.type_name() << "): ";
            for (size_t i = 0; i < data.size(); ++i) {
              std::cout << +data[i];
              if (i + 1 < data.size()) {
                std::cout << " ";
              }
            }
            std::cout << "\n";
          }
        },
        var.data);
  }

  return 0;
}