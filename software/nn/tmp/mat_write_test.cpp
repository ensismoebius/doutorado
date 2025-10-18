#include <iostream>

#include "src/dataLoaders/mat_file.h"
using namespace matio;
int main() {
  MatFile mf;
  if (!mf.create("debug_test.mat")) {
    std::cerr << "create failed\n";
    return 1;
  }
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  if (!mf.write_double_matrix("debug", data, {2, 2})) {
    std::cerr << "write failed\n";
    return 1;
  }
  mf.close();
  std::cout << "wrote debug_test.mat\n";
  MatFile rf;
  if (!rf.open("debug_test.mat")) {
    std::cerr << "open failed\n";
    return 1;
  }
  auto vars = rf.read_all_variables();
  std::cout << "vars=" << vars.size() << "\n";
  for (auto &p : vars) {
    std::cout << p.first << " type=" << p.second.type_name() << " dims=";
    for (auto d : p.second.dimensions) std::cout << d << ",";
    std::cout << "\n";
  }
}
