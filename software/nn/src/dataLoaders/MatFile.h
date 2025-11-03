#pragma once

#include <string>
#include <vector>
#include <memory>
#include <matio.h>

class MatFile {
public:
    MatFile(const std::string& filename);
    ~MatFile();

    std::vector<std::string> getVariableNames();
    matvar_t* readVariable(const std::string& varName);

private:
    mat_t* matfp_ = nullptr;
};
