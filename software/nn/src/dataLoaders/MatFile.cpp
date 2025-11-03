#include "MatFile.h"
#include <stdexcept>

MatFile::MatFile(const std::string& filename) {
    matfp_ = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (matfp_ == nullptr) {
        throw std::runtime_error("Error opening MAT file: " + filename);
    }
}

MatFile::~MatFile() {
    if (matfp_ != nullptr) {
        Mat_Close(matfp_);
    }
}

std::vector<std::string> MatFile::getVariableNames() {
    std::vector<std::string> names;
    matvar_t* matvar = nullptr;
    while ((matvar = Mat_VarReadNextInfo(matfp_)) != nullptr) {
        names.push_back(matvar->name);
        Mat_VarFree(matvar);
    }
    return names;
}

matvar_t* MatFile::readVariable(const std::string& varName) {
    return Mat_VarRead(matfp_, varName.c_str());
}
